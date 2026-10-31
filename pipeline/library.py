"""The library: everything the learning loop needs, kept across sessions.

library/                     next to work/ and output/; --force never touches it
  moments.sqlite             sessions, files, tagged windows, moments, labels,
                             picture fingerprints, ranker and settings versions
  frames/<file>/<start>/     the frames Qwen saw for each window (640 px JPEGs)
                             and prompt.txt, the exact prompt it got
  models/                    every ranker version (ranker_v1.json, ...) and every
                             tuned-settings version (settings_v1.json, ...)
  training/<date>/           training sets for fine-tuning Qwen (--export-training)
  evals/                     model checks on the test set (--evaluate)

A label says something about one moment ("worth showing: no", "funny: yes",
"really a blooper", "clip should be 12.0-26.5 s"). Each one keeps its source
and a trust weight, and a moment can collect labels from several sources:

  hand   1.0   your grade on the Grade screen (gold = part of the test set)
  cut    0.6   you set keep to n in picks.csv
  trim   0.5   you changed a clip's start or end in picks.csv
  kept   0.3   a clip you left in when you edited picks.csv
  agree  0.4   audio and vision clearly agree (made by the tool itself)

Weights live in config.yaml under learning.trust.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .common import log

if TYPE_CHECKING:
    from .config import Config
    from .fuse import Moment
    from .session import Session, SessionFile
    from .vlm import Window

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions (
    name TEXT PRIMARY KEY, folder TEXT, n_files INTEGER, footage_s REAL,
    roster TEXT, first_run TEXT, last_run TEXT);
CREATE TABLE IF NOT EXISTS files (
    stem TEXT PRIMARY KEY, session TEXT, path TEXT, duration REAL,
    offset_s REAL, wall_start REAL, width INTEGER, height INTEGER);
CREATE TABLE IF NOT EXISTS windows (
    id TEXT PRIMARY KEY, stem TEXT, start REAL, end_s REAL, event_type TEXT,
    play_active INTEGER, excitement INTEGER, funny INTEGER, peak_offset REAL,
    players TEXT, description TEXT, model TEXT, prompt TEXT,
    frame_offsets TEXT, n_frames INTEGER, updated TEXT, examples TEXT, extra TEXT);
CREATE TABLE IF NOT EXISTS moments (
    id TEXT PRIMARY KEY, session TEXT, stem TEXT, t REAL, file_time REAL,
    start REAL, end_s REAL, event_type TEXT, source TEXT, description TEXT,
    players TEXT, excitement INTEGER, funny INTEGER, features TEXT,
    base_score REAL, base_fun REAL, score REAL, hype REAL, fun REAL,
    model TEXT, prompt TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY, moment TEXT NOT NULL, kind TEXT NOT NULL,
    value TEXT NOT NULL, source TEXT NOT NULL, trust REAL NOT NULL,
    gold INTEGER NOT NULL DEFAULT 0, session TEXT, created TEXT,
    UNIQUE (moment, kind, source));
CREATE INDEX IF NOT EXISTS labels_session ON labels (session, source);
CREATE TABLE IF NOT EXISTS fingerprints (
    window TEXT PRIMARY KEY, stem TEXT, start REAL, model TEXT, vec BLOB, updated TEXT);
CREATE INDEX IF NOT EXISTS fingerprints_stem ON fingerprints (stem);
CREATE TABLE IF NOT EXISTS versions (
    name TEXT, kind TEXT, created TEXT, path TEXT, n_labels INTEGER,
    metric REAL, baseline REAL, checked_on TEXT, active INTEGER, note TEXT,
    PRIMARY KEY (kind, name));
"""
SCHEMA_VERSION = "1"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def open_library(cfg: "Config") -> "Library | None":
    """The library, or None when learning is switched off in config.yaml."""
    if not cfg.learning.enabled:
        return None
    return Library(cfg.learning.library_dir, cfg)


class Library:
    def __init__(self, root: Path, cfg: "Config"):
        self.root = root
        self.cfg = cfg
        root.mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(exist_ok=True)
        self.db = sqlite3.connect(root / "moments.sqlite", timeout=30,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (SCHEMA_VERSION,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, tuple(args)).fetchall()

    # ------------------------------------------------------------ sessions

    def record_session(self, s: "Session") -> None:
        roster = json.dumps(list(s.cfg.players))
        with self.db:
            self.db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                "folder=excluded.folder, n_files=excluded.n_files, footage_s=excluded.footage_s, "
                "roster=excluded.roster, last_run=excluded.last_run",
                (s.name, str(s.folder or ""), len(s.files), s.footage, roster, now(), now()))
            for f in s.files:
                self.db.execute(
                    "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)",
                    (f.game.name, s.name, str(f.game.video), f.info.duration, f.offset,
                     f.wall_start, f.info.width, f.info.height))

    # ------------------------------------------------------------- windows

    def frames_dir(self, stem: str, start: float) -> Path:
        return self.root / "frames" / stem / f"{int(start):05d}"

    def has_frames(self, stem: str, start: float) -> bool:
        d = self.frames_dir(stem, start)
        return d.is_dir() and any(d.glob("*.jpg"))

    def frame_files(self, stem: str, start: float) -> list[Path]:
        """The frames Qwen saw for one window, in the order it saw them."""
        return sorted(self.frames_dir(stem, start).glob("*.jpg"))

    def save_prompt(self, stem: str, start: float, text: str) -> None:
        d = self.frames_dir(stem, start)
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompt.txt").write_text(text, encoding="utf-8")

    def read_prompt(self, stem: str, start: float) -> str | None:
        path = self.frames_dir(stem, start) / "prompt.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def save_frames(self, stem: str, start: float, jpegs: list[bytes]) -> None:
        d = self.frames_dir(stem, start)
        d.mkdir(parents=True, exist_ok=True)
        for i, data in enumerate(jpegs):
            (d / f"{i:02d}.jpg").write_bytes(data)

    def record_windows(self, f: "SessionFile", windows: list["Window"], model: str,
                       prompt: str) -> None:
        rows = []
        for w in windows:
            if w.source != "vlm":
                continue
            t = w.tag
            n = len(list(self.frames_dir(f.game.name, w.start).glob("*.jpg")))
            rows.append((f"{f.game.name}@{int(w.start)}", f.game.name, w.start, w.end,
                         t.event_type, int(t.play_active), t.excitement, t.funny,
                         t.peak_offset_s, json.dumps(t.players), t.description, model,
                         prompt, json.dumps(w.frames), n, now(), json.dumps(w.examples),
                         json.dumps({"rally_start_s": t.rally_start_s,
                                     "rally_end_s": t.rally_end_s, "attack": t.attack})))
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO windows VALUES "
                                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    # -------------------------------------------------------- fingerprints

    def save_fingerprints(self, stem: str, model: str, vecs: dict[float, Any]) -> None:
        """Picture fingerprints (pipeline/learn/memory.py) by window start."""
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO fingerprints VALUES (?,?,?,?,?,?)",
                [(f"{stem}@{int(start)}", stem, float(start), model,
                  vec.astype("float16").tobytes(), now()) for start, vec in vecs.items()])

    def fingerprint_stems(self, model: str) -> set[str]:
        return {r["stem"] for r in self.q("SELECT DISTINCT stem FROM fingerprints "
                                          "WHERE model = ?", (model,))}

    # ------------------------------------------------------------- moments

    def record_moments(self, s: "Session", moments: list["Moment"], model: str,
                       prompt: str) -> None:
        rows = []
        for m in moments:
            f = s.files[m.file]
            rows.append((m.id, s.name, f.game.name, m.t, m.t - f.offset,
                         m.start - f.offset, m.end - f.offset, m.event_type, m.source,
                         m.description, json.dumps(m.players), m.excitement, m.funny,
                         json.dumps(m.features), m.base_score, m.base_fun, m.score,
                         m.hype, m.fun, model if m.source == "vlm" else "",
                         prompt if m.source == "vlm" else "", now()))
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO moments VALUES "
                                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def moments(self, session: str | None = None) -> list[sqlite3.Row]:
        if session:
            return self.q("SELECT * FROM moments WHERE session = ?", (session,))
        return self.q("SELECT * FROM moments")

    # -------------------------------------------------------------- labels

    def set_labels(self, session: str, source: str,
                   labels: list[tuple[str, str, Any]]) -> None:
        """Replace every label from one source for one session (used for labels
        re-derived on each run: picks.csv cuts, trims, keeps, and agreement)."""
        trust = getattr(self.cfg.learning.trust, source)
        with self.db:
            self.db.execute("DELETE FROM labels WHERE session = ? AND source = ?",
                            (session, source))
            self.db.executemany(
                "INSERT OR REPLACE INTO labels (moment, kind, value, source, trust, gold, "
                "session, created) VALUES (?,?,?,?,?,0,?,?)",
                [(mid, kind, json.dumps(value), source, trust, session, now())
                 for mid, kind, value in labels])

    def add_label(self, moment: str, kind: str, value: Any, source: str, *,
                  session: str, trust: float | None = None, gold: bool = False) -> None:
        if trust is None:
            trust = getattr(self.cfg.learning.trust, source)
        with self.db:
            self.db.execute(
                "INSERT INTO labels (moment, kind, value, source, trust, gold, session, created) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(moment, kind, source) DO UPDATE SET "
                "value=excluded.value, trust=excluded.trust, gold=excluded.gold, "
                "created=excluded.created",
                (moment, kind, json.dumps(value), source, trust, int(gold), session, now()))

    def remove_labels(self, moment: str, source: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM labels WHERE moment = ? AND source = ?",
                            (moment, source))

    def labels(self, kinds: Iterable[str] | None = None) -> list[sqlite3.Row]:
        if kinds is None:
            return self.q("SELECT * FROM labels")
        kinds = list(kinds)
        marks = ",".join("?" * len(kinds))
        return self.q(f"SELECT * FROM labels WHERE kind IN ({marks})", kinds)

    def hand_labels(self, session: str) -> dict[str, dict[str, Any]]:
        """Your Grade-screen answers for one session: {moment: {kind: value}}."""
        out: dict[str, dict[str, Any]] = {}
        for r in self.q("SELECT moment, kind, value FROM labels WHERE session = ? AND "
                        "source = 'hand'", (session,)):
            out.setdefault(r["moment"], {})[r["kind"]] = json.loads(r["value"])
        return out

    def counts(self) -> dict[str, int]:
        rows = self.q("SELECT source, gold, COUNT(DISTINCT moment) AS n FROM labels "
                      "GROUP BY source, gold")
        out: dict[str, int] = {}
        for r in rows:
            key = "gold" if r["gold"] else r["source"]
            out[key] = out.get(key, 0) + r["n"]
        return out

    # ------------------------------------------------------------ versions

    def versions(self, kind: str = "ranker") -> list[sqlite3.Row]:
        return self.q("SELECT * FROM versions WHERE kind = ? ORDER BY created, name", (kind,))

    def active_version(self, kind: str = "ranker") -> sqlite3.Row | None:
        rows = self.q("SELECT * FROM versions WHERE kind = ? AND active = 1", (kind,))
        return rows[0] if rows else None

    def next_version_name(self, kind: str = "ranker") -> str:
        nums = [int(r["name"][1:]) for r in self.versions(kind) if r["name"][1:].isdigit()]
        return f"v{max(nums, default=0) + 1}"

    def add_version(self, kind: str, name: str, path: Path, n_labels: int, metric: float | None,
                    baseline: float | None, checked_on: str, active: bool, note: str) -> None:
        with self.db:
            if active:
                self.db.execute("UPDATE versions SET active = 0 WHERE kind = ?", (kind,))
            self.db.execute("INSERT OR REPLACE INTO versions VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (name, kind, now(), str(path), n_labels, metric, baseline,
                             checked_on, int(active), note))

    def set_active(self, kind: str, name: str | None) -> None:
        with self.db:
            self.db.execute("UPDATE versions SET active = 0 WHERE kind = ?", (kind,))
            if name:
                self.db.execute("UPDATE versions SET active = 1 WHERE kind = ? AND name = ?",
                                (kind, name))


def summary(lib: Library) -> str:
    n_moments = lib.q("SELECT COUNT(*) AS n FROM moments")[0]["n"]
    n_sessions = lib.q("SELECT COUNT(*) AS n FROM sessions")[0]["n"]
    c = lib.counts()
    parts = [f"{c[k]} {name}" for k, name in
             (("hand", "graded by you"), ("gold", "in the test set"),
              ("cut", "cut in picks.csv"), ("trim", "trimmed"), ("kept", "kept"),
              ("agree", "auto-labelled")) if c.get(k)]
    return (f"{n_moments} moments from {n_sessions} "
            f"session{'s' if n_sessions != 1 else ''}"
            + (f"; labels: {', '.join(parts)}" if parts else "; no labels yet"))


def log_summary(lib: Library) -> None:
    log(f"[library] {summary(lib)}")
