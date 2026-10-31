"""What the Grade screen shows, in which order, and what each answer records.

Two modes:
  quick   the moments the tool is least sure about, about 10 per session.
          Optional; every answer is a training label.
  test    a spread of random moments across sessions and labels for the
          test set (learning.test_set.target, ~100). These are never trained
          on; they decide whether a new ranker version is better.

Answers (all labels are source "hand", trust 1.0):
  Y  worth showing       highlight = yes  (and funny = no, at half trust)
  F  funny               highlight = yes, funny = yes
  N  not worth showing   highlight = no, funny = no; left out of every video
  1-6                    the right label, if Qwen's is wrong
  arrows                 the clip's start / end; also used for the next render
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import ranker as ranker_mod

if TYPE_CHECKING:
    from ..library import Library

LABELS = [("great_play", "great play"), ("blooper", "blooper"),
          ("celebration", "celebration"), ("chaos", "chaos"), ("rally", "rally"),
          ("downtime", "nothing much"), ("conversation", "conversation")]
NEAR_S = 10.0           # a moment this close to one you graded counts as graded


@dataclass
class Card:
    moment: str
    session: str
    stem: str
    video: str                  # proxy path relative to the work folder
    duration: float
    offset: float               # where the file starts on the session clock
    t: float                    # key instant, seconds into the file
    start: float                # clip, seconds into the file
    end: float
    event_type: str
    description: str
    players: list[str]
    excitement: int
    funny: int
    reaction: float
    laugh: float
    why: str
    source: str = "vlm"                            # vlm | audio | comment
    window: tuple[float, float] | None = None      # what Qwen looked at, seconds into the file
    frames: list[float] = field(default_factory=list)   # ...and the frames it got
    label: str | None = None                       # your correction
    orig: tuple[float, float] = (0.0, 0.0)
    answer: str | None = None
    prev_edit: list[float] | None = field(default=None, repr=False)

    @property
    def bounds_changed(self) -> bool:
        return abs(self.start - self.orig[0]) > 0.05 or abs(self.end - self.orig[1]) > 0.05


# -------------------------------------------------------------------- queue

def _graded(lib: "Library") -> dict[str, list[float]]:
    """Per file, the times of moments you already graded."""
    out: dict[str, list[float]] = {}
    for r in lib.q("SELECT m.stem, m.file_time FROM moments m JOIN labels l "
                   "ON l.moment = m.id WHERE l.source = 'hand' GROUP BY m.id"):
        out.setdefault(r["stem"], []).append(r["file_time"])
    return out


def _candidates(lib: "Library", session: str | None) -> list[Any]:
    graded = _graded(lib)
    files = {r["stem"]: r for r in lib.q("SELECT * FROM files")}
    out = []
    for m in lib.moments(session):
        f = files.get(m["stem"])
        if f is None or not (lib.cfg.work_dir / m["stem"] / "proxy.mp4").exists():
            continue
        if any(abs(m["file_time"] - t) < NEAR_S for t in graded.get(m["stem"], [])):
            continue
        out.append((m, f))
    return out


def _windows(lib: "Library") -> dict[str, list[Any]]:
    """Every window Qwen tagged, per file."""
    out: dict[str, list[Any]] = {}
    for r in lib.q("SELECT stem, start, end_s, peak_offset, frame_offsets, description, "
                   "updated FROM windows"):
        out.setdefault(r["stem"], []).append(r)
    return out


def _qwen_window(wins: list[Any], t: float, description: str) -> Any | None:
    """The window Qwen described a moment from (a vision moment's key instant
    is that window's peak), else the tagged window centred nearest to it."""
    own = [w for w in wins if abs(w["start"] + (w["peak_offset"] or 0.0) - t) < 0.6
           and w["description"] and description.startswith(w["description"])]
    if own:
        return max(own, key=lambda w: w["updated"] or "")
    inside = [w for w in wins if w["start"] - 0.01 <= t <= w["end_s"] + 0.01]
    return min(inside, key=lambda w: abs((w["start"] + w["end_s"]) / 2 - t), default=None)


def _card(m: Any, f: Any, why: str, wins: dict[str, list[Any]] | None = None) -> Card:
    feats = json.loads(m["features"] or "{}")
    start = max(0.0, m["start"])
    end = min(f["duration"], m["end_s"])
    c = Card(m["id"], m["session"], m["stem"], f"{m['stem']}/proxy.mp4", f["duration"],
             f["offset_s"] or 0.0, m["file_time"], start, end, m["event_type"],
             m["description"] or "", json.loads(m["players"] or "[]"),
             m["excitement"] or 0, m["funny"] or 0, feats.get("reaction", 0.0),
             feats.get("laugh", 0.0), why, source=m["source"] or "vlm", orig=(start, end))
    w = _qwen_window((wins or {}).get(m["stem"], []), m["file_time"], c.description)
    if w is not None:
        a, b = max(0.0, w["start"]), min(f["duration"], w["end_s"])
        c.window = (a, b)
        c.frames = [round(a + o, 2) for o in json.loads(w["frame_offsets"] or "[]")
                    if a + o <= b]
    return c


def _cutoffs(lib: "Library", rows: list[Any]) -> dict[str, float]:
    """Per session, roughly the lowest score that still makes the reel: the
    moments near it are the ones where an answer changes the reel."""
    rc = lib.cfg.reel
    sessions = {r["name"]: r for r in lib.q("SELECT * FROM sessions")}
    by_session: dict[str, list[Any]] = {}
    for m, _ in rows:
        by_session.setdefault(m["session"], []).append(m)
    out = {}
    for name, ms in by_session.items():
        footage = sessions[name]["footage_s"] if name in sessions else 3600
        target = min(rc.target_min * 60, rc.max_fraction * footage)
        total, cut = 0.0, rc.min_score
        for m in sorted(ms, key=lambda m: m["score"] or 0, reverse=True):
            total += (m["end_s"] - m["start"])
            if total >= target:
                cut = max(rc.min_score, m["score"] or 0)
                break
        out[name] = cut
    return out


def _sureness(m: Any, feats: dict, p: float | None, cutoff: float) -> tuple[float, str]:
    """0 = no idea, 1 = sure; and why it's asking."""
    if p is not None:
        sure = abs(p - 0.5) * 2
        why = f"the ranker gives it a {p:.0%} chance of being worth showing"
    else:
        sure = min(1.0, abs((m["score"] or 0) - cutoff) / 0.2)
        why = "it's right at the edge of making the reel" if sure < 0.5 else \
            "a spot check"
    reaction = feats.get("reaction", 0.0)
    qwen = max(m["excitement"] or 0, m["funny"] or 0)
    if reaction >= 0.4 and qwen <= 4:
        return sure - 0.5, "the group reacted loudly, but the vision model rated it low"
    if reaction < 0.1 and qwen >= 7 and m["source"] == "vlm":
        return sure - 0.5, "the vision model rated it high, but nobody reacted"
    return sure, why


def build_queue(lib: "Library", mode: str, session: str | None = None) -> list[Card]:
    rows = _candidates(lib, session)
    if not rows:
        return []
    if mode == "test":
        return _test_queue(lib, rows)
    ranker = ranker_mod.load_active(lib)
    feats = [json.loads(m["features"] or "{}") for m, _ in rows]
    ps: list[float | None] = [None] * len(rows)
    if ranker is not None:
        ps = [float(x) for x in ranker.predict(feats)[0]]
    cut = _cutoffs(lib, rows)
    wins = _windows(lib)
    scored = []
    for (m, f), ft, p in zip(rows, feats, ps):
        sure, why = _sureness(m, ft, p, cut.get(m["session"], lib.cfg.reel.min_score))
        scored.append((sure, m, f, why, p))
    scored.sort(key=lambda x: x[0])
    per = lib.cfg.learning.grade.per_session
    order = [r["name"] for r in lib.q("SELECT name FROM sessions ORDER BY last_run DESC")]
    cards: list[Card] = []
    for name in order:
        picked = [x for x in scored if x[1]["session"] == name][:per]
        cards += [_card(m, f, why, wins) for _, m, f, why, _ in picked]
    return cards


def _test_queue(lib: "Library", rows: list[Any]) -> list[Card]:
    done, target = test_progress(lib)
    want = max(0, target - done)
    # A fixed shuffle (by id), then take labels and sessions in turn, so the
    # test set covers every kind of moment and every session.
    shuffled = sorted(rows, key=lambda r: hashlib.sha1(r[0]["id"].encode()).hexdigest())
    groups: dict[tuple[str, str], list[Any]] = {}
    for r in shuffled:
        groups.setdefault((r[0]["event_type"], r[0]["session"]), []).append(r)
    keys = sorted(groups)
    wins = _windows(lib)
    cards: list[Card] = []
    while len(cards) < want and any(groups.values()):
        for k in keys:
            if groups[k] and len(cards) < want:
                m, f = groups[k].pop(0)
                cards.append(_card(m, f, "a random pick for the test set", wins))
    return cards


def test_progress(lib: "Library") -> tuple[int, int]:
    done = lib.q("SELECT COUNT(DISTINCT moment) AS n FROM labels WHERE gold = 1 "
                 "AND kind = 'highlight'")[0]["n"]
    return done, lib.cfg.learning.test_set.target


# ------------------------------------------------------------------ answers

def _edits_path(lib: "Library", card: Card) -> Path:
    return lib.cfg.work_dir / card.session / "clip_edits.json"


def record(lib: "Library", card: Card, answer: str, gold: bool) -> None:
    """Save your answer (y, f or n) for a card."""
    undo(lib, card)
    card.answer = answer
    add = lambda kind, value, trust=None: lib.add_label(  # noqa: E731
        card.moment, kind, value, "hand", session=card.session, trust=trust, gold=gold)
    if answer == "f":
        add("highlight", 1)
        add("funny", 1)
    elif answer == "y":
        add("highlight", 1)
        add("funny", 0, lib.cfg.learning.trust.hand * 0.5)
    else:
        add("highlight", 0)
        add("funny", 0)
    if card.label and card.label != card.event_type:
        add("event_type", card.label)
    if card.bounds_changed:
        add("bounds", [round(card.start, 1), round(card.end, 1)])
        path = _edits_path(lib, card)
        edits = _load(path)
        card.prev_edit = edits.get(card.moment)
        edits[card.moment] = [round(card.start + card.offset, 1),
                              round(card.end + card.offset, 1)]
        _save(path, edits)


def undo(lib: "Library", card: Card) -> None:
    """Forget the answer for a card (Backspace on the Grade screen)."""
    if card.answer is None:
        return
    lib.remove_labels(card.moment, "hand")
    path = _edits_path(lib, card)
    edits = _load(path)
    if card.moment in edits:
        if card.prev_edit is None:
            del edits[card.moment]
        else:
            edits[card.moment] = card.prev_edit
        _save(path, edits)
    card.answer = None


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def stats(cards: list[Card]) -> dict[str, int]:
    return {a: sum(c.answer == a for c in cards) for a in ("y", "f", "n")}

