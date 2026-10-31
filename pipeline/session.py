"""Sessions: every video from one day of volleyball, treated as one timeline.

    python make_highlights.py "D:\\Volleyball\\06-14-2026"   (a folder)              
    python make_highlights.py game.mp4                                 (one file)

Files are put in recording order using the time in the camera's file name
(DJI_20260614181558_0007_D.MP4 = 14 Jun 2026, 6:15:58 PM), or the recording
date stored in the file when the name has none. They are laid out on one
session clock that keeps real gaps (a break between games stays a gap). When
one file starts exactly where the previous one ended, the camera split one
long recording, and the two are treated as continuous, so a rally across the
split still becomes one clip.

Each file keeps its own cache in work/<file name>/ (adding a file later only
processes that file). Session-wide results go to work/<session>/ and the
finished videos to output/<session>/.

An optional session.yaml in the folder lists who played that day (the names
the Who is who screen offers first):

    players: [Alex, Sam, Jordan]
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from . import ingest
from .common import Game, PipelineError, Probe, fmt_time, log
from .config import Config, Roster, Strict

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
STAMP_RE = re.compile(r"(?<!\d)(20\d{12})(?!\d)")     # 20260614181558
ROSTER_FILE = "session.yaml"


class SessionRoster(Strict):
    players: Roster


@dataclass
class SessionFile:
    index: int
    game: Game
    info: Probe
    wall_start: float | None      # seconds since the epoch, when known
    offset: float                 # where this file starts on the session clock
    joined_prev: bool             # continues the previous file with no gap

    @property
    def end(self) -> float:
        return self.offset + self.info.duration

    @property
    def name(self) -> str:
        return self.game.video.name


@dataclass
class Session:
    name: str
    folder: Path | None
    files: list[SessionFile]
    work: Path
    out: Path
    cfg: Config
    roster_from: str

    @property
    def duration(self) -> float:
        """Session clock length, gaps included."""
        return self.files[-1].end if self.files else 0.0

    @property
    def footage(self) -> float:
        """Recorded seconds, gaps excluded."""
        return sum(f.info.duration for f in self.files)

    def file_at(self, t: float) -> SessionFile:
        """The file playing at session time t (the nearest one inside a gap)."""
        for f in self.files:
            if f.offset <= t < f.end:
                return f
        return min(self.files, key=lambda f: min(abs(t - f.offset), abs(t - f.end)))

    def segments(self, start: float, end: float,
                 anchor: float | None = None) -> list[tuple[SessionFile, float, float]]:
        """Pieces of footage covering [start, end] on the session clock, as
        (file, local start, local end). Crosses into the next or previous file
        only where the camera split one recording; never bridges a real gap."""
        anchor = (start + end) / 2 if anchor is None else anchor
        i = self.file_at(anchor).index
        lo = i
        while lo > 0 and self.files[lo].joined_prev and start < self.files[lo].offset:
            lo -= 1
        hi = i
        while (hi + 1 < len(self.files) and self.files[hi + 1].joined_prev
               and end > self.files[hi].end):
            hi += 1
        out = []
        for f in self.files[lo:hi + 1]:
            a, b = max(start, f.offset), min(end, f.end)
            if b - a > 0.05:
                out.append((f, a - f.offset, b - f.offset))
        return out

    def clock(self, t: float) -> str:
        """Wall-clock time of session time t, e.g. '6:42 PM' ('' if unknown)."""
        f = self.file_at(t)
        if f.wall_start is None:
            return ""
        wall = f.wall_start + (t - f.offset)
        return datetime.fromtimestamp(wall).strftime("%I:%M %p").lstrip("0")


def open_session(path: Path, cfg: Config) -> Session:
    path = path.expanduser().resolve()
    if path.is_dir():
        videos = sorted(p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                        and not p.name.startswith("."))
        if not videos:
            raise PipelineError(f"No videos found in {path} "
                                f"(looked for {', '.join(sorted(VIDEO_EXTS))}).")
        name, folder = path.name, path
    elif path.is_file():
        videos, name, folder = [path], path.stem, None
    else:
        raise PipelineError(f"Video or folder not found: {path}")

    cfg, roster_from = _apply_roster(cfg, folder)
    work = cfg.work_dir / name
    out = cfg.output_dir / name
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    games = [Game.create(v, cfg, out=out) for v in videos]
    probes = [ingest.probe_game(g) for g in games]
    files = _lay_out(games, probes, cfg.session.join_tolerance_s)
    session = Session(name, folder, files, work, out, cfg, roster_from)
    _remember(session, path)
    _describe(session)
    return session


def _remember(s: Session, source: Path) -> None:
    """work/<session>/session.json: where the footage is and how the files
    line up, for the app's Run, Picks and Who is who screens."""
    data = {"name": s.name, "source": str(source),
            "files": [{"name": f.name, "stem": f.game.name, "offset": round(f.offset, 3),
                       "duration": round(f.info.duration, 3)} for f in s.files]}
    (s.work / "session.json").write_text(json.dumps(data, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ roster

def _apply_roster(cfg: Config, folder: Path | None) -> tuple[Config, str]:
    if folder is None or not (folder / ROSTER_FILE).exists():
        return cfg, "config.yaml"
    path = folder / ROSTER_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        roster = SessionRoster.model_validate(data)
    except (yaml.YAMLError, ValidationError) as err:
        raise PipelineError(f"{path} has a problem:\n{err}") from err
    return cfg.model_copy(update={"players": roster.players}), ROSTER_FILE


# ---------------------------------------------------------------- ordering

def _wall_from_name(path: Path) -> float | None:
    m = STAMP_RE.search(path.stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


def _wall_from_metadata(path: Path) -> float | None:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
           "-of", "json", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    try:
        stamp = json.loads(proc.stdout)["format"]["tags"]["creation_time"]
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except (ValueError, KeyError, TypeError):
        return None


def _wall_starts(games: list[Game]) -> list[float | None]:
    """Recording start of each file. Uses one source for every file (names or
    metadata) so the times are comparable; None everywhere if neither works."""
    by_name = [_wall_from_name(g.video) for g in games]
    if all(t is not None for t in by_name):
        return by_name
    by_meta = [_wall_from_metadata(g.video) for g in games]
    if all(t is not None for t in by_meta):
        return by_meta
    return [None] * len(games)


def _lay_out(games: list[Game], probes: list[Probe], tol: float) -> list[SessionFile]:
    walls = _wall_starts(games)
    order = sorted(range(len(games)),
                   key=lambda i: (walls[i] if walls[i] is not None else 0.0,
                                  games[i].video.name.lower()))
    files: list[SessionFile] = []
    for i in order:
        g, info, wall = games[i], probes[i], walls[i]
        if not files:
            files.append(SessionFile(0, g, info, wall, 0.0, False))
            continue
        prev = files[-1]
        joined, offset = False, prev.end
        if wall is not None and prev.wall_start is not None:
            gap = wall - (prev.wall_start + prev.info.duration)
            # Split chunks either start where the last one ended, or all carry
            # the recording's start time in their names.
            if abs(gap) <= tol or abs(wall - prev.wall_start) <= tol:
                joined = True
                wall = prev.wall_start + prev.info.duration
            elif gap > 0:
                offset = prev.end + gap
            else:
                log(f"[session] warning: {g.video.name} overlaps the previous file "
                    "(a second camera?); placing it right after.")
        files.append(SessionFile(len(files), g, info, wall, offset, joined))
    return files


def _describe(s: Session) -> None:
    if s.roster_from != "config.yaml":
        log(f"[session] roster: {len(s.cfg.players)} players from {s.roster_from}")
    if len(s.files) == 1:
        return
    log(f"[session] {s.name}: {len(s.files)} files, {s.footage / 3600:.1f} h of footage "
        f"over {s.duration / 3600:.1f} h; roster from {s.roster_from}")
    for f in s.files:
        note = "  continues the previous file" if f.joined_prev else ""
        clock = s.clock(f.offset)
        log(f"[session]   {f.index + 1}. {f.name}  {fmt_time(f.offset)}-{fmt_time(f.end)}"
            f"{'  (' + clock + ')' if clock else ''}{note}")
