"""Stage 6: select. Pick the moments for the highlight reel and each short,
write picks.csv + thumbnails so you can veto, then render every video.

output/<session>/
  <session> highlights.mp4                 the ~20 minute reel, in game order
  <session> highlights - timestamps.txt    what happens when in the reel
  <session> clip that <n>.mp4              a play someone said "clip that" about
  <session> short - <theme> <n>.mp4        30-60 s shorts (shorts.enabled), one file each
  picks.csv                                every clip in every video
  thumbs/<moment>.jpg                      a still of each picked moment

Vetoing: set keep to n on a row of picks.csv and run the same command again.
That clip leaves that video and the next best moment takes its place; the
row stays in picks.csv (keep = n) so you can undo it. Editing clip_start_s or
clip_end_s (seconds on the session clock, plain numbers so Excel leaves them
alone) changes that moment's clip everywhere it is used. Clips already
encoded are reused, so a re-run only encodes what changed.

Everything you do in picks.csv also teaches the ranker (see pipeline/library.py):
a cut says "not a highlight", a trim says "worth showing, with these edges", and
once you have edited the file, the clips you left in count as a weak "yes".

The app's Picks screen edits picks.csv the same way, and can also change the
order of a video's clips (work/<session>/order.json). Rendering adds the
nametags and vertical shorts (pipeline/polish.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from . import clipthat, polish
from .common import PipelineError, Progress, fmt_time, log
from .config import ThemeCfg
from .fuse import Moment
from .render import Clip
from .session import Session, SessionFile

if TYPE_CHECKING:
    from .library import Library

NO = {"n", "no", "x", "0", "false", "drop", "cut"}
REEL = "reel"
CLIP_THAT = "clip_that"
ORDER_FILE = "order.json"


@dataclass
class Item:
    """One clip in a video: a moment, or neighbouring moments merged."""
    start: float
    end: float
    moments: list[Moment]

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def t(self) -> float:
        return max(self.moments, key=lambda m: m.score).t


@dataclass
class Video:
    key: str                 # "reel", "bloopers", "player_alex", ...
    title: str
    path: Path
    items: list[Item] = field(default_factory=list)
    formats: tuple[str, ...] = ("landscape",)

    @property
    def length(self) -> float:
        return sum(i.length for i in self.items)

    def path_for(self, fmt: str) -> Path:
        """'... bloopers 1.mp4', plus '... bloopers 1 (vertical).mp4' when a
        short is made both ways."""
        if fmt == "vertical" and "landscape" in self.formats:
            return self.path.with_name(f"{self.path.stem} (vertical){self.path.suffix}")
        return self.path


@dataclass
class Feedback:
    """What you did in the last picks.csv."""
    vetoes: set[tuple[str, str]] = field(default_factory=set)
    edits: dict[str, tuple[float, float]] = field(default_factory=dict)
    kept: set[str] = field(default_factory=set)
    edited: bool = False                 # you changed the file since it was written


def run(session: Session, moments: list[Moment], render: bool = True,
        lib: "Library | None" = None) -> list[Video]:
    fb = _read_picks(session)
    vetoes, edits = fb.vetoes, fb.edits
    for m in moments:
        if m.id in edits:
            m.start, m.end = edits[m.id]
    if lib:
        _log_labels(session, lib, moments, fb)
    usable = [m for m in moments if not m.rejected]
    videos = [_pick_reel(session, usable, vetoes)]
    if session.cfg.shorts.enabled:
        videos += _pick_shorts(session, usable, vetoes)
    asked, extra = _pick_clip_that(session, usable, vetoes, edits)
    videos += asked
    videos = [v for v in videos if v.items]
    if not videos:
        raise PipelineError("No moments scored high enough for a reel. Check tags.csv and "
                            "moments.csv in the work folder, or lower reel.min_score.")
    _apply_order(session, videos)
    _thumbnails(session, videos)
    _write_picks(session, videos, moments + extra, vetoes)
    for v in videos:
        n = len(v.items)
        log(f"[select] {v.path.name}: {n} clip{'s' if n != 1 else ''}, {fmt_time(v.length)}")
    if render:
        try:
            for v in videos:
                _render(session, v)
        finally:
            polish.finish(session)
    else:
        log(f"[select] picks only: check {session.out / 'picks.csv'}, then run again "
            "without --pick-only to render")
    return videos


# --------------------------------------------------------------------- reel

def _pick_reel(session: Session, moments: list[Moment],
               vetoes: set[tuple[str, str]]) -> Video:
    rc, cc = session.cfg.reel, session.cfg.comments
    footage = session.footage
    target = min(rc.target_min * 60, rc.max_fraction * footage)
    ceiling = min(rc.max_min * 60, max(target, rc.max_fraction * footage))
    cands = [m for m in moments if (REEL, m.id) not in vetoes and m.score >= rc.min_score]
    cands.sort(key=lambda m: m.score, reverse=True)
    # Unhinged comments go in first, most unhinged first, up to their share.
    sure: list[Moment] = []
    if cc.enabled:
        room = cc.max_reel_share * target
        for m in sorted((m for m in moments if (REEL, m.id) not in vetoes
                         and m.unhinged >= cc.min_unhinged),
                        key=lambda m: (m.unhinged, m.score), reverse=True):
            if m.length <= room:
                sure.append(m)
                room -= m.length
        chosen = {m.id for m in sure}
        left = sum(m.unhinged >= cc.min_unhinged for m in cands if m.id not in chosen)
        if sure or left:
            log(f"[select] {len(sure)} unhinged comment{'s' if len(sure) != 1 else ''} "
                "sure of a place in the reel"
                + (f" ({left} more compete on score: over comments.max_reel_share)"
                   if left else ""))
        cands = sure + [m for m in cands if m.id not in chosen]

    items: list[Item] = []
    total = 0.0
    for m in cands:
        if total >= target:
            break
        near = next((i for i in items if m.start < i.end + rc.merge_gap_s
                     and m.end > i.start - rc.merge_gap_s
                     and session.file_at(i.t).index == m.file), None)
        if near is not None:
            s, e = min(near.start, m.start), max(near.end, m.end)
            if e - s <= rc.max_merged_s:
                total += (e - s) - near.length
                near.start, near.end = s, e
                near.moments.append(m)
                continue
            # Too long as one clip (two whole rallies back to back): keep this
            # one as its own clip, starting where the other ends.
            if m.t > near.t:
                s, e = max(m.start, near.end), m.end
            else:
                s, e = m.start, min(m.end, near.start)
            if (e - s < session.cfg.fuse.min_clip_s or total + e - s > ceiling
                    or any(s < i.end and e > i.start for i in items)):
                continue
            items.append(Item(s, e, [m]))
            total += e - s
            continue
        if total + m.length > ceiling:
            continue
        items.append(Item(m.start, m.end, [m]))
        total += m.length

    items = _merge_overlaps(items)
    if rc.order == "chronological":
        items.sort(key=lambda i: i.start)
    if total < min(rc.min_min * 60, target) - 1:
        log(f"[select] the reel is {fmt_time(total)}: only that much scored above "
            f"reel.min_score ({rc.min_score}) for {fmt_time(footage)} of footage")
    name = f"{session.name} highlights.mp4"
    return Video(REEL, "highlights", session.out / name, items)


def _merge_overlaps(items: list[Item]) -> list[Item]:
    out: list[Item] = []
    for i in sorted(items, key=lambda i: i.start):
        if out and i.start < out[-1].end:
            out[-1].end = max(out[-1].end, i.end)
            out[-1].moments += i.moments
        else:
            out.append(i)
    return out


# ---------------------------------------------------------------- clip that

def _pick_clip_that(session: Session, moments: list[Moment], vetoes: set[tuple[str, str]],
                    edits: dict[str, tuple[float, float]]) -> tuple[list[Video], list[Moment]]:
    """A video per "clip that", and the clip moments (vetoed ones too, so
    picks.csv keeps their rows)."""
    cc = session.cfg.clip_that
    videos: list[Video] = []
    made: list[Moment] = []
    for call in clipthat.find(session):
        f = session.files[call.file]
        m = clipthat.target(session, moments, call)
        end = call.t + cc.after_s
        start = min(m.start, call.t - 3.0) if m else call.t - cc.before_s
        end = max(end, m.end) if m else end
        if end - start > cc.max_s:
            start = end - cc.max_s
        if not f.joined_prev:
            start = max(start, f.offset)
        nxt = session.files[f.index + 1] if f.index + 1 < len(session.files) else None
        if not (nxt and nxt.joined_prev):
            end = min(end, f.offset + f.info.duration)
        if end - start < 2:
            continue
        said = f'Someone said "{call.text}"'
        cm = (replace(m, start=start, end=end, description=f"{said}: {m.description}",
                      players=list(m.players)) if m else
              Moment(id="", file=f.index, t=call.t, start=start, end=end,
                     event_type="reaction", description=said, source="audio"))
        cm.id = f"{f.game.video.stem}@{call.t - f.offset:.0f}+clip"
        cm.clip_that = call.text
        if cm.id in edits:
            cm.start, cm.end = edits[cm.id]
        made.append(cm)
        if (CLIP_THAT, cm.id) in vetoes:
            continue
        name = f"{session.name} clip that {len(videos) + 1}.mp4"
        videos.append(Video(CLIP_THAT, "clip that", session.out / name,
                            [Item(cm.start, cm.end, [cm])]))
    return videos, made


# ------------------------------------------------------------------- shorts

def _value(m: Moment, rank_by: str) -> float:
    return {"hype": m.hype, "fun": m.fun}.get(rank_by, m.score)


def _pick_shorts(session: Session, moments: list[Moment],
                 vetoes: set[tuple[str, str]]) -> list[Video]:
    sc = session.cfg.shorts
    formats = {"landscape": ("landscape",), "vertical": ("vertical",),
               "both": ("landscape", "vertical")}[session.cfg.vertical.shorts]
    themes = dict(sc.themes)
    videos = []
    made: set[frozenset[str]] = set()        # a short with the same clips as one made already
    for key, theme in themes.items():
        shorts = [c for c in _fill_shorts(session, moments, key, theme, vetoes)
                  if frozenset(m.id for m in c) not in made]
        made.update(frozenset(m.id for m in c) for c in shorts)
        for n, chosen in enumerate(shorts, 1):
            if sc.order == "chronological":
                chosen.sort(key=lambda m: m.t)
            name = f"{session.name} short - {theme.title} {n}.mp4"
            videos.append(Video(key, theme.title, session.out / name,
                                [Item(m.start, m.end, [m]) for m in chosen], formats))
    return videos


def _fill_shorts(session: Session, moments: list[Moment], key: str, theme: ThemeCfg,
                 vetoes: set[tuple[str, str]]) -> list[list[Moment]]:
    sc = session.cfg.shorts
    pool = [m for m in moments if not theme.labels or m.event_type in theme.labels]
    pool = [m for m in pool if (key, m.id) not in vetoes
            and _value(m, theme.rank_by) >= theme.min_value]
    pool.sort(key=lambda m: _value(m, theme.rank_by), reverse=True)

    shorts, used = [], set()
    for _ in range(sc.max_per_theme):
        chosen: list[Moment] = []
        total = 0.0
        for m in pool:
            if m.id in used or total + m.length > sc.max_s:
                continue
            if any(m.start < c.end and m.end > c.start for c in chosen):
                continue
            chosen.append(m)
            total += m.length
            if total >= sc.max_s - 5:
                break
        if len(chosen) < sc.min_moments or total < sc.min_s:
            break
        used.update(m.id for m in chosen)
        shorts.append(chosen)
    return shorts


# ------------------------------------------------------------------- render

def _clips(session: Session, item: Item) -> list[tuple[SessionFile, Clip]]:
    segs = session.segments(item.start, item.end, anchor=item.t)
    return [(f, Clip(a, b, video=f.game.video, info=f.info,
                     fade_in=(k == 0), fade_out=(k == len(segs) - 1)))
            for k, (f, a, b) in enumerate(segs)]


def _render(session: Session, video: Video) -> None:
    pieces = []
    for n, item in enumerate(video.items):
        for f, c in _clips(session, item):
            c.ref = n                    # for the reel's timestamps list
            pieces.append((f, c))
    polish.render_video(session, video, pieces)


def _apply_order(session: Session, videos: list[Video]) -> None:
    """Clip order you set on the app's Picks screen (moments not listed keep
    their place after the listed ones)."""
    order = _load_json(session.work / ORDER_FILE)
    for v in videos:
        wanted = order.get(v.key)
        if not wanted:
            continue
        rank = {mid: i for i, mid in enumerate(wanted)}
        big = len(rank)
        v.items.sort(key=lambda i: min((rank.get(m.id, big) for m in i.moments), default=big))


# ---------------------------------------------------------- picks + thumbs

def _thumb(session: Session, m: Moment) -> Path:
    return session.out / "thumbs" / (re.sub(r"[^\w@.-]+", "_", m.id) + ".jpg")


def _thumbnails(session: Session, videos: list[Video]) -> None:
    todo: dict[str, Moment] = {}
    for v in videos:
        for i in v.items:
            for m in i.moments:
                if not _thumb(session, m).exists():
                    todo[m.id] = m
    if not todo:
        return
    (session.out / "thumbs").mkdir(exist_ok=True)
    with Progress("select", "thumbnails", len(todo)) as bar:
        for n, m in enumerate(todo.values(), 1):
            f = session.files[m.file]
            src = f.game.work / "proxy.mp4"
            if not src.exists():
                src = f.game.video
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                            "-ss", f"{max(0.0, min(m.t - f.offset, f.info.duration - 0.5)):.2f}",
                            "-i", str(src),
                            "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4",
                            str(_thumb(session, m))], capture_output=True)
            bar.update(n)


COLUMNS = ["video", "theme", "order", "keep", "id", "session_time", "clock", "file",
           "file_time", "clip_start_s", "clip_end_s", "length_s", "event_type", "score",
           "hype", "fun", "players", "thumbnail", "description"]


def _row(session: Session, video: Video, order: int | str, keep: str, m: Moment) -> dict:
    f = session.files[m.file]
    return {
        "video": video.path.name, "theme": video.key, "order": order, "keep": keep,
        "id": m.id, "session_time": fmt_time(m.t), "clock": session.clock(m.t),
        "file": f.name, "file_time": fmt_time(m.t - f.offset),
        "clip_start_s": f"{m.start:.1f}", "clip_end_s": f"{m.end:.1f}",
        "length_s": f"{m.length:.1f}", "event_type": m.event_type,
        "score": f"{m.score:.2f}", "hype": f"{m.hype:.2f}", "fun": f"{m.fun:.2f}",
        "players": "; ".join(m.players),
        "thumbnail": f"thumbs/{_thumb(session, m).name}", "description": m.description,
    }


def _write_picks(session: Session, videos: list[Video], moments: list[Moment],
                 vetoes: set[tuple[str, str]]) -> None:
    by_id = {m.id: m for m in moments}
    path = session.out / "picks.csv"
    try:
        fh = open(path, "w", newline="", encoding="utf-8-sig")
    except PermissionError as err:
        raise PipelineError(f"Can't write {path}: close it in Excel and run again.") from err
    with fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for v in videos:
            for n, item in enumerate(v.items, 1):
                for m in item.moments:
                    w.writerow(_row(session, v, n, "y", m))
            for key, mid in sorted(vetoes):
                if key == v.key and mid in by_id:
                    w.writerow(_row(session, v, "-", "n", by_id[mid]))
    # What we wrote, so the next run can tell which bounds you changed and
    # whether you touched the file at all.
    (session.work / "picks_written.json").write_text(
        json.dumps({m.id: [round(m.start, 1), round(m.end, 1)] for m in moments}),
        encoding="utf-8")
    (session.work / "picks_hash.txt").write_text(_digest(path), encoding="utf-8")


def _digest(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read_picks(session: Session) -> Feedback:
    """Your edits from the last picks.csv: vetoed (theme, moment) pairs,
    changed clip bounds, and the clips you left in. Bound edits are kept in
    work/<session>/clip_edits.json so they survive later runs."""
    path = session.out / "picks.csv"
    edits_path = session.work / "clip_edits.json"
    written_path = session.work / "picks_written.json"
    fb = Feedback(edits={k: tuple(v) for k, v in _load_json(edits_path).items()})
    written = _load_json(written_path)
    if not path.exists():
        return fb
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error) as err:
        log(f"[select] warning: couldn't read {path.name} ({err}); ignoring it")
        return fb
    hash_path = session.work / "picks_hash.txt"
    fb.edited = hash_path.exists() and _digest(path) != hash_path.read_text(encoding="utf-8")
    for r in rows:
        mid, key = (r.get("id") or "").strip(), (r.get("theme") or "").strip()
        if not mid:
            continue
        if (r.get("keep") or "y").strip().lower() in NO and key:
            fb.vetoes.add((key, mid))
        else:
            fb.kept.add(mid)
        try:
            s, e = float(r.get("clip_start_s") or ""), float(r.get("clip_end_s") or "")
        except ValueError:
            continue
        was = written.get(mid)
        if e > s and was and (abs(s - was[0]) > 0.2 or abs(e - was[1]) > 0.2):
            fb.edits[mid] = (s, e)
    fb.kept -= {mid for _, mid in fb.vetoes}
    if fb.vetoes:
        log(f"[select] {len(fb.vetoes)} vetoed clips from picks.csv")
    if fb.edits:
        edits_path.write_text(json.dumps(fb.edits, indent=1), encoding="utf-8")
    return fb


def _log_labels(session: Session, lib: "Library", moments: list[Moment],
                fb: Feedback) -> None:
    """Your picks.csv edits as labels for the ranker."""
    by_id = {m.id: m for m in moments}
    cut = sorted({mid for _, mid in fb.vetoes if mid in by_id})
    lib.set_labels(session.name, "cut", [(mid, "highlight", 0) for mid in cut])
    trims = []
    for mid, (s, e) in sorted(fb.edits.items()):
        m = by_id.get(mid)
        if m is None or mid in cut:
            continue
        off = session.files[m.file].offset
        trims += [(mid, "bounds", [round(s - off, 1), round(e - off, 1)]),
                  (mid, "highlight", 1)]
    lib.set_labels(session.name, "trim", trims)
    if fb.edited:
        lib.set_labels(session.name, "kept",
                       [(mid, "highlight", 1) for mid in sorted(fb.kept) if mid in by_id])


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
