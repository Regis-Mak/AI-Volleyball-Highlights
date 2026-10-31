"""Finishing touches added while a video renders (Phase 5):

  nametags   Minecraft-style name tags over each player for the first
             nametags.seconds of the reel (and of shorts, if nametags.shorts)
  vertical   9:16 versions of the shorts that follow the action
             (vertical.shorts: landscape | vertical | both)
  timestamps "<session> highlights - timestamps.txt": what happens when in
             the reel ("1:00 - 2:00  Long rally #1")

Both need player tracking (Ultralytics YOLO). Without it, videos render
exactly as before, with a centre crop for vertical shorts.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from . import track
from .common import fmt_time, log
from . import vertical as vertical_mod
from .nametags import Frame, Planner
from .render import Clip, clamp_clips, output_size, render_reel

if TYPE_CHECKING:
    from .selection import Item, Video
    from .session import Session, SessionFile

_tagged: list[str] = []


def render_video(session: "Session", video: "Video",
                 pieces: list[tuple["SessionFile", Clip]]) -> None:
    for fmt in video.formats:
        is_vertical = fmt == "vertical"
        clips = []
        for f, c in pieces:
            kept = clamp_clips([c], f.info)
            if kept:
                clips.append((f, replace(kept[0])))
        if is_vertical:
            for f, c in clips:
                c.pan = vertical_mod.pan(session, f, c.start, c.end)
        nc = session.cfg.nametags
        if nc.enabled and (video.key == "reel" or nc.shorts) and clips:
            _add_tags(session, video, clips, is_vertical)
        title = video.title + (" (vertical)" if is_vertical and len(video.formats) > 1 else "")
        timeline: list[tuple[int | None, float]] = []
        out = render_reel(session, [c for _, c in clips], video.path_for(fmt), what=title,
                          vertical=is_vertical, transitions=not is_vertical, timeline=timeline)
        if video.key == "reel" and not is_vertical:
            write_timestamps(session, video, timeline, out)


def _add_tags(session: "Session", video: "Video", clips: list[tuple["SessionFile", Clip]],
              is_vertical: bool) -> None:
    nc = session.cfg.nametags
    fps = session.cfg.render.fps
    w, h = output_size(session, [c for _, c in clips], is_vertical)
    planner = Planner(session)
    cum = 0.0
    for k, (f, c) in enumerate(clips):
        if cum >= nc.seconds:
            break
        dur = c.duration
        show_to = min(dur, nc.seconds - cum)
        last = k == len(clips) - 1 or cum + dur >= nc.seconds
        if is_vertical:
            cw, ch = vertical_mod.window(f.info.width, f.info.height, w, h,
                                         session.cfg.vertical.zoom)
            sw, sh = cw / f.info.width, ch / f.info.height
            corners = c.pan or [(0.5 - sw / 2, 0.5 - sh / 2)]

            def to_px(i, x, y, corners=corners, sw=sw, sh=sh):
                left, top = corners[min(i, len(corners) - 1)]
                return (x - left) / sw * w, (y - top) / sh * h
        else:
            def to_px(i, x, y):
                return x * w, y * h
        c.tags = planner.plan(f, c.start, c.end, 0.0, show_to, k == 0, last,
                              Frame(w, h, fps, to_px), reel_at=cum)
        cum += dur
    planner.report(video.path_for(video.formats[0]).name if not is_vertical
                   else f"{video.title} (vertical)")
    _tagged.extend(k for k in planner.stats.keys if k not in _tagged)


def write_timestamps(session: "Session", video: "Video",
                     timeline: list[tuple[int | None, float]], out: Path) -> Path | None:
    """One line per clip of the reel, where it sits in the finished video:
    "1:00 - 2:00  Long rally #1". A dissolve or swipe counts half to each side."""
    spans: dict[int, list[float]] = {}
    t = 0.0
    for ref, secs in timeline:
        if ref is not None:
            a, b = spans.setdefault(ref, [t, t])
            spans[ref] = [min(a, t), max(b, t + secs)]
        t += secs
    if not spans:
        return None
    counts: dict[str, int] = {}
    rows = []
    for ref in sorted(spans, key=lambda r: spans[r][0]):
        what = _title(session, video.items[ref])
        counts[what] = counts.get(what, 0) + 1
        a, b = spans[ref]
        rows.append((f"{fmt_time(round(a))} - {fmt_time(round(b))}",
                     f"{what} #{counts[what]}" + _quote(video.items[ref])))
    wide = max(len(r) for r, _ in rows)
    lines = [f"{out.name}  ({fmt_time(t)})", ""]
    lines += [f"{r:<{wide}}  {title}" for r, title in rows]
    path = out.with_name(f"{out.stem} - timestamps.txt")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[render] timestamps -> {path}")
    return path


_KINDS = {"great_play": "Great play", "rally": "Rally", "chaos": "Chaos",
          "blooper": "Blooper", "celebration": "Celebration", "conversation": "Convo"}
_PLAYS = ("great_play", "rally", "chaos")


def _title(session: "Session", item: "Item") -> str:
    """What a clip of the reel is, from its best moment: for a play, a big
    spike or a long rally when it has one."""
    m = max(item.moments, key=lambda m: m.score)
    if m.source == "comment":
        return "Unhinged comment"
    if m.event_type in _PLAYS:
        plays = [x for x in item.moments if x.event_type in _PLAYS]
        if max(x.attack for x in plays) >= 7:
            return "Big spike"
        if max(x.rally_s for x in plays) >= session.cfg.fuse.rally_long_s:
            return "Long rally"
    return _KINDS.get(m.event_type, m.event_type.replace("_", " ").capitalize() or "Moment")


def _quote(item: "Item") -> str:
    """For an unhinged comment, what was said."""
    m = max(item.moments, key=lambda m: m.score)
    said = " ".join(m.comment.split())
    if m.source != "comment" or not said:
        return ""
    if len(said) > 60:
        said = said[:57].rstrip() + "..."
    return f' - "{said}"'


def finish(session: "Session") -> None:
    """After every video: free the detector and tell the app which clips have tags."""
    track.release()
    if _tagged:
        d = session.work / "tags"
        d.mkdir(parents=True, exist_ok=True)
        (d / "current.json").write_text(json.dumps(_tagged), encoding="utf-8")
        _tagged.clear()
