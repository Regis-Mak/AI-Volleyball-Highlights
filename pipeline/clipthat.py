"""When someone says "clip that" (or "clip it", "someone clip that", ...), the
play they mean gets its own video, and counts more for the reel.

The words come from the Whisper transcript (word times), matched against
clip_that.phrases in config.yaml. The play is the best tagged moment that
peaked in the clip_that.lookback_s before the words; with none, the clip is
just the clip_that.before_s before them. Either way it runs on for
clip_that.after_s, so the "clip that!" is in it.

output/<session>/<session> clip that <n>.mp4   one per call, in game order
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import vlm
from .common import fmt_time

if TYPE_CHECKING:
    from .fuse import Moment
    from .session import Session


@dataclass
class Call:
    file: int          # index into session.files
    t: float           # session clock, when "clip" was said
    text: str          # what Whisper heard around it


def find(session: "Session") -> list[Call]:
    cc = session.cfg.clip_that
    if not cc.enabled:
        return []
    pats = [re.compile(p, re.IGNORECASE) for p in cc.phrases]
    calls: list[Call] = []
    for f in session.files:
        for seg in vlm._load_transcript(f.game):
            words = seg.get("words") or []
            text = "".join(w["word"] for w in words) or seg.get("text", "")
            offsets, n = [], 0
            for w in words:
                offsets.append((n, w["start"]))
                n += len(w["word"])
            for p in pats:
                for hit in p.finditer(text):
                    t = next((ws for off, ws in reversed(offsets) if off <= hit.start()),
                             seg.get("start", 0.0))
                    calls.append(Call(f.index, f.offset + float(t),
                                      " ".join(seg.get("text", text).split())[:120]))
    calls.sort(key=lambda c: c.t)
    out: list[Call] = []
    for c in calls:                  # "clip that! clip that!" is one call
        if out and out[-1].file == c.file and c.t - out[-1].t < cc.min_gap_s:
            continue
        out.append(c)
    return out


def target(session: "Session", moments: list["Moment"], call: Call) -> "Moment | None":
    """The play the call is about: the best moment that peaked just before it."""
    cc = session.cfg.clip_that
    near = [m for m in moments if m.file == call.file and m.event_type != "conversation"
            and call.t - cc.lookback_s <= m.t <= call.t + 2]
    return max(near, key=lambda m: m.score, default=None)


def describe(calls: list[Call]) -> str:
    return ", ".join(fmt_time(c.t) for c in calls)
