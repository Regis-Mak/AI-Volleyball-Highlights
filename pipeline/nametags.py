"""Minecraft-style nametags over each player for the first seconds of the reel.

1. Find and follow players (pipeline/track.py) in the clips that make up the
   reel's opening `nametags.seconds`.
2. Put names on them: the names you type on the app's Who is who screen,
   where everyone on screen while tags show is outlined and numbered. Nobody
   gets a tag until you name them. The tracker loses people for a moment
   (behind another player, at the net, out of the picture) and finds them
   again as someone "new": that someone takes the name of the one named
   person who could be them (not on screen just then, last seen close by,
   dressed alike), so you name each person once. Nothing is guessed from a
   list of shirt colours: people wear something different every week.
3. Choose above or below: above the head unless that's off the picture, too
   bright or busy to read (sky, the net), or on top of another tag. A tag only
   changes side when the other side stays clearly better for a moment.
4. Keep it smooth: positions are smoothed, tags fade in and out at the ends,
   and a tag hides while its player can't be seen.
5. Draw it: each tag is a small picture that ffmpeg moves frame by frame over
   the full-quality video (pipeline/render.py), so nothing is encoded twice.

work/<session>/tags/<clip>.jpg + .json   a still of each tagged clip and who's in it
work/<session>/tags/<clip>_p<n>.jpg      a frame of person n when they come in after that still
work/<session>/tags/sprites/             the tag pictures
work/<session>/who_is_who.json           your answers from the app
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from . import pixelfont
from .common import log
from .render import Sprite, TagLayer
from .track import Stretch, Track, tracker

if TYPE_CHECKING:
    from .session import Session, SessionFile

WHO_FILE = "who_is_who.json"

def _de(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Colour difference, going easy on brightness (shade and sun change it)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(math.sqrt((0.6 * (a[0] - b[0])) ** 2 + (a[1] - b[1]) ** 2
                           + (a[2] - b[2]) ** 2))


# ------------------------------------------------------------------ people

@dataclass
class Person:
    """One player through a clip (tracker fragments joined up)."""
    id: int
    t: np.ndarray
    box: np.ndarray            # n x 4, shares of the frame
    bg: np.ndarray             # n x 4
    torso: list[float] | None
    legs: list[float] | None
    parts: list[int] = field(default_factory=list)

    def box_at(self, t: float, within: float = 0.5) -> np.ndarray | None:
        i = int(np.argmin(np.abs(self.t - t)))
        return self.box[i] if abs(self.t[i] - t) <= within else None

    def visible_in(self, a: float, b: float) -> float:
        """Seconds on screen between file times a and b."""
        t = np.sort(self.t[(self.t >= a) & (self.t <= b)])
        gaps = np.diff(t)
        return float(gaps[gaps <= 0.6].sum())

    def first_in(self, a: float, b: float) -> float:
        t = self.t[(self.t >= a) & (self.t <= b)]
        return float(t.min()) if len(t) else float(self.t.min())

    def last_in(self, a: float, b: float) -> float:
        t = self.t[(self.t >= a) & (self.t <= b)]
        return float(t.max()) if len(t) else float(self.t.max())

    @property
    def height(self) -> float:
        return float(np.median(self.box[:, 3] - self.box[:, 1]))


def _person(tr: Track) -> Person:
    return Person(tr.id, np.array(tr.t, float), np.array(tr.box, float),
                  np.array(tr.bg, float), tr.torso, tr.legs, [tr.id])


def people(stretch: Stretch) -> list[Person]:
    """Join track fragments that are clearly the same person (lost for a
    moment behind someone, then picked up again with a new id)."""
    out: list[Person] = []
    for tr in sorted(stretch.tracks, key=lambda tr: tr.t[0]):
        p = _person(tr)
        best, best_d = None, 1e9
        for q in out:
            gap = p.t[0] - q.t[-1]
            if not 0 < gap <= 3.0:
                continue
            qa, pa = q.box[-1], p.box[0]
            w = max(qa[2] - qa[0], 1e-3)
            d = math.hypot((qa[0] + qa[2] - pa[0] - pa[2]) / 2, (qa[3] - pa[3])) / w
            if d > 1.0 + 0.75 * gap:
                continue
            if q.torso and p.torso and _de(q.torso, p.torso) > 25:
                continue
            if d < best_d:
                best, best_d = q, d
        if best is None:
            out.append(p)
        else:
            best.t = np.concatenate([best.t, p.t])
            best.box = np.concatenate([best.box, p.box])
            best.bg = np.concatenate([best.bg, p.bg])
            best.parts.append(tr.id)
            best.torso = best.torso or p.torso
            best.legs = best.legs or p.legs
    return out


# ------------------------------------------------------------------ follow

FOLLOW_GAP_S = 8.0        # longest someone can be lost and still be followed


def _continue_cost(seen: list[Person], p: Person) -> float | None:
    """How well p carries on from one person's other sightings (lower is
    better); None when p can't be them: on screen at the same time, too far
    from where they were last seen, a different size or shirt."""
    p0, p1 = float(p.t.min()), float(p.t.max())
    for q in seen:
        if p0 < q.t.max() - 0.3 and q.t.min() < p1 - 0.3:
            return None
    best = None
    for q in seen:
        if q.t.max() <= p0 + 0.3:            # q, then p
            gap = max(0.0, p0 - float(q.t.max()))
            qa, pa = q.box[int(np.argmax(q.t))], p.box[int(np.argmin(p.t))]
        elif q.t.min() >= p1 - 0.3:          # p, then q
            gap = max(0.0, float(q.t.min()) - p1)
            qa, pa = q.box[int(np.argmin(q.t))], p.box[int(np.argmax(p.t))]
        else:
            continue
        if gap > FOLLOW_GAP_S:
            continue
        hq, hp = max(qa[3] - qa[1], 1e-3), max(pa[3] - pa[1], 1e-3)
        # Feet to feet, in body heights; people cover about 2 a second.
        d = math.hypot((qa[0] + qa[2] - pa[0] - pa[2]) / 2, qa[3] - pa[3]) / max(hq, hp)
        reach = 1.0 + 2.0 * gap
        size = abs(math.log(hp / hq))
        if d > reach or size > 0.5:
            continue
        cost = d / reach + size
        if q.torso and p.torso:
            de = _de(q.torso, p.torso)
            if de > 30:
                continue
            cost += de / 30
        best = cost if best is None else min(best, cost)
    return best


def follow(crowd: list[Person], names: dict[int, "Naming"], a: float,
           b: float) -> tuple[dict[int, tuple[str, int]], dict[int, list[str]]]:
    """Name the people the tracker found again: ({person id: (name, id of the
    named person they carry on from)}, {person id: names they might be}).
    Only clear cases get a name; the rest keep their suggestions."""
    groups: dict[str, list[Person]] = {}
    root: dict[str, int] = {}
    for p in sorted(crowd, key=lambda p: p.t.min()):
        n = names.get(p.id)
        if n is not None and n.name:
            groups.setdefault(n.name, []).append(p)
            root.setdefault(n.name, p.id)
    todo = [p for p in sorted(crowd, key=lambda p: p.t.min())
            if p.id not in names and p.height >= 0.035 and p.visible_in(a, b) >= 0.5]
    out: dict[int, tuple[str, int]] = {}
    changed = True
    while changed:                     # someone followed can lead to the next
        changed = False
        for p in todo:
            if p.id in out:
                continue
            costs = sorted((c, nm) for nm, seen in groups.items()
                           if (c := _continue_cost(seen, p)) is not None)
            if costs and costs[0][0] <= 1.0 and (len(costs) == 1
                                                 or costs[1][0] - costs[0][0] >= 0.4):
                nm = costs[0][1]
                out[p.id] = (nm, root[nm])
                groups[nm].append(p)
                changed = True
    maybe = {}
    for p in todo:
        if p.id not in out:
            costs = sorted((c, nm) for nm, seen in groups.items()
                           if (c := _continue_cost(seen, p)) is not None and c <= 2.0)
            if costs:
                maybe[p.id] = [nm for _, nm in costs[:3]]
    return out, maybe


# ------------------------------------------------------------------- names

@dataclass
class Naming:
    name: str
    how: str          # you | follow
    sure: bool
    maybe: list[str] = field(default_factory=list)
    follows: int | None = None     # follow: the named person this one carries on from


def clip_key(f: "SessionFile", start: float, end: float) -> str:
    return f"{f.game.name}@{start:.1f}-{end:.1f}"


def _safe(key: str) -> str:
    return re.sub(r"[^\w@.-]+", "_", key)


def load_who(work: Path) -> dict:
    try:
        return json.loads((work / WHO_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_who(work: Path, data: dict) -> None:
    (work / WHO_FILE).write_text(json.dumps(data, indent=1), encoding="utf-8")


def _grab(src: Path, t: float, jpg: Path) -> None:
    """One 720p frame of the video, t seconds in."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                    "-vf", "scale=-2:720", "-q:v", "3", str(jpg)], capture_output=True)


def _iou(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Namer:
    """Who's who for one session: the names you typed, and the people the
    tracker lost and found again carrying on with them."""

    def __init__(self, session: "Session"):
        self.session = session
        self.answers = load_who(session.work)

    def name(self, key: str, people_: list[Person],
             window: tuple[float, float] | None = None) -> dict[int, Naming]:
        """window: the file times that get tags (only people on screen then
        are followed)."""
        lo, hi = window or (min((float(p.t.min()) for p in people_), default=0.0),
                            max((float(p.t.max()) for p in people_), default=0.0))
        names: dict[int, Naming] = {}

        # 1. Your answers for this clip (the newest wins if two are about one person).
        for a in reversed(self.answers.get(key, {}).get("people", [])):
            best = max(people_, default=None,
                       key=lambda p: _iou(p.box_at(a["t"], 1.0) if p.box_at(a["t"], 1.0)
                                          is not None else [0, 0, 0, 0], a["box"]))
            if best is None:
                continue
            b = best.box_at(a["t"], 1.0)
            if b is None or _iou(b, a["box"]) < 0.3 or best.id in names:
                continue
            names[best.id] = Naming(a.get("name", ""), "you", True)

        # 2. People the tracker lost and found again carry on with their name.
        found, maybe = follow(people_, names, lo, hi)
        for pid, (nm, root) in found.items():
            names[pid] = Naming(nm, "follow", True, follows=root)
        for pid, options in maybe.items():
            names.setdefault(pid, Naming("", "follow", False, options))
        return names


# --------------------------------------------------------------- placement

@dataclass
class Frame:
    """Where a tag should go: everything in output pixels."""
    w: int
    h: int
    fps: float
    to_px: Callable[[int, float, float], tuple[float, float]]   # frame, x, y (shares) -> px


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < 3 or n < 2:
        return x
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def _read_cost(lum: float, busy: float) -> float:
    """How hard white text on the tag is to read over this background."""
    if lum < 0:
        return math.inf
    return 2.5 * max(0.0, lum - 0.55) + 6.0 * busy


def place(group: list[tuple[Person, Sprite, tuple[int, int]]], times: np.ndarray,
          frame: Frame) -> None:
    """Fill in each sprite's per-frame position (None = hidden)."""
    n = len(times)
    gap = 0.008 * frame.h
    state = []
    for person, sprite, (tw, th) in group:
        order = np.argsort(person.t)
        t = person.t[order]
        box = person.box[order]
        bg = person.bg[order] if len(person.bg) else np.zeros((len(t), 4))
        span = max(1, int(round(0.5 * (len(t) / max(t[-1] - t[0], 1e-3)))))
        cx = _smooth((box[:, 0] + box[:, 2]) / 2, span)
        top = _smooth(box[:, 1], span)
        bot = _smooth(box[:, 3], span)
        ca = _smooth(np.array([_read_cost(a, b) for a, b in bg[:, :2]]), span * 2) - 0.15
        cb = _smooth(np.array([_read_cost(a, b) for a, b in bg[:, 2:]]), span * 2)
        idx = np.clip(np.searchsorted(t, times), 1, max(1, len(t) - 1))
        lo = np.clip(idx - 1, 0, len(t) - 1)
        near = np.minimum(np.abs(t[lo] - times), np.abs(t[np.minimum(idx, len(t) - 1)] - times))
        inside = (times >= t[0] - 0.15) & (times <= t[-1] + 0.15)
        # Coast across short gaps; hide during longer ones.
        seen = inside & (near <= 0.6)
        state.append({
            "sprite": sprite, "tw": tw, "th": th, "seen": _drop_blips(seen, frame.fps),
            "cx": np.interp(times, t, cx), "top": np.interp(times, t, top),
            "bot": np.interp(times, t, bot), "ca": np.interp(times, t, ca),
            "cb": np.interp(times, t, cb), "side": None, "since": 0, "y": None,
        })

    for i in range(n):
        placed: list[tuple[float, float, float, float]] = []
        for s in sorted(state, key=lambda s: s["top"][i]):
            sprite = s["sprite"]
            if not s["seen"][i]:
                sprite.xy.append(None)
                s["side"] = None
                continue
            x_px, top_px = frame.to_px(i, s["cx"][i], s["top"][i])
            _, bot_px = frame.to_px(i, s["cx"][i], s["bot"][i])
            tw, th = s["tw"], s["th"]
            x = min(max(0.0, x_px - tw / 2), frame.w - tw)
            if x_px < 0 or x_px > frame.w:
                sprite.xy.append(None)
                continue
            ys = {"above": top_px - gap - th, "below": bot_px + gap}
            cost = {}
            for side, y in ys.items():
                c = s["ca"][i] if side == "above" else s["cb"][i]
                if y < 0.01 * frame.h or y + th > 0.99 * frame.h:
                    c = math.inf
                rect = (x, y, x + tw, y + th)
                if any(r[0] < rect[2] and rect[0] < r[2] and r[1] < rect[3] and rect[1] < r[3]
                       for r in placed):
                    c += 1.0
                cost[side] = c
            side = s["side"]
            if side is None:
                side = min(cost, key=cost.get)
                s["since"] = 0
            else:
                other = "below" if side == "above" else "above"
                if math.isinf(cost[side]) and not math.isinf(cost[other]):
                    side, s["since"] = other, 0
                elif cost[other] < cost[side] - 0.25:
                    s["since"] += 1
                    if s["since"] >= 0.5 * frame.fps:
                        side, s["since"] = other, 0
                else:
                    s["since"] = 0
            target = ys[side]
            if s["side"] is not None and s["side"] != side and s["y"] is not None:
                s["glide"] = (s["y"], i)
            y = target
            glide = s.get("glide")
            if glide is not None:
                k = (i - glide[1]) / max(1.0, 0.25 * frame.fps)
                if k >= 1:
                    s["glide"] = None
                else:
                    y = glide[0] + (target - glide[0]) * (k * k * (3 - 2 * k))
            s["side"], s["y"] = side, y
            y = min(max(0.0, y), frame.h - th)
            placed.append((x, y, x + tw, y + th))
            sprite.xy.append((int(round(x)), int(round(y))))


def _one_each(group: list[tuple[Person, Sprite, tuple[int, int]]], fps: float) -> None:
    """A name shows on one person at a time: while two people carry the same
    name (the tracker swapped them, or the same name was typed twice), neither
    gets the tag."""
    by_name: dict[Path, list[Sprite]] = {}
    for _, sprite, _ in group:
        by_name.setdefault(sprite.png, []).append(sprite)
    for sprites in by_name.values():
        if len(sprites) < 2:
            continue
        n = min(len(s.xy) for s in sprites)
        both = np.array([sum(s.xy[i] is not None for s in sprites) > 1 for i in range(n)])
        for s in sprites:
            seen = np.array([xy is not None for xy in s.xy[:n]]) & ~both
            seen = _drop_blips(seen, fps)
            for i in range(n):
                if not seen[i]:
                    s.xy[i] = None


def _drop_blips(seen: np.ndarray, fps: float) -> np.ndarray:
    """Hide runs shorter than 0.4 s, so a tag never flashes."""
    out = seen.copy()
    n, i = len(seen), 0
    while i < n:
        if seen[i]:
            j = i
            while j < n and seen[j]:
                j += 1
            if j - i < 0.4 * fps:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


# ------------------------------------------------------------------ clips

@dataclass
class TagStats:
    clips: int = 0
    people: int = 0                 # everyone outlined on the Who is who screen
    named: int = 0                  # ...of whom you named this many
    shown: set[str] = field(default_factory=set)
    unsure: set[str] = field(default_factory=set)
    keys: list[str] = field(default_factory=list)


class Planner:
    """Plans the tags for one video's opening clips."""

    def __init__(self, session: "Session"):
        self.session = session
        self.cfg = session.cfg.nametags
        self.namer = Namer(session)
        self.stats = TagStats()
        self.tags_dir = session.work / "tags"

    def sprite(self, name: str, unit: int) -> tuple[Path, tuple[int, int]]:
        c = self.cfg
        style = f"{name}|{unit}|{c.box_opacity}|{c.text_color}|{c.font}"
        digest = hashlib.sha1(style.encode()).hexdigest()[:10]
        path = self.tags_dir / "sprites" / f"{_safe(name)}_{digest}.png"
        img = pixelfont.render_tag(name, unit, c.box_opacity, c.text_color, c.font)
        if not path.exists():
            pixelfont.save_png(img, path)
        return path, (img.shape[1], img.shape[0])

    def plan(self, f: "SessionFile", start: float, end: float, show_from: float,
             show_to: float, fade_in: bool, fade_out: bool, frame: Frame,
             reel_at: float = 0.0) -> TagLayer | None:
        """Tags for one clip (start/end: seconds into the file). show_from /
        show_to: the part of the clip that gets tags, in clip seconds.
        reel_at: where the clip starts in the video (for the app)."""
        tr = tracker(self.session.cfg)
        if tr is None:
            return None
        stretch = tr.track(f.game, start, end)
        key = clip_key(f, start, end)
        crowd = people(stretch)
        window = (start + show_from, start + min(show_to, end - start))
        names = self.namer.name(key, crowd, window)
        outlined, named = self._write_still(f, key, stretch, crowd, names, window, reel_at)
        self.stats.people += outlined
        self.stats.named += named
        self.stats.clips += 1
        self.stats.keys.append(key)
        for n in names.values():
            if not n.sure and n.maybe:
                self.stats.unsure.update(n.maybe)

        unit = pixelfont.unit_for(frame.h, self.cfg.text_height)
        n_frames = max(1, int(round((end - start) * frame.fps)))
        times = start + np.arange(n_frames) / frame.fps
        group = []
        for p in crowd:
            naming = names.get(p.id)
            if naming is None or not naming.name:
                continue
            png, size = self.sprite(naming.name, unit)
            group.append((p, Sprite(png, []), size))
            self.stats.shown.add(naming.name)
        if not group:
            return None
        place(group, times, frame)
        _one_each(group, frame.fps)
        for _, sprite, _ in group:
            for i in range(n_frames):
                if not show_from - 1e-6 <= i / frame.fps < show_to:
                    sprite.xy[i] = None
        sprites = [s for _, s, _ in group if any(xy is not None for xy in s.xy)]
        if not sprites:
            return None
        return TagLayer(sprites, show_from, show_to, fade_in, fade_out, self.cfg.fade_s)

    def _write_still(self, f: "SessionFile", key: str, stretch: Stretch,
                     crowd: list[Person], names: dict[int, Naming],
                     window: tuple[float, float], reel_at: float) -> tuple[int, int]:
        """For the Who is who screen: a still from the part of the clip that
        gets tags, with as many people in it as possible; everyone on screen
        while tags show, numbered (left to right in the still, then by when
        they come in), with a frame of each one who comes in later. Returns
        (people, named)."""
        a, b = window
        cands = [p for p in crowd if p.height >= 0.035 and p.visible_in(a, b) >= min(
                     self.cfg.min_visible_s, 0.5 * max(1e-3, b - a))]
        cands += [p for p in crowd if p not in cands and p.id in names    # named: gets a tag
                  and names[p.id].name and p.visible_in(a, b) >= 0.5]
        if not cands:
            return 0, 0
        times = np.unique(np.concatenate([p.t for p in cands]))
        times = times[(times >= a) & (times <= b)]
        if not len(times):
            return 0, 0
        counts = [sum(p.box_at(t, 0.05) is not None for p in cands) for t in times]
        t = float(times[int(np.argmax(counts))])
        self.tags_dir.mkdir(parents=True, exist_ok=True)
        jpg = self.tags_dir / f"{_safe(key)}.jpg"
        src = f.game.work / "proxy.mp4"
        if not src.exists():
            src = f.game.video
        _grab(src, t, jpg)
        for old in self.tags_dir.glob(f"{_safe(key)}_p*.jpg"):
            old.unlink(missing_ok=True)

        def order(p: Person) -> tuple[bool, float]:
            bx = p.box_at(t, 0.05)
            return (bx is None, float(bx[0] + bx[2]) if bx is not None else p.first_in(a, b))

        rows = []
        number: dict[int, int] = {}
        for n, p in enumerate(sorted(cands, key=order), 1):
            number[p.id] = n
            bx, pt, image = p.box_at(t, 0.05), t, jpg.name
            if bx is None:
                inside = np.flatnonzero((p.t >= a) & (p.t <= b))
                if not len(inside):
                    inside = np.arange(len(p.t))
                k = int(inside[np.argmax(p.box[inside, 3] - p.box[inside, 1])])  # biggest
                bx, pt = p.box[k], float(p.t[k])
                image = f"{_safe(key)}_p{n}.jpg"
                _grab(src, pt, self.tags_dir / image)
            nm = names.get(p.id)
            rows.append({"id": p.id, "n": n, "box": [round(float(v), 4) for v in bx],
                         "t": round(pt, 2), "image": image, "in_still": image == jpg.name,
                         "comes_in": round(reel_at + max(0.0, p.first_in(a, b) - stretch.start), 1),
                         "on": [round(reel_at + float(v) - stretch.start, 1) for v in
                                (p.first_in(a, b), p.last_in(a, b))],
                         "name": nm.name if nm else "", "how": nm.how if nm else "",
                         "maybe": nm.maybe if nm else [], "follows": nm.follows if nm else None})
        for r in rows:                      # who they carry on from, by number
            r["follows"] = number.get(r["follows"]) if r["follows"] is not None else None
        info = {"key": key, "file": f.name, "clip": [stretch.start, stretch.end],
                "window": [round(a, 2), round(b, 2)],
                "reel": [round(reel_at, 1), round(reel_at + b - stretch.start, 1)],
                "session_time": f.offset + t, "t": t, "image": jpg.name,
                "roster": list(self.session.cfg.players), "people": rows}
        (self.tags_dir / f"{_safe(key)}.json").write_text(json.dumps(info, indent=1),
                                                          encoding="utf-8")
        return len(rows), sum(1 for r in rows if r["name"])

    def report(self, what: str) -> None:
        s = self.stats
        if not s.clips:
            return
        shown = ", ".join(sorted(s.shown)) or "nobody yet"
        log(f"[tags] {what}: nametags on {s.clips} clip{'s' if s.clips != 1 else ''}: {shown}")
        if s.people > s.named:
            log(f"[tags] {s.people - s.named} of {s.people} people in those clips have no name "
                "yet: type them on the Who is who screen (python make_highlights.py --gui), "
                "then make the videos again")
        unsure = sorted(s.unsure - s.shown)
        if unsure:
            log(f"[tags] someone the camera lost for a moment might be {', '.join(unsure)}: "
                "name them on the Who is who screen (python make_highlights.py --gui), "
                "then run again")
