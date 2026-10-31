"""Vertical (9:16) shorts: a window that follows the action.

The window is about a third of the landscape frame's width (less when
vertical.zoom brings it closer), so it has to pick where to look. It aims at
the players who are moving (people standing still on the sideline count for
little) and at the ball when the detector sees it, then moves like a calm
camera operator: it holds still while the action stays inside the middle of
the window (vertical.dead_zone), pans no faster than vertical.max_pan, and
eases in and out of every move. With zoom above 1 it also follows the players
up and down, so the picture isn't mostly sky and grass.

Without Ultralytics installed, vertical shorts use a fixed centre crop.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .track import tracker

if TYPE_CHECKING:
    from .session import Session, SessionFile


def window(src_w: int, src_h: int, out_w: int, out_h: int,
           zoom: float = 1.0) -> tuple[int, int]:
    """The window's width and height in source pixels (even numbers)."""
    h = min(src_h, round(src_h / max(1.0, zoom) / 2) * 2)
    w = min(src_w, round(h * out_w / out_h / 2) * 2)
    return w, h


def _targets(stretch, n: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the action is (shares of the width and height) at each tracked
    frame; NaN = unknown."""
    pts: list[list[tuple[float, float, float]]] = [[] for _ in range(n)]
    for tr in stretch.tracks:
        t = np.array(tr.t)
        box = np.array(tr.box)
        if len(t) < 2:
            continue
        cx = (box[:, 0] + box[:, 2]) / 2
        cy = (box[:, 1] + box[:, 3]) / 2
        h = box[:, 3] - box[:, 1]
        speed = np.abs(np.gradient(cx, t))
        for ti, x, y, hi, sp in zip(t, cx, cy, h, speed):
            if hi < 0.035:
                continue           # far background
            k = int(round((ti - stretch.start) * fps))
            if 0 <= k < n:
                pts[k].append((x, y, (0.25 + 4 * min(sp, 0.5)) * min(1.0, hi / 0.15)))
    for t, x, y, conf in stretch.ball:
        k = int(round((t - stretch.start) * fps))
        if 0 <= k < n and conf >= 0.3:
            pts[k].append((x, y, 3.0 * conf))
    xs, ys = np.full(n, np.nan), np.full(n, np.nan)
    for k, p in enumerate(pts):
        if p:
            xs[k] = _weighted_median([(x, w) for x, _, w in p])
            ys[k] = _weighted_median([(y, w) for _, y, w in p])
    return xs, ys


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Robust to one stray person far from the rest."""
    pairs.sort()
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def _fill(x: np.ndarray, default: float) -> np.ndarray:
    ok = ~np.isnan(x)
    if not ok.any():
        return np.full_like(x, default)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _moving(x: np.ndarray, n: int, how: str = "mean") -> np.ndarray:
    if n < 2 or len(x) < 3:
        return x
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    return np.median(win, axis=1) if how == "median" else win.mean(axis=1)


def _operator(goal: np.ndarray, share: float, dead_zone: float, max_pan: float,
              fps: float, smooth_s: float) -> np.ndarray:
    """A calm camera operator: hold inside the dead zone, pan at a capped
    speed, ease every move. Returns the window's centre per frame."""
    half_zone = dead_zone * share / 2
    step_max = max_pan / fps
    cam = np.empty(len(goal))
    c = float(np.median(goal[: max(1, int(fps))]))
    for i, g in enumerate(goal):
        d = g - c
        if abs(d) > half_zone:
            c += float(np.clip(d - np.sign(d) * half_zone, -step_max, step_max))
        cam[i] = c
    k = max(1, int(smooth_s * fps))
    return _moving(_moving(cam, k), k)


def pan(session: "Session", f: "SessionFile", start: float,
        end: float) -> list[tuple[float, float]] | None:
    """The window's top-left corner (shares of the width and height) for each
    output frame, or None for a fixed centre crop."""
    cfg = session.cfg
    vc = cfg.vertical
    src_w, src_h = f.info.width, f.info.height
    cw, ch = window(src_w, src_h, vc.width, vc.height, vc.zoom)
    sw, sh = cw / src_w, ch / src_h
    if sw >= 0.999 and sh >= 0.999:
        return None
    tr = tracker(cfg)
    if tr is None:
        return None
    stretch = tr.track(f.game, start, end)
    fps_t = stretch.fps
    n_t = max(1, int(round((end - start) * fps_t)))
    tx, ty = _targets(stretch, n_t, fps_t)
    tx = _moving(_fill(tx, 0.5), max(1, int(fps_t)), "median")
    ty = _moving(_fill(ty, 0.5), max(1, int(2 * fps_t)), "median")

    out_fps = cfg.render.fps
    n = max(1, int(round((end - start) * out_fps)))
    tt = np.arange(n_t) / fps_t
    frames = np.arange(n) / out_fps
    cx = _operator(np.interp(frames, tt, tx), sw, vc.dead_zone, vc.max_pan, out_fps,
                   vc.smooth_s)
    # Up and down moves are rarer and gentler; players sit a little below
    # the window's middle so their heads and the ball have room.
    cy = _operator(np.interp(frames, tt, ty) - 0.08 * sh, sh, 0.5, vc.max_pan / 2,
                   out_fps, vc.smooth_s * 2)
    left = np.clip(cx - sw / 2, 0.0, 1.0 - sw)
    top = np.clip(cy - sh / 2, 0.0, 1.0 - sh)
    return [(round(float(a), 4), round(float(b), 4)) for a, b in zip(left, top)]
