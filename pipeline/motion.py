"""Stage 3: motion filter. Score movement for every second of the game so the
vision stage can skip idle stretches (water breaks, ball chasing, chatting).

Frames are read from the 720p proxy at a few per second, shrunk to 160x90
grayscale and differenced; the per-second score is the mean absolute pixel
change. Cheap enough for a full game in a minute or two.

work/<game>/
  motion.npy   per-second movement score
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import FramePipe, Game, PipelineError, Progress, damaged_video, is_fresh, mark_done

W, H = 160, 90


def cache_plan(game: Game) -> dict[str, tuple[list[Path], dict]]:
    """The motion scores and the settings they depend on."""
    return {"motion": ([game.work / "motion.npy"],
                       {**game.source_id(), "fps": game.cfg.motion.sample_fps,
                        "proxy": game.cfg.ingest.model_dump()})}


def run(game: Game, duration: float) -> np.ndarray:
    mc = game.cfg.motion
    (out,), params = cache_plan(game)["motion"]
    if not is_fresh(game, "motion", [out], params):
        np.save(out, _measure(game, duration, mc.sample_fps))
        mark_done(game, "motion", params)
    return np.load(out)


def _measure(game: Game, duration: float, fps: int) -> np.ndarray:
    n_sec = int(np.ceil(duration))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", str(game.work / "proxy.mp4"),
           "-vf", f"fps={fps},scale={W}:{H},format=gray",
           "-f", "rawvideo", "-"]
    pipe = FramePipe(cmd)
    frame_bytes = W * H
    sums = np.zeros(n_sec + 1, dtype=np.float64)
    counts = np.zeros(n_sec + 1, dtype=np.int64)
    prev = None
    i = 0
    try:
        with Progress("motion", "measuring movement", n_sec) as bar:
            while True:
                buf = pipe.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
                if prev is not None:
                    sec = min(int(i / fps), n_sec)
                    sums[sec] += np.abs(frame - prev).mean()
                    counts[sec] += 1
                prev = frame
                i += 1
                if i % (fps * 10) == 0:
                    bar.update(i / fps)
            code, err = pipe.wait()
            if pipe.damaged:
                raise damaged_video(game.work / "proxy.mp4")
    except BaseException:              # Ctrl+C or a damaged copy
        pipe.stop()
        raise
    if code != 0:
        raise PipelineError(f"ffmpeg failed (motion):\n{err.strip()[-2000:]}")
    return (sums / np.maximum(counts, 1))[:n_sec].astype(np.float32)


def window_is_active(game: Game, motion: np.ndarray, reaction: np.ndarray,
                     start: float, end: float) -> bool:
    """False for windows with little movement and no audible reaction."""
    mc = game.cfg.motion
    a, b = int(start), max(int(start) + 1, int(np.ceil(end)))
    seg = motion[a:b]
    if len(seg) == 0:
        return False
    typical = float(np.median(motion[motion > 0])) if np.any(motion > 0) else 0.0
    moving = float(np.percentile(seg, 75)) >= mc.idle_fraction * typical
    reacted = float(reaction[a:b].max(initial=0.0)) >= mc.keep_if_reaction
    return moving or reacted
