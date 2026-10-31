"""Player tracking, for the nametags and the vertical shorts.

Ultralytics YOLO finds every person (and the ball, when it can) in the 720p
analysis copy, and its built-in ByteTrack tracker follows each person from
frame to frame. Only the stretches that need it are tracked: the reel's
opening seconds (nametags) and the clips in shorts (vertical crop).

For each person it also notes the shirt and shorts colour they wear in that
clip (so someone the tracker loses and finds again can keep their name), and
how bright and busy the picture is just above their head and just below their
feet (for deciding where a nametag is easy to read).

work/<file>/tracks/<start>-<end>.json   one per stretch, reused on later runs
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .common import FramePipe, Game, PipelineError, Progress, damaged_video, fmt_time, log, probe
from .config import Config

VERSION = "1"            # bump when what a stretch file holds changes
PERSON, BALL = 0, 32     # COCO classes
TAG_W, TAG_H, GAP = 0.06, 0.03, 0.008   # the area a tag would cover, as shares of the frame


@dataclass
class Track:
    id: int
    t: list[float] = field(default_factory=list)          # seconds into the file
    box: list[list[float]] = field(default_factory=list)  # x1, y1, x2, y2 (shares of the frame)
    bg: list[list[float]] = field(default_factory=list)   # above: brightness, busy; below: same
    torso: list[float] | None = None                      # shirt colour, CIELAB
    legs: list[float] | None = None                       # shorts colour, CIELAB


@dataclass
class Stretch:
    stem: str
    start: float
    end: float
    fps: float
    tracks: list[Track]
    ball: list[list[float]]     # t, x, y, confidence


# --------------------------------------------------------------------- colour

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, ... x 3) to CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


# ------------------------------------------------------------------- tracker

class Tracker:
    """Loads YOLO on first use and tracks one stretch of footage at a time."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Any = None
        self.device: Any = None
        self._sizes: dict[str, tuple[int, int]] = {}

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("ultralytics") is not None

    def _load(self) -> None:
        if self.model is not None:
            return
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset
        tc = self.cfg.tracking
        weights = Path(tc.model)
        if not weights.is_absolute() and len(weights.parts) == 1:
            tc.models_dir.mkdir(parents=True, exist_ok=True)
            weights = tc.models_dir / weights
            if not weights.exists():
                log(f"[track] downloading {weights.name} (once)")
                attempt_download_asset(weights)
        self.model = YOLO(str(weights))
        dev = tc.device
        if dev == "auto":
            import torch
            dev = 0 if torch.cuda.is_available() else "cpu"
        self.device = 0 if dev == "cuda" else dev

    def close(self) -> None:
        if self.model is None:
            return
        self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -------------------------------------------------------------- caching

    def _source(self, game: Game) -> Path:
        proxy = game.work / "proxy.mp4"
        return proxy if proxy.exists() else game.video

    def _size(self, src: Path) -> tuple[int, int]:
        """Width and height of the frames analysed (at most 720 high)."""
        key = str(src)
        if key not in self._sizes:
            info = probe(src)
            h = min(720, info.height) // 2 * 2
            w = round(info.width * h / info.height / 2) * 2
            self._sizes[key] = (w, h)
        return self._sizes[key]

    def track(self, game: Game, start: float, end: float) -> Stretch:
        tc = self.cfg.tracking
        src = self._source(game)
        st = src.stat()
        key = hashlib.sha1(json.dumps({
            "v": VERSION, "src": str(src), "size": st.st_size, "mtime": int(st.st_mtime),
            "a": round(start, 2), "b": round(end, 2), "model": tc.model, "fps": tc.fps,
            "imgsz": tc.imgsz, "conf": tc.conf}, sort_keys=True).encode()).hexdigest()[:16]
        path = game.work / "tracks" / f"{int(start * 10):06d}-{int(end * 10):06d}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("key") == key:
                    return _from_json(data)
            except (ValueError, KeyError, TypeError):
                pass
        stretch = self._run(game, src, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, **asdict(stretch)}), encoding="utf-8")
        return stretch

    # ------------------------------------------------------------- tracking

    def _run(self, game: Game, src: Path, start: float, end: float) -> Stretch:
        self._load()
        tc = self.cfg.tracking
        w, h = self._size(src)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{max(0.0, start):.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
               "-vf", f"fps={tc.fps},scale={w}:{h},format=bgr24", "-f", "rawvideo", "-"]
        pipe = FramePipe(cmd)
        size = w * h * 3
        n_frames = max(1, int((end - start) * tc.fps))
        tracks: dict[int, Track] = {}
        colours: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        ball: list[list[float]] = []
        i = 0
        # A fresh tracker per stretch, so ids never carry over between clips.
        self.model.predictor = None
        try:
            with Progress("track", f"players at {fmt_time(start)} in {game.video.name}",
                          n_frames) as bar:
                while True:
                    buf = pipe.stdout.read(size)
                    if len(buf) < size:
                        break
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                    t = start + i / tc.fps
                    self._frame(frame, t, tracks, colours, ball)
                    i += 1
                    bar.update(min(i, n_frames))
        finally:
            pipe.stop()
            code, err = pipe.wait()
        if pipe.damaged:
            raise damaged_video(src)
        if code != 0 and i == 0:
            raise PipelineError(f"ffmpeg failed (tracking):\n{err.strip()[-2000:]}")
        for tid, samples in colours.items():
            if samples:
                tracks[tid].torso = np.median([s[0] for s in samples], axis=0).round(1).tolist()
                tracks[tid].legs = np.median([s[1] for s in samples], axis=0).round(1).tolist()
        return Stretch(game.name, start, end, tc.fps,
                       [tr for tr in tracks.values() if len(tr.t) >= 3], ball)

    def _frame(self, frame: np.ndarray, t: float, tracks: dict[int, Track],
               colours: dict[int, list], ball: list[list[float]]) -> None:
        tc = self.cfg.tracking
        res = self.model.track(frame, persist=True, classes=[PERSON, BALL], conf=tc.conf,
                               imgsz=tc.imgsz, tracker="bytetrack.yaml", verbose=False,
                               device=self.device)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return
        h, w = frame.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        people = [(k, xyxy[k]) for k in range(len(cls)) if cls[k] == PERSON]
        balls = [k for k in range(len(cls)) if cls[k] == BALL]
        if balls:
            k = max(balls, key=lambda k: conf[k])
            x1, y1, x2, y2 = xyxy[k]
            ball.append([round(t, 2), round(float(x1 + x2) / 2 / w, 4),
                         round(float(y1 + y2) / 2 / h, 4), round(float(conf[k]), 2)])
        if ids is None:
            return
        gray = frame.mean(axis=2) / 255.0
        for k, (x1, y1, x2, y2) in people:
            tid = int(ids[k])
            tr = tracks.setdefault(tid, Track(tid))
            tr.t.append(round(t, 2))
            tr.box.append([round(float(x1 / w), 4), round(float(y1 / h), 4),
                           round(float(x2 / w), 4), round(float(y2 / h), 4)])
            tr.bg.append(_background(gray, x1, y1, x2, y2))
            others = [b for j, b in people if j != k]
            if len(colours.get(tid, [])) < 25 and (y2 - y1) >= 40:
                c = _colours(frame, x1, y1, x2, y2, others)
                if c is not None:
                    colours.setdefault(tid, []).append(c)


def _region_stats(gray: np.ndarray, x1: float, y1: float, x2: float,
                  y2: float) -> list[float]:
    h, w = gray.shape
    if y1 < 0 or y2 > h:
        return [-1.0, 0.0]
    a, b = max(0, int(x1)), min(w, int(x2))
    patch = gray[int(y1):int(y2), a:b]
    if patch.size < 4:
        return [-1.0, 0.0]
    busy = (np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean()) / 2
    return [round(float(patch.mean()), 3), round(float(busy), 4)]


def _background(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Brightness and busyness of where a tag would sit above the head and
    below the feet."""
    h, w = gray.shape
    cx = (x1 + x2) / 2
    tw, th, gap = TAG_W * w, TAG_H * h, GAP * h
    above = _region_stats(gray, cx - tw / 2, y1 - gap - th, cx + tw / 2, y1 - gap)
    below = _region_stats(gray, cx - tw / 2, y2 + gap, cx + tw / 2, y2 + gap + th)
    return above + below


def _colours(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
             others: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Median shirt and shorts colour, skipped when someone else overlaps."""
    bw, bh = x2 - x1, y2 - y1
    torso = (x1 + 0.3 * bw, y1 + 0.2 * bh, x2 - 0.3 * bw, y1 + 0.45 * bh)
    legs = (x1 + 0.25 * bw, y1 + 0.5 * bh, x2 - 0.25 * bw, y1 + 0.62 * bh)
    for o in others:
        ox1, oy1, ox2, oy2 = o
        if ox1 < x2 and ox2 > x1 and oy1 < torso[3] and oy2 > torso[1]:
            return None
    out = []
    for rx1, ry1, rx2, ry2 in (torso, legs):
        patch = frame[int(ry1):int(ry2) + 1, int(rx1):int(rx2) + 1]
        if patch.size == 0:
            return None
        rgb = patch[..., ::-1].reshape(-1, 3)
        out.append(np.median(rgb_to_lab(rgb), axis=0))
    return out[0], out[1]


def _from_json(data: dict) -> Stretch:
    return Stretch(data["stem"], data["start"], data["end"], data["fps"],
                   [Track(**t) for t in data["tracks"]], data["ball"])


_shared: Tracker | None = None
_warned = False


def tracker(cfg: Config) -> Tracker | None:
    """The run's tracker, or None (with one warning) when Ultralytics isn't installed."""
    global _shared, _warned
    if _shared is None:
        if not Tracker.available():
            if not _warned:
                log("[track] warning: player tracking needs Ultralytics (pip install "
                    "ultralytics); nametags are skipped and vertical shorts use a centre crop")
                _warned = True
            return None
        _shared = Tracker(cfg)
    return _shared


def release() -> None:
    """Free the detector's GPU memory."""
    global _shared
    if _shared is not None:
        _shared.close()
        _shared = None
