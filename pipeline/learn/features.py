"""The numbers the ranker sees for each moment.

All of them come from caches the pipeline already has (timeline.csv,
motion.npy, the vision tag), so adding a feature never re-runs a model.
"""
from __future__ import annotations

import numpy as np

EVENTS = ("great_play", "blooper", "celebration", "chaos", "conversation", "rally", "reaction")
AUDIO = ("laugh", "cheer", "shout", "loudness", "keyword", "speech_burst")
FEATURES = (["excitement", "funny", "rally", "attack", "unhinged", "play_active", "audio_only"]
            + [f"ev_{e}" for e in EVENTS]
            + ["reaction", *AUDIO, "wind", "motion_peak", "motion_mean",
               "clip_s", "n_players"])


def _span(x: np.ndarray, a: float, b: float) -> np.ndarray:
    i, j = max(0, int(a)), min(len(x), int(np.ceil(b)) + 1)
    return x[i:j] if j > i else np.zeros(1, dtype=np.float32)


def motion_reference(motion: np.ndarray) -> float:
    """Typical movement in this file, so motion features mean the same thing
    across cameras and courts (1.0 = an ordinary second of this file)."""
    moving = motion[motion > 0]
    return float(np.median(moving)) if len(moving) else 1.0


def compute(tl: dict[str, np.ndarray], motion: np.ndarray, motion_ref: float, lt: float,
            event_type: str, excitement: int, funny: int, play_active: bool,
            audio_only: bool, clip_s: float, n_players: int, lookback: float,
            lookahead: float, rally: float = 0.0, attack: int = 0,
            unhinged: float = 0.0) -> dict[str, float]:
    """Features of the moment whose key instant is lt seconds into its file.
    rally: fuse.rally_share of the rally's length (0-1); unhinged: the
    comment's 0-10."""
    a, b = lt - lookback, lt + lookahead
    f: dict[str, float] = {
        "excitement": excitement / 10, "funny": funny / 10,
        "rally": rally, "attack": attack / 10, "unhinged": unhinged / 10,
        "play_active": float(play_active), "audio_only": float(audio_only),
    }
    for e in EVENTS:
        f[f"ev_{e}"] = float(event_type == e)
    f["reaction"] = float(min(1.5, _span(tl["reaction"], a, b).max()))
    for k in AUDIO:
        f[k] = float(_span(tl[k], a, b).max())
    f["wind"] = float(_span(tl["wind"], a, b).mean())
    ref = max(motion_ref, 1e-6)
    around = _span(motion, lt - 6, lt + 2)
    f["motion_peak"] = float(min(5.0, around.max() / ref))
    f["motion_mean"] = float(min(5.0, around.mean() / ref))
    f["clip_s"] = float(clip_s)
    f["n_players"] = float(n_players)
    return f


def vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(k, 0.0)) for k in FEATURES]
