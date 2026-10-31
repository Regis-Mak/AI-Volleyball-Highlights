"""The example memory: when Qwen tags a window, it also sees a few similar
moments from earlier games and how your group judged them ("looked like
this, Qwen called it a great play, the group said not worth showing").
Qwen then follows your group's standards without any retraining.

"Similar" means similar pictures. SigLIP, a small image model (downloaded
once into models/, about 800 MB), gives every second of footage a fingerprint, and a
window's fingerprint is the average over its seconds. The court, the light
and the camera angle look the same for a whole file, so each file's average
look is taken away first: what's left is what makes that stretch different
(a player on the ground, everyone at the net, a group hug).

Only moments you labelled count as examples (graded on the Grade screen, or
cut or trimmed in picks), never the test set and never the session being
tagged. Hints start once there are learning.examples.min_examples of them,
so from about the third game. Fingerprints are made on every run from the
start, so earlier games are ready when the hints switch on.

work/<file>/fingerprints.npy    one fingerprint per second (cached)
library: fingerprints table     one per window; kept when work/ is deleted
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..common import FramePipe, PipelineError, Progress, damaged_video, is_fresh, log, mark_done

if TYPE_CHECKING:
    from ..common import Game
    from ..library import Library
    from ..session import Session, SessionFile
    from ..vlm import Hints

USER_SOURCES = ("hand", "cut", "trim", "kept")
NEAR_S = 30.0           # two examples from one earlier file must be this far apart
BATCH = 64
LABEL_WORDS = {"great_play": "a great play", "blooper": "a blooper",
               "celebration": "a celebration", "chaos": "chaos",
               "rally": "an ordinary rally", "downtime": "nothing much happening",
               "conversation": "a conversation",
               "reaction": "a reaction off the court"}
INTRO = ("Moments from this group's earlier games that looked similar, with what was "
         "first said about them and how the group judged them. Use them to match the "
         "group's taste in labels and scores, but describe only what you see in these "
         "frames (names in them may not apply today):")


# ------------------------------------------------------------- fingerprints

def available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


class Embedder:
    """SigLIP's image half: pictures in, one unit-length fingerprint each."""

    def __init__(self, name: str, device: str = "auto", cache_dir: Path | None = None):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        import torch
        from transformers import SiglipVisionModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()      # "text weights not used" is expected
        hf_logging.disable_progress_bar()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = SiglipVisionModel.from_pretrained(name, cache_dir=cache_dir)
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.size = int(model.config.image_size)
        mean, std = [0.5] * 3, [0.5] * 3
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(name, use_fast=False,
                                                      cache_dir=cache_dir)
            mean, std = list(proc.image_mean), list(proc.image_std)
        except Exception:  # noqa: BLE001 - every SigLIP uses 0.5 / 0.5 anyway
            pass
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        """(n, size, size, 3) uint8 pictures -> (n, d) unit vectors."""
        x = rgb.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        x = (x - self.mean) / self.std
        with self.torch.no_grad():
            t = self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device, self.dtype)
            out = self.model(pixel_values=t).pooler_output.float().cpu().numpy()
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)

    def close(self) -> None:
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


def cache_plan(game: "Game", model: str) -> dict[str, tuple[list[Path], dict]]:
    """The fingerprints and what they depend on (nothing when the proxy is gone)."""
    proxy = game.work / "proxy.mp4"
    if not proxy.exists():
        return {}
    st = proxy.stat()
    return {"fingerprints": ([game.work / "fingerprints.npy"],
                             {"model": model, "proxy_size": st.st_size,
                              "proxy_mtime": int(st.st_mtime)})}


def per_second(game: "Game", duration: float, embedder: Callable[[], Embedder],
               model: str) -> np.ndarray | None:
    """One fingerprint per second of the file, from its 720p proxy (cached).
    None when the proxy is gone."""
    proxy = game.work / "proxy.mp4"
    out = game.work / "fingerprints.npy"
    plan = cache_plan(game, model)
    if not plan:
        return np.load(out) if out.exists() else None
    _, params = plan["fingerprints"]
    if is_fresh(game, "fingerprints", [out], params):
        return np.load(out)
    emb = embedder()
    n_sec = max(1, int(np.ceil(duration)))
    s = emb.size
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(proxy),
           "-vf", f"fps=1,scale={s}:{s}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    pipe = FramePipe(cmd)
    frame_bytes = s * s * 3
    vecs: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    try:
        with Progress("memory", f"picture fingerprints, {game.name}", n_sec) as bar:
            while True:
                buf = pipe.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                batch.append(np.frombuffer(buf, dtype=np.uint8).reshape(s, s, 3))
                if len(batch) == BATCH:
                    vecs.append(emb.embed(np.stack(batch)))
                    batch = []
                    bar.update(sum(len(v) for v in vecs))
            code, err = pipe.wait()
            if pipe.damaged:
                raise damaged_video(proxy)
            if batch:
                vecs.append(emb.embed(np.stack(batch)))
    except BaseException:              # Ctrl+C, a model error or a damaged copy
        pipe.stop()
        raise
    if code != 0 or not vecs:
        raise PipelineError(f"ffmpeg failed (fingerprints):\n{err.strip()[-2000:]}")
    arr = np.concatenate(vecs)[:n_sec]
    if len(arr) < n_sec:                    # the last second or so can be missing
        arr = np.concatenate([arr, np.repeat(arr[-1:], n_sec - len(arr), axis=0)])
    arr = arr.astype(np.float16)
    np.save(out, arr)
    mark_done(game, "fingerprints", params)
    return arr


def window_vectors(per_sec: np.ndarray,
                   spans: list[tuple[float, float]]) -> dict[float, np.ndarray]:
    """Each window's fingerprint, with the file's average look taken away."""
    base = per_sec.astype(np.float32)
    file_mean = base.mean(axis=0)
    out = {}
    for start, end in spans:
        a = min(int(start), len(base) - 1)
        seg = base[a: max(a + 1, int(np.ceil(end)))]
        v = seg.mean(axis=0) - file_mean
        out[start] = v / max(float(np.linalg.norm(v)), 1e-6)
    return out


# ----------------------------------------------------------------- examples

@dataclass
class Example:
    window: str            # library window id, "<file>@<start>"
    stem: str
    t: float               # key instant, seconds into its file
    description: str
    tagged: str            # what Qwen called it
    excitement: int
    funny: int
    label: str | None      # the label you confirmed or corrected (None: not checked)
    worth: int | None      # worth showing, by your labels
    is_funny: int | None

    def line(self) -> str:
        desc = " ".join(self.description.split())
        if len(desc) > 220:
            desc = desc[:217].rstrip() + "..."
        verdict = []
        if self.label:
            verdict.append(("actually " if self.label != self.tagged else "")
                           + LABEL_WORDS.get(self.label, self.label))
        if self.worth is not None:
            verdict.append("worth showing" if self.worth else "not worth showing")
        if self.is_funny is not None:
            verdict.append("funny" if self.is_funny else "not funny")
        return (f'- "{desc}" First tagged {self.tagged}, excitement {self.excitement}, '
                f"funny {self.funny}. The group: {', '.join(verdict)}.")


def verdict(rows: list, min_trust: float) -> int | None:
    """Your yes/no for one kind: a Grade-screen answer wins, otherwise what you
    did in picks (weighted by trust). None when nothing trusted enough says."""
    hand = [r for r in rows if r["source"] == "hand"]
    use = hand or rows
    if not use or max(r["trust"] for r in use) < min_trust:
        return None
    total = sum(r["trust"] for r in use)
    vote = sum(r["trust"] * float(json.loads(r["value"])) for r in use) / total
    return int(vote >= 0.5)


def _pool(lib: "Library", model: str, skip_stems: set[str]) -> tuple[list[Example], np.ndarray]:
    ec = lib.cfg.learning.examples
    gold = {r["moment"] for r in lib.q("SELECT DISTINCT moment FROM labels WHERE gold = 1")}
    marks = ",".join("?" * len(USER_SOURCES))
    labels: dict[str, dict[str, list]] = {}
    for r in lib.q(f"SELECT moment, kind, value, source, trust FROM labels WHERE source IN "
                   f"({marks}) AND kind IN ('highlight', 'funny', 'event_type')", USER_SOURCES):
        labels.setdefault(r["moment"], {}).setdefault(r["kind"], []).append(r)
    moments = {r["id"]: r for r in lib.q("SELECT id, stem, file_time, event_type, description "
                                         "FROM moments WHERE source = 'vlm'")}
    # The window each labelled moment came from (fuse names a moment after its
    # window's peak; when overlapping windows saw the same play, the moment
    # kept the description of the one that won).
    source: dict[str, object] = {}
    for w in lib.q("SELECT w.id, w.stem, w.start, w.peak_offset, w.event_type, w.excitement, "
                   "w.funny, w.description, f.vec FROM windows w JOIN fingerprints f "
                   "ON f.window = w.id WHERE f.model = ?", (model,)):
        if w["stem"] in skip_stems:
            continue
        mid = f"{w['stem']}@{w['start'] + w['peak_offset']:.0f}"
        if mid in gold or mid not in moments or mid not in labels:
            continue
        if mid not in source or w["description"] == moments[mid]["description"]:
            source[mid] = w
    pool: list[Example] = []
    vecs: list[np.ndarray] = []
    for mid, w in sorted(source.items()):
        lab = labels[mid]
        worth = verdict(lab.get("highlight", []), ec.min_trust)
        funny = verdict([r for r in lab.get("funny", []) if r["source"] == "hand"],
                         ec.min_trust)
        hand = [r for r in lab.get("event_type", []) if r["source"] == "hand"]
        graded = any(r["source"] == "hand" for r in lab.get("highlight", []))
        m = moments[mid]
        label = json.loads(hand[0]["value"]) if hand else (m["event_type"] if graded else None)
        if worth is None and label is None:
            continue
        pool.append(Example(w["id"], w["stem"], float(m["file_time"]),
                            m["description"] or w["description"] or "", w["event_type"],
                            int(w["excitement"]), int(w["funny"]), label, worth, funny))
        vecs.append(np.frombuffer(w["vec"], dtype=np.float16).astype(np.float32))
    if not pool:
        return [], np.zeros((0, 1), dtype=np.float32)
    dim = max(set(len(v) for v in vecs), key=[len(v) for v in vecs].count)
    keep = [i for i, v in enumerate(vecs) if len(v) == dim]
    return [pool[i] for i in keep], np.stack([vecs[i] for i in keep])


class Memory:
    """Similar labelled moments for each window of this session."""

    def __init__(self, lib: "Library", pool: list[Example], matrix: np.ndarray,
                 vectors: dict[str, dict[float, np.ndarray]]):
        self.cfg = lib.cfg.learning.examples
        self.pool, self.matrix, self.vectors = pool, matrix, vectors

    def similar(self, q: np.ndarray) -> list[tuple[float, Example]]:
        ec = self.cfg
        sims = self.matrix @ q
        chosen: list[tuple[float, Example]] = []
        for i in np.argsort(-sims):
            if sims[i] < ec.min_similarity or len(chosen) >= ec.count:
                break
            ex = self.pool[i]
            if any(c.stem == ex.stem and abs(c.t - ex.t) < NEAR_S for _, c in chosen):
                continue
            chosen.append((float(sims[i]), ex))
        return chosen

    def hints_for(self, f: "SessionFile") -> "Hints | None":
        vecs = {int(s): v for s, v in self.vectors.get(f.game.name, {}).items()}
        if not vecs:
            return None

        def hints(start: float, end: float) -> tuple[str, list[str]]:
            q = vecs.get(int(start))
            if q is None or len(q) != self.matrix.shape[1]:
                return "", []
            chosen = [ex for _, ex in self.similar(q)]
            if not chosen:
                return "", []
            text = "\n".join([INTRO] + [ex.line() for ex in chosen]) + "\n\n"
            return text, [ex.window for ex in chosen]

        return hints


# -------------------------------------------------------------------- setup

def _missing_stems(lib: "Library", model: str, skip: set[str]) -> list[str]:
    """Earlier files with labelled moments but no fingerprints yet (tagged
    before the example memory existed)."""
    have = lib.fingerprint_stems(model)
    rows = lib.q("SELECT DISTINCT m.stem FROM moments m JOIN labels l ON l.moment = m.id "
                 "WHERE l.gold = 0 AND l.source IN ('hand', 'cut', 'trim')")
    return sorted(r["stem"] for r in rows if r["stem"] not in have and r["stem"] not in skip)


def prepare(session: "Session", lib: "Library") -> Memory | None:
    """Fingerprint this session (and earlier labelled files that have none yet),
    then gather the examples. None when the hints are off this run."""
    from .. import vlm
    from ..common import Game

    ec = lib.cfg.learning.examples
    if not ec.enabled:
        return None
    if not available():
        log("[memory] warning: the example memory needs transformers (pip install -r "
            "requirements.txt); Qwen gets no hints from earlier games this run")
        return None
    model = ec.model
    holder: dict[str, Embedder] = {}

    def embedder() -> Embedder:
        if "e" not in holder:
            log(f"[memory] loading the picture model {model} (downloaded once, about 800 MB)")
            holder["e"] = Embedder(model, ec.device,
                                   lib.cfg.tracking.models_dir / "huggingface")
        return holder["e"]

    stems = {f.game.name for f in session.files}
    vectors: dict[str, dict[float, np.ndarray]] = {}
    try:
        for f in session.files:
            per_sec = per_second(f.game, f.info.duration, embedder, model)
            if per_sec is None:
                continue
            vectors[f.game.name] = window_vectors(per_sec,
                                                  vlm.window_grid(f.game.cfg, f.info.duration))
            lib.save_fingerprints(f.game.name, model, vectors[f.game.name])
        missing = _missing_stems(lib, model, stems)
        gone = 0
        for stem in missing:
            work = lib.cfg.work_dir / stem
            row = lib.q("SELECT path, duration FROM files WHERE stem = ?", (stem,))
            if not row or not (work / "proxy.mp4").exists():
                gone += 1
                continue
            game = Game(Path(row[0]["path"]), stem, work, lib.cfg.output_dir / stem, lib.cfg)
            per_sec = per_second(game, float(row[0]["duration"]), embedder, model)
            starts = lib.q("SELECT start, end_s FROM windows WHERE stem = ?", (stem,))
            if per_sec is not None and starts:
                lib.save_fingerprints(stem, model, window_vectors(
                    per_sec, [(r["start"], r["end_s"]) for r in starts]))
        if gone:
            log(f"[memory] {gone} earlier file{'s' if gone != 1 else ''} with labels can't "
                "give examples: the work folder (its proxy copy) was deleted")
    except PipelineError:
        raise
    except Exception as err:  # noqa: BLE001 - a model problem must not stop the run
        log(f"[memory] warning: picture fingerprints failed ({err}); Qwen gets no hints "
            "this run")
        return None
    finally:
        if "e" in holder:
            holder["e"].close()

    pool, matrix = _pool(lib, model, stems)
    if len(pool) < ec.min_examples:
        log(f"[memory] {len(pool)} of {ec.min_examples} labelled moments from earlier "
            f"sessions so far; Qwen gets hints from similar ones from {ec.min_examples} on")
        return None
    log(f"[memory] {len(pool)} labelled moments from earlier sessions; Qwen sees up to "
        f"{ec.count} similar ones with each new window")
    return Memory(lib, pool, matrix, vectors)


def status(lib: "Library") -> str:
    """One line for the Progress screen."""
    ec = lib.cfg.learning.examples
    if not ec.enabled:
        return "Switched off (learning.examples.enabled in config.yaml)."
    pool, _ = _pool(lib, ec.model, set())
    n = len(pool)
    if n < ec.min_examples:
        return (f"Not yet: {n} of {ec.min_examples} labelled moments so far. From then on, "
                "Qwen sees similar past moments and how you judged them while it tags.")
    return (f"On: {n} labelled moments to draw from. Qwen sees up to {ec.count} similar ones, "
            "and how you judged them, with each new window it tags.")
