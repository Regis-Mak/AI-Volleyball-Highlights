"""Stage 2: audio signals -> per-second reaction timeline + top reaction peaks.

Friends' reactions are the main signal for both great plays and bloopers, so
this stage scores every second of the game on:
  laugh / cheer / shout   PANNs AudioSet tagger, max prob over class groups
  loudness                dB above a rolling-median baseline, discounted by
                          PANNs' wind probability so gusts don't count
  keyword                 transcript hits like "oh my god", "no way"
  speech_burst            words per second just after t, vs. the game's norm
and combines them with the weights in config.yaml.

work/<game>/
  transcript.json / transcript.txt   faster-whisper segments with word times
  panns.npz                          per-second AudioSet probabilities (all 527)
  timeline.csv                       every signal, every second
  peaks.csv                          top reaction moments, best first

Models run one at a time and are freed before the next loads. Raw model
outputs are cached separately from the scoring, so changing weights, class
groups or keywords re-scores in seconds without re-running any model.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import re
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import (AUDIO_ON_TIME, Game, Progress, fmt_time, is_fresh, log, mark_done,
                     on_windows)

PANNS_DIR = Path.home() / "panns_data"
PANNS_CKPT = PANNS_DIR / "Cnn14_mAP=0.431.pth"
PANNS_CKPT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_LABELS = PANNS_DIR / "class_labels_indices.csv"
PANNS_LABELS_URL = ("https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
                    "class_labels_indices.csv")
PANNS_SR = 32000

SIGNALS = ["laugh", "cheer", "shout", "loudness", "keyword", "speech_burst"]


@dataclass
class Peak:
    rank: int
    time: float           # seconds into the source video
    score: float
    dominant: str         # the signal contributing most at the peak
    signals: dict[str, float]
    transcript: str


def run(game: Game, duration: float) -> list[Peak]:
    n_sec = int(np.ceil(duration))
    words = _transcribe(game)
    probs = _panns(game)
    timeline = _build_timeline(game, n_sec, words, probs)
    peaks = _pick_peaks(game, timeline, words)
    _write_outputs(game, timeline, peaks)
    return peaks


# ------------------------------------------------------------ audio loading

def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data, sr


# --------------------------------------------------------------- transcript

def cache_plan(game: Game) -> dict[str, tuple[list[Path], dict]]:
    """The two model steps' outputs and the settings they depend on (the
    timeline is rebuilt from them on every run; it takes a moment)."""
    src = {**game.source_id(), "filter": game.cfg.ingest.audio_filter,
           "on_time": AUDIO_ON_TIME}
    return {
        "transcript": ([game.work / "transcript.json"],
                       {**src, **game.cfg.audio.whisper.model_dump()}),
        "panns": ([game.work / "panns.npz"],
                  {**src, "window_s": game.cfg.audio.panns.window_s}),
    }


def _transcribe(game: Game) -> list[dict]:
    """Returns a flat word list: [{start, end, word}] (empty if disabled)."""
    wc = game.cfg.audio.whisper
    (out_json,), params = cache_plan(game)["transcript"]
    if not wc.enabled:
        log("[audio] whisper disabled in config; no transcript signals")
        return []
    if not is_fresh(game, "transcript", [out_json], params):
        segments = _run_whisper(game)
        out_json.write_text(json.dumps(segments, indent=1), encoding="utf-8")
        with open(game.work / "transcript.txt", "w", encoding="utf-8") as f:
            for s in segments:
                f.write(f"[{fmt_time(s['start'])}] {s['text'].strip()}\n")
        mark_done(game, "transcript", params)
    else:
        log("[audio] transcript cached")
    segments = json.loads(out_json.read_text(encoding="utf-8"))
    return [w for s in segments for w in s["words"]]


def _add_windows_cuda_dlls() -> None:
    """faster-whisper (CTranslate2) needs cuBLAS + cuDNN 9 DLLs on Windows.
    Reuse the ones shipped with PyTorch or the nvidia-* pip packages."""
    if not on_windows():
        return
    dirs = []
    try:
        import torch  # noqa: F401  (loading torch also loads its CUDA DLLs)
        dirs.append(Path(torch.__file__).parent / "lib")
    except ImportError:
        pass
    try:
        import nvidia
        for root in nvidia.__path__:
            dirs += [p for p in Path(root).glob("*/bin")]
    except ImportError:
        pass
    for d in dirs:
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def missing_cuda_dll() -> str | None:
    """On Windows, the first GPU library Whisper needs that can't be loaded
    (None when they all load, and always None elsewhere). Checked before
    trying the GPU: a failed attempt can leave the next one hanging."""
    if not on_windows():
        return None
    import ctypes
    for name in ("cublas64_12.dll", "cudnn64_9.dll"):   # CTranslate2 4.x: CUDA 12, cuDNN 9
        try:
            ctypes.WinDLL(name, winmode=0)        # the usual search, PATH included
        except OSError:
            try:
                ctypes.WinDLL(name)                # the folders added with add_dll_directory
            except OSError:
                return name
    return None


_whisper_cuda_error: str | None = None     # once the GPU fails, later files go straight to CPU


def _run_whisper(game: Game) -> list[dict]:
    global _whisper_cuda_error
    wc = game.cfg.audio.whisper
    audio = str(game.work / "audio_16k.wav")
    attempts = [(wc.device, wc.compute_type)]
    if wc.device == "cuda":
        attempts.append(("cpu", "int8"))
    last_err: Exception | None = None
    for device, compute in attempts:
        if device == "cuda":
            _add_windows_cuda_dlls()
            missing = missing_cuda_dll()
            if missing and _whisper_cuda_error is None:
                _whisper_cuda_error = f"Library {missing} is not found or cannot be loaded"
            if _whisper_cuda_error is not None:
                log(f"[audio] warning: whisper can't use the GPU ({_whisper_cuda_error}). "
                    "Using the CPU, which is much slower. See README > Troubleshooting.")
                continue
        try:
            from faster_whisper import WhisperModel
            log(f"[audio] whisper {wc.model} on {device} ({compute})")
            model = WhisperModel(wc.model, device=device, compute_type=compute)
            try:
                return _whisper_segments(model, audio, wc)
            finally:
                del model
                gc.collect()
        except (RuntimeError, OSError, ValueError) as err:
            last_err = err
            if device == "cuda":
                _whisper_cuda_error = str(err)
                log(f"[audio] warning: whisper on CUDA failed ({err}). Falling back to CPU, "
                    "which is much slower. See README > Troubleshooting.")
    raise RuntimeError(f"faster-whisper failed: {last_err}")


def _whisper_segments(model, audio: str, wc) -> list[dict]:
    seg_iter, info = model.transcribe(
        audio, language=wc.language, vad_filter=wc.vad_filter,
        word_timestamps=True, beam_size=5,
        condition_on_previous_text=False,   # avoids repeated-text loops in noise
    )
    segments = []
    with Progress("audio", "transcribing", info.duration) as bar:
        for s in seg_iter:
            segments.append({
                "start": s.start, "end": s.end, "text": s.text,
                "words": [{"start": w.start, "end": w.end, "word": w.word}
                          for w in (s.words or [])],
            })
            bar.update(s.end)
    log(f"[audio] transcript: {len(segments)} segments")
    return segments


# -------------------------------------------------------------------- PANNs

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"[audio] downloading {dest.name} (one time)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, next_pct = 0, 10
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done * 100 // total >= next_pct:
                log(f"[audio]   {next_pct}%")
                next_pct += 10
    tmp.replace(dest)


def _ensure_panns_files() -> None:
    # panns_inference would fetch these with `wget`, which Windows lacks.
    if not PANNS_LABELS.exists():
        _download(PANNS_LABELS_URL, PANNS_LABELS)
    if not PANNS_CKPT.exists() or PANNS_CKPT.stat().st_size < 3e8:
        _download(PANNS_CKPT_URL, PANNS_CKPT)


def panns_labels() -> list[str]:
    with open(PANNS_LABELS, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [r[2] for r in rows[1:]]


def _panns(game: Game) -> np.ndarray | None:
    """Per-second AudioSet probabilities, shape (seconds, 527), or None."""
    pc = game.cfg.audio.panns
    (out,), params = cache_plan(game)["panns"]
    if not pc.enabled:
        log("[audio] PANNs disabled in config; no laugh/cheer/shout signals")
        return None
    if not is_fresh(game, "panns", [out], params):
        _ensure_panns_files()
        probs = _run_panns(game)
        np.savez_compressed(out, probs=probs.astype(np.float16))
        mark_done(game, "panns", params)
    else:
        log("[audio] PANNs scores cached")
    return np.load(out)["probs"].astype(np.float32)


def _run_panns(game: Game) -> np.ndarray:
    import torch
    from panns_inference.models import Cnn14

    pc = game.cfg.audio.panns
    device = "cuda" if pc.device == "cuda" and torch.cuda.is_available() else "cpu"
    if pc.device == "cuda" and device == "cpu":
        log("[audio] warning: PyTorch can't see the GPU; running PANNs on CPU. "
            "See README > Troubleshooting.")
    log(f"[audio] PANNs Cnn14 on {device}")
    model = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64,
                  fmin=50, fmax=14000, classes_num=527)
    ckpt = torch.load(PANNS_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    pcm, sr = _read_wav(game.work / "audio_32k.wav")
    assert sr == PANNS_SR, f"expected {PANNS_SR} Hz audio, got {sr}"
    n_sec = int(np.ceil(len(pcm) / sr))
    win = int(pc.window_s * sr)
    # Pad so the window for second t is centered on t + 0.5 s.
    pad = (win - sr) // 2
    padded = np.pad(pcm, (pad, pad + n_sec * sr - len(pcm) + win))

    probs = np.zeros((n_sec, 527), dtype=np.float32)
    try:
        with torch.inference_mode(), Progress("audio", "tagging sounds", n_sec) as bar:
            for b0 in range(0, n_sec, pc.batch_size):
                idx = range(b0, min(n_sec, b0 + pc.batch_size))
                batch = np.stack([padded[t * sr: t * sr + win] for t in idx])
                x = torch.from_numpy(batch.astype(np.float32) / 32768.0).to(device)
                probs[idx.start: idx.stop] = model(x)["clipwise_output"].cpu().numpy()
                bar.update(idx.stop)
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return probs


# ----------------------------------------------------------------- scoring

def _norm(x: np.ndarray, full_scale: float) -> np.ndarray:
    return np.clip(x / max(full_scale, 1e-9), 0.0, 1.0)


def _rolling_median(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(width, len(x)))
    half = width // 2
    padded = np.pad(x, (half, width - half - 1), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    return np.convolve(x, np.ones(width) / width, mode="same")


def _loudness_db(game: Game, n_sec: int) -> np.ndarray:
    pcm, sr = _read_wav(game.work / "audio_16k.wav")
    pcm = np.pad(pcm, (0, max(0, n_sec * sr - len(pcm))))[: n_sec * sr]
    frames = pcm.reshape(n_sec, sr).astype(np.float32) / 32768.0
    # Loudest 100 ms in each second: a shout or cheer is short, so a plain
    # per-second RMS would average it away.
    sub = frames.reshape(n_sec, 10, sr // 10)
    rms = np.sqrt(np.mean(sub ** 2, axis=2)).max(axis=1)
    return 20 * np.log10(rms + 1e-6)


def _normalize_word(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def _keyword_hits(words: list[dict], keywords: list[str], n_sec: int) -> np.ndarray:
    hits = np.zeros(n_sec, dtype=np.float32)
    tokens = [_normalize_word(w["word"]) for w in words]
    for phrase in keywords:
        ptoks = [_normalize_word(t) for t in phrase.split()]
        k = len(ptoks)
        for i in range(len(tokens) - k + 1):
            if tokens[i:i + k] == ptoks:
                a = int(words[i]["start"])
                b = int(words[i + k - 1]["end"]) + 1
                hits[max(0, a):min(n_sec, b)] = 1.0
    return hits


def _speech_burst(words: list[dict], n_sec: int, ahead_s: int = 4) -> np.ndarray:
    counts = np.zeros(n_sec + ahead_s, dtype=np.float32)
    for w in words:
        t = int(w["start"])
        if 0 <= t < n_sec:
            counts[t] += 1
    # Words spoken in [t, t + ahead_s): bursts of talk right after a play.
    rate = np.convolve(counts, np.ones(ahead_s), mode="full")[ahead_s - 1: ahead_s - 1 + n_sec]
    talky = rate[rate > 0]
    if len(talky) < 10:
        return np.zeros(n_sec, dtype=np.float32)
    lo, hi = np.percentile(talky, 50), np.percentile(talky, 99)
    return np.clip((rate - lo) / max(hi - lo, 1.0), 0.0, 1.0)


def _build_timeline(game: Game, n_sec: int, words: list[dict],
                    probs: np.ndarray | None) -> dict[str, np.ndarray]:
    ac = game.cfg.audio
    tl: dict[str, np.ndarray] = {"t": np.arange(n_sec, dtype=np.float32)}

    groups = {g: np.zeros(n_sec, dtype=np.float32) for g in ("laugh", "cheer", "shout", "wind")}
    if probs is not None:
        labels = panns_labels()
        index = {name: i for i, name in enumerate(labels)}
        p = np.zeros((n_sec, probs.shape[1]), dtype=np.float32)
        m = min(n_sec, len(probs))
        p[:m] = probs[:m]
        for group, names in ac.panns.classes.items():
            missing = [n for n in names if n not in index]
            if missing:
                log(f"[audio] warning: unknown AudioSet classes in '{group}': {missing}")
            cols = [index[n] for n in names if n in index]
            if cols:
                raw = p[:, cols].max(axis=1)
                if group == "wind":
                    groups[group] = raw
                else:
                    # Scale so this game's strongest reactions reach ~1, but a
                    # game with no real laughter doesn't blow noise up to 1.
                    full = max(float(np.percentile(raw, 99.5)), ac.panns.min_full_scale)
                    groups[group] = _norm(raw, full)
    tl.update(groups)

    db = _loudness_db(game, n_sec)
    excess = db - _rolling_median(db, ac.loudness.baseline_s)
    tl["loudness_db"] = db
    tl["loudness"] = _norm(excess, ac.loudness.full_scale_db) * (1.0 - groups["wind"])

    tl["keyword"] = _keyword_hits(words, ac.keywords, n_sec)
    tl["speech_burst"] = _speech_burst(words, n_sec)

    total_w = sum(ac.weights.get(s, 0.0) for s in SIGNALS) or 1.0
    raw = sum(ac.weights.get(s, 0.0) * tl[s] for s in SIGNALS) / total_w
    tl["reaction_raw"] = raw
    tl["reaction"] = _smooth(raw, ac.smooth_s)
    return tl


def _pick_peaks(game: Game, tl: dict[str, np.ndarray], words: list[dict]) -> list[Peak]:
    pc = game.cfg.audio.peaks
    weights = game.cfg.audio.weights
    score = tl["reaction"]
    order = np.argsort(-score, kind="stable")
    chosen: list[int] = []
    for i in order:
        if score[i] <= 0 or len(chosen) >= pc.top_n:
            break
        if all(abs(int(i) - c) >= pc.min_separation_s for c in chosen):
            chosen.append(int(i))

    peaks = []
    for rank, i in enumerate(chosen, 1):
        lo, hi = max(0, i - 2), i + 3
        sig = {s: float(tl[s][lo:hi].max()) for s in SIGNALS}
        dominant = max(SIGNALS, key=lambda s: weights.get(s, 0.0) * sig[s])
        said = " ".join(w["word"].strip() for w in words if i - 6 <= w["start"] <= i + 4)
        peaks.append(Peak(rank, i + 0.5, float(score[i]), dominant, sig, said))
    return peaks


def _write_outputs(game: Game, tl: dict[str, np.ndarray], peaks: list[Peak]) -> None:
    cols = ["t", "reaction", "reaction_raw", "laugh", "cheer", "shout", "wind",
            "loudness", "loudness_db", "keyword", "speech_burst"]
    with open(game.work / "timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in zip(*(tl[c] for c in cols)):
            w.writerow([f"{v:.3f}" for v in row])

    with open(game.work / "peaks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "timestamp", "time_s", "score", "dominant", *SIGNALS, "transcript"])
        for p in peaks:
            w.writerow([p.rank, fmt_time(p.time), f"{p.time:.1f}", f"{p.score:.3f}",
                        p.dominant, *(f"{p.signals[s]:.2f}" for s in SIGNALS), p.transcript])
    log(f"[audio] {len(peaks)} reaction peaks -> {game.work / 'peaks.csv'}")
    for p in peaks[:10]:
        log(f"[audio]   #{p.rank:<2} {fmt_time(p.time):>8}  {p.score:.2f}  {p.dominant:<12} "
            f"{p.transcript[:60]}")
