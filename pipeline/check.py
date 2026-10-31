"""Setup check: everything a run needs, looked over before you start a long one.

    python make_highlights.py --check
    python make_highlights.py --check "D:\\Volleyball\\06-14-2026"

    ✓ fine    ! worth a look (runs still work)    ✗ needs fixing first

It checks the tools (ffmpeg and the GPU video encoder), that PyTorch sees
the graphics card, the Python packages, that Ollama is running with the
vision model and that the model answers a test picture from the GPU, which
models are downloaded already, free disk space, and what the library has
learned so far. Given a session folder or a video, it also lists the videos
in order and which steps are already done for each, so you can see what a
run will skip and what it will do.

Nothing is changed, apart from reading each video's length once (cached).
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from . import audio, ingest, motion, vlm
from .common import (PipelineError, fmt_time, is_fresh, log, log_status, nvenc_available,
                     on_windows)

if TYPE_CHECKING:
    from .config import Config

STAGE = "check"
MIN_FREE_GB, LOW_FREE_GB = 5, 30
TORCH_FIX = ("pip uninstall -y torch torchvision, then: pip install torch torchvision "
             "--index-url https://download.pytorch.org/whl/cu128")
# (import name, what needs it, needed for a plain run)
PACKAGES = [
    ("numpy", "everything", True),
    ("yaml", "reading config.yaml", True),
    ("pydantic", "config and model answers", True),
    ("rich", "colour output and progress bars", False),
    ("faster_whisper", "the transcript", True),
    ("panns_inference", "laughter and cheering detection", True),
    ("nicegui", "the app (--gui, --grade)", False),
    ("lightgbm", "the ranker from 400 labels on", False),
    ("ultralytics", "nametags and vertical shorts", False),
    ("lap", "nametags and vertical shorts", False),
    ("transformers", "the example memory", False),
]
STEP_NAMES = {"proxy": "720p copy", "audio_extract": "audio tracks", "transcript": "transcript",
              "panns": "laughter detection", "motion": "motion",
              "fingerprints": "picture fingerprints"}


class Report:
    def __init__(self) -> None:
        self.counts = {"ok": 0, "warn": 0, "fail": 0, "info": 0}

    def __call__(self, status: str, msg: str, fix: str = "") -> None:
        self.counts[status] += 1
        log_status(STAGE, status, msg + (f"\n{fix}" if fix else ""))


def run(cfg: "Config", source: Path | None = None) -> int:
    """Prints the check and returns how many things need fixing."""
    r = Report()
    log(f"[{STAGE}] Python and tools")
    _python(r)
    has_ffmpeg = _tools(r, cfg)
    log(f"[{STAGE}] GPU and packages")
    _gpu(r)
    _packages(r, cfg)
    _whisper_gpu(r, cfg)
    log(f"[{STAGE}] Models")
    _ollama(r, cfg, has_ffmpeg)
    _internet(r, _downloads(r, cfg))
    log(f"[{STAGE}] Disk space and library")
    _disk(r, cfg)
    _library(r, cfg)
    if source is not None:
        log(f"[{STAGE}] {source.name}")
        if has_ffmpeg:
            _session(r, cfg, source)
        else:
            r("fail", "Can't look at the videos without ffprobe (part of ffmpeg)")
    c = r.counts
    if c["fail"]:
        log_status(STAGE, "fail", f"{c['fail']} thing{'s' if c['fail'] != 1 else ''} to fix "
                   "before a run" + (f", {c['warn']} worth a look" if c["warn"] else ""))
    elif c["warn"]:
        log_status(STAGE, "warn", f"Ready to run; {c['warn']} thing"
                   f"{'s' if c['warn'] != 1 else ''} worth a look")
    else:
        log_status(STAGE, "ok", "All good: ready to run")
    return c["fail"]


# ------------------------------------------------------------------ basics

def _python(r: Report) -> None:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 10) <= (v.major, v.minor) < (3, 13):
        r("ok", f"Python {ver}")
    else:
        r("warn", f"Python {ver}; the tool is tested with 3.11",
          "Make the .venv with: py -3.11 -m venv .venv")
    if sys.prefix == sys.base_prefix:
        activate = (r".venv\Scripts\activate" if on_windows()
                    else "source .venv/bin/activate")
        r("warn", "Not running inside a virtual environment",
          f"If you made the .venv, run {activate} first")


def _tools(r: Report, cfg: "Config") -> bool:
    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            r("fail", f"{tool} not found",
              "On Windows: winget install Gyan.FFmpeg, then open a new terminal")
            ok = False
            continue
        out = subprocess.run([tool, "-version"], capture_output=True, text=True,
                             errors="replace").stdout.splitlines()
        r("ok", (out[0].split(" Copyright")[0] if out else tool))
    if not ok:
        return False
    enc = cfg.render.encoder
    if enc == "libx264":
        r("info", "Videos encode on the CPU (render.encoder is libx264)")
    elif nvenc_available():
        r("ok", "The GPU video encoder (NVENC) works")
    elif enc == "h264_nvenc":
        r("fail", "render.encoder is h264_nvenc, but the GPU video encoder doesn't work",
          "Update the NVIDIA driver, or set render.encoder: auto in config.yaml")
    else:
        r("warn", "The GPU video encoder (NVENC) doesn't work, so videos encode on the CPU "
          "(slower)", "Updating the NVIDIA driver usually fixes this")
    return True


def _gpu(r: Report) -> None:
    if importlib.util.find_spec("torch") is None:
        r("fail", "PyTorch isn't installed", TORCH_FIX)
        return
    import torch
    if not torch.cuda.is_available():
        build = " (the CPU-only build)" if torch.__version__.endswith("+cpu") else ""
        r("warn", f"PyTorch {torch.__version__}{build} can't see the GPU, so the transcript, "
          "laughter detection, picture fingerprints and player tracking run on the CPU "
          "(much slower)", f"Install the GPU build: {TORCH_FIX}")
        return
    free, total = torch.cuda.mem_get_info(0)
    r("ok", f"GPU: {torch.cuda.get_device_name(0)}, {total / 2**30:.0f} GB "
      f"({free / 2**30:.1f} GB free), PyTorch {torch.__version__}")
    if free < 6 * 2**30:
        r("warn", "Less than 6 GB of GPU memory is free right now",
          "Close games or other GPU-heavy programs before a run")


def _packages(r: Report, cfg: "Config") -> None:
    skip = set()
    if not cfg.audio.whisper.enabled:
        skip.add("faster_whisper")
    if not cfg.audio.panns.enabled:
        skip.add("panns_inference")
    have, missing = [], []
    for name, why, needed in PACKAGES:
        if name in skip:
            continue
        (have if importlib.util.find_spec(name) is not None else missing).append((name, why,
                                                                               needed))
    if have:
        r("ok", f"{len(have)} of {len(have) + len(missing)} packages installed")
    for name, why, needed in missing:
        r("fail" if needed else "warn", f"{name} is missing (needed for {why})",
          "pip install -r requirements.txt")


def _whisper_gpu(r: Report, cfg: "Config") -> None:
    wc = cfg.audio.whisper
    if not wc.enabled or wc.device != "cuda" or importlib.util.find_spec("ctranslate2") is None:
        return
    audio._add_windows_cuda_dlls()
    import ctranslate2
    missing = audio.missing_cuda_dll()
    if ctranslate2.get_cuda_device_count() == 0:
        r("warn", "Whisper can't see the GPU and will run on the CPU (slower)",
          "See README > Troubleshooting (cuBLAS / cuDNN)")
    elif missing:
        r("warn", f"Whisper can't load {missing} (it borrows it from PyTorch), so it will "
          "run on the CPU (much slower)", f"Reinstall the GPU build of PyTorch: {TORCH_FIX}")
    else:
        r("ok", "Whisper can use the GPU")


# ------------------------------------------------------------------ models

def _test_picture() -> bytes | None:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                           "-f", "lavfi", "-i", "testsrc2=s=640x360:d=1", "-frames:v", "1",
                           "-f", "image2pipe", "-c:v", "mjpeg", "-"], capture_output=True)
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


def _ollama(r: Report, cfg: "Config", has_ffmpeg: bool) -> None:
    vc = cfg.vlm
    try:
        client = vlm.OllamaClient(vc)
    except vlm.ModelMissing as err:
        r("warn", str(err))
        return
    except PipelineError as err:
        r("fail", str(err))
        return
    r("ok", f"Ollama {client.version()} is running and has {vc.model}")
    if "thinking" in client.capabilities():
        r("warn", f"{vc.model} is a thinking model: it reasons at length before every "
          "answer, which is much slower and can run past vlm.timeout_s",
          f"Set vlm.model to {vlm.instruct_name(vc.model)} in config.yaml")
    picture = _test_picture() if has_ffmpeg else None
    if picture is None:
        return
    t0 = time.time()
    try:
        answer = client.ask("This is a test picture. Name two colours you can see, in a "
                            "few words.", [picture], timeout=max(vc.timeout_s, 180))
        loaded = client.loaded()
    except PipelineError as err:
        r("fail", f"{vc.model} didn't answer a test picture: {err}")
        return
    finally:
        client.unload()
    took = time.time() - t0
    if not answer:
        r("fail", f"{vc.model} gave an empty answer to a test picture",
          f"Try: ollama pull {vc.model}")
        return
    short = " ".join(answer.split())
    short = short if len(short) <= 60 else short[:57] + "..."
    r("ok", f'{vc.model} answered a test picture in {took:.0f} s, loading included: "{short}"')
    if loaded and loaded.get("size"):
        size = float(loaded["size"])
        share = float(loaded.get("size_vram", 0)) / size
        if share >= 0.99:
            r("ok", f"It fits on the GPU ({size / 2**30:.1f} GB with its working memory)")
        else:
            r("warn", f"Only {share:.0%} of it ({size / 2**30:.1f} GB with its working memory) "
              "fits on the GPU; the rest runs on the CPU, which makes tagging much slower",
              "Close other GPU programs, lower vlm.num_ctx, or try a smaller model "
              "(qwen3-vl:4b-instruct)")


def _hf_cached(repo: str, filename: str, cache_dir: Path | None = None) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    found = try_to_load_from_cache(repo, filename,
                                   cache_dir=str(cache_dir) if cache_dir else None)
    return isinstance(found, str)


def _downloads(r: Report, cfg: "Config") -> list[str]:
    """Reports which models are downloaded; returns the hosts still needed."""
    hosts: list[str] = []
    wc = cfg.audio.whisper
    if wc.enabled and importlib.util.find_spec("faster_whisper") is not None:
        from faster_whisper import utils as fw_utils
        repo = getattr(fw_utils, "_MODELS", {}).get(wc.model)
        if Path(wc.model).is_dir():
            r("ok", f"Whisper model: {wc.model}")
        elif repo and _hf_cached(repo, "model.bin"):
            r("ok", f"Whisper {wc.model} is downloaded")
        elif repo:
            r("info", f"Whisper {wc.model} downloads on the first audio run (about 1.6 GB)")
            hosts.append("huggingface.co")
    if cfg.audio.panns.enabled:
        if audio.PANNS_CKPT.exists() and audio.PANNS_CKPT.stat().st_size >= 3e8:
            r("ok", "The laughter and cheering model (PANNs) is downloaded")
        else:
            r("info", "The laughter and cheering model (PANNs) downloads on the first audio "
              "run (about 330 MB)")
            hosts.append("zenodo.org")
    ec = cfg.learning.examples
    if cfg.learning.enabled and ec.enabled:
        if Path(ec.model).exists():
            r("ok", f"Picture model for the example memory: {ec.model}")
        elif _hf_cached(ec.model, "config.json", cfg.tracking.models_dir / "huggingface"):
            r("ok", "The SigLIP picture model (example memory) is downloaded")
        else:
            r("info", "The SigLIP picture model (example memory) downloads on the first "
              "vision run (about 800 MB)")
            hosts.append("huggingface.co")
    if cfg.nametags.enabled or cfg.vertical.shorts != "landscape":
        weights = Path(cfg.tracking.model)
        if not weights.is_absolute() and len(weights.parts) == 1:
            weights = cfg.tracking.models_dir / weights
        if weights.exists():
            r("ok", "The player tracker (YOLO) is downloaded")
        else:
            r("info", "The player tracker (YOLO) downloads with the first reel (about 20 MB)")
            hosts.append("github.com")
    return hosts


def _internet(r: Report, hosts: list[str]) -> None:
    for host in sorted(set(hosts)):
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://{host}", method="HEAD"),
                                   timeout=8)
        except urllib.error.HTTPError:
            pass                                  # an answer of any kind means it's reachable
        except (urllib.error.URLError, OSError):
            r("warn", f"Can't reach {host} to download the models above",
              "Check the internet connection before the first run")
            continue
        r("ok", f"Can reach {host} for the downloads")


# ------------------------------------------------------------ disk, library

def _disk(r: Report, cfg: "Config") -> None:
    drives: dict[str, tuple[float, list[str]]] = {}
    for label, path in (("work", cfg.work_dir), ("output", cfg.output_dir),
                        ("library", cfg.learning.library_dir)):
        p = Path(path).resolve()
        while not p.exists() and p.parent != p:
            p = p.parent
        key = p.anchor if on_windows() else str(p.stat().st_dev)
        free = shutil.disk_usage(p).free / 2**30
        drives.setdefault(key, (free, []))[1].append(label)
    for free, labels in drives.values():
        status = "fail" if free < MIN_FREE_GB else "warn" if free < LOW_FREE_GB else "ok"
        r(status, f"{free:.0f} GB free for {', '.join(labels)}",
          "" if status == "ok" else "A 4-hour session needs roughly 25 GB while it runs "
          "(the 720p copies, audio tracks, frames and videos)")


def _library(r: Report, cfg: "Config") -> None:
    if not cfg.learning.enabled:
        r("info", "Learning is switched off (learning.enabled)")
        return
    if not (cfg.learning.library_dir / "moments.sqlite").exists():
        r("info", "No library yet; it starts with the first full run")
        return
    from .learn import evaluate, memory, trainset, tune
    from .library import open_library, summary
    lib = open_library(cfg)
    if lib is None:
        return
    try:
        r("info", f"Library: {summary(lib)}")
        active = lib.active_version("ranker")
        r("info", "Ranker: " + (f"{active['name']} is on" if active else
                                f"not on yet (it starts at {cfg.learning.ranker.min_labels} "
                                "labelled moments)"))
        active = lib.active_version(tune.KIND)
        r("info", "Tuned settings: " + (f"{active['name']} is on" if active
                                        else "none yet; config.yaml as written"))
        r("info", f"Example memory: {memory.status(lib)}")
        r("info", f"Fine-tune (Phase 7): {trainset.status(lib)}")
        last = evaluate.latest(lib)
        if last:
            r("info", f"Last model check: {last}")
    finally:
        lib.close()


# ------------------------------------------------------------------ session

def _session(r: Report, cfg: "Config", source: Path) -> None:
    from .learn import memory
    from .session import open_session
    try:
        session = open_session(source, cfg)
    except PipelineError as err:
        r("fail", str(err))
        return
    ec = cfg.learning.examples
    fingerprints = (cfg.learning.enabled and ec.enabled
                    and importlib.util.find_spec("transformers") is not None)
    for f in session.files:
        g = f.game
        plan = {**ingest.cache_plan(g), **audio.cache_plan(g), **motion.cache_plan(g)}
        plan.pop("probe")
        if not cfg.audio.whisper.enabled:
            plan.pop("transcript")
        if not cfg.audio.panns.enabled:
            plan.pop("panns")
        done, todo, redo = [], [], []
        for stage, (outs, params) in plan.items():
            name = STEP_NAMES[stage]
            if is_fresh(g, stage, outs, params):
                done.append(name)
            elif (g.work / f"{stage}.done.json").exists():
                redo.append(name)              # made with other settings, or the video changed
            else:
                todo.append(name)
        if fingerprints:
            fp = memory.cache_plan(g, ec.model).get("fingerprints")
            (done if fp and is_fresh(g, "fingerprints", *fp) else todo).append(
                STEP_NAMES["fingerprints"])
        tagged, total = vlm.progress(g, f.info.duration)
        if total is None:
            vision = f"vision: {tagged} windows tagged so far" if tagged else "vision: to do"
        elif tagged >= total:
            vision = f"vision: all {total} windows tagged"
        else:
            vision = f"vision: {tagged} of {total} windows tagged"
        parts = []
        if done:
            parts.append("done: " + ", ".join(done))
        if redo:
            parts.append("redo (settings or the video changed): " + ", ".join(redo))
        if todo:
            parts.append("to do: " + ", ".join(todo))
        parts.append(vision)
        finished = not todo and not redo and total is not None and tagged >= total
        r("ok" if finished else "info",
          f"{f.index + 1}. {f.name} ({fmt_time(f.info.duration)})\n" + "\n".join(parts))
    log(f"[{STAGE}] The highlights go to {session.out}")
