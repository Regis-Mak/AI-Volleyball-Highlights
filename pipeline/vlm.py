"""Stage 4: vision tagging. Qwen3-VL (via a local Ollama server) watches each
24 s window and labels it.

For every window the model gets frames, not video: ~1 fps across the window
plus a 4 fps burst over the few seconds before each audio reaction peak
(that's where the play that caused the laughter or cheering happened). The
prompt also carries that window's transcript, the audio reactions and the
player list. Answers are validated with pydantic.

Once the library has enough labelled moments, the prompt also gets a few
similar moments from earlier games with how your group judged them (the
example memory, pipeline/learn/memory.py). Windows already tagged keep their
answer; only new ones get the hints.

The exact prompt each window got is kept with its answer, so the library can
later turn graded windows into training examples (Phase 7).

work/<game>/
  vlm/<start>.json   one cached answer (and its prompt) per window; an
                     interrupted run resumes
  tags.csv           every window's labels, in time order
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from . import motion as motion_mod
from .common import Game, PipelineError, Progress, fmt_time, is_fresh, log
from .config import Config, VlmCfg

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "tag_window.txt"
LAUGH_SOME, LAUGH_CLEAR = 0.2, 0.5    # PANNs laughter, 0-1: what the prompt calls a little / clear
EVENT_TYPES = ("great_play", "blooper", "celebration", "chaos", "conversation", "rally",
               "downtime")


class WindowTag(BaseModel):
    # Qwen fills these in order: what it sees first, then the label. The rally
    # and attack fields default to 0 for library rows saved before they existed.
    description: str
    play_active: bool
    rally_start_s: float = 0.0        # the longest rally, seconds into the window
    rally_end_s: float = 0.0
    attack: int = Field(default=0, ge=0, le=10)    # the hardest attacking hit
    event_type: Literal["great_play", "blooper", "celebration", "chaos", "conversation",
                        "rally", "downtime"]
    players: list[str] = Field(default_factory=list)
    excitement: int = Field(ge=0, le=10)
    funny: int = Field(ge=0, le=10)
    peak_offset_s: float


@dataclass
class Window:
    start: float
    end: float
    tag: WindowTag
    source: str                       # "vlm" | "idle" (skipped by the motion filter)
    frames: list[float] = field(default_factory=list)
    reaction_max: float = 0.0
    examples: list[str] = field(default_factory=list)   # past windows shown as hints
    prompt: str = ""                  # the exact text Qwen got ("" in older caches)

    @property
    def peak_time(self) -> float:
        return self.start + self.tag.peak_offset_s

    @property
    def rally_s(self) -> float:
        return max(0.0, self.tag.rally_end_s - self.tag.rally_start_s)


def answer_schema() -> dict:
    """The JSON shape Ollama must answer in: every field, in order."""
    schema = WindowTag.model_json_schema()
    schema["required"] = list(schema["properties"])
    return schema


def prompt_version() -> str:
    """Short id of the current prompt text, recorded with every answer."""
    return hashlib.sha1(PROMPT_FILE.read_bytes()).hexdigest()[:8]


Hints = Callable[[float, float], tuple[str, list[str]]]


def run(game: Game, duration: float,
        keep_frames: Callable[[float, list[bytes]], None] | None = None,
        hints: Hints | None = None) -> list[Window]:
    """keep_frames(window start, jpegs) is called with the frames Qwen saw for
    each newly tagged window (the library keeps them for later training).
    hints(start, end) gives the example-memory text for a window and the ids
    of the past windows it shows."""
    vc = game.cfg.vlm
    tl = load_timeline(game)
    motion = motion_mod.run(game, duration)
    segments = _load_transcript(game)
    prompt_tpl = PROMPT_FILE.read_text(encoding="utf-8")
    cache_dir = game.work / "vlm"
    cache_dir.mkdir(exist_ok=True)

    windows: list[Window] = []
    todo: list[tuple[float, float, Path, str]] = []
    for start, end in window_grid(game.cfg, duration):
        a, b = int(start), int(np.ceil(end))
        reaction_max = float(tl["reaction"][a:b].max(initial=0.0))
        if not motion_mod.window_is_active(game, motion, tl["reaction"], start, end):
            windows.append(Window(float(start), end, _idle_tag(), "idle",
                                  reaction_max=reaction_max))
            continue
        path = cache_dir / f"{int(start):05d}.json"
        key = _cache_key(game, float(start), end, prompt_tpl)
        cached = _read_cache(path, key)
        if cached is not None:
            cached.reaction_max = reaction_max
            windows.append(cached)
        else:
            todo.append((float(start), end, path, key))
            windows.append(Window(float(start), end, _idle_tag(), "pending",
                                  reaction_max=reaction_max))

    idle = sum(w.source == "idle" for w in windows)
    log(f"[vlm] {len(windows)} windows: {idle} idle (skipped), "
        f"{len(windows) - idle - len(todo)} cached, {len(todo)} to tag")
    if vc.max_windows and len(todo) > vc.max_windows:
        log(f"[vlm] vlm.max_windows = {vc.max_windows}: tagging only the first "
            f"{vc.max_windows} this run")
        todo = todo[: vc.max_windows]

    if todo:
        client = OllamaClient(vc, pull=True)
        by_start = {w.start: i for i, w in enumerate(windows)}
        hinted = late = in_a_row = 0
        try:
            client.load()
            with Progress("vlm", f"tagging with {vc.model}", len(todo)) as bar:
                for n, (start, end, path, key) in enumerate(todo, 1):
                    try:
                        win, jpegs = _tag_window(game, client, prompt_tpl, tl, segments,
                                                 start, end, hints)
                    except QwenTimeout:
                        late += 1
                        in_a_row += 1
                        if in_a_row >= 2:
                            raise PipelineError(client.too_slow()) from None
                        log(f"[vlm] warning: no answer for the window at {fmt_time(start)} "
                            f"within {vc.timeout_s:.0f} s; skipped this run, tried again "
                            "next run")
                        bar.update(n)
                        continue
                    in_a_row = 0
                    if keep_frames is not None:
                        keep_frames(start, jpegs)
                    win.reaction_max = windows[by_start[start]].reaction_max
                    hinted += bool(win.examples)
                    windows[by_start[start]] = win
                    _write_cache(path, key, win)
                    bar.update(n)
        finally:
            client.unload()
        if late:
            log(f"[vlm] {late} window{'s' if late != 1 else ''} got no answer in time; "
                "the next run tags them")
        if hints is not None:
            log(f"[vlm] {hinted} of {len(todo)} newly tagged windows had similar past "
                "moments as hints")

    windows = [w for w in windows if w.source != "pending"]
    _write_tags(game, windows)
    return windows


def progress(game: Game, duration: float) -> tuple[int, int | None]:
    """(windows already tagged, windows to tag in all), without running
    anything. The total is None until the audio and motion stages have run."""
    tpl = PROMPT_FILE.read_text(encoding="utf-8")
    grid = window_grid(game.cfg, duration)
    done = sum(_read_cache(game.work / "vlm" / f"{int(s):05d}.json",
                           _cache_key(game, float(s), e, tpl)) is not None for s, e in grid)
    (mpath,), params = motion_mod.cache_plan(game)["motion"]
    if not is_fresh(game, "motion", [mpath], params) or not (game.work / "timeline.csv").exists():
        return done, None
    motion, reaction = np.load(mpath), load_timeline(game)["reaction"]
    return done, sum(motion_mod.window_is_active(game, motion, reaction, s, e) for s, e in grid)


def window_grid(cfg: Config, duration: float) -> list[tuple[float, float]]:
    """(start, end) of every window in a file of this length."""
    vc = cfg.vlm
    starts = np.arange(0.0, max(duration - 6.0, 0.0) + 1e-6, vc.hop_s)
    return [(float(s), min(float(s) + vc.window_s, duration)) for s in starts]


# ------------------------------------------------------------------- inputs

def load_timeline(game: Game) -> dict[str, np.ndarray]:
    path = game.work / "timeline.csv"
    if not path.exists():
        raise PipelineError("timeline.csv is missing; the audio stage must run first.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows], dtype=np.float32) for k in rows[0]}


def _load_transcript(game: Game) -> list[dict]:
    path = game.work / "transcript.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _idle_tag() -> WindowTag:
    return WindowTag(play_active=False, event_type="downtime",
                     description="idle (skipped by the motion filter)",
                     players=[], excitement=0, funny=0, peak_offset_s=0.0)


def _cache_key(game: Game, start: float, end: float, prompt_tpl: str) -> str:
    vc = game.cfg.vlm
    blob = json.dumps({
        **game.source_id(), "start": start, "end": end, "prompt": prompt_tpl,
        "vlm": vc.model_dump(exclude={"host", "max_windows", "timeout_s",
                                      "max_answer_tokens"}),
    }, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def _read_cache(path: Path, key: str) -> Window | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["key"] != key:
            return None
        return Window(data["start"], data["end"], WindowTag(**data["tag"]), "vlm",
                      frames=data.get("frames", []), examples=data.get("examples", []),
                      prompt=data.get("prompt", ""))
    except (ValueError, KeyError, ValidationError):
        return None


def _write_cache(path: Path, key: str, win: Window) -> None:
    path.write_text(json.dumps({
        "key": key, "start": win.start, "end": win.end,
        "tag": win.tag.model_dump(), "frames": win.frames, "examples": win.examples,
        "prompt": win.prompt,
    }, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ framing

def _reaction_peaks(game: Game, tl: dict[str, np.ndarray], start: float,
                    end: float) -> list[float]:
    """Seconds (absolute) of the strongest reaction peaks inside the window."""
    vc = game.cfg.vlm
    r = tl["reaction"]
    a, b = int(start), min(len(r), int(np.ceil(end)))
    peaks = []
    for i in range(a, b):
        left = r[i - 1] if i > 0 else -1
        right = r[i + 1] if i + 1 < len(r) else -1
        if r[i] >= vc.burst_min_reaction and r[i] >= left and r[i] > right:
            peaks.append((float(r[i]), i + 0.5))
    return [t for _, t in sorted(peaks, reverse=True)[:2]]


def _frame_times(game: Game, start: float, end: float, peaks: list[float]) -> list[float]:
    """Offsets (seconds from start) on the burst_fps grid that get sent."""
    vc = game.cfg.vlm
    step = 1.0 / vc.burst_fps
    every = max(1, round(vc.burst_fps / vc.base_fps))
    grid = np.arange(0.0, end - start - 1e-6, step)
    chosen = []
    for k, off in enumerate(grid):
        t = start + off
        in_burst = any(p - vc.burst_s <= t <= p for p in peaks)
        if k % every == 0 or in_burst:
            chosen.append(round(float(off), 3))
    if len(chosen) > vc.max_frames:
        idx = np.linspace(0, len(chosen) - 1, vc.max_frames).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(idx))]
    return chosen


def _extract_jpegs(game: Game, start: float, end: float) -> list[bytes]:
    """All frames of the window on the burst_fps grid, as JPEG bytes."""
    vc = game.cfg.vlm
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", str(game.work / "proxy.mp4"),
           "-vf", f"fps={vc.burst_fps},scale={vc.frame_width}:-2",
           "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise PipelineError("ffmpeg failed (frames):\n"
                            + proc.stderr.decode("utf-8", errors="replace")[-2000:])
    data, frames, pos = proc.stdout, [], 0
    while True:
        s = data.find(b"\xff\xd8", pos)
        if s < 0:
            break
        e = data.find(b"\xff\xd9", s)
        if e < 0:
            break
        frames.append(data[s:e + 2])
        pos = e + 2
    return frames


# ------------------------------------------------------------------ prompts

def _transcript_text(segments: list[dict], start: float, end: float) -> str:
    lines = [f"+{max(0.0, s['start'] - start):.0f}s: {s['text'].strip()}"
             for s in segments if s["end"] >= start and s["start"] <= end]
    return "\n".join(lines) or "(nothing was said)"


def _audio_text(tl: dict[str, np.ndarray], start: float, end: float,
                peaks: list[float]) -> str:
    lines = []
    for p in sorted(peaks):
        i = int(p)
        kinds = {k: float(tl[k][max(0, i - 2): i + 3].max()) for k in ("laugh", "cheer", "shout")}
        kind, level = max(kinds.items(), key=lambda kv: kv[1])
        word = {"laugh": "laughter", "cheer": "cheering", "shout": "shouting"}[kind]
        if level < 0.3:
            word = "loud voices"
        strength = "strong" if tl["reaction"][i] >= 0.22 else "moderate"
        lines.append(f"+{p - start:.1f}s: {strength} {word}")
    laugh = float(tl["laugh"][int(start): int(np.ceil(end))].max(initial=0.0))
    heard = ("none heard" if laugh < LAUGH_SOME else
             "a little" if laugh < LAUGH_CLEAR else "clear laughter")
    return "\n".join(lines or ["(no notable reactions)"]) + f"\nLaughter in this stretch: {heard}"


def _with_examples_slot(tpl: str) -> str:
    """Prompts without a $examples slot get the hints just before the
    instructions (an empty hint leaves the prompt exactly as written)."""
    if "$examples" in tpl or "${examples}" in tpl:
        return tpl
    anchor = "\nDescribe what happens"
    if anchor in tpl:
        return tpl.replace(anchor, "\n${examples}" + anchor[1:], 1)
    return tpl.rstrip("\n") + "\n\n${examples}"


def _build_prompt(game: Game, tpl: str, tl, segments, start, end, peaks, offsets,
                  examples: str = "") -> str:
    return Template(_with_examples_slot(tpl)).safe_substitute(
        n_frames=len(offsets), window_s=f"{end - start:.0f}",
        frame_times=", ".join(f"{o:.2f}" for o in offsets),
        transcript=_transcript_text(segments, start, end),
        audio=_audio_text(tl, start, end, peaks),
        examples=examples,
    )


def _tag_window(game: Game, client: "OllamaClient", tpl: str, tl, segments,
                start: float, end: float,
                hints: Hints | None = None) -> tuple[Window, list[bytes]]:
    vc = game.cfg.vlm
    examples, example_ids = hints(start, end) if hints is not None else ("", [])
    peaks = _reaction_peaks(game, tl, start, end)
    offsets = _frame_times(game, start, end, peaks)
    jpegs = _extract_jpegs(game, start, end)
    images = []
    for off in offsets:
        k = int(round(off * vc.burst_fps))
        if k < len(jpegs):
            images.append(jpegs[k])
    offsets = offsets[: len(images)]
    prompt = _build_prompt(game, tpl, tl, segments, start, end, peaks, offsets, examples)
    tag = client.tag(prompt, images, end - start)
    return (Window(start, end, tag, "vlm", frames=offsets, examples=example_ids, prompt=prompt),
            images)


class PromptBuilder:
    """Builds a window's prompt from the file's cached transcript and audio,
    for windows whose prompt wasn't kept (tagged before prompts were saved)
    and for re-tagging with an edited prompt. Example-memory hints aren't
    rebuilt: they depend on what the library held at the time."""

    def __init__(self, game: Game, tpl: str | None = None):
        self.game = game
        self.tpl = tpl if tpl is not None else PROMPT_FILE.read_text(encoding="utf-8")
        self.tl = load_timeline(game)
        self.segments = _load_transcript(game)

    def build(self, start: float, end: float, offsets: list[float]) -> str:
        peaks = _reaction_peaks(self.game, self.tl, start, end)
        return _build_prompt(self.game, self.tpl, self.tl, self.segments, start, end,
                             peaks, offsets)


def window_frames(game: Game, win: Window) -> list[bytes]:
    """The frames Qwen saw for a cached window, re-cut from the proxy."""
    jpegs = _extract_jpegs(game, win.start, win.end)
    step = game.cfg.vlm.burst_fps
    return [jpegs[k] for k in (int(round(o * step)) for o in win.frames) if k < len(jpegs)]


# ------------------------------------------------------------------- ollama

class QwenTimeout(PipelineError):
    """Ollama took the window but gave no answer within vlm.timeout_s."""


class ModelMissing(PipelineError):
    """Ollama is running but hasn't downloaded the model yet."""


class OllamaClient:
    def __init__(self, vc: VlmCfg, pull: bool = False):
        """pull: download the model when Ollama doesn't have it yet (a run
        does; the setup check only reports it)."""
        self.vc = vc
        self.host = self.vc.host.rstrip("/")
        self.think_supported = True
        try:
            self._check()
        except ModelMissing:
            if not pull:
                raise
            self.pull()
            self._check()

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _check(self) -> None:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as r:
                models = [m["name"] for m in json.loads(r.read().decode())["models"]]
        except (urllib.error.URLError, OSError) as err:
            raise PipelineError(
                f"Can't reach Ollama at {self.host} ({err}). Start the Ollama app "
                "(it lives in the system tray) and run the command again.") from err
        want = self.vc.model if ":" in self.vc.model else self.vc.model + ":latest"
        if want not in models:
            raise ModelMissing(f"Ollama doesn't have {self.vc.model} yet. The next run "
                               "downloads it by itself (once, a few GB), or run:\n"
                               f"    ollama pull {self.vc.model}\n"
                               f"Installed models: {', '.join(models) or 'none'}")

    def pull(self) -> None:
        """Download the model through Ollama, with a progress bar."""
        model = self.vc.model
        log(f"[vlm] Ollama doesn't have {model} yet: downloading it (only this once)")
        req = urllib.request.Request(self.host + "/api/pull",
                                     data=json.dumps({"model": model, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        sizes: dict[str, tuple[float, float]] = {}       # layer -> (done, total) bytes
        try:
            with urllib.request.urlopen(req, timeout=600) as r, \
                    Progress("vlm", f"downloading {model}", 100) as bar:
                for line in r:
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise PipelineError(f"Ollama couldn't download {model}: {msg['error']}")
                    if msg.get("digest") and msg.get("total"):
                        sizes[msg["digest"]] = (float(msg.get("completed") or 0),
                                                float(msg["total"]))
                        done, total = (sum(x) for x in zip(*sizes.values()))
                        bar.update(100 * done / total)
                    if msg.get("status") == "success":
                        bar.update(100)
        except (urllib.error.URLError, OSError, ValueError) as err:
            raise PipelineError(f"Downloading {model} through Ollama failed ({err}). Check "
                                f"the internet connection, or run:\n    ollama pull {model}"
                                ) from err
        total = sum(t for _d, t in sizes.values())
        if total:
            log(f"[vlm] downloaded {model} ({total / 2**30:.1f} GB)")

    def capabilities(self) -> list[str]:
        try:
            return list(self._post("/api/show", {"model": self.vc.model}, 30)
                        .get("capabilities") or [])
        except (urllib.error.URLError, OSError, ValueError):
            return []

    def load(self) -> None:
        """Load the model before the first window (so that window's wait isn't
        spent loading) and say how much of it sits on the graphics card."""
        model = self.vc.model
        if "thinking" in self.capabilities():
            log(f"[vlm] warning: {model} is a thinking model. It writes out its reasoning "
                "before every answer, which is much slower and can go on and on. Use "
                f"{instruct_name(model)} (vlm.model in config.yaml).")
        log(f"[vlm] loading {model} into the graphics card")
        t0 = time.time()
        try:
            self._post("/api/generate", {"model": model, "prompt": "",
                                         "options": {"num_ctx": self.vc.num_ctx}},
                       max(self.vc.timeout_s, 600))
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")
            raise PipelineError(f"Ollama couldn't load {model} (error {err.code}): "
                                f"{body[:500]}") from err
        except (urllib.error.URLError, OSError) as err:
            raise PipelineError(f"Ollama couldn't load {model} ({err}). Restart the "
                                "Ollama app (system tray > Quit Ollama, then open it "
                                "again) and press Start again.") from err
        took = time.time() - t0
        share, size = self.gpu_share()
        if share is None:
            log(f"[vlm] {model} loaded in {took:.0f} s")
        elif share >= 0.99:
            log(f"[vlm] {model} loaded in {took:.0f} s, all of it on the graphics card "
                f"({size / 2**30:.1f} GB)")
        else:
            log(f"[vlm] warning: only {share:.0%} of {model} ({size / 2**30:.1f} GB with its "
                "working memory) fits on the graphics card; the rest runs on the processor, "
                "which makes tagging much slower. Close other programs that use the "
                "graphics card, or lower vlm.num_ctx.")

    def gpu_share(self) -> tuple[float | None, float]:
        """(share of the loaded model on the GPU, its size in bytes); None if unknown."""
        loaded = self.loaded()
        if not loaded or not loaded.get("size"):
            return None, 0.0
        size = float(loaded["size"])
        return float(loaded.get("size_vram", 0)) / size, size

    def too_slow(self) -> str:
        """Why two windows in a row got no answer, for the error that stops the run."""
        model, wait = self.vc.model, f"{self.vc.timeout_s:.0f}"
        share, _size = self.gpu_share()
        if share is not None and share < 0.99:
            why = (f"Only {share:.0%} of {model} is on the graphics card, so most of the work "
                   "runs on the processor. Close other programs that use the graphics card, "
                   "or lower vlm.num_ctx in config.yaml.")
        elif "thinking" in self.capabilities():
            why = (f"{model} is a thinking model, which reasons at length before answering. "
                   f"Set vlm.model to {instruct_name(model)} in config.yaml.")
        else:
            why = ("Restart the Ollama app (system tray > Quit Ollama, then open it again). "
                   "If it happens again, raise vlm.timeout_s in config.yaml.")
        return (f"{model} didn't answer two windows in a row within {wait} s each "
                f"(vlm.timeout_s). {why} Windows already tagged are kept: press Start "
                "again to carry on.")

    def tag(self, prompt: str, images: list[bytes], window_len: float) -> WindowTag:
        payload = {
            "model": self.vc.model,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(j).decode() for j in images]}],
            "format": answer_schema(),
            "stream": False,
            "options": {"temperature": self.vc.temperature, "num_ctx": self.vc.num_ctx,
                        "num_predict": self.vc.max_answer_tokens},
        }
        last_err = ""
        for _attempt in range(3):
            if self.think_supported:
                payload["think"] = False
            else:
                payload.pop("think", None)
            try:
                resp = self._post("/api/chat", payload, self.vc.timeout_s)
            except urllib.error.HTTPError as err:
                body = err.read().decode("utf-8", errors="replace")
                if "think" in body.lower() and self.think_supported:
                    self.think_supported = False      # model has no thinking switch
                    continue
                raise PipelineError(f"Ollama error {err.code}: {body[:500]}") from err
            except (urllib.error.URLError, OSError) as err:
                raise _connection_error(err, self.vc.timeout_s) from err
            content = resp.get("message", {}).get("content", "")
            if resp.get("done_reason") == "length":
                last_err = (f"the answer ran past {self.vc.max_answer_tokens} tokens "
                            "(vlm.max_answer_tokens)")
                continue
            try:
                tag = WindowTag.model_validate(json.loads(_strip_json(content)))
            except (ValueError, ValidationError) as err:
                last_err = f"{err}; got: {content[:300]!r}"
                continue
            tag.peak_offset_s = float(min(max(tag.peak_offset_s, 0.0), window_len))
            a = float(min(max(tag.rally_start_s, 0.0), window_len))
            b = float(min(max(tag.rally_end_s, 0.0), window_len))
            tag.rally_start_s, tag.rally_end_s = (a, b) if b > a else (0.0, 0.0)
            return tag
        log(f"[vlm] warning: no valid answer for a window after 3 tries ({last_err})")
        return WindowTag(play_active=False, event_type="rally",
                         description="(the model gave no valid answer)", players=[],
                         excitement=0, funny=0, peak_offset_s=window_len / 2)

    def answer(self, prompt: str, schema: dict, num_predict: int) -> dict | None:
        """A text-only question answered in the JSON shape of schema (None when
        no valid answer came in 3 tries)."""
        payload = {
            "model": self.vc.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema, "stream": False,
            "options": {"temperature": self.vc.temperature, "num_ctx": self.vc.num_ctx,
                        "num_predict": num_predict},
        }
        for _attempt in range(3):
            if self.think_supported:
                payload["think"] = False
            else:
                payload.pop("think", None)
            try:
                resp = self._post("/api/chat", payload, self.vc.timeout_s)
            except urllib.error.HTTPError as err:
                body = err.read().decode("utf-8", errors="replace")
                if "think" in body.lower() and self.think_supported:
                    self.think_supported = False
                    continue
                raise PipelineError(f"Ollama error {err.code}: {body[:500]}") from err
            except (urllib.error.URLError, OSError) as err:
                raise _connection_error(err, self.vc.timeout_s) from err
            if resp.get("done_reason") == "length":
                continue
            try:
                data = json.loads(_strip_json(resp.get("message", {}).get("content", "")))
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
        return None

    def ask(self, prompt: str, images: list[bytes], timeout: float) -> str:
        """A plain answer with no JSON schema (the setup check's test picture)."""
        payload = {
            "model": self.vc.model,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(j).decode() for j in images]}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.vc.num_ctx,
                        "num_predict": self.vc.max_answer_tokens},
        }
        for _attempt in range(2):
            if self.think_supported:
                payload["think"] = False
            else:
                payload.pop("think", None)
            try:
                resp = self._post("/api/chat", payload, timeout)
            except urllib.error.HTTPError as err:
                body = err.read().decode("utf-8", errors="replace")
                if "think" in body.lower() and self.think_supported:
                    self.think_supported = False
                    continue
                raise PipelineError(f"Ollama error {err.code}: {body[:500]}") from err
            except (urllib.error.URLError, OSError) as err:
                raise _connection_error(err, timeout) from err
            content = resp.get("message", {}).get("content", "")
            return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        return ""

    def version(self) -> str:
        try:
            with urllib.request.urlopen(self.host + "/api/version", timeout=10) as r:
                return str(json.loads(r.read().decode()).get("version", "?"))
        except (urllib.error.URLError, OSError, ValueError):
            return "?"

    def loaded(self) -> dict | None:
        """What Ollama reports for this model while it's in memory, including
        size (bytes) and size_vram (the part on the GPU)."""
        want = self.vc.model if ":" in self.vc.model else self.vc.model + ":latest"
        try:
            with urllib.request.urlopen(self.host + "/api/ps", timeout=10) as r:
                models = json.loads(r.read().decode()).get("models", [])
        except (urllib.error.URLError, OSError, ValueError):
            return None
        return next((m for m in models if want in (m.get("name"), m.get("model"))), None)

    def unload(self) -> None:
        """Free the GPU for whatever runs next."""
        try:
            self._post("/api/generate", {"model": self.vc.model, "keep_alive": 0}, 30)
        except (urllib.error.URLError, OSError, ValueError):
            pass


def instruct_name(model: str) -> str:
    """The plain-answer version of a model: qwen3-vl:4b -> qwen3-vl:4b-instruct
    (Ollama's plain qwen3-vl tags are the thinking versions)."""
    base, _, tag = model.partition(":")
    size = []
    for part in tag.split("-"):                 # 30b-a3b-thinking-q8_0 -> 30b-a3b
        if part in ("thinking", "instruct", "latest", "") or re.match(r"(q\d|bf16|fp16)", part):
            break
        size.append(part)
    return f"{base}:{'-'.join(size) or '8b'}-instruct"


def _connection_error(err: OSError, timeout: float) -> PipelineError:
    """A request to Ollama that failed: no answer in time, or Ollama gone."""
    if isinstance(err, TimeoutError) or isinstance(getattr(err, "reason", None), TimeoutError):
        return QwenTimeout(f"Ollama gave no answer within {timeout:.0f} s")
    return PipelineError(f"Lost connection to Ollama ({err}). Check that the Ollama app is "
                         "still running (system tray), then press Start again: windows "
                         "already tagged are kept.")


def _strip_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s >= 0 and e > s else text


# ------------------------------------------------------------------ outputs

def _write_tags(game: Game, windows: list[Window]) -> None:
    path = game.work / "tags.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "timestamp", "source", "play_active", "event_type",
                    "excitement", "funny", "rally_s", "attack", "peak_time", "players",
                    "reaction_max", "description"])
        for win in windows:
            t = win.tag
            w.writerow([f"{win.start:.1f}", f"{win.end:.1f}", fmt_time(win.start), win.source,
                        t.play_active, t.event_type, t.excitement, t.funny,
                        f"{win.rally_s:.0f}", t.attack,
                        f"{win.peak_time:.1f}", "; ".join(t.players),
                        f"{win.reaction_max:.3f}", t.description])
    tagged = [w for w in windows if w.source == "vlm"]
    counts = {e: sum(w.tag.event_type == e for w in tagged) for e in EVENT_TYPES}
    long_rallies = sum(w.rally_s >= 15 for w in tagged)
    big_hits = sum(w.tag.attack >= 7 for w in tagged)
    log("[vlm] labels: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        + f"; {long_rallies} rallies of 15 s or more, {big_hits} big spikes (attack 7+)"
        + f" -> {path}")
