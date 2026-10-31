"""Shared helpers: game paths, stage caching, ffmpeg calls, time formatting."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .config import Config


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------- game paths

@dataclass
class Game:
    video: Path
    name: str
    work: Path
    out: Path
    cfg: Config

    @classmethod
    def create(cls, video: Path, cfg: Config, out: Path | None = None) -> "Game":
        """One video file. Its cache lives in work/<file name>/, so a file keeps
        its cached analysis even when it is processed as part of a session."""
        video = video.resolve()
        if not video.is_file():
            raise PipelineError(f"Video not found: {video}")
        name = video.stem
        work = cfg.work_dir / name
        out = out or cfg.output_dir / name
        work.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        return cls(video, name, work, out, cfg)

    def source_id(self) -> dict:
        """Identifies the input file so caches go stale if the video changes."""
        st = self.video.stat()
        return {"video": str(self.video), "size": st.st_size, "mtime": int(st.st_mtime)}


# ------------------------------------------------------------------ caching
# Each stage writes its outputs plus <stage>.done.json holding the parameters
# that produced them. A stage is skipped when every output exists and the
# stamp matches the current parameters, so re-runs are cheap and changing a
# setting in config.yaml re-runs only the stages that depend on it.

def _digest(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()


def is_fresh(game: Game, stage: str, outputs: list[Path], params: dict) -> bool:
    stamp = game.work / f"{stage}.done.json"
    if not stamp.exists() or not all(p.exists() for p in outputs):
        return False
    try:
        return json.loads(stamp.read_text())["digest"] == _digest(params)
    except (ValueError, KeyError):
        return False


def mark_done(game: Game, stage: str, params: dict) -> None:
    stamp = game.work / f"{stage}.done.json"
    stamp.write_text(json.dumps({"digest": _digest(params), "params": params},
                                indent=2, default=str))


def clear_stage(game: Game, stage: str) -> None:
    (game.work / f"{stage}.done.json").unlink(missing_ok=True)


# -------------------------------------------------------------- console output
# Colour and live progress bars come from the `rich` package. Without it the
# pipeline still runs, with plain text and a simple [####----] bar. When the
# app (gui/) runs the pipeline it sets VH_EVENTS=1: output is then plain lines,
# and each progress bar is reported as "@@progress {json}" lines it can draw.

EVENTS = os.environ.get("VH_EVENTS") == "1"

try:
    if EVENTS:
        raise ImportError
    from rich.console import Console
    from rich.markup import escape
    from rich.progress import BarColumn, ProgressColumn, SpinnerColumn, TaskProgressColumn
    from rich.progress import Progress as _RichProgress
    from rich.progress import TextColumn, TimeElapsedColumn
    from rich.table import Column
    from rich.text import Text
    _console: "Console | None" = Console(highlight=False)
    _err_console: "Console | None" = Console(stderr=True, highlight=False)
except ImportError:
    _console = _err_console = None

STAGE_STYLES = {
    "ingest": "cyan",
    "audio": "magenta",
    "motion": "blue",
    "vlm": "green",
    "render": "yellow",
    "review": "bright_white",
    "session": "bright_cyan",
    "fuse": "bright_magenta",
    "select": "bright_yellow",
    "library": "bright_blue",
    "learn": "bright_green",
    "track": "dark_orange",
    "tags": "orchid",
    "gui": "bright_white",
    "memory": "medium_purple1",
    "tune": "gold1",
    "check": "sky_blue1",
    "eval": "spring_green1",
    "train": "light_salmon1",
    "comments": "hot_pink",
    "queue": "aquamarine1",
}
_STAGE_RE = re.compile(r"^(\s*)\[(\w+)\]")


def _styled_line(line: str) -> "Text":
    text = Text()
    m = _STAGE_RE.match(line)
    rest = line
    if m:
        text.append(m.group(1))
        text.append(f"[{m.group(2)}]", style=f"bold {STAGE_STYLES.get(m.group(2), 'white')}")
        rest = line[m.end():]
    low = rest.lower()
    if "warning" in low:
        style = "yellow"
    elif low.startswith(("started", "finished")):
        style = "bold"
    elif ": finished" in low or "wrote " in low or " watch " in f" {low}":
        style = "green"
    elif "cached" in low or "disabled" in low:
        style = "dim"
    else:
        style = ""
    text.append(rest, style=style)
    return text


def log(msg: str) -> None:
    if _console is None:
        if Progress.active is not None:
            Progress.active.break_line()
        print(msg, flush=True)
        return
    for line in msg.split("\n"):
        _console.print(_styled_line(line), soft_wrap=True)


def error(msg: str) -> None:
    if _err_console is None:
        print(f"\nError: {msg}", file=sys.stderr, flush=True)
        return
    _err_console.print()
    _err_console.print(Text("Error: ", style="bold red") + Text(msg, style="red"))


STATUS_MARKS = {"ok": ("✓", "+", "green"), "warn": ("!", "!", "yellow"),
                "fail": ("✗", "x", "bold red"), "info": ("·", "-", "dim")}


def log_status(stage: str, status: str, msg: str) -> None:
    """One line of a check: ✓ fine, ! worth a look, ✗ needs fixing, · for information."""
    mark, plain, style = STATUS_MARKS[status]
    if _console is None:
        if Progress.active is not None:
            Progress.active.break_line()
        try:
            mark.encode(sys.stdout.encoding or "utf-8")
        except (UnicodeEncodeError, LookupError):
            mark = plain
        indent = "\n" + " " * len(f"[{stage}] {mark} ")
        print(f"[{stage}] {mark} " + msg.replace("\n", indent), flush=True)
        return
    for i, line in enumerate(msg.split("\n")):
        text = Text(f"[{stage}] ", style=f"bold {STAGE_STYLES.get(stage, 'white')}")
        text.append(f"{mark} " if i == 0 else "  ", style=style)
        text.append(line, style=style if status in ("warn", "fail") else "")
        _console.print(text, soft_wrap=True)


def notice(msg: str) -> None:
    """A highlighted one-off message (e.g. after Ctrl+C)."""
    if _console is None:
        print(f"\n{msg}", flush=True)
    else:
        _console.print()
        _console.print(Text(msg, style="bold yellow"))


# ----------------------------------------------------------------- progress

def clock(ts: float | None = None) -> str:
    """Wall-clock time like '6:52 PM'."""
    return time.strftime("%I:%M %p", time.localtime(ts)).lstrip("0")


def fmt_hm(secs: float) -> str:
    h, m = divmod(int(round(max(0.0, secs) / 60)), 60)
    return f"{h}:{m:02d}"


def fmt_hms(secs: float) -> str:
    h, rem = divmod(int(round(max(0.0, secs))), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _eta_text(frac: float, elapsed: float, now: float) -> str:
    if frac >= 0.999:
        return "done"
    if frac >= 0.01 and elapsed >= 3:
        remaining = elapsed * (1 - frac) / frac
        left = fmt_hm(remaining) if remaining >= 60 else "<0:01"
        return f"ETA {left}  (done ~{clock(now + remaining)})"
    return "ETA estimating..."


if _console is not None:
    class _EtaColumn(ProgressColumn):
        """'ETA 0:12  (done ~6:44 PM)' in hours:minutes."""
        max_refresh = 1.0

        def __init__(self) -> None:
            super().__init__(table_column=Column(no_wrap=True))

        def render(self, task) -> "Text":
            frac = (task.completed / task.total) if task.total else 0.0
            text = _eta_text(frac, task.elapsed or 0.0, time.time())
            return Text(text, style="bold green" if text == "done" else "cyan")


class Progress:
    """Live console progress bar with ETA (H:MM) and expected finish time.

        with Progress("ingest", "720p proxy", total=duration) as bar:
            ...bar.update(done)...
    """
    active: "Progress | None" = None
    WIDTH = 30

    def __init__(self, stage: str, what: str, total: float):
        self.stage = stage
        self.what = what
        self.prefix = f"[{stage}] {what}"
        self.total = max(float(total), 1e-9)
        self.done = 0.0
        self.start = time.time()
        self._last_draw = 0.0
        self._drawn = False
        self._width = 0
        self._rich = None
        self._task = None

    def __enter__(self) -> "Progress":
        log(f"{self.prefix}: started {clock(self.start)}")
        Progress.active = self
        if _console is not None:
            color = STAGE_STYLES.get(self.stage, "white")
            # Fit one line: shrink the bar, then drop the elapsed clock, on narrow windows.
            fixed = 2 + len(self.prefix) + 1 + 5 + 1 + 27 + 4
            show_elapsed = _console.width - fixed - 16 >= 20
            room = _console.width - fixed - (16 if show_elapsed else 0)
            columns = [
                SpinnerColumn(style=color, finished_text=Text("✓", style="bold green")),
                TextColumn(f"[bold {color}]{escape(f'[{self.stage}]')}[/] {escape(self.what)}",
                           table_column=Column(no_wrap=True)),
                BarColumn(bar_width=max(10, min(40, room)), style="grey30",
                          complete_style=color, finished_style="green"),
                TaskProgressColumn(),
                _EtaColumn(),
            ]
            if show_elapsed:
                columns += [TextColumn("[dim]elapsed[/]"), TimeElapsedColumn()]
            self._rich = _RichProgress(*columns, console=_console, refresh_per_second=4)
            self._rich.start()
            self._task = self._rich.add_task("", total=self.total)
        else:
            self._draw(force=True)
        return self

    def update(self, done: float) -> None:
        self.done = min(max(done, self.done), self.total)
        if self._rich is not None:
            self._rich.update(self._task, completed=self.done)
        else:
            self._draw()

    def break_line(self) -> None:
        """Move off the bar's line so another message prints cleanly (plain mode)."""
        if self._drawn and not EVENTS:
            sys.stdout.write("\n")
            self._drawn = False

    def _draw(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_draw < 0.5:
            return
        self._last_draw = now
        frac = self.done / self.total
        if EVENTS:
            print("@@progress " + json.dumps({
                "stage": self.stage, "what": self.what, "frac": round(frac, 4),
                "elapsed": round(now - self.start, 1),
                "eta": _eta_text(frac, now - self.start, now)}), flush=True)
            return
        filled = min(self.WIDTH, int(frac * self.WIDTH + 0.01))
        bar = "#" * filled + "-" * (self.WIDTH - filled)
        line = f"{self.prefix} [{bar}] {frac * 100:5.1f}%  {_eta_text(frac, now - self.start, now)}"
        sys.stdout.write("\r" + line.ljust(self._width))
        sys.stdout.flush()
        self._width = max(self._width, len(line))
        self._drawn = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.done = self.total
        if self._rich is not None:
            if exc_type is None:
                self._rich.update(self._task, completed=self.total)
            self._rich.stop()
            self._rich = None
        else:
            if exc_type is None:
                self._draw(force=True)
            self.break_line()
        Progress.active = None
        end = time.time()
        if exc_type is None:
            log(f"{self.prefix}: finished {clock(end)} (took {fmt_hms(end - self.start)})")


# ------------------------------------------------------------------- ffmpeg

# The camera sometimes skips a few audio frames while recording. The file's
# timestamps keep the gap, but plain decoding closes it up, so the sound runs
# ahead of the picture (by up to 5 s by the end of a 30-minute video). These
# aresample options fill each gap with silence, and pad the start, so the
# sound stays on time.
AUDIO_ON_TIME = "async=1:min_hard_comp=0.01:first_pts=0"


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise PipelineError(
                f"{tool} not found on PATH. On Windows: winget install Gyan.FFmpeg, "
                "then open a new terminal.")


def run_ffmpeg(args: list[str], desc: str = "",
               on_progress: Callable[[float], None] | None = None) -> None:
    """Run ffmpeg; on_progress, if given, receives seconds of output written."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if on_progress is None:
        cmd += args
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        returncode, stderr = proc.returncode, proc.stderr
    else:
        cmd += ["-progress", "pipe:1", "-nostats", *args]
        # stderr goes to a temp file so a chatty ffmpeg can never block on a full pipe.
        with tempfile.TemporaryFile() as errf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True,
                                    encoding="utf-8", errors="replace")
            assert proc.stdout is not None
            for line in proc.stdout:
                key, _, val = line.strip().partition("=")
                if key == "out_time_us" and val.isdigit():
                    on_progress(int(val) / 1e6)
            returncode = proc.wait()
            errf.seek(0)
            stderr = errf.read().decode("utf-8", errors="replace")
    if returncode != 0:
        raise PipelineError(f"ffmpeg failed{' (' + desc + ')' if desc else ''}:\n"
                            f"{' '.join(cmd)}\n{stderr.strip()[-2000:]}")


# ffmpeg's complaints about a damaged video stream. A healthy file prints none;
# a broken one prints thousands a second.
DAMAGE_RE = re.compile(r"Invalid NAL unit|Error splitting the input|Error while decoding|"
                       r"Invalid data found|corrupt|no frame!")
DAMAGE_LIMIT = 50


class FramePipe:
    """ffmpeg decoding a video to raw frames on stdout (pipe.stdout.read).

    Its messages are read as they come, by a thread: if nobody read them, a
    damaged video would fill the stderr pipe and freeze ffmpeg and the run
    for good. Once a video turns out damaged, ffmpeg is stopped (so stdout
    ends) and `damaged` is True."""

    def __init__(self, cmd: list[str]):
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert self.proc.stdout is not None
        self.stdout = self.proc.stdout
        self.errors = 0
        self._tail: deque[str] = deque(maxlen=30)
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    @property
    def damaged(self) -> bool:
        return self.errors > DAMAGE_LIMIT

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            self._tail.append(line)
            if DAMAGE_RE.search(line):
                self.errors += 1
                if self.errors == DAMAGE_LIMIT + 1 and self.proc.poll() is None:
                    self.proc.kill()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()

    def wait(self) -> tuple[int, str]:
        """ffmpeg's exit code and its last messages."""
        code = self.proc.wait()
        self._reader.join(timeout=10)
        return code, "\n".join(self._tail)


def damaged_video(src: Path) -> PipelineError:
    """The error for a video ffmpeg can't decode. A damaged 720p copy also loses
    its "done" stamp, so the next run makes it again from the original."""
    if src.name == "proxy.mp4":
        (src.parent / "proxy.done.json").unlink(missing_ok=True)
        return PipelineError(
            f"The 720p copy of {src.parent.name} is damaged (ffmpeg can't decode it). "
            "Start the run again: that copy is made again from the original first, and "
            "everything already finished is kept.")
    return PipelineError(f"{src.name} looks damaged: ffmpeg can't decode it ({src}).")


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def probe(path: Path) -> Probe:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise PipelineError(f"ffprobe could not read {path}:\n{proc.stderr}")
    info = json.loads(proc.stdout)
    vstreams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise PipelineError(f"No video stream in {path}")
    v = vstreams[0]
    num, _, den = v.get("avg_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    w, h = int(v["width"]), int(v["height"])
    # Phones store portrait video as landscape + a rotation flag; ffmpeg
    # auto-rotates when decoding, so report the size as displayed.
    rot = 0
    for sd in v.get("side_data_list", []):
        rot = int(sd.get("rotation", rot) or 0)
    rot = int(v.get("tags", {}).get("rotate", rot) or 0)
    if abs(rot) % 180 == 90:
        w, h = h, w
    return Probe(
        duration=float(info["format"]["duration"]),
        width=w, height=h, fps=fps or 30.0,
        has_audio=any(s.get("codec_type") == "audio" for s in info["streams"]),
    )


@lru_cache(maxsize=None)
def nvenc_available() -> bool:
    """Try a tiny NVENC encode; True only if the GPU encoder really works."""
    try:
        run_ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                    "-c:v", "h264_nvenc", "-f", "null", "-"], "nvenc test")
        return True
    except PipelineError:
        return False


def video_encoder_args(cfg: Config) -> list[str]:
    choice = cfg.render.encoder
    if choice == "auto":
        choice = "h264_nvenc" if nvenc_available() else "libx264"
    if choice == "h264_nvenc":
        return ["-c:v", "h264_nvenc", *cfg.render.nvenc_args, "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", *cfg.render.x264_args, "-pix_fmt", "yuv420p"]


# --------------------------------------------------------------------- time

def parse_time(text: str) -> float:
    """'754.5', '12:34', '1:02:03.5' -> seconds."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Bad timestamp: {text!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def fmt_time(secs: float) -> str:
    secs = max(0.0, secs)
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def on_windows() -> bool:
    return sys.platform.startswith("win")
