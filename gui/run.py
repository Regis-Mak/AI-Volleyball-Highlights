"""The Run screen: pick a session folder, choose what to make, start it, and
watch each stage's progress. Stop saves everything finished so far, exactly
like Ctrl+C in the terminal. Pause freezes the run where it is (the run and
every process it started) until Resume, so not even the step in progress is
lost; the time spent paused is left out of the times and ETAs shown.

A run started here is the same command you would type (it's shown on the
screen), run as a separate process: closing the page doesn't stop it, and
reopening the page picks the run back up. The "Road map" drop-down under the
log shows every step of the run and which one it's on (gui/roadmap.py).

The queue: "Add to queue" lines up more session folders (each with the options
chosen when it was added); they run one after another, each as its own run of
its own session, so every date keeps its own videos, picks, nametags and
grades (the "Your sessions" card links to each). A run that fails is noted and
the next one starts; Stop puts the queue on hold. Every run's log is also saved
as work/<session>/logs/run <date time>.txt.
"""
from __future__ import annotations

import json
import os
import re
import signal
import string
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote

from pipeline.common import _eta_text, clock, fmt_hms
from pipeline.config import Config
from pipeline.session import VIDEO_EXTS

from .roadmap import CSS as MAP_CSS
from .roadmap import STOP_CODES, Roadmap, mode_for
from .shell import find_session, frame, sessions

ROOT = Path(__file__).resolve().parent.parent

MAKE = {
    "highlights": "Highlight reel",
    "picks": "Choose clips only (picks.csv, no videos yet)",
    "peaks": "Check: top audio reactions (Phase 2 review video)",
    "tags": "Check: a sample of vision tags (Phase 3 review video)",
    "check": "Check the setup and what's already done (processes nothing)",
}
FLAGS = {"highlights": [], "picks": ["--pick-only"], "peaks": ["--review", "peaks"],
         "tags": ["--review", "tags"], "check": ["--check"]}
REDO = {"ingest": "Proxy copy and audio", "transcript": "Speech (Whisper)",
        "panns": "Laughs and cheers (PANNs)", "motion": "Motion",
        "fingerprints": "Picture fingerprints (example memory)", "vlm": "Vision tags (Qwen)",
        "comments": "Unhinged comments (Qwen reads what was said)"}

# The terminal's stage colours (pipeline/common.py), for the log.
COLOURS = {"ingest": "#22d3ee", "audio": "#e879f9", "motion": "#60a5fa", "vlm": "#4ade80",
           "render": "#facc15", "review": "#ffffff",
           "session": "#67e8f9", "fuse": "#f0abfc", "select": "#fde047",
           "library": "#93c5fd", "learn": "#86efac",
           "track": "#fb923c", "tags": "#da70d6", "gui": "#ffffff",
           "memory": "#af87ff", "tune": "#ffd700", "check": "#87d7ff", "eval": "#00ff87",
           "train": "#ffaf87", "comments": "#ff5fd7", "queue": "#87ffd7"}
STAGE_RE = re.compile(r"^\s*\[(\w+)\]")
MAP_UI = {"open": False}       # the Road map drop-down stays as you left it


class Job:
    """The one pipeline run the app looks after."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lines: deque[str] = deque(maxlen=3000)
        self.count = 0                 # lines ever added, so pages know what's new
        self.progress: dict | None = None
        self.started = self.ended = 0.0
        self.code: int | None = None
        self.stopping = False
        self.paused_at = 0.0           # when the current pause began; 0 = not paused
        self.pauses: list[tuple[float, float]] = []   # earlier pauses (start, end)
        self.session = ""
        self.what = ""
        self.command = ""
        self.roadmap: Roadmap | None = None
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.run_id = 0                # goes up with every run, so pages can start afresh
        self.log_file = None           # this run's log, saved in work/<session>/logs/
        self.log_path: Path | None = None
        self.cfg: Config | None = None
        self.queue: list[dict] = []    # {source, session, flags, what}, next first
        self.held = False              # Stop holds the queue until "Start the queue"
        self.history: list[dict] = []  # runs started from this app, newest last

    @property
    def running(self) -> bool:
        return self.proc is not None and self.code is None

    def start(self, cfg: Config, source: str, flags: list[str], what: str) -> None:
        with self.start_lock:              # Start and the queue never start two runs
            self._start(cfg, source, flags, what)

    def _start(self, cfg: Config, source: str, flags: list[str], what: str) -> None:
        if self.running:
            raise RuntimeError("A run is already going")
        src = Path(source.strip().strip('"')).expanduser()
        cmd = [sys.executable, "-u", str(ROOT / "make_highlights.py"), str(src), *flags]
        if cfg.config_path is not None:
            cmd += ["--config", str(cfg.config_path)]
        env = dict(os.environ, VH_EVENTS="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   PYTHONUTF8="1")
        kw: dict = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        session = src.name if src.is_dir() else src.stem
        shown = ["python", "make_highlights.py", f'"{src}"', *flags]
        log_path: Path | None = cfg.work_dir / session / "logs" / \
            f"run {time.strftime('%Y-%m-%d %H%M%S')}.txt"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # utf-8-sig: the BOM lets browsers and Notepad show the symbols right
            log_file = open(log_path, "w", encoding="utf-8-sig")
            log_file.write(" ".join(shown) + "\n")
        except OSError:
            log_path, log_file = None, None
        with self.lock:
            self.cfg = cfg
            self.run_id += 1
            self.log_file, self.log_path = log_file, log_path
            self.lines.clear()
            self.count = 0
            self.progress = None
            self.code = None
            self.stopping = False
            self.paused_at, self.pauses = 0.0, []
            self.started, self.ended = time.time(), 0.0
            self.session = session
            self.what = what
            self.command = " ".join(shown)
            self.roadmap = Roadmap(mode_for(flags))
        try:
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=env, **kw)
        except OSError as err:
            with self.lock:
                self.proc, self.code, self.ended = None, -1, time.time()
                if self.log_file is not None:
                    self._save(f"Couldn't start the run: {err}")
                    self.log_file.close()
                    self.log_file = None
            raise RuntimeError(f"Couldn't start the run: {err}") from err
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\r\n").split("\r")[-1]
            if line.startswith("@@progress "):
                try:
                    p = json.loads(line[11:])
                except ValueError:
                    continue
                p["_t"] = time.time()
                with self.lock:
                    self.progress = p if p.get("frac", 0) < 1 else None
                continue
            with self.lock:
                self.lines.append(line)
                self.count += 1
                if self.roadmap is not None:
                    self.roadmap.feed(line)
                self._save(line)
        code = proc.wait()
        with self.lock:
            self.code = code
            self.ended = time.time()
            self.progress = None
            if self.roadmap is not None:
                self.roadmap.finish(code, self.stopping)
            self.history.append({"session": self.session, "what": self.what,
                                 "outcome": self._outcome(), "code": code,
                                 "stopped": self.stopping, "log": self.log_path})
            if self.log_file is not None:
                done = self._outcome()
                self._save(done[:1].upper() + done[1:])
                self.log_file.close()
                self.log_file = None
            if self.stopping:
                self.held = bool(self.queue)
        self._next()

    def _save(self, line: str) -> None:
        if self.log_file is not None:
            try:
                self.log_file.write(line + "\n")
                self.log_file.flush()
            except OSError:
                self.log_file = None

    # ------------------------------------------------------------------ queue

    def add(self, cfg: Config, source: str, flags: list[str], what: str) -> bool:
        """Lines a session up; False if it's already in the queue as it is."""
        src = Path(source.strip().strip('"')).expanduser()
        with self.lock:
            self.cfg = cfg
            if any(q["source"] == str(src) and q["flags"] == flags for q in self.queue):
                return False
            self.queue.append({"source": str(src), "flags": list(flags), "what": what,
                               "session": src.name if src.is_dir() else src.stem})
        return True

    def remove(self, i: int) -> None:
        with self.lock:
            if 0 <= i < len(self.queue):
                self.queue.pop(i)
            if not self.queue:
                self.held = False

    def move(self, i: int, step: int) -> None:
        with self.lock:
            j = i + step
            if 0 <= i < len(self.queue) and 0 <= j < len(self.queue):
                self.queue[i], self.queue[j] = self.queue[j], self.queue[i]

    def run_queue(self) -> None:
        """Start the queue (after Stop held it, or when it was filled while idle)."""
        self.held = False
        self._next()

    def _next(self) -> None:
        """The next queued session, unless something is running or Stop held the queue."""
        with self.lock:
            if self.running or self.held or not self.queue or self.cfg is None:
                return
            item = self.queue.pop(0)
        try:
            self.start(self.cfg, item["source"], item["flags"], item["what"])
        except RuntimeError as err:
            with self.lock:
                if self.running:                   # Start was pressed meanwhile: wait
                    self.queue.insert(0, item)
                    return
                self.history.append({"session": item["session"], "what": item["what"],
                                     "outcome": str(err), "code": -1, "stopped": False,
                                     "log": None})
            self._next()

    def new_lines(self, seen: int) -> tuple[list[str], int]:
        with self.lock:
            n = min(self.count - seen, len(self.lines))
            return (list(self.lines)[-n:] if n > 0 else []), self.count

    @property
    def paused(self) -> bool:
        return self.running and self.paused_at > 0

    def note(self, text: str) -> None:
        """A line of the app's own in the run's log."""
        with self.lock:
            self.lines.append(f"[gui] {text}")
            self.count += 1
            self._save(f"[gui] {text}")

    def pause(self) -> str | None:
        """Freezes the run where it is. None, or why it couldn't."""
        if not self.running or self.stopping or self.paused_at or self.proc is None:
            return None
        try:
            import psutil
        except ImportError:
            return "Pause needs psutil: pip install psutil"
        try:
            root = psutil.Process(self.proc.pid)
            root.suspend()                     # first, so it starts nothing new
            for child in root.children(recursive=True):
                try:
                    child.suspend()            # ffmpeg and the like
                except psutil.Error:
                    pass
        except psutil.Error as err:
            return f"Couldn't pause the run ({err})"
        self.paused_at = time.time()
        self.note(f"paused {clock(self.paused_at)}; press Resume to carry on")
        return None

    def resume(self) -> None:
        if not self.paused_at or self.proc is None:
            return
        try:
            import psutil
            root = psutil.Process(self.proc.pid)
            for child in root.children(recursive=True):
                try:
                    child.resume()
                except psutil.Error:
                    pass
            root.resume()
        except Exception:  # noqa: BLE001 - it ended meanwhile, or psutil is gone
            pass
        now = time.time()
        self.pauses.append((self.paused_at, now))
        self.paused_at = 0.0
        self.note(f"resumed {clock(now)} after {fmt_hms(self.pauses[-1][1] - self.pauses[-1][0])} paused")

    def paused_between(self, a: float, b: float) -> float:
        """Seconds the run spent paused between times a and b."""
        spans = self.pauses + ([(self.paused_at, b)] if self.paused_at else [])
        return sum(max(0.0, min(end, b) - max(start, a)) for start, end in spans)

    def eta(self, p: dict) -> str:
        """The step's ETA, leaving out the time spent paused."""
        if self.paused_at:
            return "paused"
        if not self.pauses or "_t" not in p:
            return str(p.get("eta", ""))
        t, elapsed = float(p["_t"]), float(p.get("elapsed", 0))
        return _eta_text(float(p.get("frac", 0)),
                         elapsed - self.paused_between(t - elapsed, t), t)

    def stop(self) -> None:
        """Like Ctrl+C: the run saves its work and exits."""
        proc = self.proc
        if proc is None or not self.running:
            return
        self.resume()                          # a frozen run can't save and exit
        self.stopping = True
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(proc.pid, signal.SIGINT)
        except (OSError, ValueError):
            pass
        threading.Timer(30, self._kill, args=(proc,)).start()

    def _kill(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass

    def _outcome(self) -> str:
        """How the run that just ended went, for the queue's list."""
        took = fmt_hms(self.ended - self.started - self.paused_between(self.started, self.ended))
        if self.code == 0:
            return f"finished {clock(self.ended)}, took {took}"
        if self.code in STOP_CODES or self.stopping:
            return f"stopped {clock(self.ended)} after {took}"
        return f"failed {clock(self.ended)} after {took} (exit code {self.code})"

    def status(self) -> str:
        if self.proc is None:
            return "Nothing running."
        if self.running:
            now = time.time()
            took = fmt_hms(now - self.started - self.paused_between(self.started, now))
            if self.paused_at:
                return (f"Paused since {clock(self.paused_at)}: {self.what} for {self.session} "
                        f"({took} of work so far)")
            word = "Stopping" if self.stopping else "Running"
            return f"{word}: {self.what} for {self.session} (started {clock(self.started)}, {took} so far)"
        took = fmt_hms(self.ended - self.started - self.paused_between(self.started, self.ended))
        if self.code == 0:
            return f"Finished {clock(self.ended)}: {self.what} for {self.session} (took {took})"
        if self.code in STOP_CODES or self.stopping:
            return f"Stopped {clock(self.ended)} after {took}. Everything finished so far is saved."
        return f"Failed after {took} (exit code {self.code}). The log below says why."


JOB = Job()


def _videos_in(folder: Path) -> int:
    try:
        return sum(1 for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    except OSError:
        return 0


def _subfolders(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "$"))),
                      key=lambda p: p.name.lower())
    except OSError:
        return []


def _drives() -> list[Path]:
    if os.name != "nt":
        return [Path("/")]
    return [Path(f"{d}:\\") for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]


def _outputs(cfg: Config, session: str) -> list[Path]:
    d = cfg.output_dir / session
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.suffix.lower() == ".mp4"),
                  key=lambda p: (" highlights" not in p.name, p.name.lower()))


def _last_log(cfg: Config, session: str) -> Path | None:
    d = cfg.work_dir / session / "logs"
    try:
        logs = sorted(p for p in d.iterdir() if p.suffix == ".txt")
    except OSError:
        return None
    return logs[-1] if logs else None


def _log_url(cfg: Config, path: Path) -> str:
    rel = path.relative_to(cfg.work_dir)
    return "/media/" + "/".join(quote(part) for part in rel.parts)


def _video_label(session: str, p: Path) -> str:
    """'highlights', 'clip that 2', ... rather than the whole file name."""
    name = p.stem
    if name.startswith(session):
        name = name[len(session):].strip(" -_")
    return name or p.stem


def register(cfg: Config, default_source: str | None = None, lib=None) -> None:
    from nicegui import ui

    @ui.page("/run")
    def page(source: str = "") -> None:
        frame("/run")
        ui.add_head_html(MAP_CSS)
        recent = sessions(cfg)
        start_src = source or default_source or (recent[0].source if recent else "")
        form = {"source": start_src, "make": "highlights",
                "redo": {k: False for k in REDO}}

        def chosen(src: str | None = None) -> tuple[str, list[str], str] | None:
            """The folder and options chosen above, or None (and why) if not usable."""
            src = (src if src is not None else form["source"] or "").strip().strip('"')
            if not src:
                ui.notify("Choose a session folder (or one video) first.", type="warning")
                return None
            if not Path(src).expanduser().exists():
                ui.notify(f"Not found: {src}", type="negative")
                return None
            flags = list(FLAGS[form["make"]])
            redo = [k for k, on in form["redo"].items() if on]
            if redo and form["make"] != "check":
                flags += ["--force", *redo]
            return src, flags, MAKE[form["make"]].split(" (")[0]

        def go() -> None:
            pick = chosen()
            if pick is None:
                return
            try:
                JOB.start(cfg, *pick)
            except RuntimeError as err:
                ui.notify(str(err), type="warning")
                return
            tick()

        def queue_it(folders: list[Path] | None = None) -> None:
            added, already = [], []
            for src in ([str(f) for f in folders] if folders is not None else [None]):
                pick = chosen(src)
                if pick is None:
                    continue
                name = Path(pick[0]).expanduser()
                name = name.name if name.is_dir() else name.stem
                (added if JOB.add(cfg, *pick) else already).append(name)
            if added:
                after = "after the current run" if JOB.running else \
                    "press Start the queue to begin" if not JOB.held else "the queue is on hold"
                ui.notify(f"Queued {', '.join(added)} ({after})", type="positive")
            if already:
                ui.notify(f"Already queued: {', '.join(already)}", type="info")
            tick()

        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
            with ui.card().classes("w-full bg-slate-900 gap-3"):
                ui.label("What to run").classes("text-xl font-bold")
                with ui.row().classes("w-full items-center gap-2"):
                    src_in = ui.input("Session folder or video file", value=start_src)\
                        .bind_value(form, "source").props("outlined dark clearable")\
                        .classes("grow")
                    ui.button("Browse", icon="folder_open",
                              on_click=lambda: browse(form, src_in, queue_it))\
                        .props("no-caps outline")
                if recent:
                    with ui.row().classes("items-center gap-1"):
                        ui.label("Recent:").classes("text-sm text-slate-400")
                        for s in recent[:8]:
                            ui.chip(s.name, icon="history",
                                    on_click=lambda s=s: src_in.set_value(s.source))\
                                .props("outline clickable color=teal")
                ui.radio(MAKE, value="highlights").bind_value(form, "make")\
                    .props("dark color=teal")
                with ui.expansion("Start some stages over (ignores their saved work)")\
                        .classes("w-full text-slate-300"):
                    with ui.row().classes("gap-3"):
                        for k, name in REDO.items():
                            ui.checkbox(name).bind_value(form["redo"], k).props("dark")
                with ui.row().classes("gap-2 items-center"):
                    start_btn = ui.button("Start", icon="play_arrow", on_click=go)\
                        .props("no-caps color=teal")
                    ui.button("Add to queue", icon="playlist_add", on_click=lambda: queue_it())\
                        .props("no-caps color=teal outline")\
                        .tooltip("Line this folder up to run after the others, "
                                 "with the options chosen above")
                    pause_btn = ui.button("Pause", icon="pause", on_click=lambda: toggle_pause())\
                        .props("no-caps color=amber outline")\
                        .tooltip("Freeze the run where it is, then carry on with Resume")
                    stop_btn = ui.button("Stop", icon="stop", on_click=JOB.stop)\
                        .props("no-caps color=negative outline")
                    command = ui.label().classes("vh-key")

            @ui.refreshable
            def queue_card() -> None:
                with JOB.lock:
                    queue, history, held = list(JOB.queue), list(JOB.history), JOB.held
                if not queue and not history:
                    return
                with ui.card().classes("w-full bg-slate-900 gap-1"):
                    with ui.row().classes("w-full items-center gap-3"):
                        ui.label("Queue").classes("text-xl font-bold")
                        if queue and not JOB.running:
                            ui.button("Start the queue", icon="queue_play_next",
                                      on_click=lambda: (JOB.run_queue(), tick()))\
                                .props("no-caps color=teal")
                        if queue and held:
                            ui.label("On hold because you pressed Stop.")\
                                .classes("text-amber-300")
                        elif queue and not JOB.running:
                            ui.label("Waiting for you to start it.").classes("text-slate-400")
                    if not queue:
                        ui.label("Nothing waiting.").classes("text-slate-400")
                    for i, q in enumerate(queue):
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            ui.label(f"{i + 1}.").classes("text-slate-400 w-6")
                            with ui.column().classes("gap-0 grow min-w-0"):
                                extra = [f for f in q["flags"] if f in REDO]
                                ui.label(q["session"] + "  ·  " + q["what"] + (
                                    "  ·  starting over: " + ", ".join(REDO[f] for f in extra)
                                    if extra else "")).classes("font-bold")
                                ui.label(q["source"]).classes("vh-key break-all")
                            ui.button(icon="arrow_upward", on_click=lambda i=i: (JOB.move(i, -1), tick()))\
                                .props("flat dense round").tooltip("Run it sooner")\
                                .set_enabled(i > 0)
                            ui.button(icon="arrow_downward", on_click=lambda i=i: (JOB.move(i, 1), tick()))\
                                .props("flat dense round").tooltip("Run it later")\
                                .set_enabled(i < len(queue) - 1)
                            ui.button(icon="close", on_click=lambda i=i: (JOB.remove(i), tick()))\
                                .props("flat dense round color=negative")\
                                .tooltip("Take it out of the queue")
                    if history:
                        ui.label("Done since the app opened").classes("text-sm text-slate-400 mt-2")
                        for h in reversed(history[-12:]):
                            ok, stopped = h["code"] == 0, h.get("stopped")
                            icon, colour = (("check_circle", "text-green-400") if ok else
                                            ("stop_circle", "text-amber-300") if stopped else
                                            ("error", "text-red-400"))
                            with ui.row().classes("items-center gap-2"):
                                ui.icon(icon).classes(colour)
                                ui.label(f"{h['session']}  ·  {h['what']}: {h['outcome']}")
                                if h["log"] is not None and Path(h["log"]).exists():
                                    ui.link("log", _log_url(cfg, Path(h["log"])), new_tab=True)\
                                        .classes("text-teal-300")

            queue_card()

            with ui.card().classes("w-full bg-slate-900 gap-2"):
                status = ui.label().classes("text-lg")
                with ui.row().classes("w-full items-center gap-3"):
                    stage = ui.label().classes("font-bold min-w-[220px]")
                    bar = ui.linear_progress(value=0, show_value=False)\
                        .props("rounded size=14px color=teal").classes("grow")
                    pct = ui.label().classes("min-w-[48px] text-right")
                    eta = ui.label().classes("text-slate-400 min-w-[170px]")
                log = ui.log(max_lines=1500).classes("w-full h-[420px] bg-black vh-log")

                def map_toggled(e) -> None:
                    MAP_UI["open"] = bool(e.value)
                    state["map"] = None
                with ui.expansion("Road map", icon="map", value=MAP_UI["open"],
                                  on_value_change=map_toggled)\
                        .classes("w-full text-slate-200") as map_exp:
                    map_html = ui.html("", sanitize=False).classes("w-full pt-2")

            @ui.refreshable
            def results() -> None:
                if JOB.proc is None or JOB.running:
                    return
                outs = _outputs(cfg, JOB.session)
                info = find_session(cfg, JOB.session)
                with ui.card().classes("w-full bg-slate-900 gap-2"):
                    ui.label("Results").classes("text-xl font-bold")
                    if outs:
                        ui.label(str(cfg.output_dir / JOB.session)).classes("vh-key")
                        for p in outs:
                            ui.link(p.name, f"/out/{quote(JOB.session)}/{quote(p.name)}",
                                    new_tab=True).classes("text-teal-300")
                    with ui.row().classes("gap-2"):
                        if (cfg.output_dir / JOB.session / "picks.csv").exists():
                            ui.button("Check the picks", icon="checklist",
                                      on_click=lambda: ui.navigate.to(
                                          f"/picks?session={quote(JOB.session)}"))\
                                .props("no-caps outline color=teal")
                        if info and (info.work / "tags" / "current.json").exists():
                            ui.button("Who is who", icon="badge",
                                      on_click=lambda: ui.navigate.to(
                                          f"/who?session={quote(JOB.session)}"))\
                                .props("no-caps outline color=teal")
                        ui.button("Grade this session", icon="grading",
                                  on_click=lambda: ui.navigate.to(
                                      f"/grade?session={quote(JOB.session)}"))\
                            .props("no-caps outline color=teal")

            results()

            @ui.refreshable
            def sessions_card() -> None:
                from . import who as who_screen

                all_ = sessions(cfg)
                if not all_:
                    return
                with JOB.lock:
                    queued = [q["session"] for q in JOB.queue]
                with ui.card().classes("w-full bg-slate-900 gap-1"):
                    ui.label("Your sessions").classes("text-xl font-bold")
                    ui.label("Each date keeps its own videos, picks, nametags and grades.")\
                        .classes("text-sm text-slate-400")
                    for s in all_:
                        ui.separator().classes("my-1")
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label(s.name).classes("text-lg font-bold min-w-[130px]")\
                                .tooltip(s.source)
                            if JOB.running and JOB.session == s.name:
                                ui.badge("paused" if JOB.paused else "running", color="teal")
                            if s.name in queued:
                                ui.badge(f"queued #{queued.index(s.name) + 1}", color="blue-grey")
                            outs = _outputs(cfg, s.name)
                            if outs:
                                for p in outs:
                                    ui.link(_video_label(s.name, p),
                                            f"/out/{quote(s.name)}/{quote(p.name)}",
                                            new_tab=True).classes("text-teal-300").tooltip(p.name)
                            else:
                                ui.label("no videos yet").classes("text-slate-500")
                        with ui.row().classes("items-center gap-2 pl-[138px]"):
                            if (s.out / "picks.csv").exists():
                                ui.button("Picks", icon="checklist",
                                          on_click=lambda s=s: ui.navigate.to(
                                              f"/picks?session={quote(s.name)}"))\
                                    .props("no-caps dense outline color=teal")
                            named = who_screen.named_count(s)
                            if named is not None:
                                n, total = named
                                ui.button(f"Who is who · {n} of {total} named", icon="badge",
                                          on_click=lambda s=s: ui.navigate.to(
                                              f"/who?session={quote(s.name)}"))\
                                    .props(f"no-caps dense outline "
                                           f"color={'teal' if n >= total else 'amber'}")
                            if lib is not None:
                                graded = sum(1 for kinds in lib.hand_labels(s.name).values()
                                             if "highlight" in kinds)
                                ui.button(f"Grade · {graded} graded", icon="grading",
                                          on_click=lambda s=s: ui.navigate.to(
                                              f"/grade?session={quote(s.name)}"))\
                                    .props("no-caps dense outline color=teal")
                            last = _last_log(cfg, s.name)
                            if last is not None:
                                ui.link("last run's log", _log_url(cfg, last), new_tab=True)\
                                    .classes("text-sm text-teal-300").tooltip(last.name)
                            ui.button("Use", icon="input",
                                      on_click=lambda s=s: src_in.set_value(s.source))\
                                .props("no-caps dense flat color=white")\
                                .tooltip("Put this session's folder in the box above")

            sessions_card()

        state = {"seen": 0, "was_running": JOB.running, "map": None, "paused": None,
                 "run": JOB.run_id, "queue": None}

        def toggle_pause() -> None:
            if JOB.paused:
                JOB.resume()
            else:
                err = JOB.pause()
                if err:
                    ui.notify(err, type="warning")
            tick()

        idle_maps: dict[str, Roadmap] = {}

        def show_map(running: bool) -> None:
            if JOB.proc is not None and JOB.roadmap is not None:
                rm = JOB.roadmap
            else:                        # nothing run yet: the plan for the chosen option
                rm = idle_maps.setdefault(form["make"], Roadmap(form["make"]))
            p = JOB.progress if running and rm is JOB.roadmap else None
            if p:
                p = {k: v for k, v in p.items() if k != "_t"} | {"eta": JOB.eta(p)}
            paused = JOB.paused and rm is JOB.roadmap
            key = (id(rm), rm.version, MAP_UI["open"], paused,
                   json.dumps(p, sort_keys=True) if p and MAP_UI["open"] else "")
            if key == state["map"]:
                return
            state["map"] = key
            with JOB.lock:
                where = rm.where(paused)
                content = rm.render(cfg.vlm.window_s, p, paused) if MAP_UI["open"] else None
            map_exp._props["caption"] = where
            map_exp.update()
            if content is not None:
                map_html.set_content(content)

        def tick() -> None:
            if JOB.run_id != state["run"]:       # a new run (Start, or the queue's next one)
                state["run"], state["seen"] = JOB.run_id, 0
                log.clear()
                results.refresh()
            lines, state["seen"] = JOB.new_lines(state["seen"])
            for line in lines:
                m = STAGE_RE.match(line)
                low = line.lower()
                colour = COLOURS.get(m.group(1), "#e5e7eb") if m else "#e5e7eb"
                if low.startswith("error") or "traceback" in low or "] ✗ " in line:
                    colour = "#f87171"
                elif "] ! " in line:
                    colour = "#fbbf24"
                elif "warning" in low:
                    colour = "#fbbf24"
                elif "cached" in low:
                    colour = "#6b7280"
                log.push(line, style=f"color: {colour}")
            running = JOB.running
            start_btn.set_enabled(not running)
            stop_btn.set_enabled(running and not JOB.stopping)
            pause_btn.set_enabled(running and not JOB.stopping)
            paused = JOB.paused
            if paused != state["paused"]:
                state["paused"] = paused
                pause_btn.set_text("Resume" if paused else "Pause")
                pause_btn.props(f"icon={'play_arrow' if paused else 'pause'}")
                bar.props(f"color={'amber' if paused else 'teal'}")
            status.set_text(JOB.status())
            command.set_text(JOB.command)
            p = JOB.progress if running else None
            if p:
                stage.set_text(f"[{p.get('stage', '')}] {p.get('what', '')}")
                stage.style(f"color: {COLOURS.get(p.get('stage', ''), '#ffffff')}")
                bar.set_value(float(p.get("frac", 0)))
                pct.set_text(f"{100 * float(p.get('frac', 0)):.0f}%")
                eta.set_text(JOB.eta(p))
            else:
                stage.set_text("")
                bar.set_value(1.0 if JOB.code == 0 else 0.0)
                pct.set_text("")
                eta.set_text("")
            show_map(running)
            if state["was_running"] and not running:
                results.refresh()
            state["was_running"] = running
            with JOB.lock:
                q = (JOB.run_id, running, paused, JOB.held, len(JOB.history),
                     tuple((i["source"], tuple(i["flags"])) for i in JOB.queue))
            if q != state["queue"]:
                state["queue"] = q
                queue_card.refresh()
                sessions_card.refresh()

        tick()
        ui.timer(0.3, tick)


def browse(form: dict, target, queue_all=None) -> None:
    """A simple folder picker for folders on this PC. queue_all(folders), if
    given, adds a button that queues every folder in here that has videos."""
    from nicegui import ui

    cur = Path(form.get("source") or "").expanduser()
    if not cur.is_dir():
        cur = cur.parent if cur.parent.is_dir() and str(cur) not in ("", ".") else Path.home()
    here = {"path": cur}

    with ui.dialog() as dialog, ui.card().classes("bg-slate-900 w-[640px] max-w-full gap-2"):
        @ui.refreshable
        def body() -> None:
            p = here["path"]
            n = _videos_in(p)
            ui.label(str(p)).classes("font-bold break-all")
            ui.label(f"{n} video{'s' if n != 1 else ''} in this folder")\
                .classes("text-sm text-slate-400")
            with ui.row().classes("gap-1"):
                ui.button("Up", icon="arrow_upward",
                          on_click=lambda: move(p.parent)).props("no-caps dense flat")\
                    .set_enabled(p.parent != p)
                for d in _drives():
                    ui.button(str(d), on_click=lambda d=d: move(d)).props("no-caps dense flat")
                ui.button("Home", on_click=lambda: move(Path.home())).props("no-caps dense flat")
            with_videos = []
            with ui.scroll_area().classes("h-[340px] w-full border border-slate-700 rounded"):
                subs = _subfolders(p)
                if not subs:
                    ui.label("(no folders here)").classes("text-slate-500 p-2")
                for s in subs:
                    vids = _videos_in(s)
                    if vids:
                        with_videos.append(s)
                    label = s.name + (f"   ({vids} video{'s' if vids != 1 else ''})" if vids else "")
                    ui.item(label, on_click=lambda s=s: move(s)).props("clickable dense")
            with ui.row().classes("w-full justify-end gap-2"):
                if queue_all is not None and with_videos:
                    ui.button(f"Queue all {len(with_videos)} folders in here"
                              if len(with_videos) > 1 else f"Queue {with_videos[0].name}",
                              icon="playlist_add",
                              on_click=lambda: (queue_all(with_videos), dialog.close()))\
                        .props("no-caps outline color=teal")\
                        .tooltip("Every folder listed with videos, one after another, "
                                 "with the options chosen on the Run screen")
                ui.space()
                ui.button("Cancel", on_click=dialog.close).props("no-caps flat")
                ui.button("Use this folder", on_click=lambda: pick(p))\
                    .props("no-caps color=teal").set_enabled(n > 0)

        def move(p: Path) -> None:
            here["path"] = p
            body.refresh()

        def pick(p: Path) -> None:
            target.set_value(str(p))
            dialog.close()

        body()
    dialog.open()
