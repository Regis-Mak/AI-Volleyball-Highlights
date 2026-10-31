"""The road map under the Run screen's log: every step of a run in order,
which are done, and where the run is now ("video 2 of 7, you are here").

It follows the run's own log lines ("[session] audio: file 2 of 7, ...",
"[fuse] ...", "[render] wrote ..."), so the pipeline needs nothing extra and
the map is rebuilt exactly when the page is reopened mid-run.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

# (name, what it does); {window} is the vision window length in seconds.
STEPS = {
    "ingest": ("Ingest", "Makes a 720p copy of each video and cleans up the audio (wind noise)."),
    "audio": ("Audio", "Whisper writes down what everyone says; PANNs hears laughing, cheering "
                       "and shouting. Together they mark the group's reactions."),
    "memory": ("Example memory", "A picture fingerprint for every second, so Qwen can be shown "
                                 "similar moments from earlier games and how you judged them."),
    "vision": ("Vision", "Skips stretches with no movement, then Qwen watches each {window}-second "
                         "window and labels it, with excitement and funny scores. The slow part."),
    "comments": ("Unhinged comments", "Qwen reads what everyone said and rates how unhinged each "
                                      "comment is; a \"whaaat?!\" or laughter right after a "
                                      "line makes it count for more. Quick: no pictures."),
    "fuse": ("Score the moments", "Combines Qwen's labels with the group's reactions into "
                                  "scored moments, each with its own clip length."),
    "select": ("Choose clips", "Picks the best moments for the highlight reel, unhinged "
                               "comments included (and a clip for every \"clip that\"), and "
                               "writes picks.csv with a thumbnail for every clip."),
    "render": ("Make the videos", "Cuts the reel and the \"clip that\" clips, adds nametags over "
                                  "the players, and mixes in dissolves and swipes."),
    "learn": ("Learn", "Saves every moment to the library, then retrains the ranker and re-tunes "
                       "the scoring once there are enough of your labels."),
    "review": ("Make the review video", ""),
}
REVIEW_WHAT = {"peaks": "A video of the top reaction peaks, to check by eye.",
               "tags": "A video of a sample of Qwen's labels, to check by eye."}
PLANS = {
    "highlights": ["ingest", "audio", "memory", "vision", "comments", "fuse", "select", "render",
                   "learn"],
    "picks": ["ingest", "audio", "memory", "vision", "comments", "fuse", "select", "learn"],
    "peaks": ["ingest", "audio", "review"],
    "tags": ["ingest", "audio", "memory", "vision", "review"],
}
PER_VIDEO = ("ingest", "audio", "vision", "comments")
# A log line's [tag] -> the step it belongs to.
TAG_STEP = {"ingest": "ingest", "audio": "audio", "memory": "memory", "motion": "vision",
            "vlm": "vision", "comments": "comments", "fuse": "fuse", "select": "select",
            "render": "render", "track": "render", "tags": "render", "review": "review"}
LEARN_TAGS = ("library", "learn", "tune")
STOP_CODES = (130, -2, 3221225786)    # Ctrl+C here, on Linux, on Windows

EACH_RE = re.compile(
    r"^\[session\] (ingest|audio|vision|comments): file (\d+) of (\d+), (.+)$")
COUNT_RE = re.compile(r"^\[session\] .+: (\d+) files, ")
LIST_RE = re.compile(r"^\[session\]\s+(\d+)\. (.+?)  ")
TAG_RE = re.compile(r"^\s*\[(\w+)\] ?(.*)$")
FINGERPRINT_RE = re.compile(r"^picture fingerprints, (.+): started")
MAKING_RE = re.compile(r"^\d+ clips?, [\d.]+ min -> (.+)$")
PLANNED_RE = re.compile(r"^(.+\.mp4): \d+ clips?, ")


def mode_for(flags: list[str]) -> str:
    """Which plan a run follows, from its command-line flags."""
    if "--check" in flags:
        return "check"
    if "--pick-only" in flags:
        return "picks"
    if "--review" in flags:
        i = flags.index("--review") + 1
        return flags[i] if i < len(flags) and flags[i] in PLANS else "highlights"
    return "highlights"


@dataclass
class Step:
    key: str
    status: str = "todo"          # todo | running | done | failed | stopped
    detail: str = ""
    seen: bool = False


class Roadmap:
    def __init__(self, mode: str):
        self.mode = mode
        self.steps = [Step(k) for k in PLANS.get(mode, [])]
        self.pos = {s.key: i for i, s in enumerate(self.steps)}
        self.current = -1
        self.total = 1                       # videos in the session
        self.order: dict[str, int] = {}      # file stem -> its place in the session
        self.planned = self.made = 0
        self.learn_msgs: dict[str, str] = {}
        self.ended = ""                      # "" while running, then finished | failed | stopped
        self.version = 0                     # bumps on every change, so pages redraw only then

    # ------------------------------------------------------------ following

    def feed(self, line: str) -> None:
        if not self.steps or self.ended:
            return
        m = EACH_RE.match(line)
        if m:
            step, i, n, name = m.groups()
            self.total = int(n)
            self._go(step, f"video {i} of {n}: {name}")
            return
        m = COUNT_RE.match(line)
        if m:
            self.total = int(m.group(1))
            return
        m = LIST_RE.match(line)
        if m:
            self.order[Path(m.group(2)).stem] = int(m.group(1))
            return
        m = TAG_RE.match(line)
        if not m:
            return
        tag, rest = m.groups()
        step = self._step_for(tag)
        if step is None or self.pos[step] < self.current:
            return
        self._go(step)
        self._detail(step, tag, rest.strip())

    def _step_for(self, tag: str) -> str | None:
        if tag in LEARN_TAGS:
            # [learn] and [library] lines also appear mid-run (the ranker scoring
            # moments, frames being saved); only the ones after the last step count.
            learn = self.pos.get("learn")
            return "learn" if learn is not None and self.current >= learn - 1 else None
        step = TAG_STEP.get(tag)
        if step == "render" and "review" in self.pos:
            step = "review"
        return step if step in self.pos else None

    def _go(self, key: str, detail: str | None = None) -> None:
        i = self.pos[key]
        if i < self.current:
            return
        for s in self.steps[max(self.current, 0):i]:
            self._close(s)
        s = self.steps[i]
        if s.status != "running" or (detail and detail != s.detail):
            self.version += 1
        s.status, s.seen = "running", True
        if detail:
            s.detail = detail
        elif s.key in PER_VIDEO and self.total == 1 and not s.detail:
            s.detail = "1 video"
        self.current = i

    def _close(self, s: Step) -> None:
        if s.status in ("done", "failed", "stopped"):
            return
        if s.key in PER_VIDEO and s.seen:
            s.detail = f"all {self.total} videos" if self.total > 1 else "1 video"
        elif not s.seen:
            s.detail = "nothing to do this time"
        s.status = "done"
        self.version += 1

    def _detail(self, key: str, tag: str, rest: str) -> None:
        s = self.steps[self.pos[key]]
        old = s.detail
        if key == "memory":
            m = FINGERPRINT_RE.match(rest)
            if m:
                stem = m.group(1)
                k = self.order.get(stem)
                s.detail = f"video {k} of {self.total}: {stem}" if k else stem
            elif rest.startswith("loading") or "labelled moments" in rest:
                s.detail = rest
        elif key == "fuse" and re.match(r"^\d+ moments", rest):
            s.detail = rest
        elif key == "select":
            if PLANNED_RE.match(rest):
                self.planned += 1
                s.detail = f"{self.planned} video{'s' if self.planned != 1 else ''} planned"
            elif rest.startswith("picks only"):
                s.detail = "picks.csv is ready to check"
        elif key in ("render", "review"):
            m = MAKING_RE.match(rest)
            if m:
                done = (f" ({self.made} of {self.planned} done)"
                        if self.planned and self.made <= self.planned else "")
                s.detail = f"making {m.group(1)}{done}"
            elif rest.startswith("wrote "):
                self.made += 1
                name = Path(rest[6:].strip()).name
                s.detail = (f"{self.made} of {self.planned} videos made, last: {name}"
                            if self.planned and self.made <= self.planned
                            else f"{self.made} video{'s' if self.made != 1 else ''} made, "
                                 f"last: {name}")
        elif key == "learn" and tag in ("learn", "tune") and rest:
            self.learn_msgs[tag] = rest
            s.detail = "\n".join(self.learn_msgs[t] for t in ("learn", "tune")
                                if t in self.learn_msgs)
        if s.detail != old:
            self.version += 1

    def finish(self, code: int | None, stopped: bool) -> None:
        if not self.steps or self.ended:
            return
        cur = self.steps[self.current] if self.current >= 0 else None
        if code == 0 and not stopped:
            for s in self.steps:
                self._close(s)
            self.ended = "finished"
        elif stopped or code in STOP_CODES:
            if cur:
                cur.status = "stopped"
            self.ended = "stopped"
        else:
            if cur:
                cur.status = "failed"
            self.ended = "failed"
        self.version += 1

    # ------------------------------------------------------------- showing

    def where(self, paused: bool = False) -> str:
        """One line for the drop-down's caption: where the run is."""
        if not self.steps:
            return ""
        n = len(self.steps)
        if self.current < 0:
            return f"{n} steps"
        s = self.steps[self.current]
        name = STEPS[s.key][0]
        if self.ended == "finished":
            return f"all {n} steps done"
        if self.ended == "stopped":
            return f"stopped at step {self.current + 1} of {n}, {name}"
        if self.ended == "failed":
            return f"failed at step {self.current + 1} of {n}, {name}"
        detail = s.detail.split(":")[0] if s.detail.startswith("video ") else ""
        return (f"{'paused at ' if paused else ''}step {self.current + 1} of {n}, {name}"
                + (f" ({detail})" if detail else ""))

    def render(self, window_s: float, progress: dict | None = None,
               paused: bool = False) -> str:
        if self.mode == "check":
            return ('<div class="vh-map"><p class="vh-map-note">The setup check processes '
                    'nothing, so it has no road map. Pick another option to see its steps.</p>'
                    '</div>')
        rows = []
        for i, s in enumerate(self.steps):
            name, what = STEPS[s.key]
            what = REVIEW_WHAT.get(self.mode, "") if s.key == "review" else what
            what = what.replace("{window}", f"{window_s:g}")
            pill = {"todo": "to do", "running": "paused" if paused else "running",
                    "done": "done", "failed": "stopped by an error",
                    "stopped": "stopped"}[s.status]
            dot = "✓" if s.status == "done" else str(i + 1)
            here = ('<span class="vh-map-here">◀ you are here</span>'
                    if s.status == "running" else "")
            extra = ""
            if s.detail:
                extra += f'<p class="vh-map-detail">{html.escape(s.detail)}</p>'
            if s.status == "running" and progress:
                frac = float(progress.get("frac", 0))
                extra += (
                    '<div class="vh-map-prog"><div class="vh-map-track"><div class="vh-map-fill" '
                    f'style="width:{100 * frac:.1f}%"></div></div>'
                    f'<span>{html.escape(str(progress.get("what", "")))} · {100 * frac:.0f}%'
                    f'{" · " + html.escape(str(progress.get("eta", ""))) if progress.get("eta") else ""}'
                    '</span></div>')
            if s.status == "stopped":
                extra += ('<p class="vh-map-detail">Press Start again (no boxes ticked) to carry '
                          'on from here.</p>')
            elif s.status == "failed":
                extra += '<p class="vh-map-detail">The log above says why.</p>'
            rows.append(
                f'<li class="vh-map-{s.status}"><div class="vh-map-dot">{dot}</div>'
                f'<div class="vh-map-box"><div class="vh-map-row"><span class="vh-map-name">'
                f'{name}</span><span class="vh-map-pill">{pill}</span>{here}</div>'
                f'<p class="vh-map-what">{html.escape(what)}</p>{extra}</div></li>')
        return f'<div class="vh-map"><ol>{"".join(rows)}</ol></div>'


CSS = """
<style>
  .vh-map ol { list-style: none; margin: 0; padding: 0; }
  .vh-map li { position: relative; display: flex; gap: 12px; padding-bottom: 10px; }
  .vh-map li:not(:last-child)::before { content: ''; position: absolute; left: 15px; top: 32px;
    bottom: 0; width: 2px; background: #334155; }
  .vh-map-dot { flex: 0 0 32px; height: 32px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-weight: 700; font-size: 14px;
    background: #1e293b; color: #94a3b8; border: 2px solid #334155; box-sizing: border-box; }
  .vh-map-box { flex: 1; background: #111827; border: 1px solid #334155; border-radius: 10px;
    padding: 8px 12px; }
  .vh-map-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .vh-map-name { font-weight: 650; font-size: 0.95rem; color: #e2e8f0; }
  .vh-map-pill { font-size: 0.72rem; font-weight: 650; padding: 1px 8px; border-radius: 999px;
    background: #1e293b; color: #94a3b8; }
  .vh-map-here { font-size: 0.75rem; font-weight: 700; color: #fbbf24; }
  .vh-map-what { margin: 3px 0 0; font-size: 0.83rem; color: #94a3b8; line-height: 1.4; }
  .vh-map-detail { margin: 4px 0 0; font-size: 0.85rem; color: #e2e8f0; overflow-wrap: anywhere; white-space: pre-line; }
  .vh-map-note { margin: 0; color: #94a3b8; }
  .vh-map-done .vh-map-dot { background: #14532d; color: #4ade80; border-color: #22c55e; }
  .vh-map-done .vh-map-pill { background: #14532d; color: #4ade80; }
  .vh-map-running .vh-map-dot { background: #134e4a; color: #5eead4; border-color: #2dd4bf; }
  .vh-map-running .vh-map-box { border: 2px solid #2dd4bf; }
  .vh-map-running .vh-map-pill { background: #134e4a; color: #5eead4; }
  .vh-map-stopped .vh-map-dot, .vh-map-stopped .vh-map-pill { background: #451a03; color: #fbbf24;
    border-color: #f59e0b; }
  .vh-map-failed .vh-map-dot, .vh-map-failed .vh-map-pill { background: #450a0a; color: #f87171;
    border-color: #ef4444; }
  .vh-map-prog { display: flex; align-items: center; gap: 10px; margin-top: 6px;
    font-size: 0.8rem; color: #cbd5e1; }
  .vh-map-track { flex: 0 0 160px; height: 8px; border-radius: 4px; background: #1e293b;
    overflow: hidden; }
  .vh-map-fill { height: 100%; background: #2dd4bf; }
</style>
"""
