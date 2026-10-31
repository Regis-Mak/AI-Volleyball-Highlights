"""What every screen shares: the top bar with the screen links, the colours,
and the list of sessions the tool has seen."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pipeline.config import Config

SCREENS = [("/run", "Run"), ("/grade", "Grade"), ("/test", "Test set"), ("/picks", "Picks"),
           ("/who", "Who is who"), ("/progress", "Progress")]

STYLE = """
<style>
  .vh-key { font-family: ui-monospace, monospace; font-size: 0.75rem; opacity: 0.7; }
  .vh-nav a { text-decoration: none; }
  .vh-log { font-family: ui-monospace, monospace; font-size: 0.8rem; }
  .vh-log .q-scrollarea__content { width: 100%; }
  .vh-log .q-scrollarea__content > div { white-space: pre-wrap; overflow-wrap: anywhere; }
  .vh-drop { outline: 2px dashed #2dd4bf; outline-offset: 2px; }
</style>
"""


def frame(active: str) -> None:
    """The page background and top bar; call first on every screen."""
    from nicegui import ui

    ui.add_head_html(STYLE)
    ui.query("body").classes("bg-slate-950")
    with ui.header().classes("bg-slate-900 items-center gap-1 px-4 py-1 vh-nav"):
        ui.label("🏐 Volleyball highlights").classes("text-lg font-bold mr-4")
        for path, name in SCREENS:
            on = path == active
            ui.button(name, on_click=lambda path=path: ui.navigate.to(path))\
                .props(f"flat no-caps dense {'color=teal' if on else 'color=white'}")\
                .classes("font-bold" if on else "opacity-80")


@dataclass
class SessionInfo:
    name: str
    source: str
    files: list[dict]
    work: Path
    out: Path
    modified: float

    def offset(self, file_name: str) -> float:
        for f in self.files:
            if f["name"] == file_name:
                return float(f["offset"])
        return 0.0

    def span(self, file_name: str) -> tuple[float, float] | None:
        """Where a file starts and ends on the session clock."""
        for f in self.files:
            if f["name"] == file_name:
                return float(f["offset"]), float(f["offset"]) + float(f["duration"])
        return None

    def stem(self, file_name: str) -> str:
        for f in self.files:
            if f["name"] == file_name:
                return f["stem"]
        return Path(file_name).stem


def sessions(cfg: Config) -> list[SessionInfo]:
    """Every session the tool has opened (work/<session>/session.json), newest first."""
    out = []
    root = cfg.work_dir
    if not root.is_dir():
        return out
    for d in root.iterdir():
        meta = d / "session.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(SessionInfo(d.name, data.get("source", ""), data.get("files", []), d,
                               cfg.output_dir / d.name, meta.stat().st_mtime))
    out.sort(key=lambda s: s.modified, reverse=True)
    return out


def find_session(cfg: Config, name: str | None) -> SessionInfo | None:
    all_ = sessions(cfg)
    if name:
        return next((s for s in all_ if s.name == name), None)
    return all_[0] if all_ else None


def session_picker(cfg: Config, current: SessionInfo | None, path: str,
                   want=None) -> None:
    """A drop-down of sessions that reloads the screen on another one.
    want(s) filters the list (e.g. only sessions with picks)."""
    from nicegui import ui

    options = {s.name: s.name for s in sessions(cfg) if want is None or want(s)}
    if current is not None and current.name not in options:
        options[current.name] = current.name
    if not options:
        return
    ui.select(options, value=current.name if current else None, label="Session",
              on_change=lambda e: ui.navigate.to(f"{path}?session={quote(e.value)}"))\
        .props("dense outlined dark").classes("min-w-[240px]")
