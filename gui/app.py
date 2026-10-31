"""The local browser app: opens at http://localhost:8765 and runs entirely on
this PC.

    python make_highlights.py --gui                   the app, on the Run screen
    python make_highlights.py --gui "...\\06-14-2026"   with that session filled in
    python make_highlights.py --grade                 straight to Grade
    python make_highlights.py --grade test            straight to the Test set

Screens (links along the top):
  Run          pick a session folder (or queue several), choose what to make,
               watch it work, stop it; every session's videos and screens
  Grade        the moments the tool is least sure about (gui/grade.py)
  Test set     grade the fixed set every ranker version is checked against
  Picks        every clip in every video: take out, trim, reorder, remake
  Who is who   name the people the tool outlined, for the nametags
  Progress     ranker versions and how well each one picks

Stop it with Ctrl+C in the terminal. Every answer and edit is saved the
moment you make it, so closing the page or stopping the app never loses work.
A run started on the Run screen keeps going if you close the page.
"""
from __future__ import annotations

from pipeline.common import PipelineError, log
from pipeline.config import Config
from pipeline.library import open_library, summary


def serve(cfg: Config, start: str = "/run", session: str | None = None,
          source: str | None = None) -> None:
    try:
        from nicegui import app, ui
    except ImportError as err:
        raise PipelineError("The app needs NiceGUI. Install it with:\n"
                            "    pip install nicegui") from err
    from urllib.parse import quote

    from . import grade, picks, progress, run, who
    from .shell import frame

    lib = open_library(cfg)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    app.add_media_files("/media", cfg.work_dir)
    app.add_media_files("/out", cfg.output_dir)

    run.register(cfg, source, lib)
    picks.register(cfg)
    who.register(cfg)
    if lib is not None:
        grade.register(lib)
        progress.register(lib)
        log(f"[gui] library: {summary(lib)}")
    else:
        for path in ("/grade", "/test", "/progress"):
            @ui.page(path)
            def off(path: str = path) -> None:
                frame(path)
                ui.label("Learning is switched off (learning.enabled in config.yaml), so "
                         "there's nothing to grade or track.").classes("p-6 text-lg")

    first = start + (f"?session={quote(session)}" if session else "")

    @ui.page("/")
    def home() -> None:
        ui.navigate.to(first)

    port = cfg.learning.grade.port
    log(f"[gui] opening http://localhost:{port}{start} in your browser; Ctrl+C here to stop")
    app.on_shutdown(run.JOB.stop)
    ui.run(title="Volleyball highlights", host="127.0.0.1", port=port, dark=True,
           reload=False, show=first, show_welcome_message=False, favicon="🏐")
