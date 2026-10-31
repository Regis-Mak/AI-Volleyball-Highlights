"""The Progress screen: is the tool getting better at picking? Every ranker
version with its score next to the config weights' score on the same
moments, how many labels you've given, and buttons to retrain or switch
version. Below that, the tuned settings (pipeline/learn/tune.py), whether
the example memory is giving Qwen hints yet (pipeline/learn/memory.py), and
how close the library is to fine-tuning Qwen (pipeline/learn/trainset.py),
with the latest model checks on the test set (pipeline/learn/evaluate.py).

The score is how well a version puts the moments you said were worth showing
above the ones you said weren't (0.5 = a coin flip, 1.0 = perfect). Once the
test set has 20 graded moments it's measured there; before that it's a
five-way cross-check on your labels.
"""
from __future__ import annotations

from datetime import datetime
from urllib.parse import quote

from pipeline.common import fmt_hms
from pipeline.learn import evaluate, grading, memory, ranker, trainset, tune
from pipeline.library import Library, summary

from .shell import frame

SOURCES = [("hand", "graded by you"), ("gold", "test set"), ("cut", "cut in picks"),
           ("trim", "trimmed"), ("kept", "kept"), ("agree", "auto-labelled")]


def _when(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %I:%M %p").replace(" 0", " ")
    except ValueError:
        return iso


def _num(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def register(lib: Library) -> None:
    from nicegui import run, ui

    @ui.page("/progress")
    def page() -> None:
        frame("/progress")
        msg = {"text": "", "tune": "", "train": ""}

        async def retrain() -> None:
            msg["text"] = "Training..."
            body.refresh()
            msg["text"] = await run.io_bound(ranker.retrain, lib, True)
            body.refresh()

        def use(name: str | None) -> None:
            lib.set_active("ranker", name)
            msg["text"] = (f"Ranker {name} is active from the next run." if name else
                           "Scoring with the config weights only from the next run.")
            body.refresh()

        async def retune() -> None:
            msg["tune"] = "Tuning..."
            body.refresh()
            msg["tune"] = await run.io_bound(tune.retune, lib, True)
            body.refresh()

        async def export() -> None:
            msg["train"] = "Writing the training set..."
            body.refresh()
            try:
                folder = await run.io_bound(trainset.export, lib)
                msg["train"] = f"Wrote the training set to {folder}"
            except Exception as err:  # noqa: BLE001 - shown on the page
                msg["train"] = str(err)
            body.refresh()

        def use_settings(name: str | None) -> None:
            lib.set_active(tune.KIND, name)
            msg["tune"] = (f"Settings {name} are on from the next run." if name else
                           "config.yaml as written from the next run.")
            body.refresh()

        @ui.refreshable
        def body() -> None:
            counts = lib.counts()
            done, target = grading.test_progress(lib)
            versions = lib.versions("ranker")
            active = lib.active_version("ranker")

            ui.label(summary(lib)).classes("text-slate-300")
            with ui.row().classes("w-full gap-3"):
                for key, name in SOURCES:
                    with ui.card().classes("bg-slate-900 min-w-[120px] items-center gap-0 p-3"):
                        ui.label(str(counts.get(key, 0))).classes("text-2xl font-bold")
                        ui.label(name).classes("text-xs text-slate-400")
                with ui.card().classes("bg-slate-900 min-w-[200px] gap-1 p-3"):
                    ui.label(f"Test set: {done} of {target}").classes("font-bold")
                    ui.linear_progress(value=min(1.0, done / max(1, target)), show_value=False)\
                        .props("rounded color=purple")
                    ui.button("Grade more for it", on_click=lambda: ui.navigate.to("/test"))\
                        .props("no-caps dense flat color=purple")

            with ui.card().classes("w-full bg-slate-900 gap-2"):
                with ui.row().classes("w-full items-center"):
                    ui.label("Ranker versions").classes("text-xl font-bold")
                    ui.space()
                    ui.label("Active: " + (active["name"] if active else "config weights only"))\
                        .classes("text-teal-300")
                if not versions:
                    ui.label("No versions yet. The first one trains once you've given about "
                             f"{lib.cfg.learning.ranker.min_labels} labels (Grade screen, or "
                             "edits in picks).").classes("text-slate-400")
                else:
                    names = [v["name"] for v in versions]
                    ui.echart({
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis"},
                        "legend": {"textStyle": {"color": "#cbd5e1"}},
                        "xAxis": {"type": "category", "data": names,
                                  "axisLabel": {"color": "#cbd5e1"}},
                        "yAxis": {"type": "value", "min": 0.5, "max": 1.0,
                                  "axisLabel": {"color": "#cbd5e1"},
                                  "splitLine": {"lineStyle": {"color": "#334155"}}},
                        "series": [
                            {"name": "this version", "type": "line", "smooth": True,
                             "data": [v["metric"] for v in versions], "color": "#2dd4bf"},
                            {"name": "config weights", "type": "line", "smooth": True,
                             "lineStyle": {"type": "dashed"},
                             "data": [v["baseline"] for v in versions], "color": "#94a3b8"},
                        ],
                    }).classes("w-full h-[280px]")
                    with ui.grid(columns=7).classes("w-full gap-x-4 gap-y-1 items-center text-sm"):
                        for h in ("Version", "Made", "Labels", "Score", "Config weights",
                                  "Checked on", ""):
                            ui.label(h).classes("text-slate-400 font-bold")
                        for v in reversed(versions):
                            on = bool(v["active"])
                            ui.label(v["name"] + (" (active)" if on else ""))\
                                .classes("font-bold text-teal-300" if on else "")
                            ui.label(_when(v["created"]))
                            ui.label(str(v["n_labels"]))
                            ui.label(_num(v["metric"]))
                            ui.label(_num(v["baseline"]))
                            ui.label(v["checked_on"] or "").tooltip(v["note"] or "")
                            if on:
                                ui.label("")
                            else:
                                ui.button("Use this one", on_click=lambda n=v["name"]: use(n))\
                                    .props("no-caps dense outline color=teal")
                with ui.row().classes("gap-2 mt-2"):
                    ui.button("Retrain now", icon="model_training", on_click=retrain)\
                        .props("no-caps color=teal")
                    ui.button("Use the config weights only", on_click=lambda: use(None))\
                        .props("no-caps flat").set_enabled(active is not None)
                if msg["text"]:
                    ui.label(msg["text"]).classes("text-slate-200")

            tuned = lib.versions(tune.KIND)
            on = lib.active_version(tune.KIND)
            tc = lib.cfg.learning.tune
            with ui.card().classes("w-full bg-slate-900 gap-2"):
                with ui.row().classes("w-full items-center"):
                    ui.label("Tuned settings").classes("text-xl font-bold")
                    ui.space()
                    ui.label("On: " + (on["name"] if on else "config.yaml as written"))\
                        .classes("text-teal-300")
                ui.label("Scoring weights, clip lengths and the reel cutoff, re-picked from "
                         "your labels. config.yaml itself never changes, and anything you "
                         "edit there later wins.").classes("text-sm text-slate-400")
                if not tuned:
                    ui.label(f"No versions yet. The scoring is tuned once about "
                             f"{tc.min_labels} moments have labels, and a label's clip length "
                             f"after {tc.min_trims} trims.").classes("text-slate-400")
                else:
                    with ui.element("div").classes("w-full gap-x-4 gap-y-1 items-center "
                                                   "text-sm").style(
                            "display: grid; grid-template-columns: auto auto auto 1fr auto"):
                        for h in ("Version", "Made", "Labels", "What changed", ""):
                            ui.label(h).classes("text-slate-400 font-bold")
                        for v in reversed(tuned):
                            active = bool(v["active"])
                            ui.label(v["name"] + (" (on)" if active else ""))\
                                .classes("font-bold text-teal-300" if active else "")
                            ui.label(_when(v["created"]))
                            ui.label(str(v["n_labels"]))
                            ui.label(v["note"] or "")
                            if active:
                                ui.label("")
                            else:
                                ui.button("Use this one",
                                          on_click=lambda n=v["name"]: use_settings(n))\
                                    .props("no-caps dense outline color=teal")
                with ui.row().classes("gap-2 mt-2"):
                    ui.button("Tune now", icon="tune", on_click=retune)\
                        .props("no-caps color=teal")
                    ui.button("Use config.yaml as written", on_click=lambda: use_settings(None))\
                        .props("no-caps flat").set_enabled(on is not None)
                if msg["tune"]:
                    ui.label(msg["tune"]).classes("text-slate-200")

            with ui.card().classes("w-full bg-slate-900 gap-1"):
                ui.label("Example memory").classes("text-xl font-bold")
                ui.label(memory.status(lib)).classes("text-slate-300")

            n, need = trainset.readiness(lib)
            with ui.card().classes("w-full bg-slate-900 gap-2"):
                with ui.row().classes("w-full items-center"):
                    ui.label("Fine-tuning Qwen (Phase 7)").classes("text-xl font-bold")
                    ui.space()
                    ui.label(f"{n} of about {need} labelled moments").classes("text-amber-300")
                ui.linear_progress(value=min(1.0, n / max(1, need)), show_value=False)\
                    .props("rounded color=amber")
                ui.label(trainset.status(lib)).classes("text-slate-300")
                runs = evaluate.history(lib, 3)
                if runs:
                    ui.label("Latest model checks on the test set (0.5 = guessing, 1 = "
                             "perfect)").classes("text-sm text-slate-400 mt-2")
                    with ui.element("div").classes("w-full gap-x-4 gap-y-1 items-center "
                                                   "text-sm").style(
                            "display: grid; grid-template-columns: auto 1fr auto auto auto auto"):
                        for h in ("When", "Model", "Label", "Highlight", "Funny", "Overall"):
                            ui.label(h).classes("text-slate-400 font-bold")
                        for data in runs:
                            for model, res in data["results"].items():
                                s = res["scores"]
                                ui.label(_when(data["created"]))
                                ui.label(model).classes("font-bold")
                                for key in ("label", "highlight", "funny", "overall"):
                                    ui.label(_num(s.get(key)))
                else:
                    ui.label("No model checks yet. Once the test set has 10 graded moments, "
                             "run: python make_highlights.py --evaluate")\
                        .classes("text-sm text-slate-400")
                with ui.row().classes("gap-2 mt-2"):
                    ui.button("Write the training set", icon="model_training", on_click=export)\
                        .props("no-caps color=amber text-color=black")
                if msg["train"]:
                    ui.label(msg["train"]).classes("text-slate-200")

            rows = lib.q("SELECT * FROM sessions ORDER BY last_run DESC")
            if rows:
                with ui.card().classes("w-full bg-slate-900 gap-1"):
                    ui.label("Sessions").classes("text-xl font-bold")
                    with ui.grid(columns=5).classes("w-full gap-x-4 gap-y-1 items-center text-sm"):
                        for h in ("Session", "Files", "Footage", "Last run", ""):
                            ui.label(h).classes("text-slate-400 font-bold")
                        for r in rows:
                            ui.label(r["name"]).classes("font-bold")
                            ui.label(str(r["n_files"]))
                            ui.label(fmt_hms(r["footage_s"] or 0))
                            ui.label(_when(r["last_run"]))
                            with ui.row().classes("gap-1"):
                                for path, text in (("/picks", "Picks"), ("/grade", "Grade"),
                                                   ("/who", "Who is who")):
                                    ui.button(text, on_click=lambda p=path, n=r["name"]:
                                              ui.navigate.to(f"{p}?session={quote(n)}"))\
                                        .props("no-caps dense flat color=teal")

        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-3"):
            ui.label("Progress").classes("text-2xl font-bold")
            body()
