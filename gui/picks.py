"""The Picks screen: every clip in every video, with its thumbnail. Take clips
out, trim them, drag them into a different order, then make the videos again.

It edits the same picks.csv you can open in Excel (keep, clip_start_s,
clip_end_s), so everything there still applies: a clip taken out is replaced
by the next best moment on the next run, a trimmed clip changes everywhere it
is used, and both teach the ranker. The order goes to work/<session>/order.json.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from urllib.parse import quote

from pipeline.common import fmt_time, parse_time
from pipeline.config import Config
from pipeline.selection import NO, ORDER_FILE

from .grade import MIN_CLIP_S, NUDGE_S, SCRIPT
from .run import JOB
from .shell import SessionInfo, find_session, frame, session_picker



@dataclass
class Row:
    data: dict
    keep: bool
    start: float
    end: float

    @property
    def id(self) -> str:
        return self.data.get("id", "")


@dataclass
class ItemRows:
    rows: list[Row]


@dataclass
class VideoRows:
    name: str
    key: str
    items: list[ItemRows] = field(default_factory=list)
    out: list[Row] = field(default_factory=list)      # taken out earlier (keep = n)

    def length(self) -> float:
        return sum(max(r.end for r in i.rows if r.keep) - min(r.start for r in i.rows if r.keep)
                   for i in self.items if any(r.keep for r in i.rows))


class Picks:
    def __init__(self, cfg: Config, s: SessionInfo):
        self.cfg, self.s = cfg, s
        self.path = s.out / "picks.csv"
        self.columns: list[str] = []
        self.videos: list[VideoRows] = []
        self.dirty = False
        self.load()

    def load(self) -> None:
        with open(self.path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            self.columns = list(reader.fieldnames or [])
            raw = list(reader)
        by_name: dict[str, VideoRows] = {}
        groups: dict[tuple[str, str], ItemRows] = {}
        for r in raw:
            v = by_name.get(r["video"])
            if v is None:
                v = by_name[r["video"]] = VideoRows(r["video"], r.get("theme", ""))
            try:
                start, end = float(r["clip_start_s"]), float(r["clip_end_s"])
            except (KeyError, ValueError):
                continue
            keep = (r.get("keep") or "y").strip().lower() not in NO
            row = Row(r, keep, start, end)
            order = (r.get("order") or "").strip()
            if not keep or order in ("", "-"):
                v.out.append(row)
                continue
            g = groups.get((v.name, order))
            if g is None:
                g = groups[(v.name, order)] = ItemRows([])
                v.items.append(g)
            g.rows.append(row)
        self.videos = list(by_name.values())
        self.dirty = False

    def all_rows(self) -> list[Row]:
        return [r for v in self.videos for i in v.items for r in i.rows] + \
               [r for v in self.videos for r in v.out]

    def set_bounds(self, mid: str, start: float, end: float) -> None:
        """A moment's clip is the same in every video it's in."""
        for r in self.all_rows():
            if r.id == mid:
                r.start, r.end = start, end
        self.dirty = True

    def save(self) -> None:
        rows = []
        for v in self.videos:
            for n, item in enumerate(v.items, 1):
                for r in item.rows:
                    rows.append(self._out(r, n if r.keep else "-"))
            for r in v.out:
                rows.append(self._out(r, "-"))
        with open(self.path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        order: dict[str, list[str]] = {}
        for v in self.videos:
            ids = order.setdefault(v.key, [])
            for item in v.items:
                ids += [r.id for r in item.rows if r.id not in ids]
        (self.s.work / ORDER_FILE).write_text(json.dumps(order, indent=1), encoding="utf-8")
        self.dirty = False

    @staticmethod
    def _out(r: Row, order) -> dict:
        d = dict(r.data)
        d["keep"] = "y" if r.keep else "n"
        d["order"] = order
        d["clip_start_s"], d["clip_end_s"] = f"{r.start:.1f}", f"{r.end:.1f}"
        d["length_s"] = f"{r.end - r.start:.1f}"
        return d


def _has_picks(s: SessionInfo) -> bool:
    return (s.out / "picks.csv").exists()


def register(cfg: Config) -> None:
    from nicegui import ui

    @ui.page("/picks")
    def page(session: str = "") -> None:
        frame("/picks")
        ui.add_body_html(SCRIPT)
        s = find_session(cfg, session or None)
        if s is not None and not _has_picks(s) and not session:
            s = next((x for x in _sessions_with_picks(cfg)), s)
        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-3"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("Picks").classes("text-2xl font-bold")
                ui.space()
                session_picker(cfg, s, "/picks", want=_has_picks)
            if s is None or not _has_picks(s):
                ui.label("No picks yet. Run a session first (Run screen), then come back "
                         "to check its clips.").classes("text-lg text-slate-300")
                return
            try:
                picks = Picks(cfg, s)
            except OSError as err:
                ui.label(f"Couldn't read {s.out / 'picks.csv'}: {err}").classes("text-red-400")
                return
            body(cfg, s, picks)


def _sessions_with_picks(cfg: Config):
    from .shell import sessions
    return [s for s in sessions(cfg) if _has_picks(s)]


def body(cfg: Config, s: SessionInfo, picks: Picks) -> None:
    from nicegui import ui

    drag: dict = {}

    def save() -> bool:
        try:
            picks.save()
        except PermissionError:
            ui.notify("picks.csv is open in another program (Excel?). Close it there, "
                      "then save again.", type="negative", timeout=8000)
            return False
        ui.notify("Saved. The next run uses these picks.", type="positive")
        bar.refresh()
        return True

    def save_and_make() -> None:
        if picks.dirty and not save():
            return
        if JOB.running:
            ui.notify("A run is already going (Run screen).", type="warning")
            return
        if not s.source:
            ui.notify("Don't know where this session's videos are; start it from the "
                      "Run screen.", type="warning")
            return
        JOB.start(cfg, s.source, [], "Highlight reel")
        ui.navigate.to("/run")

    def reload() -> None:
        picks.load()
        for r in lists.values():
            r.refresh()
        bar.refresh()

    @ui.refreshable
    def bar() -> None:
        with ui.row().classes("w-full items-center gap-2"):
            n = sum(len(v.items) for v in picks.videos)
            ui.label(f"{len(picks.videos)} videos, {n} clips").classes("text-slate-300")
            if picks.dirty:
                ui.badge("unsaved changes", color="orange")
            ui.space()
            ui.button("Undo changes", icon="undo", on_click=reload)\
                .props("no-caps flat").set_enabled(picks.dirty)
            ui.button("Save", icon="save", on_click=save).props("no-caps outline color=teal")
            ui.button("Save and make the videos", icon="movie", on_click=save_and_make)\
                .props("no-caps color=teal")

    def changed() -> None:
        picks.dirty = True
        bar.refresh()

    def preview(r: Row) -> None:
        f_name = r.data.get("file", "")
        offset = s.offset(f_name) if s.files else (
            parse_time(r.data.get("session_time", "0")) - parse_time(r.data.get("file_time", "0")))
        src = f"/media/{quote(s.stem(f_name))}/proxy.mp4"
        with ui.dialog() as dialog, ui.card().classes("bg-slate-900 w-[900px] max-w-full"):
            ui.element("video").props("id=gv controls playsinline preload=auto")\
                .classes("w-full rounded bg-black").style("aspect-ratio: 16 / 9")
            label = ui.label().classes("text-slate-300")

            def show() -> None:
                label.set_text(f"Clip {fmt_time(r.start)} to {fmt_time(r.end)} on the session "
                               f"clock ({r.end - r.start:.1f} s)")
                ui.run_javascript(f"gvBounds({r.start - offset:.2f}, {r.end - offset:.2f})")

            def nudge(edge: str, d: float) -> None:
                lo, hi = s.span(f_name) or (0.0, float("inf"))
                if edge == "start":
                    start = min(max(lo, r.start + d), r.end - MIN_CLIP_S)
                    picks.set_bounds(r.id, start, r.end)
                else:
                    end = max(min(hi, r.end + d), r.start + MIN_CLIP_S)
                    picks.set_bounds(r.id, r.start, end)
                show()
                changed()

            with ui.row().classes("gap-2 items-center"):
                ui.label("Start").classes("text-slate-400")
                ui.button("−0.5 s", on_click=lambda: nudge("start", -NUDGE_S)).props("no-caps dense outline")
                ui.button("+0.5 s", on_click=lambda: nudge("start", NUDGE_S)).props("no-caps dense outline")
                ui.label("End").classes("text-slate-400 ml-4")
                ui.button("−0.5 s", on_click=lambda: nudge("end", -NUDGE_S)).props("no-caps dense outline")
                ui.button("+0.5 s", on_click=lambda: nudge("end", NUDGE_S)).props("no-caps dense outline")
                ui.space()
                ui.button("Done", on_click=dialog.close).props("no-caps color=teal")

            def closed() -> None:
                ui.run_javascript("gvStop()")
                refresh_all()
                dialog.delete()

            dialog.on("hide", closed)
        dialog.open()
        label.set_text(f"Clip {fmt_time(r.start)} to {fmt_time(r.end)}")
        ui.run_javascript(f"setTimeout(() => gvLoad({json.dumps(src)}, {r.start - offset:.2f}, "
                          f"{r.end - offset:.2f}), 150)")

    def refresh_all() -> None:
        for r in lists.values():
            r.refresh()

    def move(v: VideoRows, i: int, j: int) -> None:
        if i == j or not (0 <= j < len(v.items)):
            return
        item = v.items.pop(i)
        v.items.insert(j, item)
        lists[v.name].refresh()
        changed()

    def toggle(r: Row, on: bool) -> None:
        r.keep = on
        changed()
        refresh_all()

    def moment_line(r: Row) -> None:
        thumb = r.data.get("thumbnail", "")
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            look = "aspect-ratio: 16 / 9" + ("; opacity: 0.35" if not r.keep else "")
            if thumb and (s.out / thumb).is_file():
                ui.image(f"/out/{quote(s.name)}/{quote(thumb)}")\
                    .classes("w-[160px] rounded cursor-pointer shrink-0").style(look)\
                    .on("click", lambda r=r: preview(r))
            else:
                with ui.element("div").classes("w-[160px] rounded cursor-pointer shrink-0 "
                                               "bg-slate-800 flex items-center justify-center")\
                        .style(look).on("click", lambda r=r: preview(r)):
                    ui.icon("movie").classes("text-3xl text-slate-500")
            with ui.column().classes("grow gap-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(r.data.get("event_type", ""), color="blue")
                    when = r.data.get("session_time", "")
                    clock_ = r.data.get("clock", "")
                    ui.label(f"{when}{f' ({clock_})' if clock_ else ''} · "
                             f"{r.end - r.start:.1f} s · score {r.data.get('score', '')}")\
                        .classes("text-sm text-slate-400")
                ui.label(r.data.get("description", "")).classes("text-sm")
                if r.data.get("players"):
                    ui.label(r.data["players"]).classes("text-xs text-slate-400")
            with ui.column().classes("items-end gap-1 shrink-0"):
                ui.switch("In" if r.keep else "Out", value=r.keep,
                          on_change=lambda e, r=r: toggle(r, e.value))\
                    .props("dark color=teal dense")
                ui.button("Preview / trim", icon="content_cut", on_click=lambda r=r: preview(r))\
                    .props("no-caps dense flat size=sm")

    lists: dict = {}

    def video_block(v: VideoRows) -> None:
        @ui.refreshable
        def items() -> None:
            ui.label(f"{len(v.items)} clips · {fmt_time(v.length())}")\
                .classes("text-sm text-slate-400")
            for i, item in enumerate(v.items):
                card = ui.card().classes("w-full bg-slate-900 gap-1 p-2")\
                    .props("draggable")
                card.on("dragstart", lambda i=i: drag.update(video=v.name, i=i))
                card.on("dragover", js_handler="(e) => { e.preventDefault(); "
                        "e.currentTarget.classList.add('vh-drop'); }")
                card.on("dragleave", js_handler="(e) => e.currentTarget.classList.remove('vh-drop')")
                card.on("drop", lambda i=i: drop(v, i),
                        js_handler="(e) => { e.preventDefault(); "
                        "e.currentTarget.classList.remove('vh-drop'); emit(); }")
                with card:
                    with ui.row().classes("w-full items-start gap-2 no-wrap"):
                        with ui.column().classes("items-center gap-0 shrink-0"):
                            ui.icon("drag_indicator").classes("text-slate-500 cursor-move")
                            ui.label(str(i + 1)).classes("text-lg font-bold")
                            ui.button(icon="arrow_upward", on_click=lambda i=i: move(v, i, i - 1))\
                                .props("flat dense size=sm").set_enabled(i > 0)
                            ui.button(icon="arrow_downward", on_click=lambda i=i: move(v, i, i + 1))\
                                .props("flat dense size=sm").set_enabled(i < len(v.items) - 1)
                        with ui.column().classes("grow gap-2"):
                            for r in item.rows:
                                moment_line(r)
            if v.out:
                with ui.expansion(f"Taken out ({len(v.out)})").classes("w-full text-slate-400"):
                    for r in v.out:
                        with ui.card().classes("w-full bg-slate-900 p-2"):
                            moment_line(r)

        lists[v.name] = items
        with ui.expansion(v.name, icon="movie", value=v.key == "reel")\
                .classes("w-full bg-slate-800 rounded").props("header-class=text-lg"):
            items()

    def drop(v: VideoRows, j: int) -> None:
        if drag.get("video") != v.name:
            return
        move(v, drag["i"], j)

    bar()
    ui.label("Click a thumbnail to watch and trim it. Switch a clip Out to replace it with "
             "the next best moment. Drag clips (or use the arrows) to change the order.")\
        .classes("text-sm text-slate-400")
    for v in picks.videos:
        video_block(v)
