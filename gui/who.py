"""The Who is who screen: for each clip at the start of the reel (the part
that gets nametags), a still with everyone in it outlined and numbered, a
box for each name next to it, and below that the people who come into the
picture later while the tags are up. Type a name (roster names and names
you used before fill in as you type), or leave the box empty for no tag.
Answers are saved straight away to work/<session>/who_is_who.json; make the
videos again to use them.

Someone who comes in later and is plainly a person you already named (the
tracker lost them for a moment: same place, size and shirt) gets that name
by themselves; they're listed under "Followed automatically", where a typed
name or "Not them" overrides it.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from pipeline.common import fmt_time
from pipeline.config import Config
from pipeline.nametags import _iou, _safe, load_who, save_who

from .run import JOB
from .shell import SessionInfo, find_session, frame, session_picker, sessions

# One colour per number, so a box on the still and its close-up match.
PALETTE = ["#f87171", "#60a5fa", "#4ade80", "#facc15", "#c084fc", "#fb923c", "#2dd4bf",
           "#f472b6", "#a3e635", "#38bdf8", "#e879f9", "#fbbf24"]


def _colour(n: int) -> str:
    return PALETTE[(n - 1) % len(PALETTE)]


def _has_tags(s: SessionInfo) -> bool:
    return (s.work / "tags" / "current.json").exists()


def _stills(s: SessionInfo) -> list[dict]:
    tags = s.work / "tags"
    try:
        keys = json.loads((tags / "current.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for key in keys:
        try:
            info = json.loads((tags / f"{_safe(key)}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (tags / info.get("image", "")).is_file():
            out.append(info)
    return out


def _people(st: dict) -> list[dict]:
    """Everyone in a clip, numbered (stills from older versions: in the
    still, in order)."""
    out = []
    for i, p in enumerate(st.get("people", []), 1):
        image = p.get("image", st["image"])
        out.append({**p, "n": p.get("n", i), "t": p.get("t", st["t"]), "image": image,
                    "in_still": p.get("in_still", image == st["image"])})
    return out


def _size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001 - a missing or odd still just uses 720p
        return 1280, 720


def _same(a: dict, p: dict) -> bool:
    """Is answer a about person p (same box at the same moment)?"""
    return abs(a.get("t", p["t"]) - p["t"]) <= 0.5 and _iou(a["box"], p["box"]) > 0.5


def _answer_for(answers: dict, key: str, p: dict) -> dict | None:
    best = None
    for a in answers.get(key, {}).get("people", []):
        if _same(a, p) and (best is None or _iou(a["box"], p["box"]) > _iou(best["box"], p["box"])):
            best = a
    return best


def _name_of(answers: dict, st: dict, p: dict) -> str:
    """The name person p's tag will show: your answer, else the name they were
    followed to."""
    a = _answer_for(answers, st["key"], p)
    if a is None and p.get("how") == "you":
        # Named on an older still of this clip: the last run matched it to them.
        return p.get("name", "")
    if a is not None:
        return a.get("name", "")
    return p.get("name", "") if p.get("how") == "follow" else ""


def named_count(s: SessionInfo) -> tuple[int, int] | None:
    """(people named, people listed) on this session's Who is who screen, or
    None when it has no nametags yet."""
    if not _has_tags(s):
        return None
    answers = load_who(s.work)
    everyone = [(st, p) for st in _stills(s) for p in _people(st)]
    return sum(1 for st, p in everyone if _name_of(answers, st, p)), len(everyone)


def _closeup(src: str, box: list[float], height: int = 120, on_click=None) -> None:
    """The person, cut out of their frame with a little room around them,
    and outlined (there's often someone else right next to them)."""
    from nicegui import ui

    bx1, by1, bx2, by2 = box
    pad_x, pad_y = 0.6 * (bx2 - bx1), 0.15 * (by2 - by1)
    x1, x2 = max(0.0, bx1 - pad_x), min(1.0, bx2 + pad_x)
    y1, y2 = max(0.0, by1 - pad_y), min(1.0, by2 + pad_y)
    bw, bh = max(1e-3, x2 - x1), max(1e-3, y2 - y1)
    width = round(height * bw * 16 / 9 / bh)
    pos_x = 100 * x1 / (1 - bw) if bw < 1 else 0
    pos_y = 100 * y1 / (1 - bh) if bh < 1 else 0
    tile = ui.element("div").classes("rounded shrink-0 relative").style(
        f"width:{width}px;height:{height}px;background-image:url('{src}');"
        f"background-size:{100 / bw:.1f}% {100 / bh:.1f}%;"
        f"background-position:{pos_x:.1f}% {pos_y:.1f}%;image-rendering:auto"
        + (";cursor:zoom-in" if on_click else ""))
    with tile:
        ui.element("div").classes("absolute rounded-sm").style(
            f"left:{100 * (bx1 - x1) / bw:.1f}%;top:{100 * (by1 - y1) / bh:.1f}%;"
            f"width:{100 * (bx2 - bx1) / bw:.1f}%;height:{100 * (by2 - by1) / bh:.1f}%;"
            "border:2px solid #fde047;box-shadow:0 0 0 1px #000")
    if on_click:
        tile.on("click", on_click)


def register(cfg: Config) -> None:
    from nicegui import ui

    @ui.page("/who")
    def page(session: str = "") -> None:
        frame("/who")
        s = find_session(cfg, session or None)
        if s is not None and not _has_tags(s) and not session:
            s = next((x for x in sessions(cfg) if _has_tags(x)), s)
        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-3"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("Who is who").classes("text-2xl font-bold")
                ui.space()
                session_picker(cfg, s, "/who", want=_has_tags)
            if s is None or not _has_tags(s):
                ui.label("No nametags yet. They're made with the highlight reel; run a "
                         "session first.").classes("text-lg text-slate-300")
                return
            body(cfg, s)


def body(cfg: Config, s: SessionInfo) -> None:
    from nicegui import ui

    stills = _stills(s)
    answers = load_who(s.work)
    media = f"/media/{quote(s.name)}/tags/"
    typed = {a.get("name", "") for v in answers.values() for a in v.get("people", [])}
    names = sorted(({n for st in stills for n in st.get("roster", [])} | typed) - {""},
                   key=str.lower)
    order: list = []                      # every name box, top to bottom (Enter moves on)
    slots: dict[tuple[str, int], int] = {}   # (clip, number) -> place in order

    def remake() -> None:
        if JOB.running:
            ui.notify("A run is already going (Run screen).", type="warning")
            return
        if not s.source:
            ui.notify("Don't know where this session's videos are; start it from the "
                      "Run screen.", type="warning")
            return
        JOB.start(cfg, s.source, [], "Highlight reel")
        ui.navigate.to("/run")

    def name_of(st: dict, p: dict) -> str:
        """The name the tag will show."""
        return _name_of(answers, st, p)

    def followed(p: dict) -> bool:
        return p.get("how") == "follow" and bool(p.get("name"))

    @ui.refreshable
    def summary() -> None:
        everyone = [(st, p) for st in stills for p in _people(st)]
        named = sum(1 for st, p in everyone if name_of(st, p))
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(f"{named} of {len(everyone)} people named · tags show for the first "
                     f"{cfg.nametags.seconds:g} s of the reel").classes("text-slate-300")
            ui.space()
            ui.button("Make the videos again with these names", icon="movie",
                      on_click=remake).props("no-caps color=teal")

    def store(st: dict, p: dict, name: str | None) -> None:
        """name None: forget the answer (a followed person goes back to automatic)."""
        key = st["key"]
        entry = answers.setdefault(key, {"t": st["t"], "people": []})
        entry["people"] = [a for a in entry["people"] if not _same(a, p)]
        if name is not None:
            # Kept even when empty ("no tag"), and last, so it beats an older answer.
            entry["people"].append({"box": p["box"], "t": p["t"], "name": name})
        save_who(s.work, answers)
        if name:
            typed.add(name)
        refresh[key]()
        summary.refresh()

    def save(st: dict, p: dict, value: str) -> None:
        name = " ".join((value or "").split())
        if followed(p):
            a = _answer_for(answers, st["key"], p)
            if not name and (a is None or a.get("name")):
                if a is not None:
                    store(st, p, None)         # emptied: back to automatic
                return
            if a is not None and a.get("name", "") == name:
                return
            store(st, p, name)
            return
        if name == name_of(st, p):
            return
        store(st, p, name)

    def zoom(p: dict) -> None:
        w, h = _size(s.work / "tags" / p["image"])
        x1, y1, x2, y2 = p["box"]
        rect = (f'<rect x="{x1 * w:.0f}" y="{y1 * h:.0f}" width="{(x2 - x1) * w:.0f}" '
                f'height="{(y2 - y1) * h:.0f}" fill="none" stroke="#fde047" stroke-width="3"/>')
        with ui.dialog() as dlg, ui.card().classes("bg-slate-900 p-2").style("max-width:90vw"):
            ui.label(f"#{p['n']}" + (f" · on screen at {fmt_time(p['comes_in'])} in the reel"
                                     if not p["in_still"] and "comes_in" in p else ""))\
                .classes("text-sm text-slate-300")
            ui.interactive_image(media + quote(p["image"]), content=rect)\
                .classes("rounded").style("width:min(85vw, 1100px)")
        dlg.open()

    def name_box(st: dict, p: dict, placeholder: str = "no tag"):
        a = _answer_for(answers, st["key"], p)
        value = (a or {}).get("name", "") if followed(p) else name_of(st, p)
        box = ui.input(placeholder=placeholder, value=value, autocomplete=names)\
            .props("dense outlined dark").classes("w-full")
        i = slots.setdefault((st["key"], p["n"]), len(order))
        if i == len(order):
            order.append(box)
        else:
            order[i] = box                 # drawn again

        def enter(_, st=st, p=p, box=box, i=i) -> None:
            save(st, p, box.value)
            if i + 1 < len(order):
                order[i + 1].run_method("focus")

        box.on("blur", lambda _, st=st, p=p, box=box: save(st, p, box.value))
        box.on("keydown.enter", enter)
        return box

    refresh: dict = {}

    def clip_card(k: int, st: dict) -> None:
        w, h = _size(s.work / "tags" / st["image"])
        people = _people(st)
        by_n = {p["n"]: p for p in people}
        here = [p for p in people if p["in_still"]]
        later = [p for p in people if not p["in_still"] and not followed(p)]
        auto = [p for p in people if not p["in_still"] and followed(p)]
        inputs: dict[int, object] = {}
        focus = {"n": None}
        img = {}

        def shapes() -> str:
            out = []
            for p in here:
                x1, y1, x2, y2 = p["box"]
                c, name = _colour(p["n"]), name_of(st, p)
                wide = 5 if focus["n"] == p["n"] else 3
                dash = "" if name else ' stroke-dasharray="6 4"'
                out.append(f'<rect x="{x1 * w:.0f}" y="{y1 * h:.0f}" '
                           f'width="{(x2 - x1) * w:.0f}" height="{(y2 - y1) * h:.0f}" '
                           f'fill="{c}" fill-opacity="{0.25 if wide == 5 else 0.08}" '
                           f'stroke="{c}" stroke-width="{wide}"{dash}/>')
                out.append(f'<text x="{(x1 + x2) / 2 * w:.0f}" y="{max(24, y1 * h - 6):.0f}" '
                           f'fill="{c}" font-size="24" font-weight="bold" text-anchor="middle" '
                           f'style="paint-order: stroke" stroke="#000" stroke-width="4">'
                           f'{p["n"]}</text>')
            return "".join(out)

        def show_focus(n: int | None) -> None:
            focus["n"] = n
            if "img" in img:
                img["img"].content = shapes()

        def click(e) -> None:
            x, y = e.image_x / w, e.image_y / h
            hits = [p for p in here
                    if p["box"][0] <= x <= p["box"][2] and p["box"][1] <= y <= p["box"][3]]
            if hits:
                p = min(hits, key=lambda p: (p["box"][2] - p["box"][0])
                        * (p["box"][3] - p["box"][1]))
                inputs[p["n"]].run_method("focus")

        def dupes() -> list[tuple[int, int, str]]:
            """People with the same name who are on screen at the same time
            (they can't both be that person)."""
            out = []
            for i, p in enumerate(people):
                for q in people[i + 1:]:
                    nm = name_of(st, p)
                    if not nm or nm.lower() != name_of(st, q).lower():
                        continue
                    if "on" in p and "on" in q:
                        together = min(p["on"][1], q["on"][1]) - max(p["on"][0], q["on"][0]) > 0.3
                    else:                      # older stills: both in the picture
                        together = p["in_still"] and q["in_still"]
                    if together:
                        out.append((p["n"], q["n"], nm))
            return out

        @ui.refreshable
        def warnings() -> None:
            for n, m, nm in dupes():
                ui.label(f"#{n} and #{m} are both {nm} but are on screen together, so one "
                         "of them is someone else (neither gets the tag while both are there).")\
                    .classes("text-xs text-amber-300")

        def redraw() -> None:
            if "img" in img:
                img["img"].content = shapes()
            warnings.refresh()

        refresh[st["key"]] = redraw

        def row(p: dict) -> None:
            c = _colour(p["n"])
            with ui.row().classes("items-center gap-1 no-wrap w-full rounded pr-1")\
                    .style(f"border-left: 4px solid {c}"):
                ui.label(f"{p['n']}").classes("text-base font-bold w-6 text-right shrink-0")\
                    .style(f"color: {c}")
                with ui.element("div").classes("flex justify-center shrink-0").style("width:44px"):
                    _closeup(media + quote(p["image"]), p["box"], 52,
                             on_click=lambda _, p=p: zoom(p))
                box = name_box(st, p)
                box.on("focus", lambda _, n=p["n"]: show_focus(n))
                inputs[p["n"]] = box

        def card(p: dict, note: str):
            """Someone who comes in later: a bigger close-up, when, and what we
            know. Returns the column to put their name box in."""
            c = _colour(p["n"])
            with ui.row().classes("items-center gap-2 no-wrap rounded p-1 w-full")\
                    .style(f"border: 2px solid {c}55"):
                _closeup(media + quote(p["image"]), p["box"], 104,
                         on_click=lambda _, p=p: zoom(p))
                with ui.column().classes("gap-1 grow min-w-0") as col:
                    with ui.row().classes("items-baseline gap-x-2 gap-y-0"):
                        ui.label(f"#{p['n']}").classes("text-lg font-bold").style(f"color: {c}")
                        ui.label(note).classes("text-xs text-slate-400")
            return col

        reel = st.get("reel")
        where = (f"reel {fmt_time(reel[0])}–{fmt_time(reel[1])}" if reel
                 else f"{st.get('file', '')} · {st['key'].split('@')[-1]} s")
        with ui.card().classes("w-full bg-slate-900 gap-3 p-3"):
            with ui.row().classes("w-full items-baseline gap-2"):
                ui.label(f"Clip {k}").classes("font-bold")
                ui.label(f"{where} · {len(here)} in the picture"
                         + (f", {len(later) + len(auto)} come in later" if later or auto else ""))\
                    .classes("text-xs text-slate-400")
            with ui.element("div").classes("w-full grid gap-3 items-start who-grid"):
                img["img"] = ui.interactive_image(
                    media + quote(st["image"]), content=shapes(), on_mouse=click,
                    events=["click"], cross="#2dd4bf").classes("w-full rounded")
                with ui.column().classes("gap-1 w-full"):
                    with ui.element("div").classes("w-full grid gap-1")\
                            .style("grid-template-columns: repeat(auto-fill, minmax(190px, 1fr))"):
                        for p in here:
                            row(p)
                    warnings()

            grid = "grid-template-columns: repeat(auto-fill, minmax(280px, 1fr))"
            if later:
                ui.label("Came into the picture later").classes("font-bold mt-1")
                with ui.element("div").classes("w-full grid gap-2").style(grid):
                    for p in later:
                        when = (f"on screen at {fmt_time(p['comes_in'])}" if "comes_in" in p
                                else "not in the still")
                        with card(p, when):
                            box = name_box(st, p)
                            inputs[p["n"]] = box
                            options = [m for m in p.get("maybe", []) if m]
                            if options and not name_of(st, p):
                                with ui.row().classes("gap-1 items-center"):
                                    ui.label("Maybe").classes("text-xs text-slate-500")
                                    for m in options:
                                        ui.button(m, on_click=lambda _, p=p, m=m, box=box:
                                                  (box.set_value(m), save(st, p, m)))\
                                            .props("dense flat no-caps size=sm color=teal")

            if auto:
                with ui.expansion(f"Followed automatically ({len(auto)})", icon="link")\
                        .classes("w-full text-slate-300").props("dense"):
                    ui.label("Each of these is the same person as someone you named (the "
                             "camera lost them for a moment), so they get that name. Type "
                             "another name to change it, or press Not them for no tag.")\
                        .classes("text-xs text-slate-400")
                    with ui.element("div").classes("w-full grid gap-2 mt-1").style(grid):
                        for p in auto:
                            root = by_n.get(p.get("follows"))
                            who = (f"#{root['n']} {name_of(st, root) or p['name']}" if root
                                   else p["name"])

                            @ui.refreshable
                            def auto_card(p: dict = p, who: str = who) -> None:
                                a = _answer_for(answers, st["key"], p)
                                off = a is not None and not a.get("name")
                                with card(p, f"on screen at {fmt_time(p.get('comes_in', 0))}"):
                                    ui.label(f"same person as {who}")\
                                        .classes("text-xs text-slate-400 -mt-1")
                                    box = name_box(st, p, "no tag" if off
                                                   else f"{p['name']} (automatic)")
                                    inputs[p["n"]] = box
                                    if off:
                                        ui.button("Follow again", on_click=lambda _, p=p, r=auto_card:
                                                  (store(st, p, None), r.refresh()))\
                                            .props("dense flat no-caps size=sm color=teal")
                                    else:
                                        ui.button("Not them", on_click=lambda _, p=p, r=auto_card:
                                                  (store(st, p, ""), r.refresh()))\
                                            .props("dense flat no-caps size=sm color=grey")

                            auto_card()

    ui.add_head_html("<style>.who-grid{grid-template-columns:minmax(0,3fr) minmax(0,2fr)}"
                     "@media (max-width: 900px){.who-grid{grid-template-columns:1fr}}</style>")
    summary()
    ui.label("Type a name for each numbered person: names fill in as you type (Tab accepts), "
             "Enter goes to the next box, and an empty box means no tag. Click someone on the "
             "picture to jump to their box, or a close-up to see the whole frame. Only people "
             "on screen while the tags are up are listed.").classes("text-sm text-slate-400")
    for k, st in enumerate(stills, 1):
        clip_card(k, st)
