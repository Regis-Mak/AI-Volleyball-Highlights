"""The Grade and Test set screens: one moment at a time, playing in a loop
from the 720p analysis copy, with Qwen's label and description beside it.
The loop covers the whole window Qwen looked at for that moment as well as
the clip (W switches to just the clip); the bar under the video shows both,
with a tick for every frame Qwen got.
Grade asks about the moments the tool is least sure of; Test set builds the
fixed set of moments every new ranker version is checked against.

Keys
  Y          worth showing
  F          funny (and worth showing)
  N          not worth showing (left out of every video from now on)
  1 to 7     the right label, if Qwen's is wrong (press before Y/F/N)
  Left/Right         move the clip's start by half a second
  Shift+Left/Right   move the clip's end
  Space      replay from the start
  W          play Qwen's whole window too, or just the clip
  S          skip
  Backspace  undo the last answer
"""
from __future__ import annotations

import json
from urllib.parse import quote

from pipeline.common import fmt_time
from pipeline.learn import grading, ranker
from pipeline.learn.grading import LABELS, Card
from pipeline.library import Library, summary

from .shell import frame

NAMES = dict(LABELS) | {"reaction": "group reaction"}
NUDGE_S = 0.5
MIN_CLIP_S = 2.0
SCALE = 1.5                  # the whole screen, 50% larger than the other screens
PLAY = {"whole": True}       # play Qwen's whole window too; stays as you left it

SCRIPT = """
<script>
window.gv = {start: 0, end: 0, clip: null, win: null, frames: []};
function gvEl() { return document.getElementById('gv'); }
function gvPlay(v) {
  const p = v.play();
  if (p) p.catch(() => { v.muted = true; v.play().catch(() => {}); });
}
// m: {lo, hi, clip, win, frames} on Grade; Picks passes just a start and end.
function gvMarks(m, e) {
  if (typeof m === 'number') m = {lo: m, hi: e, clip: [m, e], win: null, frames: []};
  gv.start = m.lo; gv.end = m.hi; gv.clip = m.clip; gv.win = m.win; gv.frames = m.frames || [];
  gvDraw();
  return m;
}
function gvLoad(src, m, e) {
  const v = gvEl(); if (!v) return;
  m = gvMarks(m, e);
  const s = m.lo;
  gvAudioFor(src);
  const go = () => { v.currentTime = s; gvPlay(v); };
  if (v.dataset.src !== src) {
    v.dataset.src = src; v.src = src;
    v.addEventListener('loadedmetadata', go, {once: true});
  } else { go(); }
}
function gvBounds(m, end) {
  m = gvMarks(m, end);
  const s = m.lo, e = m.hi;
  const v = gvEl(); if (v && (v.currentTime < s - 0.3 || v.currentTime > e)) v.currentTime = s;
}
// The bar under the video: what Qwen looked at (blue, a tick per frame it
// got) and the clip (teal), across the stretch that's playing.
function gvPct(t) {
  const span = Math.max(0.1, gv.end - gv.start);
  return Math.min(100, Math.max(0, 100 * (t - gv.start) / span));
}
function gvDraw() {
  const bar = document.getElementById('gvt'); if (!bar) return;
  let html = '';
  const band = (a, b, cls) => {
    const l = gvPct(a), r = gvPct(b);
    if (r > l) html += `<div class="${cls}" style="left:${l}%;width:${r - l}%"></div>`;
  };
  if (gv.win) {
    band(gv.win[0], gv.win[1], 'gvt-win');
    for (const f of gv.frames) {
      if (f >= gv.start && f <= gv.end) html += `<div class="gvt-f" style="left:${gvPct(f)}%"></div>`;
    }
  }
  if (gv.clip) band(gv.clip[0], gv.clip[1], 'gvt-clip');
  html += '<div class="gvt-head" id="gvth"></div>';
  bar.innerHTML = html;
  if (!bar.dataset.ready) {
    bar.dataset.ready = '1';
    bar.title = 'Blue: what Qwen looked at (a tick for every frame it got). Teal: the clip. Click to jump.';
    bar.addEventListener('pointerdown', (ev) => {
      const v = gvEl(); if (!v || !(gv.end > gv.start)) return;
      const r = bar.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
      v.currentTime = gv.start + frac * (gv.end - gv.start);
    });
  }
}
function gvHead() {
  const v = gvEl(), h = document.getElementById('gvth');
  if (v && h && gv.end > gv.start) h.style.left = gvPct(v.currentTime) + '%';
  requestAnimationFrame(gvHead);
}
requestAnimationFrame(gvHead);
function gvReplay() { const v = gvEl(); if (v) { v.currentTime = gv.start; gvPlay(v); } }
function gvStop() {
  const v = gvEl(); if (v) v.pause();
  const a = document.getElementById('gva'); if (a) a.pause();
}
// The 720p copies are silent, so the sound comes from the video's cleaned-up
// audio file (audio_32k.wav next to proxy.mp4), kept in step with the picture.
window.gvSound = (() => { try { return localStorage.getItem('gvSound') !== 'off'; }
                          catch (e) { return true; } })();
function gvAudioFor(src) {
  let a = document.getElementById('gva');
  if (!a) {
    a = document.createElement('audio'); a.id = 'gva'; a.preload = 'auto';
    a.addEventListener('error', () => { a.dataset.missing = '1'; gvSoundButton(); });
    a.addEventListener('loadedmetadata', () => { delete a.dataset.missing; gvSoundButton(); });
    document.body.appendChild(a);
  }
  const asrc = src.replace(/proxy[.]mp4$/, 'audio_32k.wav');
  if (a.dataset.src !== asrc) { a.dataset.src = asrc; delete a.dataset.missing; a.src = asrc; }
  gvSoundButton();
}
function gvSoundButton() {
  const v = gvEl(); if (!v) return;
  let b = document.getElementById('gvs');
  const after = document.getElementById('gvt') || v;    // under the bar, if there is one
  if (!b || b.previousElementSibling !== after) {
    if (b) b.remove();
    b = document.createElement('button');
    b.id = 'gvs'; b.type = 'button'; b.tabIndex = -1;
    b.style.cssText = 'align-self:flex-start;margin-top:4px;padding:2px 10px;border-radius:6px;' +
      'border:1px solid #64748b;background:transparent;color:#cbd5e1;font-size:0.85rem;cursor:pointer';
    b.addEventListener('click', () => {
      gvSound = !gvSound;
      try { localStorage.setItem('gvSound', gvSound ? 'on' : 'off'); } catch (e) {}
      gvSoundButton(); gvAudioSync(true);
    });
    after.insertAdjacentElement('afterend', b);
  }
  const a = document.getElementById('gva');
  const missing = a && a.dataset.missing;
  b.disabled = !!missing;
  b.textContent = missing ? "No sound: this video's audio file was deleted"
    : String.fromCodePoint(gvSound ? 0x1F50A : 0x1F507) + (gvSound ? ' Sound on' : ' Sound off');
}
function gvAudioSync(force) {
  const v = gvEl(), a = document.getElementById('gva');
  if (!v || !a || !a.src || a.dataset.missing) return;
  if (!gvSound || v.paused || v.seeking) { if (!a.paused) a.pause(); }
  if (force || Math.abs(a.currentTime - v.currentTime) > 0.15) {
    try { a.currentTime = v.currentTime; } catch (e) {}
  }
  a.playbackRate = v.playbackRate;
  if (gvSound && !v.paused && !v.seeking && a.paused) {
    const p = a.play(); if (p) p.catch(() => {});
  }
}
['play', 'playing', 'pause', 'seeking', 'seeked', 'waiting', 'ratechange', 'ended']
  .forEach(t => document.addEventListener(t, (ev) => {
    if (ev.target && ev.target.id === 'gv') gvAudioSync(t !== 'ratechange');
  }, true));
document.addEventListener('timeupdate', (ev) => {
  const v = ev.target; if (!v || v.id !== 'gv' || !gv.end) return;
  if (v.currentTime >= gv.end || v.currentTime < gv.start - 0.5) v.currentTime = gv.start;
  else gvAudioSync(false);
}, true);
const gvUnmute = () => { const v = gvEl(); if (v) v.muted = false; gvAudioSync(false); };
document.addEventListener('keydown', gvUnmute);
document.addEventListener('pointerdown', gvUnmute);
</script>
"""

STYLE = """
<style>
  .gv-key { font-family: ui-monospace, monospace; font-size: 0.75rem; opacity: 0.7; }
  .gv-desc { font-size: 1.05rem; line-height: 1.4; }
  .gv-big { zoom: {SCALE}; }
  #gvt { position: relative; height: 26px; background: #0f172a; border-radius: 6px;
         cursor: pointer; overflow: hidden; touch-action: none; }
  #gvt .gvt-win { position: absolute; top: 2px; height: 10px; background: #3b82f6; opacity: 0.75;
                  border-radius: 3px; }
  #gvt .gvt-f { position: absolute; top: 2px; height: 10px; width: 1px; background: #dbeafe; }
  #gvt .gvt-clip { position: absolute; top: 14px; height: 10px; background: #14b8a6;
                   border-radius: 3px; }
  #gvt .gvt-head { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px;
                   background: #ffffff; }
  .gv-sw { display: inline-block; width: 0.8em; height: 0.8em; border-radius: 2px;
           vertical-align: -0.05em; margin-right: 0.3em; }
</style>
""".replace("{SCALE}", f"{SCALE:g}")


def _tenths(secs: float) -> str:
    m, s = divmod(max(0.0, secs), 60)
    return f"{int(m)}:{s:04.1f}"


def _marks(c: Card) -> dict:
    """What the player loops over (lo to hi) and what the bar under it shows."""
    lo, hi = c.start, c.end
    if PLAY["whole"] and c.window:
        lo, hi = min(lo, c.window[0]), max(hi, c.window[1])
    return {"lo": round(lo, 2), "hi": round(hi, 2), "clip": [round(c.start, 2), round(c.end, 2)],
            "win": list(c.window) if c.window else None, "frames": c.frames}


class State:
    def __init__(self, lib: Library, mode: str, session: str | None):
        self.lib, self.mode, self.session = lib, mode, session
        self.cards: list[Card] = grading.build_queue(lib, mode, session)
        self.i = 0
        self.message = ""

    @property
    def card(self) -> Card | None:
        return self.cards[self.i] if self.i < len(self.cards) else None

    @property
    def gold(self) -> bool:
        return self.mode == "test"

    def more(self) -> None:
        self.cards = grading.build_queue(self.lib, self.mode, self.session)
        self.i = 0
        self.message = ""


def register(lib: Library) -> None:
    from nicegui import ui

    @ui.page("/grade")
    async def grade_page(client, session: str = "") -> None:
        await build(lib, client, "quick", session or None)

    @ui.page("/test")
    async def test_page(client) -> None:
        await build(lib, client, "test", None)


async def build(lib: Library, client, mode: str, session: str | None) -> None:
    from nicegui import run, ui

    state = State(lib, mode, session)
    frame("/test" if mode == "test" else "/grade")
    ui.add_head_html(STYLE)
    ui.add_body_html(SCRIPT)

    @ui.refreshable
    def header() -> None:
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Grade moments").classes("text-2xl font-bold")
            if state.mode == "test":
                done, target = grading.test_progress(lib)
                ui.badge("Test set", color="purple")
                ui.label(f"{done} of {target} graded").classes("text-lg")
                frac = min(1.0, done / max(1, target))
            else:
                ui.badge("Quick review", color="teal")
                n = len(state.cards)
                ui.label(f"{min(state.i + 1, n)} of {n}" if n else "nothing to grade")\
                    .classes("text-lg")
                frac = state.i / max(1, n)
            ui.space()
            if state.mode != "test":
                # Each session's moments are graded on their own; "All sessions"
                # mixes them (the ranker learns from every session either way).
                names = [r["session"] for r in lib.q(
                    "SELECT DISTINCT session FROM moments ORDER BY session DESC")]
                options = {"": "All sessions", **{n: n for n in names}}
                ui.select(options, value=state.session or "", label="Session",
                          on_change=lambda e: ui.navigate.to(
                              f"/grade?session={quote(e.value)}" if e.value else "/grade"))\
                    .props("dense outlined dark").classes("min-w-[200px]")
        ui.linear_progress(value=frac, show_value=False).props("rounded color=teal")

    @ui.refreshable
    def bounds() -> None:
        c = state.card
        if c is None:
            return
        changed = " (changed)" if c.bounds_changed else ""
        with ui.row().classes("w-full items-center gap-x-4 gap-y-0 text-slate-300"):
            ui.html('<span class="gv-sw" style="background:#14b8a6"></span>'
                    f"Clip {_tenths(c.start)} to {_tenths(c.end)} "
                    f"({c.end - c.start:.1f} s){changed}", sanitize=False)
            if c.window:
                a, b = c.window
                ui.html('<span class="gv-sw" style="background:#3b82f6"></span>'
                        f"What Qwen looked at{'' if c.source == 'vlm' else ' around then'}: "
                        f"{_tenths(a)} to {_tenths(b)} ({b - a:.0f} s, {len(c.frames)} frames)",
                        sanitize=False)
                ui.space()
                ui.switch("Play all of it (W)", value=PLAY["whole"],
                          on_change=lambda e: set_whole(bool(e.value)))\
                    .props("dense dark color=blue tabindex=-1")\
                    .tooltip("On: the loop covers everything Qwen looked at as well as the clip. "
                             "Off: just the clip.")
            else:
                ui.label("No Qwen window on record for this moment.").classes("text-slate-500")

    @ui.refreshable
    def info() -> None:
        c = state.card
        if c is None:
            done_panel()
            return
        with ui.card().classes("w-full bg-slate-900 gap-2"):
            ui.label(f"{c.session} · {c.stem} · {fmt_time(c.t)} into the file")\
                .classes("text-xs text-slate-400")
            with ui.row().classes("items-center gap-2"):
                ui.label("Qwen says").classes("text-slate-400")
                ui.badge(NAMES.get(c.event_type, c.event_type), color="blue")\
                    .classes("text-base")
                if c.label and c.label != c.event_type:
                    ui.label("you say").classes("text-slate-400")
                    ui.badge(NAMES[c.label], color="orange").classes("text-base")
            ui.label(c.description or "(no description)").classes("gv-desc")
            if c.players:
                ui.label("Players: " + ", ".join(c.players)).classes("text-slate-300")
            ui.label(f"hype {c.excitement}/10 · funny {c.funny}/10 · "
                     f"reaction {c.reaction:.2f} · laughter {c.laugh:.2f}")\
                .classes("text-sm text-slate-400")
            ui.label(f"Asking because {c.why}.").classes("text-sm italic text-slate-400")

        ui.label(f"Is it the right label? (1 to {len(LABELS)})").classes("text-sm text-slate-400 mt-2")
        with ui.row().classes("gap-1"):
            for k, (key, name) in enumerate(LABELS, 1):
                on = (c.label or c.event_type) == key
                ui.button(f"{k} {name}", on_click=lambda key=key: set_label(key))\
                    .props(f"tabindex=-1 dense no-caps {'' if on else 'outline'} "
                           f"color={'orange' if on and c.label else 'blue-grey'}")

        ui.label("Worth showing?").classes("text-sm text-slate-400 mt-2")
        with ui.row().classes("gap-2"):
            ui.button("Y  yes", on_click=lambda: answer("y"))\
                .props("tabindex=-1 no-caps color=positive")
            ui.button("F  funny", on_click=lambda: answer("f"))\
                .props("tabindex=-1 no-caps color=amber text-color=black")
            ui.button("N  no", on_click=lambda: answer("n"))\
                .props("tabindex=-1 no-caps color=negative")
            ui.button("S  skip", on_click=skip).props("tabindex=-1 no-caps flat")
            ui.button("Undo", on_click=undo).props("tabindex=-1 no-caps flat")\
                .bind_enabled_from(state, "i", backward=lambda i: i > 0)

    def done_panel() -> None:
        st = grading.stats(state.cards)
        with ui.card().classes("w-full bg-slate-900 gap-3"):
            if state.cards:
                ui.label(f"Done: {st['y']} worth showing, {st['f']} funny, "
                         f"{st['n']} not worth showing.").classes("text-lg")
            elif state.mode == "test":
                ui.label("The test set is complete, or there's nothing left to add.")\
                    .classes("text-lg")
            else:
                ui.label("Nothing to grade yet. Run a session first.").classes("text-lg")
            ui.label(summary(lib)).classes("text-sm text-slate-400")
            with ui.row().classes("gap-2"):
                ui.button("Update the ranker now", on_click=retrain)\
                    .props("tabindex=-1 no-caps color=teal")
                ui.button("Grade more", on_click=more).props("tabindex=-1 no-caps outline")
            if state.message:
                ui.label(state.message).classes("text-slate-200")

    def show() -> None:
        header.refresh()
        info.refresh()
        bounds.refresh()
        c = state.card
        if c is None:
            ui.run_javascript("gvStop()")
            return
        src = "/media/" + quote(c.video)
        ui.run_javascript(f"gvLoad({json.dumps(src)}, {json.dumps(_marks(c))})")

    def answer(a: str) -> None:
        c = state.card
        if c is None:
            return
        grading.record(lib, c, a, gold=state.gold)
        state.i += 1
        show()

    def skip() -> None:
        if state.card is not None:
            state.i += 1
            show()

    def undo() -> None:
        if state.i == 0:
            return
        state.i -= 1
        grading.undo(lib, state.cards[state.i])
        show()

    def set_label(key: str) -> None:
        c = state.card
        if c is not None:
            c.label = None if key == c.event_type else key
            info.refresh()

    def nudge(edge: str, delta: float) -> None:
        c = state.card
        if c is None:
            return
        if edge == "start":
            c.start = min(max(0.0, c.start + delta), c.end - MIN_CLIP_S)
        else:
            c.end = max(min(c.duration, c.end + delta), c.start + MIN_CLIP_S)
        bounds.refresh()
        ui.run_javascript(f"gvBounds({json.dumps(_marks(c))})")

    def set_whole(on: bool) -> None:
        if PLAY["whole"] == on:
            return
        PLAY["whole"] = on
        bounds.refresh()
        c = state.card
        if c is not None:
            ui.run_javascript(f"gvBounds({json.dumps(_marks(c))})")

    async def retrain() -> None:
        state.message = "Training..."
        info.refresh()
        state.message = await run.io_bound(ranker.retrain, lib, True)
        info.refresh()

    def more() -> None:
        state.more()
        show()

    def on_key(e) -> None:
        if not e.action.keydown:
            return
        name = e.key.name
        arrows = name in ("ArrowLeft", "ArrowRight")
        if e.action.repeat and not arrows:
            return
        key = name.lower() if len(name) == 1 else name
        if arrows:
            delta = -NUDGE_S if name == "ArrowLeft" else NUDGE_S
            nudge("end" if e.modifiers.shift else "start", delta)
        elif key in ("y", "f", "n"):
            answer(key)
        elif key == "s":
            skip()
        elif key == "w":
            set_whole(not PLAY["whole"])
        elif key == "Backspace":
            undo()
        elif key == " ":
            ui.run_javascript("gvReplay()")
        elif key in "1234567" and len(key) == 1:
            set_label(LABELS[int(key) - 1][0])

    ui.keyboard(on_key=on_key, ignore=["input", "select", "textarea"])
    with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-3 gv-big"):
        header()
        with ui.row().classes("w-full gap-4 items-start"):
            with ui.column().classes("grow min-w-[320px] gap-1").style("flex: 3 1 0"):
                ui.element("video").props("id=gv controls playsinline preload=auto")\
                    .classes("w-full rounded-lg bg-black").style("aspect-ratio: 16 / 9")
                ui.element("div").props("id=gvt").classes("w-full")
                bounds()
            with ui.column().classes("min-w-[300px] gap-1").style("flex: 2 1 0"):
                info()
        ui.label("Y worth showing · F funny · N not worth showing · 1-7 fix the label · "
                 "Left/Right move the start · Shift+Left/Right move the end · "
                 "Space replay · W all Qwen looked at or just the clip · S skip · "
                 "Backspace undo").classes("gv-key")

    await client.connected()
    show()
