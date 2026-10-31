"""Stage 7: render. Cut clips from the full-quality source and stitch them.

Every clip is re-encoded with identical settings (frame-accurate cuts, same
size/fps/codec), so the final concat is a fast stream copy. Encoded clips are
cached in <work>/clips/, keyed by their exact parameters, so re-rendering a
reel after vetoing one clip only encodes what changed, and the reel and the
shorts share clips.

A clip can come from any file of a session (Clip.video); one that crosses a
camera split is two Clips, the first without a fade-out and the second
without a fade-in, so the join plays through.

Between clips (render.transitions): a dip to black (each clip's own fades),
a cross-dissolve, or a swipe (the next clip slides in from the right). A
dissolve or swipe is its own short piece, both clips' footage blended over
render.transition_s, and the clips either side give up that much, so the
concat stays a stream copy. Which one a cut gets is fixed by the two clips,
so a re-render reuses every piece it can.

Two extras are drawn in the same pass, so nothing is encoded twice:
  tags   nametags (pipeline/nametags.py): small pictures that ffmpeg moves
         frame by frame (sendcmd + overlay)
  pan    vertical videos: a 9:16 window that follows the action
         (pipeline/vertical.py), moved frame by frame the same way
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .common import (AUDIO_ON_TIME, Game, PipelineError, Probe, Progress, fmt_time, log,
                     on_windows, run_ffmpeg, video_encoder_args)


@dataclass
class Sprite:
    """One nametag picture and where it goes on each output frame."""
    png: Path
    xy: list[tuple[int, int] | None] = field(default_factory=list)   # top-left, None = hidden


@dataclass
class TagLayer:
    sprites: list[Sprite]
    show_from: float         # clip seconds
    show_to: float
    fade_in: bool
    fade_out: bool
    fade_s: float

    def digest(self) -> str:
        blob = json.dumps([[s.png.name, s.xy] for s in self.sprites]
                          + [self.show_from, self.show_to, self.fade_in, self.fade_out,
                             self.fade_s])
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


@dataclass
class Transition:
    """A dissolve or swipe from the end of clip a to the start of clip b."""
    kind: str                # "dissolve" | "swipe"
    a: "Clip"                # the two clips' footage that gets blended
    b: "Clip"

    @property
    def duration(self) -> float:
        return self.a.duration


XFADE = {"dissolve": "fade", "swipe": "slideleft"}


@dataclass
class Clip:
    start: float             # seconds into the clip's own video file
    end: float
    label: str = ""          # burned into the top-left corner when non-empty
    video: Path | None = None    # None = the game's video
    info: Probe | None = None
    fade_in: bool = True
    fade_out: bool = True
    tags: TagLayer | None = None
    pan: list[tuple[float, float]] | None = None   # vertical: crop's top-left per frame (shares)
    ref: int | None = None   # which item of the video this footage belongs to (timestamps list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def clamp_clips(clips: list[Clip], default_info: Probe | None) -> list[Clip]:
    out = []
    for c in clips:
        info = c.info or default_info
        assert info is not None
        s, e = max(0.0, c.start), min(info.duration, c.end)
        if e - s >= 0.5:
            out.append(replace(c, start=s, end=e, info=info))
        else:
            log(f"[render] skipping clip at {fmt_time(c.start)}: past the end of the video")
    return out


def output_size(game: Game | Any, clips: list[Clip], vertical: bool = False) -> tuple[int, int]:
    """Width and height of the finished video. One size for the whole video
    (sources of different sizes must still concatenate); landscape is never
    upscaled past the smallest source. Vertical is the 9:16 window's own
    pixels (a straight crop, nothing resized) unless vertical.resize is on."""
    vc = game.cfg.vertical
    if vertical and vc.resize:
        return vc.width, vc.height
    small = min(clips, key=lambda c: c.info.height)
    if vertical:
        from .vertical import window
        return window(small.info.width, small.info.height, vc.width, vc.height, vc.zoom)
    height = min(game.cfg.render.height, small.info.height) // 2 * 2
    width = round(small.info.width * height / small.info.height / 2) * 2
    return width, height


def render_reel(game: Game | Any, clips: list[Clip], out_path: Path,
                info: Probe | None = None, what: str = "cutting clips",
                vertical: bool = False, transitions: bool = False,
                timeline: list[tuple[int | None, float]] | None = None) -> Path:
    """game: anything with .cfg and .work (a Game or a Session); clips without
    their own video come from game.video. transitions: mix dissolves and
    swipes in with the dips to black (render.transitions). timeline: filled
    with (clip ref, seconds) in output order, a blend split half and half."""
    global _xfade_broken
    clips = clamp_clips(clips, info)
    if not clips:
        raise PipelineError("Nothing to render: every clip was empty or out of range.")
    clip_dir = game.work / "clips"
    clip_dir.mkdir(exist_ok=True)
    size = output_size(game, clips, vertical)
    pieces: list[Clip | Transition] = (
        _with_transitions(game, clips) if transitions and not vertical
        and not _xfade_broken else list(clips))
    files = []
    if timeline is not None:
        timeline.clear()
    total = sum(p.duration for p in pieces)
    blends = [p.kind for p in pieces if isinstance(p, Transition)]
    mix = (f" ({blends.count('dissolve')} dissolves, {blends.count('swipe')} swipes)"
           if blends else "")
    log(f"[render] {len(clips)} clips, {total / 60:.1f} min{mix} -> {out_path.name}")
    with Progress("render", what, total) as bar:
        done = 0.0
        for p in pieces:
            step = lambda t, base=done: bar.update(base + t)  # noqa: E731
            if isinstance(p, Transition):
                try:
                    files.append(_encode_transition(game, p, clip_dir, size, step))
                except PipelineError as err:
                    said = str(err).split("\n", 2)[-1]       # ffmpeg's own words
                    if "xfade" in said or "No such filter" in said:
                        _xfade_broken = True
                        log(f"[render] warning: this ffmpeg can't do dissolves or swipes "
                            f"({err}); using dips to black instead")
                        return render_reel(game, clips, out_path, info, what, vertical,
                                           timeline=timeline)
                    # Only this cut: play its two halves straight through.
                    log(f"[render] warning: the {p.kind} at {fmt_time(p.a.end)} didn't render, "
                        f"so that cut is a straight cut ({said.strip()[-300:]})")
                    for half in (p.a, p.b):
                        files.append(_encode_clip(
                            game, replace(half, fade_in=False, fade_out=False),
                            half.video or game.video, clip_dir, size, vertical, step))
                        if timeline is not None:
                            timeline.append((half.ref, half.duration))
                else:
                    if timeline is not None:
                        timeline += [(p.a.ref, p.duration / 2), (p.b.ref, p.duration / 2)]
            else:
                video = p.video or game.video
                files.append(_encode_clip(game, p, video, clip_dir, size, vertical, step))
                if timeline is not None:
                    timeline.append((p.ref, p.duration))
            done += p.duration
            bar.update(done)

    list_file = game.work / f"{out_path.stem}.concat.txt"
    lines = []
    for f in files:
        safe = f.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", "-movflags", "+faststart", str(out_path)], "concat")
    finally:
        list_file.unlink(missing_ok=True)
    log(f"[render] wrote {out_path}")
    return out_path


# ----------------------------------------------------------------- internals

_drawtext_broken = False
_xfade_broken = False


def _pick(game: Any, a: Clip, b: Clip, last: str) -> str:
    """The transition for the cut from a to b: random-looking, but always the
    same for the same two clips, and never two swipes in a row."""
    weights = {k: max(0.0, v) for k, v in game.cfg.render.transitions.items()
               if k in ("fade", *XFADE)}
    total = sum(weights.values())
    if total <= 0:
        return "fade"
    seed = f"{a.video}|{a.end:.2f}|{b.video}|{b.start:.2f}"
    u = int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16) / 2 ** 32 * total
    for kind, w in weights.items():
        if u < w:
            break
        u -= w
    return "dissolve" if kind == "swipe" and last == "swipe" else kind


def _with_transitions(game: Any, clips: list[Clip]) -> list:
    """The clips with a dissolve or swipe piece at some cuts. A cut gets one
    only between two separate clips (not a camera split), neither with
    nametags, both long enough to give up the blend."""
    rc = game.cfg.render
    d = round(rc.transition_s * rc.fps) / rc.fps
    out: list = []
    last = ""
    for k, c in enumerate(replace(c) for c in clips):     # copies: the edges move
        if k == 0:
            out.append(c)
            continue
        a, b = out[-1], c
        kind = "fade"
        if (d > 0 and a.fade_out and b.fade_in and a.tags is None and b.tags is None
                and a.pan is None and b.pan is None and min(a.duration, b.duration) >= 3 * d):
            kind = _pick(game, a, b, last)
        last = kind
        if kind == "fade":
            out.append(b)
            continue
        va, vb = a.video, b.video
        blend = Transition(kind, Clip(a.end - d, a.end, video=va, info=a.info, ref=a.ref),
                           Clip(b.start, b.start + d, video=vb, info=b.info, ref=b.ref))
        a.end -= d
        a.fade_out = False
        b.start += d
        b.fade_in = False
        out += [blend, b]
    return out


def _encode_transition(game: Any, tr: Transition, clip_dir: Path, size: tuple[int, int],
                       on_progress: Callable[[float], None] | None = None) -> Path:
    rc = game.cfg.render
    enc = video_encoder_args(game.cfg)
    d = tr.duration
    key = {"transition": tr.kind, "d": round(d, 4), "size": list(size), "fps": rc.fps,
           "ab": rc.audio_bitrate, "enc": enc, "on_time": AUDIO_ON_TIME, "apad": True}
    args: list[str] = []
    for side, c in (("a", tr.a), ("b", tr.b)):
        video = c.video or game.video
        st = video.stat()
        key[side] = [str(video), st.st_size, int(st.st_mtime), round(c.start, 3)]
        args += ["-ss", f"{c.start:.3f}", "-t", f"{d:.3f}", "-i", str(video)]
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    out = clip_dir / f"x{digest}.mp4"
    if out.exists():
        return out
    w, h = size
    audio, n = [], 2
    for i, c in enumerate((tr.a, tr.b)):
        if c.info is not None and not c.info.has_audio:   # silence, so the blend has sound
            args += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            audio.append(f"[{n}:a]")
            n += 1
        else:
            audio.append(f"[{i}:a]")
    graph = [f"[{i}:v]scale={w}:{h},setsar=1,fps={rc.fps},format=yuv420p[v{i}]"
             for i in (0, 1)]
    graph.append(f"[v0][v1]xfade=transition={XFADE[tr.kind]}:duration={d:.4f}:offset=0[v]")
    # Exactly d of sound from each side, even where the camera skipped some
    # (acrossfade gives nothing at all when a side is short).
    graph += [f"{src}aresample=48000:{AUDIO_ON_TIME},aformat=channel_layouts=stereo,"
              f"apad=whole_dur={d:.4f},atrim=0:{d:.4f}[a{i}]" for i, src in enumerate(audio)]
    graph.append(f"[a0][a1]acrossfade=d={d:.4f}[a]")
    args += ["-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]",
             *enc, "-c:a", "aac", "-b:a", rc.audio_bitrate, "-ac", "2",
             "-movflags", "+faststart", "-avoid_negative_ts", "make_zero"]
    tmp = out.with_suffix(".tmp.mp4")
    run_ffmpeg([*args, str(tmp)], f"{tr.kind} {tr.a.end:.1f}s", on_progress)
    tmp.replace(out)
    return out


def _encode_clip(game: Any, clip: Clip, video: Path, clip_dir: Path, size: tuple[int, int],
                 vertical: bool, on_progress: Callable[[float], None] | None = None) -> Path:
    global _drawtext_broken
    rc = game.cfg.render
    info = clip.info
    assert info is not None
    label = clip.label if (clip.label and not _drawtext_broken) else ""
    enc = video_encoder_args(game.cfg)
    st = video.stat()
    key = {"video": str(video), "size": st.st_size, "mtime": int(st.st_mtime),
           "clip": [round(clip.start, 3), round(clip.end, 3), clip.fade_in, clip.fade_out],
           "label": label, "h": size[1], "fps": rc.fps, "fade": rc.fade_s,
           "ab": rc.audio_bitrate, "enc": enc, "on_time": AUDIO_ON_TIME}
    if vertical:
        key["vertical"] = [size, game.cfg.vertical.zoom,
                           hashlib.sha1(json.dumps(clip.pan).encode()).hexdigest()]
    if clip.tags is not None:
        key["tags"] = clip.tags.digest()
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    out = clip_dir / f"{digest}.mp4"
    if out.exists():
        return out

    try:
        _ffmpeg_clip(game, clip, video, out, info, size, vertical, label, enc, on_progress)
    except PipelineError as err:
        msg = str(err).lower()
        if not label or ("drawtext" not in msg and "font" not in msg):
            raise
        _drawtext_broken = True
        log("[render] warning: could not draw clip labels (font problem); "
            "rendering without them. Set review.font_file in config.yaml to fix.")
        return _encode_clip(game, clip, video, clip_dir, size, vertical, on_progress)
    return out


def _ffmpeg_clip(game: Any, clip: Clip, video: Path, out: Path, info: Probe,
                 size: tuple[int, int], vertical: bool, label: str, enc: list[str],
                 on_progress: Callable[[float], None] | None = None) -> None:
    rc = game.cfg.render
    d = clip.duration
    fade = min(rc.fade_s, d / 4)
    width, height = size
    cmds: list[str] = []
    vf = []
    if vertical:
        # A 9:16 window that pans (and with zoom, tilts) with the action.
        from .vertical import window
        cw, ch = window(info.width, info.height, width, height, game.cfg.vertical.zoom)
        corners = clip.pan or [(0.5 - cw / info.width / 2, 0.5 - ch / info.height / 2)]
        x0, y0 = _corner(corners[0], info, cw, ch)
        vf.append(f"crop@pan=w={cw}:h={ch}:x={x0}:y={y0}")
        if (cw, ch) != (width, height):
            vf.append(f"scale={width}:{height}")
        cmds += _pan_commands(corners, info, cw, ch, rc.fps)
    else:
        vf.append(f"scale=-2:{height}")
    vf += ["setsar=1", f"fps={rc.fps}"]
    af = [f"aresample=48000:{AUDIO_ON_TIME}"]
    if fade > 0 and clip.fade_in:
        vf.append(f"fade=t=in:st=0:d={fade:.3f}")
        af.append(f"afade=t=in:st=0:d={fade:.3f}")
    if fade > 0 and clip.fade_out:
        vf.append(f"fade=t=out:st={d - fade:.3f}:d={fade:.3f}")
        af.append(f"afade=t=out:st={d - fade:.3f}:d={fade:.3f}")
    # Edge fades (and the label) go on last, after any tags are laid over.
    finish = [f for f in vf if f.startswith("fade=")]
    vf = [f for f in vf if not f.startswith("fade=")]
    if label:
        finish.append(_drawtext(label, height, game.cfg.review.font_file))

    args = ["-ss", f"{clip.start:.3f}", "-t", f"{d:.3f}", "-i", str(video)]
    if info.has_audio:
        amap = "0:a:0"
    else:  # keep an audio track so every clip concatenates cleanly
        args += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = "1:a:0"
    graph = []
    chain = list(vf)
    tags = clip.tags
    if tags is not None and tags.sprites:
        first = 2 if not info.has_audio else 1
        last = "base"
        for k, sp in enumerate(tags.sprites):
            args += ["-loop", "1", "-framerate", str(rc.fps), "-t", f"{d:.3f}", "-i", str(sp.png)]
            sf = ["format=rgba"]
            ts = min(tags.fade_s, max(0.05, (tags.show_to - tags.show_from) / 3))
            if tags.fade_in:
                sf.append(f"fade=t=in:st={tags.show_from:.3f}:d={ts:.3f}:alpha=1")
            if tags.fade_out:
                sf.append(f"fade=t=out:st={max(0.0, tags.show_to - ts):.3f}:d={ts:.3f}:alpha=1")
            graph.append(f"[{first + k}:v]{','.join(sf)}[s{k}]")
            graph.append(f"[{last}][s{k}]overlay@t{k}=x=-10000:y=-10000:eval=frame:"
                         f"eof_action=pass[o{k}]")
            last = f"o{k}"
            cmds += _tag_commands(f"t{k}", sp.xy, rc.fps)
        tail = f"[{last}]" + ",".join(finish or ["null"]) + "[v]"
    else:
        tail = None
        chain += finish
    if cmds:
        cmd_file = out.with_suffix(".cmd")
        cmd_file.write_text("\n".join(cmds) + "\n", encoding="utf-8")
        chain.insert(0, f"sendcmd=f='{_escape(str(cmd_file.resolve()))}'")
    if tail is None:
        args += ["-map", "0:v:0", "-vf", ",".join(chain)]
    else:
        graph.insert(0, f"[0:v]{','.join(chain)}[base]")
        args += ["-filter_complex", ";".join(graph + [tail]), "-map", "[v]"]
    args += ["-map", amap, "-af", ",".join(af),
             *enc, "-c:a", "aac", "-b:a", rc.audio_bitrate, "-ac", "2",
             "-movflags", "+faststart", "-avoid_negative_ts", "make_zero"]
    tmp = out.with_suffix(".tmp.mp4")
    run_ffmpeg([*args, str(tmp)], f"clip {clip.start:.1f}-{clip.end:.1f}s", on_progress)
    tmp.replace(out)


def _corner(corner: tuple[float, float], info: Probe, cw: int, ch: int) -> tuple[int, int]:
    x = int(min(max(0, round(corner[0] * info.width / 2) * 2), info.width - cw))
    y = int(min(max(0, round(corner[1] * info.height / 2) * 2), info.height - ch))
    return x, y


def _when(i: int, fps: float) -> str:
    """Send a frame's command a moment before it, so it lands on that frame."""
    return f"{max(0.0, (i - 0.4) / fps):.4f}"


def _pan_commands(corners: list[tuple[float, float]], info: Probe, cw: int, ch: int,
                  fps: float) -> list[str]:
    out, last = [], None
    for i, corner in enumerate(corners):
        p = _corner(corner, info, cw, ch)
        if p != last:
            out.append(f"{_when(i, fps)} crop@pan x {p[0]}, crop@pan y {p[1]};")
            last = p
    return out


def _tag_commands(name: str, xy: list[tuple[int, int] | None], fps: float) -> list[str]:
    out, last = [], None
    for i, p in enumerate(xy):
        p = p or (-10000, -10000)
        if p != last:
            out.append(f"{_when(i, fps)} overlay@{name} x {p[0]}, overlay@{name} y {p[1]};")
            last = p
    return out


def _escape(value: str) -> str:
    """Escape a value for use inside single quotes in an ffmpeg filter option."""
    return value.replace("\\", "/").replace("'", "").replace(":", "\\:")


def _drawtext(label: str, height: int, font_file: str) -> str:
    size = max(18, height // 22)
    opts = [f"text='{_escape(label)}'", f"fontsize={size}", "fontcolor=white",
            "box=1", "boxcolor=black@0.55", f"boxborderw={size // 3}",
            f"x={size // 2}", f"y={size // 2}", "expansion=none"]
    if font_file == "auto":
        font_file = "C:/Windows/Fonts/arial.ttf" if on_windows() else ""
    if font_file:
        opts.insert(0, f"fontfile='{_escape(font_file)}'")
    return "drawtext=" + ":".join(opts)
