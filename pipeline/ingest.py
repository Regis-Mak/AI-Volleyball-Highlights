"""Stage 1: ingest. Probe the source, build the 720p analysis proxy and the
cleaned mono audio tracks the audio stage reads.

work/<game>/
  meta.json        duration, size, fps of the source
  proxy.mp4        720p, keyframe every second (fast seeking for motion/VLM)
  audio_16k.wav    16 kHz mono, wind-cleaned (Whisper)
  audio_32k.wav    32 kHz mono, wind-cleaned (PANNs expects 32 kHz)

Gaps the camera left in the sound are filled with silence (AUDIO_ON_TIME), so
a time in these files is the same time in the video.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .common import (AUDIO_ON_TIME, Game, PipelineError, Probe, Progress, is_fresh, log,
                     mark_done, probe, run_ffmpeg, video_encoder_args)


def run(game: Game) -> Probe:
    info = probe_game(game)
    log(f"[ingest] {game.video.name}: {info.duration / 60:.1f} min, "
        f"{info.width}x{info.height} @ {info.fps:.2f} fps")

    _make_proxy(game, info)
    _extract_audio(game, info)
    return info


def cache_plan(game: Game) -> dict[str, tuple[list[Path], dict]]:
    """Each ingest step's outputs and the settings they depend on. A step is
    skipped when its outputs exist and were made with these settings."""
    cfg, w = game.cfg.ingest, game.work
    src = game.source_id()
    return {
        "probe": ([w / "meta.json"], src),
        "proxy": ([w / "proxy.mp4"],
                  {**src, "height": cfg.proxy_height, "fps": cfg.proxy_fps}),
        "audio_extract": ([w / "audio_16k.wav", w / "audio_32k.wav"],
                          {**src, "filter": cfg.audio_filter, "on_time": AUDIO_ON_TIME}),
    }


def probe_game(game: Game) -> Probe:
    """Duration, size and fps of the source (cached in meta.json)."""
    (meta_path,), params = cache_plan(game)["probe"]
    if is_fresh(game, "probe", [meta_path], params):
        return Probe(**json.loads(meta_path.read_text()))
    info = probe(game.video)
    meta_path.write_text(json.dumps(asdict(info), indent=2))
    mark_done(game, "probe", params)
    return info


def _make_proxy(game: Game, info: Probe) -> None:
    cfg = game.cfg.ingest
    (out,), params = cache_plan(game)["proxy"]
    if is_fresh(game, "proxy", [out], params):
        log("[ingest] proxy.mp4 cached")
        return
    tmp = out.with_suffix(".tmp.mp4")
    with Progress("ingest", f"{cfg.proxy_height}p analysis proxy", info.duration) as bar:
        run_ffmpeg([
            "-i", str(game.video), "-map", "0:v:0", "-an",
            "-vf", f"scale=-2:{cfg.proxy_height},fps={cfg.proxy_fps}",
            *video_encoder_args(game.cfg), "-g", str(cfg.proxy_fps),
            "-movflags", "+faststart", str(tmp),
        ], "proxy", on_progress=bar.update)
    tmp.replace(out)
    mark_done(game, "proxy", params)


def _extract_audio(game: Game, info: Probe) -> None:
    if not info.has_audio:
        raise PipelineError(f"{game.video.name} has no audio track; the reaction "
                            "signals need one.")
    cfg = game.cfg.ingest
    paths, params = cache_plan(game)["audio_extract"]
    outs = dict(zip((16000, 32000), paths))
    if is_fresh(game, "audio_extract", paths, params):
        log("[ingest] audio tracks cached")
        return
    args = ["-i", str(game.video)]
    for rate, path in outs.items():
        args += ["-map", "0:a:0", "-vn", "-af", f"aresample={AUDIO_ON_TIME},{cfg.audio_filter}",
                 "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(path)]
    with Progress("ingest", "wind-cleaned audio", info.duration) as bar:
        run_ffmpeg(args, "audio extract", on_progress=bar.update)
    mark_done(game, "audio_extract", params)
