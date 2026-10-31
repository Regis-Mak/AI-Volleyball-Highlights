"""Volleyball highlights: one command from a day of footage to highlight videos.

    python make_highlights.py "D:\\Volleyball\\06-14-2026"               (every video in a folder)
    python make_highlights.py game.mp4                                  (one video)
    python make_highlights.py "...\\06-14-2026" "...\\06-21-2026"       (a queue: one after another)

Runs ingest, audio reactions, motion filter, vision tagging and unhinged-comment
rating on each file, then scores every moment across the whole session and
renders, into output/<session>/, a ~20 minute highlight reel (long rallies and
big spikes first, plus the most unhinged things people said), a clip of every
play someone said "clip that" about (and 30-60 s themed shorts with
shorts.enabled). picks.csv lists every clip; set keep to n
on a row and run again to swap that clip out.

    --pick-only          choose the clips and write picks.csv, render nothing
    --at 3:10 1:12:45    stitch hand-picked moments (session clock)
    --review peaks       Phase 2 check: top audio reactions only
    --review tags        Phase 3 check: a sample of vision-tagged moments

The app (in your browser, on this PC only):
    --gui                Run, Grade, Test set, Picks, Who is who and Progress screens

Learning (library/ keeps every moment and label across sessions):
    --grade              open the Grade screen: the moments it's least sure about
    --grade test         grade moments for the test set
    --learn              retrain the ranker now and list its versions
    --use-ranker v3      switch ranker version ("off" = config weights only)
    --tune               re-tune the scoring, clip lengths and reel cutoff now
    --use-settings v2    switch tuned-settings version ("off" = config.yaml as written)
    --evaluate           score the vision model on your test set (or name models:
                         --evaluate qwen3-vl:4b qwen3-vl:8b)
    --export-training    write the training set for fine-tuning Qwen (Phase 7)

Before a long run:
    --check [folder]     check the setup (GPU, Ollama, models, disk) and, given a
                         folder, which steps are already done for each video
"""
from __future__ import annotations

import argparse
import csv
import shutil
import signal
import sys
import time
from pathlib import Path

from pipeline import audio, check, comments, fuse, ingest, selection, vlm
from pipeline.common import (Game, PipelineError, Progress, clear_stage, clock, error,
                             fmt_hms, fmt_time, log, notice, parse_time, require_ffmpeg)
from pipeline.config import Config, load_config
from pipeline.learn import evaluate, memory, ranker, trainset, tune
from pipeline.library import Library, log_summary, open_library
from pipeline.render import Clip, render_reel
from pipeline.session import Session, SessionFile, open_session

HERE = Path(__file__).resolve().parent
STAGES = {"ingest": ["probe", "proxy", "audio_extract"], "transcript": ["transcript"],
          "panns": ["panns"], "motion": ["motion"], "fingerprints": ["fingerprints"],
          "vlm": [], "comments": []}
REVIEW_ORDER = ["blooper", "great_play", "celebration", "chaos", "conversation", "rally",
                "downtime"]


def _default_config() -> Path:
    """config.yaml (yours, kept out of git), else the shipped config.example.yaml."""
    own = HERE / "config.yaml"
    return own if own.exists() else HERE / "config.example.yaml"


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", type=Path, nargs="*", metavar="source",
                    help="a session folder (every video in it) or one video file; give "
                         "several to run them one after another")
    ap.add_argument("--config", type=Path, default=_default_config())
    ap.add_argument("--pick-only", action="store_true",
                    help="choose the clips and write picks.csv + thumbnails, but render "
                         "no videos (veto in picks.csv, then run again)")
    ap.add_argument("--at", nargs="+", metavar="TIME",
                    help="hand-picked moments on the session clock: 12:34 (uses default "
                         "pre/post) or 12:30-12:45")
    ap.add_argument("--picks", type=Path,
                    help="text file of moments, one per line, same format as --at")
    ap.add_argument("--review", choices=["tags", "peaks", "none"], default="none",
                    help="render a check video instead of the highlights: peaks = top "
                         "audio reactions (Phase 2), tags = a sample of vision-tagged "
                         "moments (Phase 3)")
    ap.add_argument("--force", nargs="+", choices=[*STAGES, "all"], default=[],
                    help="ignore the cache for these stages and recompute them")
    ap.add_argument("--gui", action="store_true",
                    help="open the app in your browser (Run, Grade, Test set, Picks, "
                         "Who is who, Progress)")
    ap.add_argument("--grade", nargs="?", const="quick", choices=["quick", "test"],
                    help="open the Grade screen in your browser: quick = the moments the "
                         "tool is least sure about (default), test = build the test set")
    ap.add_argument("--learn", action="store_true",
                    help="retrain the ranker from the library now and list its versions")
    ap.add_argument("--use-ranker", metavar="VERSION",
                    help="switch the active ranker version (e.g. v3), or 'off' to score "
                         "with the config weights only")
    ap.add_argument("--tune", action="store_true",
                    help="re-tune the scoring weights, clip lengths and reel cutoff from "
                         "your labels now and list the tuned-settings versions")
    ap.add_argument("--use-settings", metavar="VERSION",
                    help="switch the tuned-settings version (e.g. v2), or 'off' to use "
                         "config.yaml exactly as written")
    ap.add_argument("--evaluate", nargs="*", metavar="MODEL",
                    help="re-tag your test set and score the answers against your grades: "
                         "the model in config.yaml, or the Ollama models named")
    ap.add_argument("--export-training", action="store_true",
                    help="write the training set for fine-tuning Qwen (Phase 7) into "
                         "library/training/")
    ap.add_argument("--check", action="store_true",
                    help="check the setup before a run; with a folder or file, also which "
                         "steps are already done for each video")
    args = ap.parse_args(argv)
    args.source = args.sources[0] if args.sources else None
    if len(args.sources) > 1 and (args.at or args.picks):
        ap.error("--at / --picks take one session folder or file")
    if args.source is None and not (args.gui or args.grade or args.learn or args.use_ranker
                                    or args.tune or args.use_settings or args.check
                                    or args.evaluate is not None or args.export_training):
        ap.error("give a session folder or a video file")
    return args


def picks_to_clips(entries: list[str], pre: float, post: float) -> list[Clip]:
    clips = []
    for raw in entries:
        entry = raw.split("#", 1)[0].strip()
        if not entry:
            continue
        if "-" in entry:
            a, b = entry.split("-", 1)
            start, end = parse_time(a), parse_time(b)
            if end <= start:
                raise PipelineError(f"Pick {entry!r}: end is before start")
        else:
            t = parse_time(entry)
            start, end = t - pre, t + post
        clips.append(Clip(start, end))
    return clips


def render_picks(session: Session, entries: list[str]) -> None:
    rc = session.cfg.render
    picks = picks_to_clips(entries, rc.default_pre_s, rc.default_post_s)
    picks.sort(key=lambda c: c.start)
    clips = []
    for p in picks:
        segs = session.segments(p.start, p.end)
        if not segs:
            log(f"[render] skipping {fmt_time(p.start)}: no footage there")
        clips += [Clip(a, b, video=f.game.video, info=f.info,
                       fade_in=(k == 0), fade_out=(k == len(segs) - 1))
                  for k, (f, a, b) in enumerate(segs)]
    render_reel(session, clips, session.out / "picks_test.mp4")


def _out_name(session: Session, f: SessionFile, name: str) -> Path:
    """review_tags.mp4 for one file; '<file> review_tags.mp4' in a session."""
    return session.out / (name if len(session.files) == 1 else f"{f.game.name} {name}")


def render_review(game: Game, info, peaks: list[audio.Peak], out: Path) -> None:
    rv = game.cfg.review
    top = peaks[: rv.top_n]
    if not top:
        log("[review] no reaction peaks found")
        return
    clips = []
    for p in top:
        label = f"#{p.rank}  {fmt_time(p.time)}  {p.dominant}" if rv.label_clips else ""
        clips.append(Clip(p.time - rv.pre_s, p.time + rv.post_s, label))
    out = render_reel(game, clips, out, info)

    # Where each clip lands in the review video, so you can note good/bad ones.
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "review_video_at", "game_time", "dominant", "score",
                    "transcript", "real_moment (y/n)"])
        offset = 0.0
        for p, c in zip(top, clips):
            w.writerow([p.rank, fmt_time(offset), fmt_time(p.time), p.dominant,
                        f"{p.score:.2f}", p.transcript, ""])
            offset += min(c.end, info.duration) - max(0.0, c.start)
    log(f"[review] watch {out}\n[review] and mark each row in {csv_path}")


def pick_tag_review(windows: list[vlm.Window], n: int) -> list[vlm.Window]:
    """A spread of labelled moments: the best of each label in turn, so the
    spot-check covers bloopers, great plays, celebrations and ordinary play."""
    groups = {e: sorted((w for w in windows if w.source == "vlm" and w.tag.event_type == e),
                        key=lambda w: (w.tag.funny + w.tag.excitement, w.reaction_max),
                        reverse=True)
              for e in REVIEW_ORDER}
    chosen: list[vlm.Window] = []
    while len(chosen) < n and any(groups.values()):
        for e in REVIEW_ORDER:
            while groups[e]:
                w = groups[e].pop(0)
                if all(abs(w.peak_time - c.peak_time) >= 10 for c in chosen):
                    chosen.append(w)
                    break
            if len(chosen) >= n:
                break
    return sorted(chosen, key=lambda w: w.peak_time)


def render_tag_review(game: Game, info, windows: list[vlm.Window], out: Path) -> None:
    rv, rc = game.cfg.review, game.cfg.render
    picked = pick_tag_review(windows, rv.top_n)
    if not picked:
        log("[review] no tagged windows to review")
        return
    clips = []
    for i, w in enumerate(picked, 1):
        t = w.tag
        label = (f"#{i}  {fmt_time(w.peak_time)}  {t.event_type}  fun {t.funny}  hype {t.excitement}"
                 if rv.label_clips else "")
        clips.append(Clip(w.peak_time - rc.default_pre_s, w.peak_time + rc.default_post_s, label))
    out = render_reel(game, clips, out, info)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["#", "review_video_at", "game_time", "event_type", "funny", "excitement",
                     "players", "description", "label_right (y/n)"])
        offset = 0.0
        for i, (w, c) in enumerate(zip(picked, clips), 1):
            wr.writerow([i, fmt_time(offset), fmt_time(w.peak_time), w.tag.event_type,
                         w.tag.funny, w.tag.excitement, "; ".join(w.tag.players),
                         w.tag.description, ""])
            offset += min(c.end, info.duration) - max(0.0, c.start)
    log(f"[review] watch {out}\n[review] and mark each row in {csv_path}")


def _need_library(cfg: Config) -> Library:
    lib = open_library(cfg)
    if lib is None:
        raise PipelineError("Learning is switched off (learning.enabled in config.yaml).")
    return lib


def learning_commands(args: argparse.Namespace, cfg: Config) -> None:
    """--grade, --learn and --use-ranker: no video processing."""
    if args.gui or args.grade:
        from gui.app import serve
        session = source = None
        if args.source is not None:
            src = args.source.expanduser()
            session = src.name if src.is_dir() else src.stem
            source = str(src.resolve())
        if args.grade:
            serve(cfg, "/test" if args.grade == "test" else "/grade", session)
        else:
            serve(cfg, "/run", source=source)
        return
    lib = _need_library(cfg)
    if args.use_ranker:
        name = args.use_ranker.strip().lower()
        if name in ("off", "none", "config"):
            lib.set_active("ranker", None)
            log("[learn] scoring with the config weights only from the next run")
        elif name not in {r["name"] for r in lib.versions("ranker")}:
            raise PipelineError(f"No ranker version {name!r}.\n"
                                + ranker.describe_versions(lib))
        else:
            lib.set_active("ranker", name)
            log(f"[learn] ranker {name} is active from the next run")
    if args.use_settings:
        name = args.use_settings.strip().lower()
        if name in ("off", "none", "config"):
            lib.set_active(tune.KIND, None)
            log("[tune] using config.yaml as written from the next run")
        elif name not in {r["name"] for r in lib.versions(tune.KIND)}:
            raise PipelineError(f"No tuned-settings version {name!r}.\n"
                                + tune.describe_versions(lib))
        else:
            lib.set_active(tune.KIND, name)
            log(f"[tune] settings {name} are on from the next run")
    if args.learn:
        log(f"[learn] {ranker.retrain(lib, force=True)}")
    if args.tune:
        log(f"[tune] {tune.retune(lib, force=True)}")
    if args.export_training:
        trainset.export(lib)
        log(f"[train] {trainset.status(lib)}")
    if args.evaluate is not None:
        evaluate.run(lib, args.evaluate)
    log_summary(lib)
    if args.learn or args.use_ranker:
        log("[learn] ranker versions:\n" + ranker.describe_versions(lib))
    if args.tune or args.use_settings:
        log("[tune] tuned-settings versions:\n" + tune.describe_versions(lib))


def keep_frames(lib: Library, f: SessionFile, windows: list[vlm.Window]) -> None:
    """Make sure the library has the frames and the prompt Qwen saw for every
    tagged window (windows tagged before the library existed get their frames
    re-cut from the proxy, and ones tagged before prompts were kept get theirs
    rebuilt the same way)."""
    stem = f.game.name
    tagged = [w for w in windows if w.source == "vlm" and w.frames]
    missing = [w for w in tagged if not lib.has_frames(stem, w.start)]
    if missing:
        with Progress("library", "saving frames", len(missing)) as bar:
            for n, w in enumerate(missing, 1):
                lib.save_frames(stem, w.start, vlm.window_frames(f.game, w))
                bar.update(n)
    builder = None
    for w in tagged:
        if lib.read_prompt(stem, w.start) is not None:
            continue
        if not w.prompt:
            builder = builder or vlm.PromptBuilder(f.game)
            w.prompt = builder.build(w.start, w.end, w.frames)
        lib.save_prompt(stem, w.start, w.prompt)
    lib.record_windows(f, windows, f.game.cfg.vlm.model, vlm.prompt_version())


def _each(session: Session, what: str):
    """The session's files in order, announcing each one when there are several."""
    n = len(session.files)
    for f in session.files:
        if n > 1:
            log(f"[session] {what}: file {f.index + 1} of {n}, {f.name}")
        yield f


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if hasattr(signal, "SIGBREAK"):
        # The app's Stop button sends Ctrl+Break on Windows: stop like Ctrl+C.
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    t0 = time.time()
    log(f"Started {clock(t0)}")
    try:
        cfg = load_config(args.config)
        if args.check:
            return 1 if check.run(cfg, args.source) else 0
        if (args.source is None or args.learn or args.use_ranker or args.grade or args.gui
                or args.tune or args.use_settings or args.evaluate is not None
                or args.export_training):
            learning_commands(args, cfg)
            return 0
        require_ffmpeg()
        lib = open_library(cfg)
        if len(args.sources) == 1:
            run_session(args, cfg, lib, args.source, t0)
            return 0
        return run_queue(args, cfg, lib, t0)
    except PipelineError as err:
        error(str(err))
        return 1
    except KeyboardInterrupt:
        notice("Stopped. Everything finished so far is saved; run the same command to pick up.")
        return 130


def run_queue(args: argparse.Namespace, cfg: Config, lib: "Library | None", t0: float) -> int:
    """Several sessions, one after another. One that fails is reported and
    the next one starts; Ctrl+C stops them all."""
    n, failed = len(args.sources), []
    for i, src in enumerate(args.sources, 1):
        log(f"[queue] session {i} of {n}: {src}")
        try:
            run_session(args, cfg, lib, src, time.time())
        except PipelineError as err:
            error(f"{src.name}: {err}")
            failed.append(src.name)
            log(f"[queue] {src.name} failed; going on with the next one" if i < n
                else f"[queue] {src.name} failed")
    done = n - len(failed)
    log(f"[queue] {done} of {n} sessions done" + (f"; failed: {', '.join(failed)}" if failed else "")
        + f" (took {fmt_hms(time.time() - t0)})")
    return 1 if failed else 0


def run_session(args: argparse.Namespace, cfg: Config, lib: "Library | None", source: Path,
                t0: float) -> None:
    """Everything for one session folder (or one video)."""
    session = open_session(source, tune.apply(cfg, lib))
    if lib:
        lib.record_session(session)
    for f in session.files:
        for name in (STAGES if "all" in args.force else args.force):
            for stamp in STAGES[name]:
                clear_stage(f.game, stamp)
            if name == "vlm":
                shutil.rmtree(f.game.work / "vlm", ignore_errors=True)
            if name == "comments":
                (f.game.work / comments.CACHE).unlink(missing_ok=True)

    # Stage by stage across all files, so each model loads once per sweep
    # and every file's cheap steps finish before the slow ones start.
    for f in _each(session, "ingest"):
        ingest.run(f.game)

    if args.at or args.picks:
        entries = list(args.at or [])
        if args.picks:
            entries += args.picks.read_text(encoding="utf-8").splitlines()
        render_picks(session, entries)
        log(f"Finished {clock()} (took {fmt_hms(time.time() - t0)})")
        return

    review = args.review
    peaks = {f.index: audio.run(f.game, f.info.duration)
             for f in _each(session, "audio")}
    if review == "peaks":
        for f in session.files:
            render_review(f.game, f.info, peaks[f.index],
                          _out_name(session, f, "review_peaks.mp4"))
        log(f"Finished {clock()} (took {fmt_hms(time.time() - t0)})")
        return

    mem = memory.prepare(session, lib) if lib else None
    windows = {}
    for f in _each(session, "vision"):
        keep = None
        if lib and cfg.learning.save_frames:
            keep = (lambda start, jpegs, stem=f.game.name:
                    lib.save_frames(stem, start, jpegs))
        windows[f.index] = vlm.run(f.game, f.info.duration, keep_frames=keep,
                                   hints=mem.hints_for(f) if mem else None)
        if lib:
            if cfg.learning.save_frames:
                keep_frames(lib, f, windows[f.index])
            else:
                lib.record_windows(f, windows[f.index], cfg.vlm.model,
                                   vlm.prompt_version())
    if review == "tags":
        for f in session.files:
            render_tag_review(f.game, f.info, windows[f.index],
                              _out_name(session, f, "review_tags.mp4"))
        log(f"Finished {clock()} (took {fmt_hms(time.time() - t0)})")
        return

    said = {}
    if cfg.comments.enabled:
        rater = comments.Rater(cfg)
        try:
            for f in _each(session, "comments"):
                said[f.index] = rater.run(f.game)
        finally:
            rater.close()

    moments = fuse.run(session, windows, peaks, lib, said)
    selection.run(session, moments, render=not args.pick_only, lib=lib)
    if lib:
        log_summary(lib)
        log(f"[learn] {ranker.retrain(lib)}")
        log(f"[tune] {tune.retune(lib)}")
    log(f"Finished {clock()} (took {fmt_hms(time.time() - t0)})")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
