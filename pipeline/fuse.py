"""Stage 5: fuse. Turn every file's audio reactions and vision tags into
scored moments on the session clock.

A moment is one thing worth showing: when it peaks, what kind it is, how
exciting and how funny it is, and the clip that shows it. Candidates are
every vision-tagged window that isn't downtime, plus strong audio reactions
the tags missed (a joke between points still makes people laugh).
Overlapping windows that saw the same play are merged into one moment.

Unhinged comments (pipeline/comments.py) join the moment they were said in,
widening its clip to the comment and the reaction to it, or become a
moment of their own.

Scores, all roughly 0-1 (weights live in config.yaml under fuse:):
  rally  = how long the rally lasts (rally_short_s -> 0 ... rally_long_s -> 1)
  attack = the hardest spike Qwen saw / 10
  hype   = excitement / 10
  fun    = fun_mix x funny / 10 + (1 - fun_mix) x laughter around the moment
  score  = w_rally x rally + w_attack x attack + w_excitement x hype
           + w_funny x fun + w_reaction x reaction
           + a small bonus by label (great plays first)
           + comments.score_weight x unhinged/10 for an unhinged comment
           + clip_that.reel_bonus when someone said "clip that" about it

Clips start a few seconds before the key instant (how many depends on the
label) and run on while the group is still reacting, within min/max lengths.
A rally or great play (fuse.whole_rally_labels) gets the whole rally instead:
from just before the serve, through the celebration, stopping just before
the next serve (up to fuse.max_rally_clip_s). Rallies come from every window
Qwen tagged, the same rally seen by overlapping windows joined into one, so
the next rally's start is the next serve.

Once the library has enough labels, a learned ranker (pipeline/learn/ranker.py)
blends its own judgement into score and fun, and these settings may be tuned
versions of config.yaml's (pipeline/learn/tune.py). Your Grade-screen answers apply
directly: a corrected label replaces Qwen's, and a moment you marked "not
worth showing" is left out of every video.

work/<session>/moments.csv   every moment, best first
work/<session>/comments.csv  every rated comment, most unhinged first
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import audio, clipthat, comments, vlm
from . import motion as motion_mod
from .common import fmt_time, log
from .learn import features as feat
from .learn import ranker
from .learn.labels import agreement_labels
from .session import Session, SessionFile

if TYPE_CHECKING:
    from .library import Library


@dataclass
class Moment:
    id: str                  # "<file name>@<seconds into that file>", stable across runs
    file: int                # index into session.files
    t: float                 # session time of the key instant
    start: float             # clip, on the session clock
    end: float
    event_type: str
    description: str
    players: list[str] = field(default_factory=list)
    excitement: int = 0
    funny: int = 0
    attack: int = 0
    rally_s: float = 0.0
    rally: tuple[float, float] | None = None   # the longest rally, on the session clock
    clip_that: str = ""      # what was said, when someone asked for this play to be clipped
    unhinged: float = 0.0    # 0-10, the most unhinged comment said in this clip
    comment: str = ""        # ...and what it was
    reaction: float = 0.0
    laugh: float = 0.0
    hype: float = 0.0
    fun: float = 0.0
    score: float = 0.0
    source: str = "vlm"      # vlm | audio | comment
    play_active: bool = False
    features: dict[str, float] = field(default_factory=dict)
    base_score: float = 0.0  # score from the config weights alone
    base_fun: float = 0.0
    p_highlight: float | None = None   # the ranker's estimate, when one is active
    rejected: bool = False   # you marked it "not worth showing" on the Grade screen

    @property
    def length(self) -> float:
        return self.end - self.start


def run(session: Session, windows: dict[int, list[vlm.Window]],
        peaks: dict[int, list[audio.Peak]], lib: "Library | None" = None,
        said: "dict[int, list[comments.Comment]] | None" = None) -> list[Moment]:
    """windows / peaks / said: per file index, as returned by vlm.run /
    audio.run / comments.Rater.run."""
    graded = lib.hand_labels(session.name) if lib else {}
    moments: list[Moment] = []
    ctxs = {}
    for f in session.files:
        tl = vlm.load_timeline(f.game)
        motion = motion_mod.run(f.game, f.info.duration)
        ctx = ctxs[f.index] = (tl, motion, feat.motion_reference(motion))
        moments += _from_windows(session, f, ctx, windows.get(f.index, []), graded)
        moments += _from_peaks(session, f, ctx, peaks.get(f.index, []),
                               [m for m in moments if m.file == f.index], graded)
    moments = _merge(session, moments)
    if said:
        moments += _add_comments(session, moments, said, ctxs, graded)
    calls = clipthat.find(session)
    for c in calls:
        m = clipthat.target(session, moments, c)
        if m is not None and not m.clip_that:
            m.clip_that = c.text
            m.score += session.cfg.clip_that.reel_bonus
    if calls:
        log(f'[fuse] "clip that" heard {len(calls)} time{"s" if len(calls) != 1 else ""}: '
            + clipthat.describe(calls))
    for m in moments:
        m.base_score, m.base_fun = m.score, m.fun
        m.rejected = graded.get(m.id, {}).get("highlight") == 0
    if lib:
        ranker.apply(lib, moments)
        lib.record_moments(session, moments, session.cfg.vlm.model, vlm.prompt_version())
        lib.set_labels(session.name, "agree", agreement_labels(moments))
        fixed = sum("event_type" in graded.get(m.id, {}) for m in moments)
        rejected = sum(m.rejected for m in moments)
        if fixed or rejected:
            log(f"[fuse] from your grades: {fixed} labels corrected, "
                f"{rejected} moments left out")
    moments.sort(key=lambda m: m.score, reverse=True)
    _write(session, moments)
    counts: dict[str, int] = {}
    for m in moments:
        counts[m.event_type] = counts.get(m.event_type, 0) + 1
    log(f"[fuse] {len(moments)} moments: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    return moments


# -------------------------------------------------------------- candidates

def _window_max(x: np.ndarray, a: float, b: float) -> float:
    i, j = max(0, int(a)), min(len(x), int(np.ceil(b)) + 1)
    return float(x[i:j].max()) if j > i else 0.0


def _reaction_at(session: Session, tl: dict[str, np.ndarray], lt: float) -> tuple[float, float]:
    fc = session.cfg.fuse
    a, b = lt - fc.reaction_lookback_s, lt + fc.reaction_lookahead_s
    reaction = min(1.5, _window_max(tl["reaction"], a, b))
    laugh = min(1.0, _window_max(tl["laugh"], a, b))
    return reaction, laugh


def rally_share(session: Session, rally_s: float) -> float:
    """0 for a rally of fuse.rally_short_s or less, 1 from fuse.rally_long_s."""
    fc = session.cfg.fuse
    span = max(1e-6, fc.rally_long_s - fc.rally_short_s)
    return float(min(1.0, max(0.0, (rally_s - fc.rally_short_s) / span)))


def _score(session: Session, m: Moment) -> None:
    fc = session.cfg.fuse
    w = fc.weights
    m.hype = m.excitement / 10
    m.fun = fc.fun_mix * m.funny / 10 + (1 - fc.fun_mix) * m.laugh
    m.score = (w.get("rally", 0) * rally_share(session, m.rally_s)
               + w.get("attack", 0) * m.attack / 10
               + w.get("excitement", 0) * m.hype + w.get("funny", 0) * m.fun
               + w.get("reaction", 0) * min(1.0, m.reaction)
               + fc.label_bonus.get(m.event_type, 0.0)
               + session.cfg.comments.score_weight * m.unhinged / 10)
    if m.source == "audio":
        m.score *= fc.audio_only_factor


def _features(session: Session, ctx: tuple, m: Moment, lt: float) -> None:
    tl, motion, ref = ctx
    fc = session.cfg.fuse
    m.features = feat.compute(tl, motion, ref, lt, m.event_type, m.excitement, m.funny,
                              m.play_active, m.source == "audio", m.length, len(m.players),
                              fc.reaction_lookback_s, fc.reaction_lookahead_s,
                              rally_share(session, m.rally_s), m.attack, m.unhinged)


MAX_RALLY_S = 90.0    # overlapping windows' rallies are never joined past this


def _rallies(windows: list[vlm.Window]) -> list[tuple[float, float]]:
    """Every rally Qwen saw in a file, (serve, ball dead) in local seconds, in
    order. Overlapping windows see the same rally: those spans are joined."""
    spans = sorted((w.start + w.tag.rally_start_s, w.start + w.tag.rally_end_s)
                   for w in windows if w.source == "vlm" and w.rally_s > 0)
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a <= out[-1][1] + 1.0 and max(b, out[-1][1]) - out[-1][0] <= MAX_RALLY_S:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _rally_at(rallies: list[tuple[float, float]], own: tuple[float, float] | None,
              lt: float) -> tuple[float, float] | None:
    """The whole rally a window's own rally (or, without one, its key instant) is part of."""
    if own is not None:
        hit = [r for r in rallies if r[0] <= own[1] + 1.0 and own[0] <= r[1] + 1.0]
    else:
        hit = [r for r in rallies if r[0] - 2.0 <= lt <= r[1] + 2.0]
    return max(hit, key=lambda r: r[1] - r[0], default=None)


def _after_rally(session: Session, tl: dict[str, np.ndarray], rally: tuple[float, float],
                 rallies: list[tuple[float, float]], cheers: list[float]) -> float:
    """Where a whole rally's clip ends: after the celebration (at least
    celebrate_min_s, longer while the group reacts or a celebration Qwen saw
    is still going), and before the next serve."""
    fc = session.cfg.fuse
    dead = rally[1]
    end, limit = dead + fc.celebrate_min_s, dead + fc.celebrate_max_s
    for p in cheers:
        if dead < p <= limit:
            end = max(end, p + 3.0)
    r = tl["reaction"]
    while end < limit and int(end) < len(r) and r[int(end)] >= fc.extend_while_reaction:
        end += 1.0
    end = min(end, limit)
    nxt = next((a for a, _b in rallies if a > dead + 1.0), None)
    if nxt is not None:
        end = min(end, max(dead + 1.0, nxt - fc.next_serve_gap_s))
    return end


def _clip_bounds(session: Session, f: SessionFile, tl: dict[str, np.ndarray],
                 lt: float, event_type: str,
                 rally: tuple[float, float] | None = None,
                 rallies: list[tuple[float, float]] | None = None,
                 cheers: list[float] | None = None) -> tuple[float, float]:
    """Clip around local time lt, returned on the session clock. rally: the
    whole rally the moment is part of (local); rallies: every rally in the
    file, for the next serve; cheers: when Qwen saw celebrations."""
    fc = session.cfg.fuse
    shape = fc.clip.get(event_type) or fc.clip.get("reaction")
    pre, post = (shape.pre_s, shape.post_s) if shape else (6.0, 6.0)
    start, end = lt - pre, lt + post
    whole = rally is not None and event_type in fc.whole_rally_labels
    long_rally = rally is not None and rally[1] - rally[0] >= fc.rally_short_s
    if whole:
        assert rally is not None
        start = rally[0] - fc.serve_pre_s
        end = _after_rally(session, tl, rally, rallies or [rally], cheers or [])
    else:
        if long_rally:
            assert rally is not None
            start, end = min(start, rally[0] - 1.5), max(end, rally[1] + 2.0)
        r = tl["reaction"]
        limit = end + fc.max_extend_s
        while end < limit and int(end) < len(r) and r[int(end)] >= fc.extend_while_reaction:
            end += 1.0
        end = min(end, limit)
    cap = fc.max_rally_clip_s if whole or long_rally else fc.max_clip_s
    if end - start > cap:
        start = end - cap
    if end - start < fc.min_clip_s:
        start = end - fc.min_clip_s
    start, end = _in_file(session, f, start, end)
    return f.offset + start, f.offset + end


def _in_file(session: Session, f: SessionFile, start: float,
             end: float) -> tuple[float, float]:
    """Stay inside this file (local times) unless the camera split the recording here."""
    files = session.files
    if not f.joined_prev:
        start = max(0.0, start)
    if not (f.index + 1 < len(files) and files[f.index + 1].joined_prev):
        end = min(f.info.duration, end)
    return start, end


def _from_windows(session: Session, f: SessionFile, ctx: tuple,
                  windows: list[vlm.Window], graded: dict[str, dict[str, Any]]) -> list[Moment]:
    tl = ctx[0]
    rallies = _rallies(windows)
    cheers = [w.peak_time for w in windows
              if w.source == "vlm" and w.tag.event_type == "celebration"]
    out = []
    for w in windows:
        tag = w.tag
        if w.source != "vlm" or tag.event_type == "downtime":
            continue
        lt = w.peak_time
        mid = f"{f.game.video.stem}@{lt:.0f}"
        event = graded.get(mid, {}).get("event_type", tag.event_type)
        if event == "downtime":
            continue
        own = ((w.start + tag.rally_start_s, w.start + tag.rally_end_s)
               if w.rally_s > 0 else None)
        rally = _rally_at(rallies, own, lt) or own
        start, end = _clip_bounds(session, f, tl, lt, event, rally, rallies, cheers)
        reaction, laugh = _reaction_at(session, tl, lt)
        m = Moment(id=mid, file=f.index, t=f.offset + lt,
                   start=start, end=end, event_type=event,
                   description=tag.description, players=list(tag.players),
                   excitement=tag.excitement, funny=tag.funny,
                   attack=tag.attack, rally_s=rally[1] - rally[0] if rally else 0.0,
                   rally=(f.offset + rally[0], f.offset + rally[1]) if rally else None,
                   reaction=reaction, laugh=laugh, source="vlm",
                   play_active=tag.play_active)
        _score(session, m)
        _features(session, ctx, m, lt)
        out.append(m)
    return out


def _from_peaks(session: Session, f: SessionFile, ctx: tuple, peaks: list[audio.Peak],
                tagged: list[Moment], graded: dict[str, dict[str, Any]]) -> list[Moment]:
    """Strong reactions with no tagged moment nearby (often off-court fun)."""
    fc = session.cfg.fuse
    tl = ctx[0]
    out = []
    for p in peaks:
        t = f.offset + p.time
        if p.score < fc.audio_min_reaction:
            continue
        if any(abs(m.t - t) < fc.merge_within_s for m in tagged):
            continue
        start, end = _clip_bounds(session, f, tl, p.time, "reaction")
        reaction, laugh = _reaction_at(session, tl, p.time)
        cheer = _window_max(tl["cheer"], p.time - 2, p.time + 8)
        shout = _window_max(tl["shout"], p.time - 2, p.time + 8)
        what = {"laugh": "laughter", "cheer": "cheering", "shout": "shouting"}.get(
            p.dominant, "a loud reaction")
        desc = f"The group reacts with {what}." + (f' "{p.transcript}"' if p.transcript else "")
        mid = f"{f.game.video.stem}@{p.time:.0f}"
        event = graded.get(mid, {}).get("event_type", "reaction")
        if event == "downtime":
            continue
        m = Moment(id=mid, file=f.index, t=t,
                   start=start, end=end, event_type=event, description=desc,
                   excitement=round(10 * min(1.0, max(cheer, shout))),
                   funny=round(10 * min(1.0, laugh)), reaction=reaction, laugh=laugh,
                   source="audio")
        _score(session, m)
        _features(session, ctx, m, p.time)
        out.append(m)
    return out


def _add_comments(session: Session, moments: list[Moment],
                  said: "dict[int, list[comments.Comment]]", ctxs: dict[int, tuple],
                  graded: dict[str, dict[str, Any]]) -> list[Moment]:
    """Each comment unhinged enough for the reel widens the clip of the moment
    it was said in, so the comment and the reaction to it are both in. One
    said away from every moment becomes a moment of its own; comments in a
    row share one. Every rated comment goes in comments.csv."""
    cc, fc = session.cfg.comments, session.cfg.fuse
    new: list[Moment] = []
    rows = []
    for f in session.files:
        tl = ctxs[f.index][0]
        for c in said.get(f.index, []):
            rows.append((session.clock(f.offset + c.t), f.name, fmt_time(c.t), c))
            if c.unhinged < cc.min_unhinged:
                continue
            a, b = c.t - cc.pre_s, max(c.end, c.react_end) + cc.post_s
            r = tl["reaction"]
            limit = b + fc.max_extend_s
            while b < limit and int(b) < len(r) and r[int(b)] >= fc.extend_while_reaction:
                b += 1.0
            b = min(b, a + cc.max_clip_s, limit)
            a, b = _in_file(session, f, a, b)
            if b - a < 2:
                continue
            a, b = f.offset + a, f.offset + b
            gap = session.cfg.reel.merge_gap_s
            near = [m for m in moments + new if m.file == f.index
                    and m.start < b + gap and m.end > a - gap
                    and not (m.source == "comment" and max(m.end, b) - min(m.start, a) > 60)]
            m = max(near, key=lambda m: min(m.end, b) - max(m.start, a), default=None)
            desc = f'Said: "{c.quote}"' + (f' Reaction: "{c.reaction}"' if c.reaction else "")
            if m is None:
                mid = f"{f.game.video.stem}@{c.t:.0f}+said"
                event = graded.get(mid, {}).get("event_type", "conversation")
                if event == "downtime":
                    continue
                reaction, laugh = _reaction_at(session, tl, c.t)
                m = Moment(id=mid, file=f.index, t=f.offset + c.t, start=a, end=b,
                           event_type=event, description=desc, reaction=reaction,
                           laugh=laugh, source="comment", unhinged=c.unhinged,
                           comment=c.quote)
                _score(session, m)
                _features(session, ctxs[f.index], m, c.t)
                new.append(m)
                continue
            m.start, m.end = min(m.start, a), max(m.end, b)
            if c.unhinged > m.unhinged:
                m.unhinged, m.comment = c.unhinged, c.quote
            if f'"{c.quote}"' not in m.description:    # the same line said twice
                m.description = (f"{m.description} {desc}" if m.source != "comment"
                                 else f"{m.description} / {desc[6:]}")
            _score(session, m)
            m.features.update(unhinged=m.unhinged / 10, clip_s=m.length)
    comments.write_csv(session.work / "comments.csv", rows)
    hot = sum(c.unhinged >= cc.min_unhinged for _, _, _, c in rows)
    if rows:
        log(f"[fuse] {hot} unhinged comment{'s' if hot != 1 else ''} "
            f"({cc.min_unhinged:g}+) of {len(rows)} rated -> {session.work / 'comments.csv'}")
    return new


def _same_rally(a: Moment, b: Moment) -> bool:
    return (a.file == b.file and a.rally is not None and b.rally is not None
            and a.rally[0] <= b.rally[1] + 1.0 and b.rally[0] <= a.rally[1] + 1.0)


def _other_rally(a: Moment, b: Moment) -> bool:
    return a.rally is not None and b.rally is not None and not _same_rally(a, b)


def _merge(session: Session, moments: list[Moment]) -> list[Moment]:
    """Overlapping windows often saw the same play: keep the best of each
    cluster, adding the players the others named. A rally longer than one
    window shows up in two: those become one moment covering all of it."""
    fc = session.cfg.fuse
    kept: list[Moment] = []
    for m in sorted(moments, key=lambda m: m.score, reverse=True):
        twin = next((k for k in kept if (abs(k.t - m.t) < fc.merge_within_s
                                         and not _other_rally(k, m))
                     or _same_rally(k, m)), None)
        if twin is None:
            kept.append(m)
            continue
        twin.players += [p for p in m.players if p not in twin.players]
        if _same_rally(twin, m):
            a, b = twin.rally, m.rally
            assert a is not None and b is not None
            twin.rally = (min(a[0], b[0]), max(a[1], b[1]))
            twin.rally_s = twin.rally[1] - twin.rally[0]
            twin.attack = max(twin.attack, m.attack)
            twin.start, twin.end = min(twin.start, m.start), max(twin.end, m.end)
            if twin.length > fc.max_rally_clip_s:
                twin.start = twin.end - fc.max_rally_clip_s
            _score(session, twin)
            twin.features.update(rally=rally_share(session, twin.rally_s),
                                 attack=twin.attack / 10, clip_s=twin.length)
    return kept


# ------------------------------------------------------------------ output

def _write(session: Session, moments: list[Moment]) -> None:
    path = session.work / "moments.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "session_time", "clock", "file", "file_time", "event_type",
                    "score", "config_score", "ranker", "hype", "fun", "excitement", "funny",
                    "rally_s", "attack", "unhinged", "reaction", "laugh", "clip_s", "players",
                    "source", "graded_out", "clip_that", "comment", "description"])
        for m in moments:
            f = session.files[m.file]
            w.writerow([m.id, fmt_time(m.t), session.clock(m.t), f.name,
                        fmt_time(m.t - f.offset), m.event_type, f"{m.score:.3f}",
                        f"{m.base_score:.3f}",
                        "" if m.p_highlight is None else f"{m.p_highlight:.2f}",
                        f"{m.hype:.2f}", f"{m.fun:.2f}", m.excitement, m.funny,
                        f"{m.rally_s:.0f}", m.attack, f"{m.unhinged:g}",
                        f"{m.reaction:.2f}", f"{m.laugh:.2f}", f"{m.length:.1f}",
                        "; ".join(m.players), m.source, "y" if m.rejected else "",
                        m.clip_that, m.comment, m.description])
