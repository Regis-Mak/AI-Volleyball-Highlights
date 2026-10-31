"""Auto-tuned settings: the scoring weights, each label's clip length and the
reel's cutoff, re-picked from your labels by replaying past games.

  scoring      fuse.weights, fuse.fun_mix, fuse.label_bonus, fuse.audio_only_factor
  clip length  fuse.clip (pre_s / post_s per label), fuse.max_clip_s
  reel cutoff  reel.min_score

Every moment in the library keeps the numbers its score was made from
(Qwen's excitement and funny, laughter, the audio reaction, its label), so
its score can be recomputed for any settings in a blink, with no model
re-run. The tuner tries many settings and keeps the ones that best put the
moments you said were worth showing above the ones you said weren't (and
the same for funny). Clip lengths come from your trims, replayed against
each game's cached audio: if you keep starting bloopers 2 s earlier,
bloopers start 2 s earlier. The reel's cutoff goes just under 9 in 10 of the
moments you said were worth showing, so it never keeps those out.

A new version is used only if it beats the current settings: on the test set
once it has learning.test_set.min_to_check moments, before that on your other
labels (five-way cross-check, so no label checks settings tuned on it).
Every version is kept in library/models/settings_vN.json.

config.yaml stays exactly as you wrote it. Tuned values are applied on top at
the start of each run, except where you've edited config.yaml since that
version was made: your edit wins.

    python make_highlights.py --tune             tune now and list versions
    python make_highlights.py --use-settings v2  switch version ("off" = config.yaml only)
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..common import log
from . import ranker as ranker_mod
from .labels import targets

if TYPE_CHECKING:
    from ..config import Config
    from ..library import Library

KIND = "settings"
MARGIN = 0.005          # a new version must beat the current settings by this much
CLIP_GAIN_S = 0.25      # ...and a clip length must fit your trims this much better
CUTOFF_KEEP = 0.9       # the reel cutoff lets this share of your "worth showing" moments in
SCORING_KEYS = ("fuse.weights", "fuse.fun_mix", "fuse.label_bonus", "fuse.audio_only_factor")
CLIP_KEYS = ("fuse.clip", "fuse.max_clip_s")
KEYS = SCORING_KEYS + CLIP_KEYS + ("reel.min_score",)
NAMES = {"fuse.weights": "scoring weights", "fuse.fun_mix": "what counts as funny",
         "fuse.label_bonus": "label bonuses", "fuse.audio_only_factor": "audio-only moments",
         "fuse.clip": "clip lengths", "fuse.max_clip_s": "longest clip",
         "reel.min_score": "reel cutoff"}


# --------------------------------------------------------------- settings

def _get(cfg: "Config", key: str) -> Any:
    obj: Any = cfg
    for part in key.split("."):
        obj = getattr(obj, part)
    return _plain(obj)


def _plain(v: Any) -> Any:
    if hasattr(v, "model_dump"):
        return v.model_dump()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, float):
        return round(v, 4)
    return v


def _same(a: Any, b: Any) -> bool:
    return json.dumps(_plain(a), sort_keys=True) == json.dumps(_plain(b), sort_keys=True)


def _set(cfg: "Config", key: str, value: Any) -> None:
    from ..config import ClipShape
    section, name = key.split(".")
    obj = getattr(cfg, section)
    if key == "fuse.clip":
        value = {k: ClipShape(**v) for k, v in value.items()}
    setattr(obj, name, value)


def _load_version(lib: "Library", row) -> dict | None:
    path = Path(row["path"])
    if not path.exists():
        path = lib.root / "models" / path.name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def effective(cfg: "Config", lib: "Library", quiet: bool = True) -> "Config":
    """config.yaml with the active tuned settings on top (your later edits win)."""
    if not cfg.learning.tune.enabled:
        return cfg
    row = lib.active_version(KIND)
    data = _load_version(lib, row) if row is not None else None
    if not data:
        return cfg
    out = cfg.model_copy(deep=True)
    used, yours = [], []
    for key, value in data.get("settings", {}).items():
        if key not in KEYS:
            continue
        if not _same(_get(cfg, key), data.get("from", {}).get(key)):
            yours.append(NAMES[key])          # you've edited it in config.yaml since
            continue
        _set(out, key, value)
        used.append(NAMES[key])
    if not quiet and (used or yours):
        msg = f"[tune] using tuned settings {row['name']}: {', '.join(used) or 'none'}"
        if yours:
            msg += f" (kept your config.yaml {', '.join(yours)})"
        log(msg + "; --use-settings off to use config.yaml only")
    return out


def apply(cfg: "Config", lib: "Library | None") -> "Config":
    """What a run uses. Logs which tuned settings are on."""
    return effective(cfg, lib, quiet=False) if lib is not None else cfg


# ------------------------------------------------------------------- data

WEIGHT_KEYS = ("excitement", "funny", "reaction", "rally", "attack")
SEARCHED = ("excitement", "funny", "reaction")   # rally and attack stay as you set them


@dataclass
class Scoring:
    weights: dict[str, float]
    fun_mix: float
    label_bonus: dict[str, float]
    audio_only_factor: float

    @classmethod
    def of(cls, cfg: "Config") -> "Scoring":
        fc = cfg.fuse
        return cls({k: float(fc.weights.get(k, 0.0)) for k in WEIGHT_KEYS},
                   float(fc.fun_mix), {k: float(v) for k, v in fc.label_bonus.items()},
                   float(fc.audio_only_factor))


@dataclass
class Data:
    ids: list[str]
    rows: list[Any]                 # library moment rows
    feats: list[dict[str, float]]
    exc: np.ndarray
    funny: np.ndarray
    reaction: np.ndarray
    laugh: np.ndarray
    rally: np.ndarray
    attack: np.ndarray
    audio: np.ndarray
    events: list[str]
    event_idx: np.ndarray
    y: dict[str, np.ndarray] = field(default_factory=dict)   # h, f, gh, gf (nan = none)


def _load(lib: "Library") -> Data:
    moments = {r["id"]: r for r in lib.moments()}
    use = {}
    for kind, short in (("highlight", "h"), ("funny", "f")):
        train, test = targets(lib, kind)
        use[short] = [t for t in train if t.tier != "agree" and t.moment in moments]
        use["g" + short] = [t for t in test if t.moment in moments]
    ids = sorted({t.moment for rows in use.values() for t in rows})
    index = {m: i for i, m in enumerate(ids)}
    rows = [moments[m] for m in ids]
    feats = [json.loads(r["features"] or "{}") for r in rows]
    col = lambda k: np.array([f.get(k, 0.0) for f in feats], dtype=float)  # noqa: E731
    events = sorted({r["event_type"] for r in rows})
    d = Data(ids, rows, feats, col("excitement"), col("funny"), col("reaction"),
             np.minimum(1.0, col("laugh")), col("rally"), col("attack"),
             col("audio_only") > 0.5, events,
             np.array([events.index(r["event_type"]) for r in rows], dtype=int))
    for key, targets_ in use.items():
        y = np.full(len(ids), np.nan)
        for t in targets_:
            y[index[t.moment]] = t.y
        d.y[key] = y
    return d


def _score(d: Data, s: Scoring) -> tuple[np.ndarray, np.ndarray]:
    """Recompute fuse's score and fun for these settings."""
    fun = s.fun_mix * d.funny + (1 - s.fun_mix) * d.laugh
    bonus = np.array([s.label_bonus.get(e, 0.0) for e in d.events])[d.event_idx] \
        if d.events else np.zeros(len(d.ids))
    w = s.weights
    score = (w.get("excitement", 0) * d.exc + w.get("funny", 0) * fun
             + w.get("reaction", 0) * np.minimum(1.0, d.reaction)
             + w.get("rally", 0) * d.rally + w.get("attack", 0) * d.attack + bonus)
    return np.where(d.audio, score * s.audio_only_factor, score), fun


def _auc(s: np.ndarray, y: np.ndarray) -> float | None:
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 < 3 or n0 < 3:
        return None
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = (first + (counts + 1) / 2)[inv]          # average rank over ties
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _objective(d: Data, s: Scoring, mh: np.ndarray, mf: np.ndarray,
               yh: str = "h", yf: str = "f") -> float | None:
    score, fun = _score(d, s)
    vals = [_auc(score[mh], d.y[yh][mh]), _auc(fun[mf], d.y[yf][mf])]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- scoring

def _search(d: Data, s0: Scoring, mh: np.ndarray, mf: np.ndarray) -> Scoring:
    """Coordinate search from the current settings; ties keep the current value."""
    best = _objective(d, s0, mh, mf)
    if best is None:
        return s0
    s = s0
    total = sum(s0.weights.get(k, 0.0) for k in SEARCHED) or 1.0
    step = 0.05
    simplex = [(a * step * total, b * step * total, max(0.0, total - (a + b) * step * total))
               for a in range(21) for b in range(21 - a)]
    labelled = {d.events[i] for i in d.event_idx[mh | mf]}
    has_audio = bool(d.audio[mh | mf].any())

    def better(cand: Scoring) -> bool:
        nonlocal best, s
        v = _objective(d, cand, mh, mf)
        if v is not None and v > best + 1e-6:
            best, s = v, cand
            return True
        return False

    for _ in range(3):
        start = s
        for mix in np.arange(0.2, 0.91, 0.05):
            better(replace(s, fun_mix=round(float(mix), 2)))
        for e, f, r in simplex:
            better(replace(s, weights={**s.weights, "excitement": round(e, 3),
                                       "funny": round(f, 3), "reaction": round(r, 3)}))
        for label in sorted(labelled):
            for b in np.arange(0.0, 0.301, 0.025):
                better(replace(s, label_bonus={**s.label_bonus, label: round(float(b), 3)}))
        if has_audio:
            for a in np.arange(0.5, 1.201, 0.05):
                better(replace(s, audio_only_factor=round(float(a), 2)))
        if s == start:
            break
    return s


def _folds(d: Data, k: int = 5) -> np.ndarray:
    """A fold per labelled moment, yes and no answers spread evenly."""
    rng = np.random.default_rng(7)
    y = np.where(np.isnan(d.y["h"]), d.y["f"], d.y["h"])
    order = rng.permutation(len(d.ids))
    order = np.concatenate([order[y[order] == 1], order[y[order] != 1]])
    fold = np.empty(len(d.ids), dtype=int)
    fold[order] = np.arange(len(order)) % k
    return fold


def _cross_check(d: Data, s0: Scoring, k: int = 5) -> tuple[float | None, float | None]:
    """(tuned, current) on your labels, each moment scored by settings that
    were tuned without it."""
    mh, mf = ~np.isnan(d.y["h"]), ~np.isnan(d.y["f"])
    fold = _folds(d, k)
    held_s, held_f = np.full(len(d.ids), np.nan), np.full(len(d.ids), np.nan)
    for i in range(k):
        test = fold == i
        s = _search(d, s0, mh & ~test, mf & ~test)
        sc, fn = _score(d, s)
        held_s[test], held_f[test] = sc[test], fn[test]
    vals = [_auc(held_s[mh], d.y["h"][mh]), _auc(held_f[mf], d.y["f"][mf])]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)) if vals else None), _objective(d, s0, mh, mf)


# ------------------------------------------------------------ clip lengths

class _Timelines:
    def __init__(self, work_dir: Path):
        self.work_dir, self.cache = work_dir, {}

    def reaction(self, stem: str) -> np.ndarray | None:
        if stem not in self.cache:
            path = self.work_dir / stem / "timeline.csv"
            try:
                with open(path, newline="", encoding="utf-8") as fh:
                    self.cache[stem] = np.array([float(r["reaction"])
                                                 for r in csv.DictReader(fh)])
            except (OSError, KeyError, ValueError):
                self.cache[stem] = None
        return self.cache[stem]


def _end_after(cfg: "Config", r: np.ndarray | None, lt: float, post: float) -> float:
    """Where fuse ends a clip (it runs on while the group still reacts)."""
    fc = cfg.fuse
    end = lt + post
    if r is None:
        return end
    limit = end + fc.max_extend_s
    while end < limit and int(end) < len(r) and r[int(end)] >= fc.extend_while_reaction:
        end += 1.0
    return min(end, limit)


def _tune_clips(lib: "Library", cfg: "Config") -> tuple[dict[str, Any], list[str]]:
    """New fuse.clip / fuse.max_clip_s values from your trims, and notes."""
    tc, fc = cfg.learning.tune, cfg.fuse
    gold = {r["moment"] for r in lib.q("SELECT DISTINCT moment FROM labels WHERE gold = 1")}
    rows = lib.q("SELECT l.moment, l.value, l.source, m.stem, m.file_time, m.event_type "
                 "FROM labels l JOIN moments m ON m.id = l.moment WHERE l.kind = 'bounds' "
                 "AND l.source IN ('hand', 'trim')")
    best: dict[str, Any] = {}
    for r in rows:                              # a Grade-screen trim wins over picks.csv
        if r["moment"] in gold:
            continue
        if r["moment"] not in best or r["source"] == "hand":
            best[r["moment"]] = r
    by_label: dict[str, list] = {}
    for r in best.values():
        b0, b1 = json.loads(r["value"])
        if b1 > b0:
            by_label.setdefault(r["event_type"], []).append((r["stem"], r["file_time"], b0, b1))
    tl = _Timelines(cfg.work_dir)
    clip = {k: v.model_dump() for k, v in fc.clip.items()}
    notes, changed = [], False
    for label, trims in sorted(by_label.items()):
        if len(trims) < tc.min_trims or label not in clip:
            continue
        cur_pre, cur_post = clip[label]["pre_s"], clip[label]["post_s"]
        pre = float(np.clip(round(2 * np.median([t - b0 for _, t, b0, _ in trims])) / 2, 1, 20))

        def end_err(post: float) -> float:
            return float(np.median([abs(_end_after(cfg, tl.reaction(st), t, post) - b1)
                                    for st, t, _, b1 in trims]))
        post = min(np.arange(1.0, 15.01, 0.5), key=lambda p: (end_err(p), abs(p - cur_post)))
        post = float(post)
        start_err = lambda p: float(np.median([abs(t - p - b0) for _, t, b0, _ in trims]))  # noqa: E731
        before = start_err(cur_pre) + end_err(cur_post)
        after = start_err(pre) + end_err(post)
        if after < before - CLIP_GAIN_S and (pre, post) != (cur_pre, cur_post):
            clip[label] = {"pre_s": pre, "post_s": post}
            changed = True
            notes.append(f"{label} clips {cur_pre:g}+{cur_post:g} s -> {pre:g}+{post:g} s "
                         f"(from {len(trims)} trims)")
    out: dict[str, Any] = {}
    if changed:
        out["fuse.clip"] = clip
    lengths = [b1 - b0 for trims in by_label.values() for _, _, b0, b1 in trims]
    if len(lengths) >= tc.min_trims:
        want = float(np.ceil(np.percentile(lengths, 75)))
        if want > fc.max_clip_s + 1:
            out["fuse.max_clip_s"] = min(45.0, want)
            notes.append(f"longest clip {fc.max_clip_s:g} -> {min(45.0, want):g} s")
    return out, notes


# ------------------------------------------------------------- reel cutoff

def _tune_cutoff(lib: "Library", d: Data, s: Scoring, current: float) -> float | None:
    """reel.min_score just under 9 in 10 of your 'worth showing' moments, scored
    the way the next run will score them (tuned weights, then the ranker)."""
    mh = ~np.isnan(d.y["h"])
    y = d.y["h"][mh]
    if (y == 1).sum() < 10 or (y == 0).sum() < 5:
        return None
    score, fun = _score(d, s)
    model = ranker_mod.load_active(lib)
    if model is not None:
        score, _, _ = model.blend([d.feats[i] for i in np.where(mh)[0]], score[mh], fun[mh])
    else:
        score = score[mh]
    thr = float(np.floor(100 * np.percentile(score[y == 1], 100 * (1 - CUTOFF_KEEP))) / 100)
    thr = min(0.6, max(0.1, thr))
    return thr if abs(thr - current) >= 0.02 else None


# ----------------------------------------------------------------- retune

def _fingerprint(lib: "Library", cfg: "Config") -> str:
    h = hashlib.sha1()
    for r in lib.q("SELECT moment, kind, value, source, gold FROM labels WHERE source != "
                   "'agree' ORDER BY moment, kind, source"):
        h.update(json.dumps(list(r)).encode())
    h.update(json.dumps({k: _get(cfg, k) for k in KEYS}, sort_keys=True).encode())
    h.update(json.dumps(cfg.learning.tune.model_dump(), sort_keys=True).encode())
    active = lib.active_version("ranker")
    h.update((active["name"] if active else "-").encode())
    h.update(str(lib.q("SELECT COUNT(*) AS n FROM moments")[0]["n"]).encode())
    return h.hexdigest()


def retune(lib: "Library", force: bool = False) -> str:
    """Tune from the whole library and keep the result only if it beats the
    current settings. Returns a one-line summary."""
    raw = lib.cfg
    tc = raw.learning.tune
    if not tc.enabled:
        return "auto-tune is switched off (learning.tune.enabled)"
    fp = _fingerprint(lib, raw)
    last = lib.q("SELECT value FROM meta WHERE key = 'tune_fingerprint'")
    if not force and last and last[0]["value"] == fp:
        active = lib.active_version(KIND)
        return ("no new labels since the last tune; "
                + (f"settings {active['name']} stay on" if active else
                   "config.yaml stays as written"))
    result = _retune(lib)
    with lib.db:
        lib.db.execute("INSERT OR REPLACE INTO meta VALUES ('tune_fingerprint', ?)", (fp,))
    return result


def _retune(lib: "Library") -> str:
    raw = lib.cfg
    tc = raw.learning.tune
    cur_cfg = effective(raw, lib)
    d = _load(lib)
    n = int((~np.isnan(d.y["h"])).sum())
    s0 = Scoring.of(cur_cfg)
    new: dict[str, Any] = {}
    notes: list[str] = []
    metric = baseline = None
    checked = ""
    if n >= tc.min_labels:
        mh, mf = ~np.isnan(d.y["h"]), ~np.isnan(d.y["f"])
        tuned = _search(d, s0, mh, mf)
        gh, gf = ~np.isnan(d.y["gh"]), ~np.isnan(d.y["gf"])
        if int(gh.sum()) >= raw.learning.test_set.min_to_check:
            metric = _objective(d, tuned, gh, gf, "gh", "gf")
            baseline = _objective(d, s0, gh, gf, "gh", "gf")
            checked = f"test set ({int(gh.sum())} moments)"
        else:
            metric, baseline = _cross_check(d, s0)
            checked = f"cross-check on your labels ({n} moments)"
        if tuned != s0 and metric is not None and baseline is not None \
                and metric > baseline + MARGIN:
            for key, value in (("fuse.weights", tuned.weights), ("fuse.fun_mix", tuned.fun_mix),
                               ("fuse.label_bonus", tuned.label_bonus),
                               ("fuse.audio_only_factor", tuned.audio_only_factor)):
                if not _same(value, _get(cur_cfg, key)):
                    new[key] = _plain(value)
            notes.append(f"scoring {baseline:.3f} -> {metric:.3f} on the {checked}")
            s0 = tuned
    clips, clip_notes = _tune_clips(lib, cur_cfg)
    new.update(clips)
    notes += clip_notes
    if n >= tc.min_labels:
        cutoff = _tune_cutoff(lib, d, s0, cur_cfg.reel.min_score)
        if cutoff is not None:
            new["reel.min_score"] = cutoff
            notes.append(f"reel cutoff {cur_cfg.reel.min_score:.2f} -> {cutoff:.2f}")

    if not new:
        if n < tc.min_labels:
            return (f"{n} of {tc.min_labels} labelled moments so far; the scoring is tuned "
                    f"from then on (clip lengths after {tc.min_trims} trims of a label)")
        return "tried new settings; none beat the current ones"

    # A version holds every tuned value, so switching versions is all-or-nothing.
    settings = {k: _get(cur_cfg, k) for k in KEYS if not _same(_get(cur_cfg, k), _get(raw, k))}
    settings.update(new)
    settings = {k: v for k, v in settings.items() if not _same(v, _get(raw, k))}
    name = lib.next_version_name(KIND)
    path = lib.root / "models" / f"settings_{name}.json"
    path.write_text(json.dumps({
        "name": name, "settings": settings, "from": {k: _get(raw, k) for k in settings},
        "notes": notes, "metric": metric, "baseline": baseline, "n_labels": n,
    }, indent=1), encoding="utf-8")
    lib.add_version(KIND, name, path, n, metric, baseline, checked or "your trims", True,
                    "; ".join(notes))
    return f"settings {name} are on from the next run: " + "; ".join(notes)


def describe_versions(lib: "Library") -> str:
    rows = lib.versions(KIND)
    if not rows:
        return "  none yet"
    lines = []
    for r in rows:
        mark = "*" if r["active"] else " "
        lines.append(f"  {mark} {r['name']:<4} {r['created'][:16].replace('T', ' ')}  "
                     f"{r['n_labels']:>4} labels  {r['note']}")
    return "\n".join(lines) + "\n  (* = on)"
