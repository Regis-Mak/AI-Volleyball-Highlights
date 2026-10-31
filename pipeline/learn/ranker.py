"""The ranker: two small models trained on the library's labels.

  highlight   is this moment worth a spot in the reel?
  funny       is it funny?

Both read the same numbers (pipeline/learn/features.py): Qwen's scores and
label, the audio reaction, laughter, cheering, movement, clip length. They
start as logistic regression (a learned version of the hand-set weights in
config.yaml) and switch to LightGBM once there are enough labels.

The learned score takes over from the config weights gradually: with few
labels it only nudges the hand-set score, and at learning.ranker.blend_full_at
labels it replaces it completely.

After every run the ranker retrains on the whole library (seconds, on the CPU).
A new version is used only if it beats the current scoring on the test set
(moments you graded with --grade test). Until the test set is big enough, it
has to beat the hand-set weights on your other labels instead (cross-checked,
so it is never scored on labels it trained on). Every version is kept in
library/models/; --use-ranker switches between them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..common import log
from . import features as feat
from .labels import Target, targets

if TYPE_CHECKING:
    from ..config import Config
    from ..fuse import Moment
    from ..library import Library

KINDS = ("highlight", "funny")
MARGIN = 0.002          # a new version must beat the current one by this much (AUC)


# ------------------------------------------------------------------- models

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _balanced(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weights that give both classes equal say (most moments aren't highlights)."""
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    if pos <= 0 or neg <= 0:
        return w
    total = pos + neg
    return w * np.where(y == 1, total / (2 * pos), total / (2 * neg))


def _fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1.0) -> dict:
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mean) / sd]
    cw = _balanced(y, w)
    beta = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[0] = 0.0
    for _ in range(100):
        p = _sigmoid(Z @ beta)
        grad = Z.T @ (cw * (p - y)) + reg * beta
        hess = (Z.T * (cw * p * (1 - p))) @ Z + np.diag(reg) + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return {"type": "logistic", "mean": mean.tolist(), "sd": sd.tolist(),
            "beta": beta.tolist()}


def _fit_lightgbm(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    clf.fit(X, y, sample_weight=_balanced(y, w))
    return {"type": "lightgbm", "model": clf.booster_.model_to_string()}


def _lightgbm_ok() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


class Fitted:
    def __init__(self, spec: dict):
        self.spec = spec
        self._booster = None
        if spec["type"] == "lightgbm":
            import lightgbm as lgb
            self._booster = lgb.Booster(model_str=spec["model"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._booster is not None:
            return np.asarray(self._booster.predict(X), dtype=float)
        s = self.spec
        Z = (X - np.array(s["mean"])) / np.array(s["sd"])
        beta = np.array(s["beta"])
        return _sigmoid(beta[0] + Z @ beta[1:])


@dataclass
class Ranker:
    name: str
    alpha: float                          # share of the score the ranker decides
    highlight: Fitted
    funny: Fitted | None
    n_labels: int
    features: list[str]
    trained_on: list[str]                 # moment ids, so tests stay fair

    def predict(self, rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.array([[r.get(k, 0.0) for k in self.features] for r in rows], dtype=float)
        if len(X) == 0:
            return np.zeros(0), None
        return self.highlight.predict(X), (self.funny.predict(X) if self.funny else None)

    def blend(self, rows: list[dict[str, float]], base_score: np.ndarray,
              base_fun: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_h, p_f = self.predict(rows)
        a = self.alpha
        score = (1 - a) * base_score + a * p_h
        fun = (1 - a) * base_fun + a * p_f if p_f is not None else base_fun
        return score, fun, p_h

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "name": self.name, "alpha": self.alpha, "n_labels": self.n_labels,
            "features": self.features, "highlight": self.highlight.spec,
            "funny": self.funny.spec if self.funny else None,
            "trained_on": self.trained_on}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Ranker":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["name"], d["alpha"], Fitted(d["highlight"]),
                   Fitted(d["funny"]) if d.get("funny") else None, d["n_labels"],
                   d["features"], d.get("trained_on", []))


def load_active(lib: "Library") -> Ranker | None:
    if not lib.cfg.learning.ranker.enabled:
        return None
    row = lib.active_version("ranker")
    if row is None:
        return None
    path = Path(row["path"])
    if not path.exists():
        path = lib.root / "models" / path.name
    try:
        return Ranker.load(path)
    except (OSError, ValueError, KeyError, ImportError) as err:
        log(f"[learn] warning: couldn't load ranker {row['name']} ({err}); "
            "using the config weights")
        return None


def apply(lib: "Library", moments: list["Moment"]) -> Ranker | None:
    """Re-score moments with the active ranker (if there is one)."""
    ranker = load_active(lib)
    if ranker is None or not moments:
        return None
    score, fun, p_h = ranker.blend([m.features for m in moments],
                                   np.array([m.base_score for m in moments]),
                                   np.array([m.base_fun for m in moments]))
    for m, s, f, p in zip(moments, score, fun, p_h):
        m.score, m.fun, m.p_highlight = float(s), float(f), float(p)
    log(f"[learn] scoring with ranker {ranker.name} "
        f"(learned from {ranker.n_labels} labels, {ranker.alpha:.0%} of the score)")
    return ranker


# ------------------------------------------------------------------ metrics

def auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Chance that a random 'yes' moment outranks a random 'no' (0.5 = guessing)."""
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) < 3 or len(neg) < 3:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):                    # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _mean(vals: list[float | None]) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------- training

@dataclass
class Data:
    kind: str
    ids: list[str]
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    tier: list[str]
    base: np.ndarray        # hand-set score, for comparison


def _dataset(lib: "Library", kind: str, rows: list[Target],
             moments: dict[str, Any]) -> Data:
    lc = lib.cfg.learning
    rows = [t for t in rows if t.moment in moments]
    w = np.array([t.weight for t in rows], dtype=float)
    tier = [t.tier for t in rows]
    self_made = np.array([t == "agree" for t in tier], dtype=bool)
    # Tool-made labels never outweigh yours (learning.self_made_cap).
    own, auto = w[~self_made].sum(), w[self_made].sum()
    cap = lc.self_made_cap / max(1e-6, 1 - lc.self_made_cap) * own
    if auto > cap:
        w[self_made] *= cap / auto if auto > 0 else 0.0
    # No single session dominates: cap each at twice the typical session.
    sessions = np.array([t.session for t in rows])
    if len(rows):
        totals = {s: w[sessions == s].sum() for s in set(sessions)}
        limit = 2 * float(np.median(list(totals.values())))
        for s, tot in totals.items():
            if tot > limit > 0:
                w[sessions == s] *= limit / tot
    base_key = "base_score" if kind == "highlight" else "base_fun"
    return Data(kind, [t.moment for t in rows],
                np.array([feat.vector(json.loads(moments[t.moment]["features"] or "{}"))
                          for t in rows], dtype=float).reshape(len(rows), len(feat.FEATURES)),
                np.array([t.y for t in rows], dtype=float), w, tier,
                np.array([moments[t.moment][base_key] or 0.0 for t in rows], dtype=float))


def _trainable(d: Data, min_labels: int) -> bool:
    mine = [i for i, t in enumerate(d.tier) if t != "agree"]
    y = d.y[d.w > 0]
    return len(mine) >= min_labels and (y == 1).sum() >= 5 and (y == 0).sum() >= 5


def _fit(cfg: "Config", d: Data, idx: np.ndarray | None = None) -> Fitted:
    rc = cfg.learning.ranker
    X, y, w = (d.X, d.y, d.w) if idx is None else (d.X[idx], d.y[idx], d.w[idx])
    keep = w > 0
    X, y, w = X[keep], y[keep], w[keep]
    use_lgb = rc.model == "lightgbm" or (rc.model == "auto" and len(y) >= rc.lightgbm_from)
    if use_lgb and _lightgbm_ok():
        return Fitted(_fit_lightgbm(X, y, w))
    if use_lgb and rc.model == "lightgbm":
        log("[learn] warning: lightgbm isn't installed; using logistic regression")
    return Fitted(_fit_logistic(X, y, w))


def _alpha(cfg: "Config", d: Data) -> float:
    mine = sum(t != "agree" for t in d.tier)
    return min(1.0, mine / max(1, cfg.learning.ranker.blend_full_at))


def _cross_check(cfg: "Config", d: Data, folds: int = 5) -> tuple[float | None, float | None]:
    """Learned vs hand-set score on your labels, each moment scored by a model
    that never saw it. Returns (learned AUC, hand-set AUC)."""
    mine = np.array([t != "agree" for t in d.tier])
    idx = np.where(mine)[0]
    if len(idx) < 10:
        return None, None
    rng = np.random.default_rng(7)
    order = idx[np.argsort(rng.random(len(idx)))]
    # Stratify: alternate yes/no into the folds so each fold sees both.
    order = np.concatenate([order[d.y[order] == 1], order[d.y[order] == 0]])
    fold_of = {int(i): k % folds for k, i in enumerate(order)}
    held_scores = np.full(len(idx), np.nan)
    alpha = _alpha(cfg, d)
    for k in range(folds):
        test = np.array([i for i in idx if fold_of[int(i)] == k])
        if len(test) == 0:
            continue
        train = np.array([i for i in range(len(d.y)) if fold_of.get(i, -1) != k])
        sub = Data(d.kind, [], d.X[train], d.y[train], d.w[train],
                   [d.tier[i] for i in train], d.base[train])
        yt = sub.y[sub.w > 0]
        if (yt == 1).sum() < 2 or (yt == 0).sum() < 2:
            continue
        model = _fit(cfg, sub)
        p = model.predict(d.X[test])
        pos = np.searchsorted(idx, test)
        held_scores[pos] = (1 - alpha) * d.base[test] + alpha * p
    ok = ~np.isnan(held_scores)
    return auc(held_scores[ok], d.y[idx][ok]), auc(d.base[idx][ok], d.y[idx][ok])


def _fingerprint(datas: dict[str, Data], cfg: "Config") -> str:
    h = hashlib.sha1()
    h.update(json.dumps(cfg.learning.ranker.model_dump(), sort_keys=True).encode())
    for k, d in sorted(datas.items()):
        h.update(k.encode())
        h.update(json.dumps(d.ids).encode())
        for arr in (d.X, d.y, d.w):
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def retrain(lib: "Library", force: bool = False) -> str:
    """Train a new ranker version from the whole library and keep it only if it
    beats the current scoring. Returns a one-line summary."""
    cfg = lib.cfg
    rc, tc = cfg.learning.ranker, cfg.learning.test_set
    if not rc.enabled:
        return "the ranker is switched off (learning.ranker.enabled)"
    moments = {r["id"]: r for r in lib.moments()}
    datas: dict[str, Data] = {}
    golds: dict[str, list[Target]] = {}
    for kind in KINDS:
        train, gold = targets(lib, kind)
        datas[kind] = _dataset(lib, kind, train, moments)
        golds[kind] = [t for t in gold if t.moment in moments]

    hd = datas["highlight"]
    mine = sum(t != "agree" for t in hd.tier)
    if not _trainable(hd, rc.min_labels):
        return (f"{mine} of {rc.min_labels} labelled moments so far; the ranker starts "
                "learning at that point (grade a few with --grade, or veto clips in picks.csv)")

    fp = _fingerprint(datas, cfg)
    last = lib.q("SELECT value FROM meta WHERE key = 'ranker_fingerprint'")
    if not force and last and last[0]["value"] == fp:
        active = lib.active_version("ranker")
        return ("no new labels since the last training; "
                + (f"ranker {active['name']} stays active" if active else
                   "the config weights stay in charge"))

    name = lib.next_version_name("ranker")
    fd = datas["funny"]
    funny = _fit(cfg, fd) if _trainable(fd, max(10, rc.min_labels // 2)) else None
    trained = sorted({i for d in datas.values() for i, w in zip(d.ids, d.w) if w > 0})
    cand = Ranker(name, _alpha(cfg, hd), _fit(cfg, hd), funny, mine, list(feat.FEATURES),
                  trained)

    gold_h = golds["highlight"]
    current = load_active(lib)
    if len(gold_h) >= tc.min_to_check:
        metric, baseline, cur = _on_test_set(cand, current, golds, moments)
        checked = f"test set ({len(gold_h)} moments)"
        bar = cur if cur is not None else baseline
    else:
        metric, baseline = _cross_check(cfg, hd)
        checked = f"cross-check on your labels ({mine} moments)"
        bar = baseline
    better = metric is not None and (bar is None or metric > bar + MARGIN)

    path = lib.root / "models" / f"ranker_{name}.json"
    cand.save(path)
    note = "better than the current scoring" if better else "not better; kept for reference"
    lib.add_version("ranker", name, path, mine, metric, baseline, checked, better, note)
    with lib.db:
        lib.db.execute("INSERT OR REPLACE INTO meta VALUES ('ranker_fingerprint', ?)", (fp,))

    m = f"{metric:.3f}" if metric is not None else "n/a"
    b = f"{bar:.3f}" if bar is not None else "n/a"
    who = current.name if current else "the config weights"
    if better:
        return (f"ranker {name} is now active: scored {m} vs {b} for {who} on the {checked}; "
                "used from the next run")
    return f"ranker {name} scored {m} vs {b} for {who} on the {checked}; keeping {who}"


def _on_test_set(cand: Ranker, current: Ranker | None, golds: dict[str, list[Target]],
                 moments: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """(candidate, hand-set, current) AUC on the test set, averaged over
    'highlight' and 'funny'. Moments the current version trained on (labelled
    some other way before they joined the test set) are left out, so it gets
    no unfair head start."""
    res: dict[str, list[float | None]] = {"cand": [], "base": [], "cur": []}
    if current is not None:
        seen = set(current.trained_on)
        fair = {k: [t for t in rows if t.moment not in seen] for k, rows in golds.items()}
        if len(fair["highlight"]) >= len(golds["highlight"]) // 2:
            golds = fair
    for kind, rows in golds.items():
        if not rows:
            continue
        feats = [json.loads(moments[t.moment]["features"] or "{}") for t in rows]
        y = np.array([t.y for t in rows])
        bs = np.array([moments[t.moment]["base_score"] or 0.0 for t in rows])
        bf = np.array([moments[t.moment]["base_fun"] or 0.0 for t in rows])
        base = bs if kind == "highlight" else bf
        s, f, _ = cand.blend(feats, bs, bf)
        res["cand"].append(auc(s if kind == "highlight" else f, y))
        res["base"].append(auc(base, y))
        if current is not None:
            s, f, _ = current.blend(feats, bs, bf)
            res["cur"].append(auc(s if kind == "highlight" else f, y))
    return _mean(res["cand"]), _mean(res["base"]), _mean(res["cur"]) if current else None


def describe_versions(lib: "Library") -> str:
    rows = lib.versions("ranker")
    if not rows:
        return "  none yet"
    lines = []
    for r in rows:
        mark = "*" if r["active"] else " "
        metric = f"{r['metric']:.3f}" if r["metric"] is not None else "  n/a"
        base = f"{r['baseline']:.3f}" if r["baseline"] is not None else "n/a"
        lines.append(f"  {mark} {r['name']:<4} {r['created'][:16].replace('T', ' ')}  "
                     f"{r['n_labels']:>4} labels  score {metric} (config weights {base})  "
                     f"{r['checked_on']}")
    return "\n".join(lines) + "\n  (* = active)"
