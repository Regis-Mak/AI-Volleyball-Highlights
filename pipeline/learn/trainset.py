"""The training set for fine-tuning Qwen (Phase 7): every window behind a
moment you labelled, as the frames Qwen saw, the exact prompt it got, and the
answer it should have given.

The answer starts from what Qwen said and takes your corrections:

  event_type   the label you set on the Grade screen, or the one you saw and
               left as it was while grading
  excitement   at least 6 when you said worth showing, at most 3 when not
  funny        at least 6 when you marked it funny, at most 3 when not
  play_active  false only for downtime

Description, players and peak time stay as Qwen gave them (nobody has written
better ones). Only trusted labels count (your grades, cuts and trims; see
learning.finetune.min_trust), and each example is weighted by the most trusted
one behind it. The test set is never in it: no window that shows a test-set
moment is used, so a fine-tuned model can be checked fairly on the test set
afterwards (--evaluate).

    python make_highlights.py --export-training

library/training/<date>/
  train.jsonl     one example per line, in the chat format Unsloth's vision
                  fine-tuning reads: the frames first, then the prompt, then the
                  answer. Image paths are relative to this folder.
  val.jsonl       about 1 in 10 moments held back to watch the training
  test.jsonl      the test-set windows with your grades (for checking only)
  manifest.json   counts and the settings it was made with
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..common import PipelineError, log
from ..vlm import EVENT_TYPES, WindowTag, answer_schema
from .memory import verdict

if TYPE_CHECKING:
    from ..library import Library

SOURCES = ("hand", "cut", "trim", "kept")
WORTH_MIN, NOT_WORTH_MAX = 6, 3          # excitement for "worth showing: yes / no"
FUNNY_MIN, NOT_FUNNY_MAX = 6, 3          # funny for "funny: yes / no"


@dataclass
class Example:
    window: str                    # library window id, "<file>@<start>"
    moment: str
    session: str
    images: list[Path]
    prompt: str
    answer: dict                   # what Qwen said
    target: dict                   # what it should have said
    weight: float
    sources: list[str] = field(default_factory=list)
    truth: dict = field(default_factory=dict)       # test set: your grades


def _labels(lib: "Library") -> dict[str, dict[str, list]]:
    marks = ",".join("?" * len(SOURCES))
    out: dict[str, dict[str, list]] = {}
    for r in lib.q(f"SELECT moment, kind, value, source, trust, gold FROM labels WHERE source "
                   f"IN ({marks}) AND kind IN ('highlight', 'funny', 'event_type')", SOURCES):
        out.setdefault(r["moment"], {}).setdefault(r["kind"], []).append(r)
    return out


def _answer(w: Any) -> dict:
    extra = json.loads(w["extra"] or "{}")
    return WindowTag(**extra, play_active=bool(w["play_active"]), event_type=w["event_type"],
                     description=w["description"] or "",
                     players=json.loads(w["players"] or "[]"),
                     excitement=int(w["excitement"]), funny=int(w["funny"]),
                     peak_offset_s=round(float(w["peak_offset"]), 1)).model_dump()


def judged(lab: dict[str, list], moment_label: str, min_trust: float,
           gold: bool) -> tuple[str | None, int | None, int | None, float, list[str]]:
    """(label, worth showing, funny, weight, sources) from one moment's labels.
    Test-set moments use only their test-set grades; the rest never use those."""
    lab = {k: [r for r in rows if bool(r["gold"]) == gold] for k, rows in lab.items()}
    hand_label = [r for r in lab.get("event_type", []) if r["source"] == "hand"]
    graded = any(r["source"] == "hand" for r in lab.get("highlight", []))
    label = (json.loads(hand_label[0]["value"]) if hand_label
             else moment_label if graded else None)
    if label not in EVENT_TYPES:
        label = None
    worth = verdict(lab.get("highlight", []), min_trust)
    funny = verdict([r for r in lab.get("funny", []) if r["source"] == "hand"], min_trust)
    used = [r for rows in lab.values() for r in rows if r["trust"] >= min_trust]
    weight = max((float(r["trust"]) for r in used), default=0.0)
    return label, worth, funny, weight, sorted({r["source"] for r in used})


def corrected(answer: dict, label: str | None, worth: int | None, funny: int | None) -> dict:
    """Qwen's answer with your corrections applied."""
    t = dict(answer)
    if label:
        t["event_type"] = label
        t["play_active"] = label != "downtime"
    if worth == 1:
        t["excitement"] = max(t["excitement"], WORTH_MIN)
    elif worth == 0:
        t["excitement"] = min(t["excitement"], NOT_WORTH_MAX)
    if funny == 1:
        t["funny"] = max(t["funny"], FUNNY_MIN)
    elif funny == 0:
        t["funny"] = min(t["funny"], NOT_FUNNY_MAX)
    return WindowTag.model_validate(t).model_dump()


def collect(lib: "Library") -> tuple[list[Example], list[Example], Counter]:
    """(training examples, test-set examples, why other windows were left out)."""
    fc = lib.cfg.learning.finetune
    labels = _labels(lib)
    gold = {mid for mid, lab in labels.items()
            if any(r["gold"] for rows in lab.values() for r in rows)}
    moments = {r["id"]: r for r in lib.q("SELECT id, session, stem, file_time, event_type "
                                         "FROM moments WHERE source = 'vlm'")}
    gold_times: dict[str, list[float]] = {}
    for mid in gold:
        if mid in moments:
            gold_times.setdefault(moments[mid]["stem"], []).append(moments[mid]["file_time"])

    train: list[Example] = []
    test: list[Example] = []
    skipped: Counter = Counter()
    for w in lib.q("SELECT * FROM windows ORDER BY stem, start"):
        mid = f"{w['stem']}@{w['start'] + w['peak_offset']:.0f}"
        if mid not in moments or mid not in labels:
            continue
        is_gold = mid in gold
        if not is_gold and any(w["start"] <= t <= w["end_s"]
                               for t in gold_times.get(w["stem"], [])):
            skipped["shows a test-set moment"] += 1
            continue
        label, worth, funny, weight, sources = judged(
            labels[mid], moments[mid]["event_type"], fc.min_trust, is_gold)
        if label is None and worth is None and funny is None:
            skipped["labels not trusted enough"] += 1
            continue
        images = lib.frame_files(w["stem"], w["start"])
        prompt = lib.read_prompt(w["stem"], w["start"])
        if not images or not prompt:
            skipped["frames or prompt not kept"] += 1
            continue
        answer = _answer(w)
        ex = Example(w["id"], mid, moments[mid]["session"], images, prompt, answer,
                     corrected(answer, label, worth, funny), weight, sources,
                     {"label": label, "worth": worth, "funny": funny})
        (test if is_gold else train).append(ex)
    return train, test, skipped


def _held_back(moment: str, share: float) -> bool:
    return int(hashlib.sha1(moment.encode()).hexdigest()[:8], 16) % 1000 < share * 1000


def _line(ex: Example, folder: Path, with_truth: bool = False) -> str:
    content: list[dict] = [{"type": "image", "image": Path(os.path.relpath(p, folder)).as_posix()}
                           for p in ex.images]
    content.append({"type": "text", "text": ex.prompt})
    row: dict[str, Any] = {
        "id": ex.window, "moment": ex.moment, "session": ex.session, "weight": ex.weight,
        "messages": [{"role": "user", "content": content},
                     {"role": "assistant",
                      "content": [{"type": "text", "text": json.dumps(ex.target)}]}],
        "qwen_said": ex.answer, "sources": ex.sources,
    }
    if with_truth:
        row["truth"] = ex.truth
    return json.dumps(row, ensure_ascii=False)


def export(lib: "Library", folder: Path | None = None) -> Path:
    fc = lib.cfg.learning.finetune
    train, test, skipped = collect(lib)
    if not train:
        raise PipelineError("Nothing to export yet: no window has trusted labels (grade "
                            "some moments with --grade, or cut and trim clips in picks.csv).")
    folder = folder or lib.root / "training" / datetime.now().strftime("%Y-%m-%d_%H%M")
    folder.mkdir(parents=True, exist_ok=True)
    val = [ex for ex in train if _held_back(ex.moment, fc.val_share)]
    train = [ex for ex in train if not _held_back(ex.moment, fc.val_share)]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for ex in rows:
                fh.write(_line(ex, folder, with_truth=name == "test") + "\n")
    prompts = {r["prompt"] for r in lib.q("SELECT DISTINCT prompt FROM windows")}
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "base_model": lib.cfg.vlm.model,
        "prompt_versions": sorted(p for p in prompts if p),
        "frame_width": lib.cfg.vlm.frame_width,
        "counts": {"train": len(train), "val": len(val), "test": len(test),
                   "moments": len({ex.moment for ex in train + val}),
                   "sessions": len({ex.session for ex in train + val})},
        "left_out": dict(skipped),
        "labels": dict(Counter(s for ex in train + val for s in ex.sources)),
        "labels_changed": dict(Counter(k for ex in train + val
                                       for k in ("event_type", "excitement", "funny")
                                       if ex.target[k] != ex.answer[k])),
        "min_trust": fc.min_trust,
        "answer_schema": answer_schema(),
        "format": "messages: user = images (in the order Qwen saw them) then the prompt; "
                  "assistant = the JSON answer. weight = how much to trust the example.",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    c = manifest["counts"]
    log(f"[train] wrote {c['train']} training and {c['val']} held-back examples from "
        f"{c['moments']} moments in {c['sessions']} session{'s' if c['sessions'] != 1 else ''}, "
        f"plus {c['test']} test-set windows for checking -> {folder}")
    if skipped:
        log("[train] left out: " + ", ".join(f"{n} ({why})" for why, n in skipped.items()))
    return folder


def readiness(lib: "Library") -> tuple[int, int]:
    """(moments outside the test set with trusted labels, learning.finetune.min_labels)."""
    fc = lib.cfg.learning.finetune
    row = lib.q("SELECT COUNT(DISTINCT moment) AS n FROM labels WHERE gold = 0 AND trust >= ? "
                "AND source IN ('hand', 'cut', 'trim') AND moment NOT IN "
                "(SELECT moment FROM labels WHERE gold = 1)", (fc.min_trust,))
    return int(row[0]["n"]), fc.min_labels


def status(lib: "Library") -> str:
    """One line for the Progress screen and --check."""
    n, need = readiness(lib)
    if n < need:
        return (f"Not yet: {n} of about {need} labelled moments. A fine-tuned Qwen needs that "
                "many to be worth training; --export-training writes the training set any time.")
    return (f"Ready: {n} labelled moments, enough to try fine-tuning Qwen. "
            "--export-training writes the training set.")
