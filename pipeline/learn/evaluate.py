"""Model check: how well a vision model tags your test set.

It re-tags the windows behind your test-set moments (the ones you graded with
--grade test, never trained on) and compares the answers with your grades:

  label      share of moments given the label you set, or saw and kept
  highlight  how well its excitement puts the moments you said were worth
             showing above the others (0.5 = no better than guessing, 1 = perfect)
  funny      the same for the moments you marked funny, from its funny score
  overall    the average of the three

The answers from when the footage was first tagged are scored too ("as
tagged"), so a different model or an edited prompt can be compared without
re-running the old one. Use it to try a smaller or bigger model, to check a
prompt change before re-tagging everything, and in Phase 7 to decide whether
a fine-tuned Qwen is better than the one you have.

Each window gets the exact prompt it was first tagged with. If
prompts/tag_window.txt has changed since, the prompt is rebuilt from the
current file instead (from the cached transcript and audio in work/), so the
check shows what the edited prompt would do.

    python make_highlights.py --evaluate                         the model in config.yaml
    python make_highlights.py --evaluate qwen3-vl:4b qwen3-vl:8b  several, side by side

Every check is saved in library/evals/.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import vlm
from ..common import Game, PipelineError, Progress, log
from .ranker import auc
from .trainset import collect

if TYPE_CHECKING:
    from ..library import Library

MIN_MOMENTS = 10
AS_TAGGED = "as tagged"


@dataclass
class Case:
    moment: str
    window: str
    stem: str
    start: float
    end: float
    images: list[Path]
    prompt: str
    answer: dict          # the answer from when it was first tagged
    truth: dict           # label, worth, funny (None where not graded)


def cases(lib: "Library") -> tuple[list[Case], int]:
    """One window per test-set moment (the one whose answer made the moment),
    and how many test-set moments had no usable window."""
    _, test, _ = collect(lib)
    descs = {r["id"]: r["description"] for r in lib.q("SELECT id, description FROM moments")}
    rows = {r["id"]: r for r in lib.q("SELECT id, stem, start, end_s FROM windows")}
    best: dict[str, Any] = {}
    for ex in test:
        if ex.moment not in best or ex.answer["description"] == descs.get(ex.moment):
            best[ex.moment] = ex
    out = []
    for mid, ex in sorted(best.items()):
        w = rows[ex.window]
        out.append(Case(mid, ex.window, w["stem"], float(w["start"]), float(w["end_s"]),
                        ex.images, ex.prompt, ex.answer, ex.truth))
    graded = {r["moment"] for r in lib.q("SELECT DISTINCT moment FROM labels WHERE gold = 1")}
    return out, len(graded) - len(out)


def score(answers: list[dict], truths: list[dict]) -> dict[str, float | None]:
    lab = [(a["event_type"], t["label"]) for a, t in zip(answers, truths) if t["label"]]
    out: dict[str, float | None] = {
        "label": float(np.mean([a == t for a, t in lab])) if lab else None}
    for key, field, truth_key in (("highlight", "excitement", "worth"),
                                  ("funny", "funny", "funny")):
        pairs = [(a[field], t[truth_key]) for a, t in zip(answers, truths)
                 if t[truth_key] is not None]
        out[key] = (auc(np.array([p[0] for p in pairs], dtype=float),
                        np.array([p[1] for p in pairs], dtype=int)) if pairs else None)
    vals = [v for v in out.values() if v is not None]
    out["overall"] = float(np.mean(vals)) if vals else None
    return out


class _Prompts:
    """The prompt to send for each case: as first sent, or rebuilt from the
    current prompts/tag_window.txt when that file has changed since."""

    def __init__(self, lib: "Library"):
        self.lib = lib
        self.current = vlm.prompt_version()
        self.versions = {r["id"]: r["prompt"] for r in lib.q("SELECT id, prompt FROM windows")}
        self.builders: dict[str, vlm.PromptBuilder | None] = {}
        self.rebuilt = self.kept = 0

    def _builder(self, stem: str) -> "vlm.PromptBuilder | None":
        if stem not in self.builders:
            lib, cfg = self.lib, self.lib.cfg
            row = lib.q("SELECT path FROM files WHERE stem = ?", (stem,))
            builder = None
            if row:
                game = Game(Path(row[0]["path"]), stem, cfg.work_dir / stem,
                            cfg.output_dir / stem, cfg)
                try:
                    builder = vlm.PromptBuilder(game)
                except PipelineError:
                    builder = None            # its work folder (audio) was deleted
            self.builders[stem] = builder
        return self.builders[stem]

    def get(self, c: Case) -> str:
        if self.versions.get(c.window) in (None, self.current):
            return c.prompt
        builder = self._builder(c.stem)
        if builder is None:
            self.kept += 1
            return c.prompt
        self.rebuilt += 1
        offsets = json.loads(self.lib.q("SELECT frame_offsets FROM windows WHERE id = ?",
                                        (c.window,))[0]["frame_offsets"] or "[]")
        return builder.build(c.start, c.end, offsets[: len(c.images)])


def run(lib: "Library", models: list[str]) -> Path:
    cfg = lib.cfg
    found, missing = cases(lib)
    if len(found) < MIN_MOMENTS:
        raise PipelineError(
            f"The check needs at least {MIN_MOMENTS} test-set moments with their frames kept; "
            f"there are {len(found)}. Grade some with: python make_highlights.py --grade test")
    models = models or [cfg.vlm.model]
    truths = [c.truth for c in found]
    prompts = _Prompts(lib)
    texts = [prompts.get(c) for c in found]
    log(f"[eval] test set: {len(found)} moments"
        + (f" ({missing} more had no frames kept)" if missing > 0 else ""))
    if prompts.rebuilt:
        log(f"[eval] prompts/tag_window.txt has changed since tagging: {prompts.rebuilt} "
            "prompts rebuilt from the current file (without example-memory hints)"
            + (f"; {prompts.kept} kept as first sent (their work folder is gone)"
               if prompts.kept else ""))

    results: dict[str, dict] = {AS_TAGGED: {"scores": score([c.answer for c in found], truths),
                                            "seconds_per_window": None}}
    answers: dict[str, list[dict]] = {AS_TAGGED: [c.answer for c in found]}
    for model in models:
        vc = cfg.vlm.model_copy(update={"model": model})
        try:
            client = vlm.OllamaClient(vc)
        except PipelineError as err:
            log(f"[eval] warning: skipping {model}: {str(err).splitlines()[0]}")
            continue
        got: list[dict] = []
        t0 = time.time()
        try:
            with Progress("eval", f"{model} on the test set", len(found)) as bar:
                for n, (c, text) in enumerate(zip(found, texts), 1):
                    images = [p.read_bytes() for p in c.images]
                    got.append(client.tag(text, images, c.end - c.start).model_dump())
                    bar.update(n)
        finally:
            client.unload()
        answers[model] = got
        results[model] = {"scores": score(got, truths),
                          "seconds_per_window": (time.time() - t0) / len(found)}

    _print(results)
    folder = lib.root / "evals"
    folder.mkdir(exist_ok=True)
    path = folder / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    path.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "models": [m for m in results if m != AS_TAGGED], "results": results,
        "prompt_version": prompts.current, "prompts_rebuilt": prompts.rebuilt,
        "moments": [{"moment": c.moment, "window": c.window, "truth": c.truth,
                     **{m: a[i] for m, a in answers.items()}} for i, c in enumerate(found)],
    }, indent=1), encoding="utf-8")
    log(f"[eval] saved -> {path}")
    return path


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "  - "


def _print(results: dict[str, dict]) -> None:
    width = max(len(m) for m in results) + 2
    log(f"[eval]   {'':<{width}}label  highlight  funny  overall  s/window")
    for model, res in results.items():
        s = res["scores"]
        spw = res["seconds_per_window"]
        log(f"[eval]   {model:<{width}}{_fmt(s['label']):>5}  {_fmt(s['highlight']):>9}  "
            f"{_fmt(s['funny']):>5}  {_fmt(s['overall']):>7}  "
            f"{(f'{spw:.1f}' if spw is not None else '-'):>8}")
    log("[eval] label = share labelled as you said; highlight and funny: 0.5 = guessing, "
        "1 = perfect order")


def history(lib: "Library", limit: int = 5) -> list[dict]:
    """The most recent checks, newest first."""
    folder = lib.root / "evals"
    out = []
    for path in sorted(folder.glob("*.json"), reverse=True)[:limit] if folder.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(data)
    return out


def latest(lib: "Library") -> str | None:
    """One line about the last check, for --check and the Progress screen."""
    runs = history(lib, 1)
    if not runs:
        return None
    data = runs[0]
    parts = []
    for model, res in data["results"].items():
        overall = res["scores"].get("overall")
        parts.append(f"{model} {_fmt(overall).strip()}")
    n = len(data.get("moments", []))
    return f"{data['created'][:16].replace('T', ' ')}, {n} moments: " + ", ".join(parts)
