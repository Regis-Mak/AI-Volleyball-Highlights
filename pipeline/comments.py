"""Unhinged comments: Qwen reads what everyone said and rates how unhinged
each comment is, so the wild things people say make the reel too.

The Whisper transcript is read in overlapping stretches of
comments.lines_per_read lines. Each line carries its time and what the
microphone heard right after it (laughter, shouting or cheering, from PANNs).
Qwen lists the comments that are wild, savage, absurd or out of pocket,
rated 0-10, with the line that reacts to each one. No pictures are sent, so
this is quick next to the vision stage.

A reaction right after a comment makes it count for more: a "whaaat?!"
(comments.cues) adds comments.cue_boost, clear laughter comments.laugh_boost.
Comments at comments.min_unhinged or more get a place in the reel, from just
before the comment to just after the reaction (see pipeline/fuse.py and
pipeline/selection.py).

work/<game>/comments.json     Qwen's answer for each stretch (an interrupted
                              run resumes; a new transcript re-reads it)
work/<session>/comments.csv   every rated comment, most unhinged first
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from . import vlm
from .audio import _normalize_word
from .common import Game, PipelineError, Progress, fmt_time, log
from .config import Config

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "rate_comments.txt"
CACHE = "comments.json"


class Said(BaseModel):
    line: int
    quote: str
    reaction_line: int = 0
    why: str = ""
    unhinged: int = Field(ge=0, le=10)


class Answer(BaseModel):
    comments: list[Said] = Field(default_factory=list)


def answer_schema() -> dict:
    """Every field required, so Ollama fills them all in, in order."""
    schema = Answer.model_json_schema()
    schema["required"] = ["comments"]
    said = schema["$defs"]["Said"]
    said["required"] = list(said["properties"])
    return schema


@dataclass
class Comment:
    t: float              # seconds into the file, when the comment starts
    end: float            # when it ends
    react_end: float      # when the reaction to it ends (= end when there is none)
    quote: str
    reaction: str         # what was said in reaction, if anything
    why: str
    rated: int            # Qwen's 0-10
    unhinged: float       # after the reaction boosts, 0-10


class Rater:
    """Rates one file at a time, loading Qwen only when a stretch isn't cached."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        cc = cfg.comments
        self.vc = cfg.vlm.model_copy(update={"model": cc.model or cfg.vlm.model})
        self.tpl = PROMPT_FILE.read_text(encoding="utf-8")
        self.cues = [re.compile(p, re.IGNORECASE) for p in cc.cues]
        self.client: vlm.OllamaClient | None = None

    def close(self) -> None:
        if self.client is not None:
            self.client.unload()
            self.client = None

    def _client(self) -> vlm.OllamaClient:
        if self.client is None:
            self.client = vlm.OllamaClient(self.vc, pull=True)
            self.client.load()
        return self.client

    def run(self, game: Game) -> list[Comment]:
        cc = self.cfg.comments
        segs = [s for s in vlm._load_transcript(game) if s.get("text", "").strip()]
        if not cc.enabled or not segs:
            if cc.enabled:
                log("[comments] no transcript, so no comments to rate")
            return []
        tl = vlm.load_timeline(game)
        lines = [_line_text(s, tl) for s in segs]
        step = max(1, cc.lines_per_read - cc.overlap_lines)
        reads = [list(range(i, min(i + cc.lines_per_read, len(segs))))
                 for i in range(0, max(1, len(segs) - cc.overlap_lines), step)]

        path = game.work / CACHE
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        prompts = [Template(self.tpl).safe_substitute(lines="\n".join(
            f"{n}. {lines[i]}" for n, i in enumerate(idx, 1))) for idx in reads]
        keys = [hashlib.sha1((p + self.vc.model).encode()).hexdigest() for p in prompts]
        todo = [k for k in range(len(reads)) if keys[k] not in cache]
        log(f"[comments] {len(segs)} lines in {len(reads)} stretches: "
            f"{len(reads) - len(todo)} cached, {len(todo)} to read")
        answers = {k: cache[keys[k]] for k in range(len(reads)) if keys[k] in cache}
        if todo:
            client = self._client()
            late = 0
            with Progress("comments", f"reading what was said with {self.vc.model}",
                          len(todo)) as bar:
                for n, k in enumerate(todo, 1):
                    try:
                        data = client.answer(prompts[k], answer_schema(),
                                             cc.max_answer_tokens)
                    except vlm.QwenTimeout:
                        late += 1
                        if late >= 3:
                            raise PipelineError(client.too_slow()) from None
                        log(f"[comments] warning: no answer for the stretch at "
                            f"{fmt_time(segs[reads[k][0]]['start'])}; read next run")
                        bar.update(n)
                        continue
                    late = 0
                    answers[k] = data if data is not None else {"comments": []}
                    cache[keys[k]] = answers[k]
                    path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                    bar.update(n)
        # Keep only the stretches this transcript still has.
        kept = {keys[k]: answers[k] for k in answers}
        if kept != cache:
            path.write_text(json.dumps(kept, indent=1), encoding="utf-8")

        found: dict[tuple[int, float], Comment] = {}
        for k, data in answers.items():
            try:
                said = Answer.model_validate(data).comments
            except ValidationError:
                continue
            for s in said:
                c = self._place(segs, reads[k], s, tl)
                if c is None:
                    continue
                key = (round(c.t * 2) / 2, c.end)
                if key not in found or found[key].unhinged < c.unhinged:
                    found[key] = c
        out = _dedupe(sorted(found.values(), key=lambda c: c.t))
        top = sorted((c for c in out if c.unhinged >= cc.min_unhinged),
                     key=lambda c: c.unhinged, reverse=True)
        log(f"[comments] {len(out)} comments rated, {len(top)} unhinged enough for the reel "
            f"({cc.min_unhinged:g}+)"
            + "".join(f'\n[comments]   {fmt_time(c.t)}  {c.unhinged:g}  "{c.quote[:70]}"'
                      for c in top[:3]))
        return out

    # ------------------------------------------------------------- placing

    def _place(self, segs: list[dict], idx: list[int], s: Said,
               tl: dict[str, np.ndarray]) -> Comment | None:
        """Where the comment was said (word times), what reacted to it, and
        how unhinged it counts once the reaction is in."""
        cc = self.cfg.comments
        # Qwen sometimes copies the line's time or [then: laughter] note into the quote.
        quote = " ".join(re.sub(r"\[[^\]]*\]", " ", s.quote).split())
        if not 1 <= s.line <= len(idx) or not quote:
            return None
        # Qwen sometimes numbers a line off by one or two: look around it.
        near = [idx[s.line - 1]] + [idx[j] for j in (s.line - 2, s.line, s.line - 3, s.line + 1)
                                    if 0 <= j < len(idx)]
        span = None
        for i in near:
            span = _find_quote(segs[i], quote)
            if span is not None:
                break
        if span is None:
            return None             # not in the transcript: made up
        start, end = span
        react_end, reaction = end, ""
        if 1 <= s.reaction_line <= len(idx) and s.reaction_line != s.line:
            r = segs[idx[s.reaction_line - 1]]
            if end - 1 <= r["start"] <= end + cc.cue_window_s:
                react_end = min(float(r["end"]), end + cc.cue_window_s + 4)
                reaction = r["text"].strip()
        score = float(s.unhinged)
        cue = self._cue_after(segs, end)
        if cue is not None:
            score += cc.cue_boost
            react_end = max(react_end, cue[0])
            reaction = reaction or cue[1]
        laugh = tl.get("laugh")
        if laugh is not None:
            a = int(end)
            if float(laugh[a: a + 6].max(initial=0.0)) >= vlm.LAUGH_CLEAR:
                score += cc.laugh_boost
        return Comment(t=start, end=end, react_end=react_end, quote=quote,
                       reaction=reaction[:120], why=s.why.strip(), rated=s.unhinged,
                       unhinged=min(10.0, score))

    def _cue_after(self, segs: list[dict], end: float) -> tuple[float, str] | None:
        """A "whaaat?!"-type reaction in the cue_window_s after a comment:
        (when it ends, what was said)."""
        window = self.cfg.comments.cue_window_s
        for seg in segs:
            if seg["end"] < end or seg["start"] > end + window:
                continue
            words = [w for w in seg.get("words") or []
                     if end - 0.05 <= w["start"] <= end + window]
            if not words:
                continue
            text, ends, n = "", [], 0
            for w in words:
                text += w["word"]
                ends.append((n + len(w["word"]), w["end"]))
            lead = len(text) - len(text.lstrip())
            text = text.strip()
            for p in self.cues:
                hit = p.search(text)
                if hit:
                    t = next((e for off, e in ends if off - lead >= hit.end()), ends[-1][1])
                    return float(t), text[:120]
        return None


# ------------------------------------------------------------------ helpers

def _line_text(seg: dict, tl: dict[str, np.ndarray]) -> str:
    """'[3:41] what was said [then: clear laughter, shouting]'"""
    a = int(seg["end"])
    heard = []
    for k, words in (("laugh", ("a little laughter", "clear laughter")),
                     ("shout", (None, "shouting")), ("cheer", (None, "cheering"))):
        level = float(tl[k][a: a + 5].max(initial=0.0))
        if level >= vlm.LAUGH_CLEAR and words[1]:
            heard.append(words[1])
        elif level >= vlm.LAUGH_SOME and words[0]:
            heard.append(words[0])
    note = f" [then: {', '.join(heard)}]" if heard else ""
    return f"[{fmt_time(seg['start'])}] {' '.join(seg['text'].split())}{note}"


def _find_quote(seg: dict, quote: str) -> tuple[float, float] | None:
    """(start, end) of the quote's words in the segment, or None when most of
    the quote isn't there."""
    q = [t for t in (_normalize_word(x) for x in quote.split()) if t]
    words = seg.get("words") or []
    toks = [_normalize_word(w["word"]) for w in words]
    if not q:
        return None
    if not words:
        text = {_normalize_word(x) for x in seg.get("text", "").split()}
        if sum(t in text for t in q) < 0.5 * len(q):
            return None
        return float(seg["start"]), float(seg["end"])
    if sum(t in toks for t in q) < 0.5 * len(q):
        return None
    for k in (min(3, len(q)), 1):
        for i in range(len(toks) - k + 1):
            if toks[i:i + k] == q[:k]:
                j = min(len(words) - 1, i + len(q) - 1)
                return float(words[i]["start"]), float(words[j]["end"])
    first = next(i for i, t in enumerate(toks) if t in q)
    last = max(i for i, t in enumerate(toks) if t in q)
    return float(words[first]["start"]), float(words[last]["end"])


def _dedupe(comments: list[Comment]) -> list[Comment]:
    """The same comment found in two overlapping stretches, kept once."""
    out: list[Comment] = []
    for c in comments:
        twin = next((o for o in out if c.t < o.end and o.t < c.end), None)
        if twin is None:
            out.append(c)
        elif c.unhinged > twin.unhinged:
            out[out.index(twin)] = c
    return out


def write_csv(path: Path, rows: list[tuple[str, str, str, Comment]]) -> None:
    """rows: (session clock, file, time in file, comment), most unhinged first."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clock", "file", "file_time", "unhinged", "qwen", "quote", "reaction",
                    "why"])
        for clock, name, ft, c in sorted(rows, key=lambda r: r[3].unhinged, reverse=True):
            w.writerow([clock, name, ft, f"{c.unhinged:g}", c.rated, c.quote, c.reaction,
                        c.why])
