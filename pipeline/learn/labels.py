"""From the library's labels to training targets, plus the labels the tool
makes itself when its independent signals clearly agree.

When a moment has labels from several sources, the most trusted kind wins:
your Grade-screen answer, then what you did in picks.csv (cut, trim, kept,
combined by trust), then the tool's own agreement label. Moments in the test
set (gold) are never trained on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..fuse import Moment
    from ..library import Library

USER_SOURCES = ("cut", "trim", "kept")
SHOWABLE = ("great_play", "blooper", "celebration", "chaos")

# Agreement thresholds. A tool-made label needs the vision tag, the audio
# reaction and the movement all pointing the same way.
HIGH_SCORE, LOW_SCORE = 7, 2          # Qwen's excitement / funny, 0-10
LOUD, QUIET = 0.4, 0.1                # audio reaction around the moment
LAUGHING = 0.5                        # laughter (PANNs), 0-1
MOVING = 1.2                          # movement vs. this file's typical second


@dataclass
class Target:
    moment: str
    session: str
    y: int
    weight: float
    tier: str          # gold | hand | user | agree


def targets(lib: "Library", kind: str) -> tuple[list[Target], list[Target]]:
    """(training targets, test-set targets) for one label kind."""
    gold_moments = {r["moment"] for r in lib.q("SELECT DISTINCT moment FROM labels "
                                               "WHERE gold = 1")}
    by_moment: dict[str, list] = {}
    for r in lib.labels([kind]):
        by_moment.setdefault(r["moment"], []).append(r)

    train: list[Target] = []
    test: list[Target] = []
    for mid, labs in sorted(by_moment.items()):
        session = labs[0]["session"] or ""
        gold = [r for r in labs if r["gold"]]
        if gold:
            test.append(Target(mid, session, int(json.loads(gold[0]["value"])), 1.0, "gold"))
            continue
        if mid in gold_moments:
            continue
        hand = [r for r in labs if r["source"] == "hand"]
        user = [r for r in labs if r["source"] in USER_SOURCES]
        agree = [r for r in labs if r["source"] == "agree"]
        if hand:
            r = hand[0]
            train.append(Target(mid, session, int(json.loads(r["value"])), r["trust"], "hand"))
        elif user:
            total = sum(r["trust"] for r in user)
            vote = sum(r["trust"] * float(json.loads(r["value"])) for r in user) / total
            train.append(Target(mid, session, int(vote >= 0.5),
                                max(r["trust"] for r in user), "user"))
        elif agree:
            r = agree[0]
            train.append(Target(mid, session, int(json.loads(r["value"])), r["trust"], "agree"))
    return train, test


def agreement_labels(moments: list["Moment"]) -> list[tuple[str, str, int]]:
    """Labels the tool is confident enough to make itself, only where vision,
    audio and movement agree. They are capped in training (learning.self_made_cap)
    and always lose to a label from you."""
    out: list[tuple[str, str, int]] = []
    for m in moments:
        if m.source != "vlm" or not m.features:
            continue
        f = m.features
        if (m.event_type in SHOWABLE and max(m.excitement, m.funny) >= HIGH_SCORE
                and f["reaction"] >= LOUD and f["motion_peak"] >= MOVING):
            out.append((m.id, "highlight", 1))
        elif (m.event_type == "rally" and m.excitement <= LOW_SCORE + 1
              and m.funny <= LOW_SCORE and f["reaction"] < QUIET):
            out.append((m.id, "highlight", 0))
        if m.funny >= HIGH_SCORE and f["laugh"] >= LAUGHING:
            out.append((m.id, "funny", 1))
        elif m.funny <= LOW_SCORE and f["laugh"] < QUIET and f["reaction"] < QUIET:
            out.append((m.id, "funny", 0))
    return out
