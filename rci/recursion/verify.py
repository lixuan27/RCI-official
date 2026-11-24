"""The protocol's label-free check of a first turn, and the training row of a captured first span."""
from __future__ import annotations

import re

from rci.recursion.turns import content, has_tool_call
from rci.tasks.hiddenbench.facts import SENTENCE, VERDICT, holds, keys_of, verdict_of, words

CRITERIA = ("loose", "strict")


def first_turn_ok(text: str, mine: list, row) -> tuple:
    """Whether a first message writes every private fact of the member and states no verdict, with the reason."""
    said, keys = words(text), keys_of(row)
    if not all(holds(said, keys[h]) for h in mine):
        return False, "a private fact left out"
    if verdict_of(text, row) is not None:
        return False, "verdict on an option"
    return True, "ok"


def independent_ok(text: str, mine: list, row, called: bool = False) -> bool:
    """The strict check: every fact's key fully covered and no sentence naming an option beside a stance phrase."""
    if called:
        return False
    said, keys = words(text), keys_of(row)
    if not all(holds(said, keys[h], 1.0) for h in mine):
        return False
    opts = [re.compile(rf"\b{re.escape(str(o).lower())}\b") for o in row["possible_answers"]]
    for s in SENTENCE.split(text):
        low = s.lower()
        if any(o.search(low) for o in opts) and VERDICT.search(low):
            return False
    return True


def first_span_row(chains: list) -> dict | None:
    """A member's first generation as a row: its prompt and its first span only."""
    if not chains or not chains[0].get("spans"):
        return None
    start, n = chains[0]["spans"][0]
    return {"ids": chains[0]["ids"][:start + n], "spans": [[start, n]]}


def judge(got, mine: list, row, submit: str = "reject", criterion: str = "loose") -> str:
    """'ok' when a sampled first turn finished and passes the check; otherwise why not."""
    if got is None or not got[1]:
        return "failed sample"
    text, _, finish = got
    if finish not in (None, "stop"):
        return "unfinished sample"
    msg = content(text)
    called = has_tool_call(text)
    if criterion == "strict":
        if not independent_ok(msg, mine, row, called=called and submit == "reject"):
            return "fails the strict check"
        return "ok"
    v = first_turn_ok(msg, mine, row)[1]
    if v == "ok" and submit == "reject" and called:
        return "first turn submits an answer"
    return v
