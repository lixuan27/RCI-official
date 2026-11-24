"""SplitBench writer: the writing prompt for a hidden-profile task, the reply parser and the novelty check."""
from __future__ import annotations

import json
import random
import re

DOMAINS = [
    "emergency evacuation", "hospital triage", "hiring a senior engineer", "choosing a supplier", "siting a wind farm",
    "picking a vaccine distribution hub", "selecting a research collaborator", "choosing a ship route", "wildfire response",
    "selecting a school site", "investing in a startup", "choosing a drug candidate", "planning a rescue mission",
    "picking a data center location", "choosing a contractor for a bridge", "selecting an expedition base camp",
    "choosing a software vendor", "picking a festival venue", "allocating a donor organ", "choosing a flood barrier design",
    "selecting a lead for a criminal case", "choosing an archaeological dig site", "picking a satellite launch window",
    "choosing a water source for a village", "selecting a shelter for refugees", "choosing a negotiation partner",
    "picking a pilot for a mission", "choosing a crop for a drought year", "selecting a merger target", "choosing a court venue",
]
LETTERS = "ABC"
BANNED = re.compile(r"correct (answer|option|choice)|you should (choose|pick|select)|the best (option|choice) is|"
                    r"\b(decoy|hidden profile)\b", re.I)

PROMPT = """Write one group decision task in the style of a hidden-profile experiment, as a single JSON object and nothing else.

Domain: {domain}. Invent a concrete situation with names, places and numbers.

Keys:
- "name": a short title.
- "description": two to four sentences that set the scene, name the three options and say that the group must pick one.
- "possible_answers": the three options, short names, in the order A, B, C.
- "correct_answer": option {right} exactly as written in possible_answers.
- "shared_information": {n_shared} facts that every member sees. Taken alone they clearly favour option {decoy}.
- "hidden_information": {n_hidden} facts, each one sentence, each held by only one member. No single one of them is enough to
  change the decision, but all of them together show that option {decoy} has a serious problem and that option {right} is best.
- "explanation": one or two sentences on why option {right} is right once every fact is known.

Rules: every fact is one plain, concrete sentence under 40 words, stated as an observation, with no verdict on which option to
pick. Do not mention the words decoy, hidden, private or correct anywhere in the facts or the description."""


PROMPT_CONTROL = """Write one group decision task as a single JSON object and nothing else.

Domain: {domain}. Invent a concrete situation with names, places and numbers.

Keys:
- "name": a short title.
- "description": two to four sentences that set the scene, name the three options and say that the group must pick one.
- "possible_answers": the three options, short names, in the order A, B, C.
- "correct_answer": option {right} exactly as written in possible_answers.
- "shared_information": {n_shared} facts that every member sees. Taken alone they favour option {right}.
- "hidden_information": {n_hidden} facts, each one sentence, each held by only one member. They add detail; one or two may look
  like small points against option {right} or for another option, but all of them together leave option {right} clearly best.
- "explanation": one or two sentences on why option {right} is right once every fact is known.

Rules: every fact is one plain, concrete sentence under 40 words, stated as an observation, with no verdict on which option to
pick. Do not mention the words decoy, hidden, private or correct anywhere in the facts or the description."""


def prompt(rng: random.Random, control: bool = False, n_hidden: int | None = None) -> tuple[str, dict]:
    right, decoy = rng.sample(range(3), 2)
    draw = {"domain": rng.choice(DOMAINS), "right": LETTERS[right], "decoy": None if control else LETTERS[decoy],
            "n_shared": rng.choice([3, 4]), "n_hidden": rng.choice([4, 4, 6, 8]), "control": control}
    if n_hidden is not None:
        draw["n_hidden"] = int(n_hidden)
    return (PROMPT_CONTROL if control else PROMPT).format(**draw), draw


def _json_object(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def parse(text: str, draw: dict) -> dict | None:
    """The task in a reply, or None when it lacks the shape the draw asked for."""
    t = _json_object(text)
    if t is None:
        return None
    opts = t.get("possible_answers")
    if not (isinstance(opts, list) and len(opts) == 3 and all(isinstance(o, str) and o.strip() for o in opts)):
        return None
    opts = [o.strip() for o in opts]
    if len({o.lower() for o in opts}) != 3:
        return None
    shared, hidden = t.get("shared_information"), t.get("hidden_information")
    if not (isinstance(shared, list) and isinstance(hidden, list) and len(hidden) == draw["n_hidden"] and 2 <= len(shared) <= 5):
        return None
    facts = shared + hidden
    if not all(isinstance(f, str) and 0 < len(f.split()) <= 60 for f in facts) or any(BANNED.search(f) for f in facts):
        return None
    desc, key = t.get("description"), t.get("correct_answer")
    if not (isinstance(desc, str) and desc.strip() and isinstance(key, str)) or BANNED.search(desc):
        return None
    match = [o for o in opts if o.lower() == key.strip().lower()]
    if not match or opts.index(match[0]) != LETTERS.index(draw["right"]):
        return None
    return {"name": str(t.get("name") or "")[:120], "description": desc.strip(), "possible_answers": opts,
            "correct_answer": match[0], "shared_information": [f.strip() for f in shared],
            "hidden_information": [f.strip() for f in hidden],
            "decoy": opts[LETTERS.index(draw["decoy"])] if draw.get("decoy") else None, "control": bool(draw.get("control")),
            "explanation": str(t.get("explanation") or ""), "domain": draw["domain"]}


def _words(row: dict) -> set[str]:
    text = " ".join([row["description"]] + row["shared_information"] + row["hidden_information"])
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def novelty(row: dict, bench: list[dict]) -> float:
    """The largest word-set Jaccard between a task and any benchmark task."""
    w = _words(row)
    return max((len(w & b) / max(1, len(w | b)) for b in map(_words, bench)), default=0.0)


def to_benchmark(rows: list[dict], start: int = 1000) -> list[dict]:
    return [{**r, "id": start + i, "writer_answer": r["correct_answer"]} for i, r in enumerate(rows)]
