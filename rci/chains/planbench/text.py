"""The PlanBench plan-generation prompt (one-shot, Blocksworld) and the official plan extraction rule.

The prompt and the extraction follow the PlanBench reference implementation (see THIRD_PARTY_NOTICES.md):

    query = intro + statement(example, its plan) + statement(target, no plan)
    statement = "\\n[STATEMENT]\\nAs initial conditions I have that, INIT.\\nMy goal is to have that GOAL."
                "\\n\\nMy plan is as follows:\\n\\n[PLAN]" + PLAN, then "-" -> " " and "ontable" -> "on the table"

Extraction: the whole response is lower-cased, "pick up" / "put down" become their PDDL names, and a line is an action
line when one of its whitespace tokens equals an action name (the first in schema order wins); its objects are the
object names that occur in it as substrings, by first position; a line whose object count does not match the action's
arity is dropped. The judge validates the extracted plan with the simulator (domain.validate) and also requires the
same plan when the response is cut at the first [PLAN END] / [STATEMENT] (strict_ok).
"""
from __future__ import annotations

import re

from rci.chains.planbench import domain as D

INTRO = (
    "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can"
    " do\n"
    "\n"
    "Pick up a block\n"
    "Unstack a block from on top of another block\n"
    "Put down a block\n"
    "Stack a block on top of another block\n"
    "\n"
    "I have the following restrictions on my actions:\n"
    "I can only pick up or unstack one block at a time.\n"
    "I can only pick up or unstack a block if my hand is empty.\n"
    "I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block"
    " has no other blocks on top of it and if the block is not picked up.\n"
    "I can only unstack a block from on top of another block if the block I am unstacking was really on top of the"
    " other block.\n"
    "I can only unstack a block from on top of another block if the block I am unstacking is clear.\n"
    "Once I pick up or unstack a block, I am holding the block.\n"
    "I can only put down a block that I am holding.\n"
    "I can only stack a block on top of another block if I am holding the block being stacked.\n"
    "I can only stack a block on top of another block if the block onto which I am stacking the block is clear.\n"
    "Once I put down or stack a block, my hand becomes empty.\n"
    "Once you stack a block on top of a second block, the second block is no longer clear.\n"
)
STATEMENT = "\n[STATEMENT]\n"
CUE = "\n\nMy plan is as follows:\n\n[PLAN]"
PLAN_END = "[PLAN END]"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def sorted_facts(facts) -> list:
    return sorted(facts, key=lambda f: f[0] + " " + " ".join(f[1:]))


def _is_fact(fact) -> bool:
    ar = D.PREDICATE_ARITY.get(fact[0])
    return ar is not None and ar == len(fact) - 1 and all(a in D.OBJECT_NAME for a in fact[1:])


def fact_text(fact) -> str:
    if not _is_fact(fact):
        raise ValueError(f"not a Blocksworld fact: {fact!r}")
    return D.PREDICATE_TEXT[fact[0]].format(*[D.OBJECT_NAME[a] for a in fact[1:]])


def facts_text(facts) -> str:
    facts = [tuple(f) for f in facts]
    if not facts or len(set(facts)) != len(facts):
        raise ValueError("facts_text needs distinct facts")
    parts = [fact_text(f) for f in sorted_facts(facts)]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]


def action_text(action) -> str:
    tpl = D.ACTION_TEXT.get(action[0])
    if tpl is None or tpl.count("{}") != len(action) - 1 or any(a not in D.OBJECT_NAME for a in action[1:]):
        raise ValueError(f"not a Blocksworld action: {action!r}")
    return tpl.format(*[D.OBJECT_NAME[a] for a in action[1:]])


def plan_text(plan) -> str:
    return "\n" + "".join(action_text(tuple(a)) + "\n" for a in plan) + PLAN_END + "\n"


def statement_text(initial, goal, plan) -> str:
    text = STATEMENT + f"As initial conditions I have that, {facts_text(initial).strip()}."
    text += f"\nMy goal is to have that {facts_text(goal)}." + CUE + ("" if plan is None else plan_text(plan))
    return text.replace("-", " ").replace("ontable", "on the table")


def render_query(example_initial, example_goal, example_plan, initial, goal) -> str:
    return INTRO + statement_text(example_initial, example_goal, example_plan) + statement_text(initial, goal, None)


# ------------------------------------------------------------------------------------------------- extraction
def _forms() -> dict:
    out = {}
    for k, v in D.ACTION_TEXT.items():
        word = v.split(" ")[0]
        out[k] = k.replace("-", " ") if word in k else word
    return out


def extract(text: str) -> dict:
    """{"actions", "lines": [(kind, line)], "n_dropped", "n_other", "n_multi"} of the official rule on `text`."""
    names = [x.lower() for x in D.OBJECT_NAME.values()]
    letter = {v: k for k, v in D.OBJECT_NAME.items()}
    text = (text or "").lower().strip()
    for raw, form in _forms().items():
        text = text.replace(form, raw)
    actions, lines, dropped, other, multi = [], [], 0, 0, 0
    for line in (ln.strip() for ln in text.split("\n")):
        tokens = line.split()
        hits = [a for a in D.ACTIONS if a in tokens]
        multi += len(hits) > 1
        if not hits:
            if line:
                other += 1
                lines.append(("other", line))
            continue
        objs = [o for _, o in sorted((line.index(o), o) for o in names if o in line)]
        if len(objs) != D.ARITY[hits[0]]:
            dropped += 1
            lines.append(("dropped", line))
            continue
        actions.append((hits[0],) + tuple(letter[o] for o in objs))
        lines.append(("action", line))
    return {"actions": actions, "lines": lines, "n_dropped": dropped, "n_other": other, "n_multi": multi}


_CUT = re.compile(r"\[plan end\]|\[statement\]", re.I)
_MARKERS = ("[plan]", "[plan end]")


def cut_region(text: str) -> str:
    m = _CUT.search(text or "")
    return (text or "") if m is None else text[:m.start()]


def tail_after_cut(text: str) -> bool:
    """Something other than whitespace follows the first [PLAN END] / [STATEMENT]."""
    m = _CUT.search(text or "")
    return bool(m) and bool(text[m.end():].strip())


def has_think_tag(text) -> bool:
    low = (text or "").lower()
    return THINK_OPEN in low or THINK_CLOSE in low


def judge(content, initial, goal) -> dict:
    """Verdicts on a served message content: official_ok (whole text), strict_ok (also the cut text, no line with two
    action names), clean (strict_ok and the cut text holds only action lines and plan markers, no think tag)."""
    text = content if isinstance(content, str) else ""
    full, cut = extract(text), extract(cut_region(text))
    f_ok, f_err, _ = D.validate(initial, goal, full["actions"])
    c_ok, _, _ = D.validate(initial, goal, cut["actions"])
    empty = text.strip() == ""
    if empty:
        f_ok = c_ok = False
        f_err = "empty response"
    tag = has_think_tag(text)
    strict = bool(f_ok and c_ok and full["n_multi"] == 0)
    tidy = all(kind == "action" or (kind == "other" and line in _MARKERS) for kind, line in cut["lines"])
    return {"official_ok": bool(f_ok), "strict_ok": strict, "clean": bool(strict and tidy and not tag),
            "plan": [list(a) for a in full["actions"]], "error": f_err, "n_actions": len(full["actions"]),
            "empty": empty, "think_tag": tag, "tail": tail_after_cut(text), "end": PLAN_END.lower() in text.lower()}
