"""The four-call PlanBench protocol of four agents and its commitment rule.

One game = one problem P (the one-shot query) x one seed, four seats of one model, four calls a seat:
    call 1  draft    [user: P]                                                    thinks
    call 2  check    AUDIT(P, [own draft]): a step-by-step simulation              no thinking
    call 3  revise   P + own draft and check + the three other seats' drafts and checks   thinks
    call 4  review   AUDIT(P, the four finals), anonymous, in a per-reviewer hashed order  no thinking
Every review block ends with one verdict line ("VERDICT A: VALID" or "VERDICT A: INVALID ..."). The group submits the
final plan with the highest share of positive reviews from the three other seats (then the mean score; ties by hash).
"""
from __future__ import annotations

import hashlib
import re

SEATS = 4
LABELS = "ABCD"
THINK_CALLS = (1, 3)
SALT = "planbench"

AUDIT_HEAD_1 = "Check the plan below against the planning problem it answers."
AUDIT_HEAD_N = ("Check the four plans below against the planning problem they answer. They were written independently; "
                "any of them may be wrong.")
_AUDIT_RULES = (
    "The problem text ends with the [STATEMENT] to solve; the [STATEMENT] before it is a solved example. Check {what} "
    "against the LAST [STATEMENT] only, by simulating {it}. Start from that statement's initial conditions and take "
    "the plan's actions one at a time, in the order written. For every action, test each condition the rules require "
    "against the current state: if they all hold, write OK and the facts that change; if one does not hold, write "
    "FAIL and the condition that is missing, and stop simulating that plan. If every action is OK, test the goal "
    "against the final state. Do not repair or rewrite {a_plan}, and do not write a plan of your own. Use exactly this "
    "format{blocks}:\n\n"
    "PLAN A\n"
    "1. <action as written> | OK | <facts that become true or stop being true>\n"
    "2. <action as written> | FAIL | <the condition that is missing>\n"
    "GOAL: <MET, NOT MET or NOT CHECKED>\n"
    "VERDICT A: <VALID or INVALID>\n\n"
    "The verdict line is the last line of {the_block}. Write `VERDICT A: VALID` only when every action is OK and the "
    "goal is MET. Otherwise write `VERDICT A: INVALID at step <k>`, with k the number of the first action that fails, "
    "or `VERDICT A: INVALID goal not met` when every action is OK but the goal does not hold at the end, or "
    "`VERDICT A: INVALID no plan` when the text holds no plan.")
AUDIT_TAIL_1 = _AUDIT_RULES.format(what="the plan", it="it", a_plan="the plan", blocks="", the_block="the block")
AUDIT_TAIL_N = _AUDIT_RULES.format(what="each plan separately", it="it", a_plan="any plan",
                                   blocks=", with one block for each of A, B, C and D, in this order",
                                   the_block="each block (with the plan's own letter)")
REVISE_HEAD = (
    "You wrote a plan for the planning problem below and then checked it by simulation. Three other planners wrote "
    "plans for the same problem on their own and checked them in the same way. Their plans and checks are shown too; "
    "any plan or check may contain mistakes.")
REVISE_TAIL = (
    "Now write your final plan for the LAST [STATEMENT] of the problem. Every action must be allowed by the rules in "
    "the state where it is taken, and the goal must hold at the end. Use whatever is right in the other plans and "
    "checks (a step you got wrong, a condition you missed, a shorter way) and do not copy their mistakes. Reply with "
    "the plan only, in the wording of the example's plan, one action per line, and end with [PLAN END]: no preamble, "
    "no notes, no simulation.")


def thinks(call: int, mode: str = "think") -> bool:
    return mode == "think" and call in THINK_CALLS


def audit_prompt(p: str, texts: list) -> str:
    if len(texts) not in (1, SEATS):
        raise ValueError(f"a review reads 1 or {SEATS} plans, not {len(texts)}")
    head, tail = (AUDIT_HEAD_1, AUDIT_TAIL_1) if len(texts) == 1 else (AUDIT_HEAD_N, AUDIT_TAIL_N)
    parts = [head, f"<problem>\n{p}\n</problem>"]
    parts += [f"<plan_{lab}>\n{x}\n</plan_{lab}>" for lab, x in zip(LABELS, texts)]
    parts.append(tail)
    return "\n\n".join(parts)


def revise_prompt(p: str, seat: int, drafts: list, checks: list) -> str:
    if len(drafts) != SEATS or len(checks) != SEATS:
        raise ValueError("a revision reads the four drafts and checks")
    parts = [REVISE_HEAD, f"<problem>\n{p}\n</problem>", f"<your_plan>\n{drafts[seat]}\n</your_plan>",
             f"<your_check>\n{checks[seat]}\n</your_check>"]
    for i in (1, 2, 3):
        o = (seat + i) % SEATS
        parts += [f"<other_plan_{i}>\n{drafts[o]}\n</other_plan_{i}>", f"<other_check_{i}>\n{checks[o]}\n</other_check_{i}>"]
    parts.append(REVISE_TAIL)
    return "\n\n".join(parts)


def _h(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


def request_seed(set_name: str, seed: int, key: str, seat: int, call: int) -> int:
    """Holds no model name: every generation meets the same random numbers on the same game."""
    return int.from_bytes(_h(f"{SALT}|{set_name}|{seed}|{key}|{seat}|{call}")[:4], "big") & 0x7FFFFFFF


def jury_order(set_name: str, seed: int, key: str, reviewer: int) -> list:
    return sorted(range(SEATS), key=lambda t: _h(f"{SALT}|jury|{set_name}|{seed}|{key}|{reviewer}|{t}"))


_VERDICT = re.compile(r"^VERDICT\s+([A-D])\s*:\s*(VALID|INVALID)\b(.*)$", re.IGNORECASE)
_BLOCK = re.compile(r"^PLAN\s+([A-D])\s*:?\s*$", re.IGNORECASE)
_STEP = re.compile(r"\bstep\s*#?\s*(\d+)", re.IGNORECASE)
_FAIL = re.compile(r"\|\s*FAIL\b", re.IGNORECASE)
_OK = re.compile(r"\|\s*OK\b", re.IGNORECASE)
_NUM = re.compile(r"^\s*(\d+)\s*[.)]")
_GOAL = re.compile(r"^GOAL\s*:\s*(NOT\s+MET|NOT\s+CHECKED|MET)\b", re.IGNORECASE)


def _clean(line: str) -> str:
    return line.replace("*", "").replace("#", "").replace("`", "").strip()


def _why(rest: str) -> tuple:
    low = rest.lower()
    m = _STEP.search(low)
    if m:
        return "step", int(m.group(1))
    if "no plan" in low:
        return "noplan", None
    if "goal" in low:
        return "goal", None
    return "other", None


def parse_verdicts(text: str, labels: str) -> dict:
    """{label: {"valid", "why", "step", "how"} or None}: the label's last VERDICT line, else its block's step lines."""
    lines = [_clean(x) for x in (text or "").splitlines()]
    out = {}
    for lab in labels:
        best = None
        for ln in lines:
            m = _VERDICT.match(ln)
            if m and m.group(1).upper() == lab:
                if m.group(2).upper() == "VALID":
                    best = {"valid": True, "why": None, "step": None, "how": "verdict"}
                else:
                    why, step = _why(m.group(3))
                    best = {"valid": False, "why": why, "step": step, "how": "verdict"}
        if best is None:
            inside, n_ok, fail_at, goal = False, 0, None, None
            for ln in lines:
                m = _BLOCK.match(ln)
                if m:
                    if inside:
                        break
                    inside = m.group(1).upper() == lab
                    continue
                if not inside:
                    continue
                if _FAIL.search(ln):
                    if fail_at is None:
                        k = _NUM.match(ln)
                        fail_at = int(k.group(1)) if k else n_ok + 1
                elif _OK.search(ln):
                    n_ok += 1
                else:
                    g = _GOAL.match(ln)
                    if g:
                        goal = " ".join(g.group(1).upper().split())
            if fail_at is not None:
                best = {"valid": False, "why": "step", "step": fail_at, "how": "fallback"}
            elif goal == "MET" and n_ok >= 1:
                best = {"valid": True, "why": None, "step": None, "how": "fallback"}
            elif goal == "NOT MET":
                best = {"valid": False, "why": "goal", "step": None, "how": "fallback"}
        out[lab] = best
    return out


def reviews_of(set_name: str, seed: int, key: str, review_texts: list) -> dict:
    """{reviewer: {final t: verdict or None}} from the four call-4 outputs."""
    out = {}
    for a, txt in enumerate(review_texts):
        order = jury_order(set_name, seed, key, a)
        parsed = parse_verdicts(txt, LABELS)
        out[a] = {t: parsed[LABELS[i]] for i, t in enumerate(order)}
    return out


def commit(reviews: dict, set_name: str, seed: int, key: str) -> dict:
    """The committed final: the highest (share of VALID verdicts, mean score) over the three other reviewers."""
    keys = {}
    for t in range(SEATS):
        got = [1.0 if rv[t]["valid"] else 0.0 for a, rv in reviews.items() if a != t and rv.get(t) is not None]
        keys[t] = (sum(got) / len(got), sum(got) / len(got)) if got else (-1.0, -1.0)
    best = max(keys.values())
    tied = [t for t in range(SEATS) if keys[t] == best]
    t_star = min(tied, key=lambda t: _h(f"{SALT}|commit|{set_name}|{seed}|{key}|{t}"))
    return {"t": t_star, "tie_width": len(tied), "keys": {str(t): keys[t][0] for t in range(SEATS)}}
