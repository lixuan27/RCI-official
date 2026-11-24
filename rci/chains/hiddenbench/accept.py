"""The HiddenBench check and rows: a turn is accepted when its committed answer is the base model's full-information
reference and it passes the protocol's guards. No answer label is read: the reference is the base model's own answer.

Per member of a game (record + captured token chains): its generations (one captured span each, aligned with its
model calls and transcript messages), its tentative answers (the submit_answer call of each turn), its final answer and
whether the framework forced it. Guards on a generation: (a) the reasoning closes before the first tool call, (b) at
least `min_tokens` completion tokens, (c) its text holds a private fact of another member (two thirds of the fact's key
words). A game in which any member was forced, or (debate) made more than three calls, gives no row.

Row families (collective chain, three-round debate):
    D   the generation that holds an on-time member's final answer (exactly three calls), final = reference, guards
        a, b, c
    R2  the second generation of an on-time member whose second-turn answer and final are the reference, guards a, b
    P   the first turn that writes every private fact of the member and names no option, credited: another member
        whose final is the reference holds one of those facts in its final generation; the call's parameter values are
        not trained
    S   (single-agent self-training chain) the generation holding a solo member's final, final = reference, guards a, b
Every row ends with <|im_end|>; the tokens of its submit_answer call, less the tokens of the parameter values, are its
call skeleton (the mixer twins it to a quarter of the loss tokens).

The band (per question, over every seat and game): r = share of finals equal to the reference; archive when r > rho_hi,
trainable when 0 < r <= rho_hi, frontier when r = 0 (re-harvested at most max_age times, at most f_max kept). The chain
stops when no question was newly solved or fewer than t_min are trainable.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from rci.chains import health as HL
from rci.chains.util import hkey, order
from rci.tasks.hiddenbench.facts import holds, keys_of, verdict_of, words

SUBMIT = "submit_answer"
ON_TIME_CALLS = 3
CALL_OPEN, THINK_END, IM_END = "<tool_call>", "</think>", "<|im_end|>"
PARAM_XML = re.compile(r"<parameter=[^>\n]+>\s*(.*?)\s*</parameter>", re.S)
PARAM_JSON = (re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"'), re.compile(r'"confidence"\s*:\s*("?[-+0-9.eE]+"?)'))
ANSWER_XML = re.compile(r"<parameter=answer>\s*(.*?)\s*</parameter>", re.S)
DETECT = 2 / 3


def norm(x) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().lower().strip(" .,:;!?\"'`")


def records(root, proto_dir: str, n: int = 4, seeds=None) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(str(root), proto_dir, f"N{n}", "s*", "*.json.gz"))):
        seed = int(Path(p).parent.name[1:])
        if seeds is not None and seed not in seeds:
            continue
        with gzip.open(p, "rt") as f:
            out.append((p, seed, json.load(f)))
    return out


def submits(turn: dict) -> list:
    out = []
    for m in turn.get("messages") or []:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") == SUBMIT:
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out.append(args.get("answer"))
    return out


def generations(transcript: list) -> list:
    """The assistant messages in order: one a model call."""
    out = []
    for turn in transcript or []:
        for m in turn.get("messages") or []:
            if m.get("role") == "assistant":
                out.append(m)
    return out


def member(rec: dict, seat: int) -> dict:
    agent = rec["agents"][seat]
    fin = rec["finals"][seat]
    final = fin.get("final") or {}
    turns = [t for t in (rec.get("transcripts") or {}).get(str(seat)) or (rec.get("transcripts") or {}).get(seat) or []
             if isinstance(t, dict) and t.get("messages")]
    answers = [(submits(t) or [None])[-1] for t in turns]
    gens = generations(turns)
    calls = [c for c in agent.get("calls") or [] if isinstance(c, dict)]
    return {"seat": seat, "final": norm(final.get("answer")) if isinstance(final, dict) else norm(final),
            "forced": bool(fin.get("forced")), "failing": bool(fin.get("failing")), "n_calls": len(calls),
            "ct": [int(c.get("completion_tokens") or 0) for c in calls], "answers": [norm(a) if a else None for a in answers],
            "tentative": next((norm(a) for a in answers if a), None), "gens": gens}


def gen_text(m: dict) -> str:
    return " ".join(str(m.get(k) or "") for k in ("reasoning_content", "content"))


def mine_of(row: dict, n: int, seat: int) -> list:
    return [h for h in range(len(keys_of(row))) if h % n == seat % n]


def holds_other(text: str, row: dict, n: int, seat: int) -> bool:
    said, keys = words(text), keys_of(row)
    return any(holds(said, keys[h], DETECT) for h in range(len(keys)) if h % n != seat % n)


def first_turn_ok(text: str, mine: list, row: dict) -> bool:
    said, keys = words(text), keys_of(row)
    return all(holds(said, keys[h]) for h in mine) and verdict_of(text, row) is None


def game_excluded(ms: list, debate: bool) -> bool:
    return any(m["forced"] for m in ms) or (debate and any(m["n_calls"] > ON_TIME_CALLS for m in ms))


# ------------------------------------------------------------------------------------------------------- tokens
def spans_of(chains: list) -> list:
    """[(chain index, start, n)] of every generation, in order."""
    return [(ci, int(s), int(n)) for ci, c in enumerate(chains) for s, n in c["spans"]]


def call_parts(pieces: list, start: int, answer_value: bool = False, decision_out: bool = False) -> dict | None:
    """Of one generation's token texts (pieces[i] is the text of token start + i): None without a tool call, else the
    absolute indices of the call skeleton (from the first <tool_call> to the end, less the parameter-value tokens) and of
    the value tokens. answer_value: the answer's value tokens join the skeleton; decision_out: in a silent generation
    (only whitespace between the last </think> and the call) the <tool_call> token leaves the skeleton."""
    if CALL_OPEN not in pieces:
        return None
    k = pieces.index(CALL_OPEN)
    seg = pieces[k:]
    text = "".join(seg)
    spans = [(m.start(1), m.end(1)) for m in PARAM_XML.finditer(text) if m.end(1) > m.start(1)]
    if not spans:
        spans = [(m.start(1), m.end(1)) for rx in PARAM_JSON for m in rx.finditer(text) if m.end(1) > m.start(1)]
    ans = [(m.start(1), m.end(1)) for m in ANSWER_XML.finditer(text)] or \
          [(m.start(1), m.end(1)) for m in PARAM_JSON[0].finditer(text)]
    skel, vals, pos = [], [], 0
    for i, p in enumerate(seg):
        a, b = pos, pos + len(p)
        pos = b
        is_val = any(a < e and s < b for s, e in spans)
        if is_val and answer_value and any(a < e and s < b for s, e in ans):
            skel.append(start + k + i)
        elif is_val:
            vals.append(start + k + i)
        else:
            skel.append(start + k + i)
    if decision_out:
        before = "".join(pieces[:k])
        if THINK_END in before and before.rsplit(THINK_END, 1)[1].strip() == "":
            skel = [i for i in skel if i != start + k]
    return {"open": start + k, "skel": skel, "values": vals}


def to_spans(indices) -> list:
    out = []
    for i in sorted(indices):
        if out and out[-1][0] + out[-1][1] == i:
            out[-1][1] += 1
        else:
            out.append([i, 1])
    return out


def make_row(chain: dict, start: int, n: int, piece, rid: str, family: str, sw: dict, im_end: int,
             train_values: bool = True, min_tokens: int = 0, max_len: int = 16384) -> dict | None:
    ids = chain["ids"][:start + n]
    gen = ids[start:start + n]
    if not gen or gen[-1] != im_end or len(ids) > max_len or n < min_tokens:
        return None
    pieces = [piece(t) for t in gen]
    if CALL_OPEN in pieces and THINK_END not in "".join(pieces[:pieces.index(CALL_OPEN)]):
        return None
    cp = call_parts(pieces, start, sw.get("answer_value") and family in sw.get("answer_value_families", ()),
                    sw.get("decision_out") and family in sw.get("decision_families", ()))
    loss = set(range(start, start + n))
    if cp and not train_values:
        loss -= set(cp["values"])
    spans = to_spans(loss)
    row = {"row_id": rid, "ids": ids, "spans": spans, "w": 1.0, "family": family}
    if cp:
        row["skel"] = to_spans(cp["skel"])
        row["c0"] = len([i for i in range(cp["open"], start + n) if i in loss])
    return row


# ------------------------------------------------------------------------------------------------------- the band
def band(outcomes: dict, ref: dict, frontier: dict, rho_hi: float, f_max: int, max_age: int, salt: str) -> dict:
    """outcomes {qid: [final answers over seat x game]}; frontier {qid: age} from the previous round."""
    rep = {"r": {}, "train": [], "archive": [], "frontier": [], "expired": [], "new_solved": 0}
    ages = dict(frontier)
    for q in sorted(outcomes):
        finals = outcomes[q]
        r = sum(1 for a in finals if a == ref[q]) / len(finals)
        rep["r"][q] = r
        rep["new_solved"] += r > 0
        if r > rho_hi:
            rep["archive"].append(q)
            ages.pop(q, None)
        elif r > 0:
            rep["train"].append(q)
            ages.pop(q, None)
        else:
            age = ages[q] + 1 if q in ages else 0
            if age >= max_age:
                rep["expired"].append(q)
                ages.pop(q, None)
            else:
                ages[q] = age
    keep = order(list(ages), salt + "frontier/")[:max(0, int(f_max))]
    rep["frontier_ages"] = {q: ages[q] for q in sorted(keep)}
    rep["frontier"] = sorted(keep)
    return rep


def stagnated(rep: dict, t_min: int) -> str | None:
    if rep["new_solved"] == 0:
        return "frontier stagnation: no question newly solved"
    if len(rep["train"]) < t_min:
        return f"frontier stagnation: {len(rep['train'])} trainable questions, fewer than {t_min}"
    return None


# ------------------------------------------------------------------------------------------------------- rows
def build(recs: list, chains_of, bench: dict, ref: dict, seats: list, kind: str, piece, im_end: int, c: dict,
          r: int, salt: str, sw: dict | None = None, questions=None) -> tuple:
    """(rows, counts) of the given seats. kind "collective" (D, R2, P) or "solo" (S); questions limits the questions
    (the band's trainable set, or the archive's control questions for keep rows)."""
    sw, rows, why = sw or {}, [], Counter()
    debate = kind == "collective"
    fams = c.get("families", ["D", "R2", "P"] if debate else ["S"])
    for p, seed, rec in recs:
        q = str(rec["example_id"])
        if q not in bench or (questions is not None and q not in questions):
            continue
        row = bench[q]
        n_ag = len(rec["agents"])
        ms = [member(rec, j) for j in range(n_ag)]
        if game_excluded(ms, debate):
            why["game excluded"] += 1
            continue
        try:
            chains = chains_of(p)
        except (OSError, ValueError):
            why["no captured chains"] += 1
            continue
        ok_final = {m["seat"] for m in ms if m["final"] == ref[q] and not m["forced"]}
        for m in ms:
            j = m["seat"]
            if j not in seats:
                continue
            sp = spans_of(chains.get(j, []))
            if len(sp) != m["n_calls"] or len(m["gens"]) != m["n_calls"]:
                why["chains do not align with calls"] += 1
                continue
            rid = lambda fam: f"{salt}:r{r}:{q}:s{seed}:a{j}:{fam}"  # noqa: E731

            def put(fam, g, guard_c, train_values=True):
                ci, start, n = sp[g]
                if guard_c and not holds_other(gen_text(m["gens"][g]), row, n_ag, j):
                    why[f"{fam}: no fact of another member"] += 1
                    return
                x = make_row(chains[j][ci], start, n, piece, rid(fam), fam, sw, im_end, train_values,
                             int(c.get("min_tokens", 300)) if fam != "P" else 0, int(c.get("max_len", 16384)))
                if x is None:
                    why[f"{fam}: guard (a), length or end"] += 1
                    return
                x.update(qid=q, seat=j, game=seed)
                rows.append(x)

            if not debate:
                if "S" in fams and j in ok_final:
                    put("S", m["n_calls"] - 1, False)
                continue
            on_time = m["n_calls"] == ON_TIME_CALLS and not m["forced"]
            if "D" in fams and on_time and j in ok_final:
                put("D", ON_TIME_CALLS - 1, True)
            if "R2" in fams and on_time and j in ok_final and len(m["answers"]) > 1 and m["answers"][1] == ref[q]:
                put("R2", 1, False)
            if "P" in fams and m["gens"]:
                mine = mine_of(row, n_ag, j)
                credited = mine and any(
                    any(holds(words(gen_text(o["gens"][-1])), keys_of(row)[h], DETECT) for h in mine)
                    for o in ms if o["seat"] != j and o["seat"] in ok_final and o["gens"])
                if mine and first_turn_ok(str(m["gens"][0].get("content") or ""), mine, row) and credited:
                    put("P", 0, False, train_values=False)
    rows = cap(rows, c, salt, r)
    return rows, dict(why, rows=len(rows), families=dict(Counter(x["family"] for x in rows)))


def cap(rows: list, c: dict, salt: str, r: int) -> list:
    """At most per_question rows of a family a question (hash order), then P rows cut to p_share of the loss tokens."""
    per = int(c.get("per_question", 4))
    by = defaultdict(list)
    for x in rows:
        by[(x["qid"], x["family"])].append(x)
    kept = []
    for k in sorted(by):
        kept += sorted(by[k], key=lambda x: hkey(salt, "cap", r, x["row_id"]))[:per]
    share = c.get("p_share")
    if share is not None:
        loss = lambda xs: sum(n for x in xs for _, n in x["spans"])  # noqa: E731
        ps = sorted((x for x in kept if x["family"] == "P"), key=lambda x: hkey(salt, "p", r, x["row_id"]))
        while ps and loss(ps) > float(share) * loss(kept):
            gone = ps.pop()
            kept = [x for x in kept if x is not gone]
    return sorted(kept, key=lambda x: hkey(salt, "rows", r, x["row_id"]))


# ------------------------------------------------------------------------------------------------------- health
def cell_stats(recs: list, bench: dict, control: set, z_tokens: int = 120, min_tokens: int = 300) -> dict:
    """Group rates (hidden-type and control-type, per question over seeds), the budget a game and the ledger:
    z (on-time third calls under z_tokens completion tokens), f (forced members), r (moves whose final generation
    passes guards b and c, of all moves; a move is a final answer that differs from the member's first one)."""
    from rci.ladder.commit import metrics

    per_q = defaultdict(list)
    budget, forced, n_m, c3, moves, counted = [], 0, 0, [], 0, 0
    for _, _, rec in recs:
        q = str(rec["example_id"])
        if q not in bench:
            continue
        v = metrics(rec)["vote_stake"]
        per_q[q].append(0.0 if v is None else float(v))
        ms = [member(rec, j) for j in range(len(rec["agents"]))]
        budget.append(sum(sum(m["ct"]) for m in ms))
        for m in ms:
            n_m += 1
            forced += m["forced"]
            if not m["forced"] and m["n_calls"] >= ON_TIME_CALLS:
                c3.append(m["ct"][ON_TIME_CALLS - 1])
            if not m["forced"] and m["tentative"] and m["final"] and m["tentative"] != m["final"]:
                moves += 1
                g = m["gens"][-1] if m["gens"] else {}
                if m["ct"] and m["ct"][-1] >= min_tokens and holds_other(gen_text(g), bench[q], len(ms), m["seat"]):
                    counted += 1
    rate = {q: sum(v) / len(v) for q, v in per_q.items()}
    hid = [v for q, v in rate.items() if q not in control]
    ctl = [v for q, v in rate.items() if q in control]
    return {"games": len(budget), "per_item": rate, "rate": sum(rate.values()) / len(rate) if rate else None,
            "hidden": sum(hid) / len(hid) if hid else None, "control": sum(ctl) / len(ctl) if ctl else None,
            "budget_per_game": sum(budget) / len(budget) if budget else None,
            "ledger": {"z": sum(x < z_tokens for x in c3) / len(c3) if c3 else None,
                       "f": forced / n_m if n_m else None, "r": counted / moves if moves else None}}


def health(cand: dict, base: dict, h: dict) -> list:
    ratio = HL.ratio(cand["budget_per_game"], base["budget_per_game"])
    lo, hi = h["budget_band"]
    lines = [HL.within("budget a game / base", ratio, lo if h.get("budget_low_gates", True) else None, hi),
             HL.at_most("ledger f (forced finals)", cand["ledger"]["f"], h["f_max"])]
    if h.get("z_max") is not None:
        lines.append(HL.at_most("ledger z (near-zero final reasoning)", cand["ledger"]["z"], h["z_max"]))
    if h.get("r_min") is not None:
        lines.append(HL.at_least("ledger r (guarded moves)", cand["ledger"]["r"], h["r_min"]))
    if cand.get("control") is not None and base.get("control") is not None:
        lines.append(HL.at_least("control-task rate", cand["control"], base["control"] - h["control_floor"]))
    return lines
