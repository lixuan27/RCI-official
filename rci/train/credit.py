"""Credit rules for members that share one policy: team reward, Shapley shares, and shares plus citation flow."""
from __future__ import annotations

import random
from collections import defaultdict
from functools import reduce
from math import factorial

from rci.tasks.base import stake

MODES = ("team", "shapley", "shapley_flow")
SET_TASKS = ("audit",)


def _pairs(final: dict | None, exact: bool = True) -> list[tuple[int, int]]:
    out = []
    for p in ((final or {}).get("pairs") or []):
        try:
            a, b = int(p.get("a")), int(p.get("b"))
        except (TypeError, ValueError, AttributeError):
            continue
        out.append((a, b))
    return out


def _shapley(outputs: list, combine, value, empty, max_exact: int = 12, samples: int = 2000) -> list[float]:
    n = len(outputs)
    if n == 0:
        return []
    if n <= max_exact:
        acc = [empty] * (1 << n)
        val = [value(empty)] * (1 << n)
        for m in range(1, 1 << n):
            low = (m & -m).bit_length() - 1
            acc[m] = combine(acc[m & (m - 1)], outputs[low])
            val[m] = value(acc[m])
        w = [factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)]
        return [sum(w[bin(m).count("1")] * (val[m | (1 << i)] - val[m]) for m in range(1 << n) if not m & (1 << i))
                for i in range(n)]
    rng = random.Random(n)
    phi = [0.0] * n
    for _ in range(samples):
        order = list(range(n))
        rng.shuffle(order)
        cur, prev = empty, value(empty)
        for i in order:
            cur = combine(cur, outputs[i])
            v = value(cur)
            phi[i] += (v - prev) / samples
            prev = v
    return phi


def _top_value(acc) -> float:
    return acc[1] / acc[2] if acc[2] else 0.0


def _top_combine(x, y):
    if x[2] == 0 or y[0] > x[0]:
        return y
    if y[2] == 0 or x[0] > y[0]:
        return x
    return (x[0], x[1] + y[1], x[2] + y[2])


COMMITS = ("top_confidence", "vote_stake", "vote_free")


def _answer_key(rec: dict, i: int):
    if rec.get("task") in ("fate", "swe"):
        return ("member", i)
    from rci.ladder.commit import _vote_key

    k = _vote_key(rec, i)
    return k if k is not None else ("member", i)


def _vs_combine(x: dict, y: dict) -> dict:
    out = dict(x)
    for k, (c, st, sc) in y.items():
        c0, st0, sc0 = out.get(k, (0, 0.0, 0.0))
        out[k] = (c0 + c, st0 + st, sc0 + sc)
    return out


def _vs_value(d: dict) -> float:
    if not d:
        return 0.0
    top = max((c, st) for c, st, _ in d.values())
    tied = [(c, sc) for c, st, sc in d.values() if (c, st) == top]
    return sum(sc / c for c, sc in tied) / len(tied)


def _commit_game(rec: dict, cand_idx: list[int], conf: list[float], scores: list[float], commit: str):
    if commit == "top_confidence":
        return [(b, s, 1) for b, s in zip(conf, scores)], _top_combine, _top_value, (0.0, 0.0, 0)
    if commit in ("vote_stake", "vote_free"):
        outs = [{_answer_key(rec, i): (1, b, s)} for i, b, s in zip(cand_idx, conf, scores)]
        return outs, _vs_combine, _vs_value, {}
    raise ValueError(commit)


def game_values(rec: dict, example: dict, commit: str = "top_confidence") -> tuple:
    """(group reward R, Shapley share per member, stated confidence per member or None) of one record."""
    finals = [f for f in rec["finals"] if not f.get("failing")]
    score = rec["score"]
    if rec["task"] in SET_TASKS:
        cons = example["contradictions"]
        if score.get("recall") is not None:
            true = {frozenset((c["edited"], c["partner"])) for c in cons}
            outs = [frozenset(frozenset(p) for p in _pairs(f.get("final")) if p[0] != p[1]) for f in finals]
            pen = score.get("penalty")

            def value(u):
                hit = len(u & true)
                if pen is not None:
                    return (hit - pen * (len(u) - hit)) / max(1, len(true))
                return 0.0 if not hit else 2 * hit / (len(u) + len(true))
        else:
            strict = bool(score.get("strict"))
            outs = []
            for f in finals:
                ps = _pairs(f.get("final"))
                keys = {frozenset(p) for p in ps} if strict else {x for p in ps for x in p}
                outs.append(frozenset(j for j, c in enumerate(cons)
                                      if (frozenset((c["edited"], c["partner"])) if strict else c["edited"]) in keys))

            def value(u):
                return len(u) / max(1, len(cons))
        phi = _shapley(outs, frozenset.union, value, frozenset())
        return value(frozenset().union(*outs)), {f["agent"]: v for f, v in zip(finals, phi)}, None
    per = score.get("per_agent") or []
    idx = {f["agent"]: i for i, f in enumerate(rec["finals"])}
    cand = [f for f in finals if f.get("final") and per[idx[f["agent"]]] is not None]
    from rci.ladder.commit import VERIFIED_TASKS

    stated = any("confidence" in (f.get("final") or {}) for f in cand)
    if rec["task"] in VERIFIED_TASKS:
        outs = [frozenset([f["agent"]]) if float(per[idx[f["agent"]]]) > 0 else frozenset() for f in cand]
        phi = _shapley(outs, frozenset.union, lambda u: 1.0 if u else 0.0, frozenset())
        conf = {f["agent"]: stake(f["final"]) for f in cand} if stated else None
        return (1.0 if any(outs) else 0.0), {f["agent"]: v for f, v in zip(cand, phi)}, conf
    if stated:
        if commit == "vote_free":
            cand = [f for f in cand if not f.get("forced")] or cand
        conf = {f["agent"]: stake(f["final"]) for f in cand}
        outs, combine, value, empty = _commit_game(rec, [idx[f["agent"]] for f in cand],
                                                   [conf[f["agent"]] for f in cand],
                                                   [float(per[idx[f["agent"]]]) for f in cand], commit)
        phi = _shapley(outs, combine, value, empty)
        R = value(reduce(combine, outs, empty))
        return R, {f["agent"]: v for f, v in zip(cand, phi)}, conf
    outs = [float(per[idx[f["agent"]]]) for f in cand]
    phi = _shapley(outs, max, lambda x: x, 0.0)
    return (max(outs) if outs else 0.0), {f["agent"]: v for f, v in zip(cand, phi)}, None


def citation_flow(shares: dict[int, float], finals: list[dict], messages: list[dict], beta: float = 0.5,
                  max_depth: int = 32, first_hop: dict[int, float] | None = None) -> dict[int, float]:
    """Pass a fraction beta of each positive share back along the messages a member's answer cites."""
    flow: dict[int, float] = defaultdict(float)
    by_id = {m["id"]: m for m in messages}
    final_refs = {f["agent"]: [int(r[7:]) for r in ((f.get("final") or {}).get("refs") or [])
                               if isinstance(r, str) and r.startswith("Message") and r[7:].isdigit()
                               and int(r[7:]) in by_id]
                  for f in finals}
    for agent, val in shares.items():
        refs = final_refs.get(agent, [])
        if not refs:
            flow[agent] += val
            continue
        b0 = beta if first_hop is None else first_hop.get(agent, beta)
        flow[agent] += (1 - b0) * val
        frontier = [(m, b0 * val / len(refs)) for m in refs]
        depth = 0
        while frontier and depth < max_depth:
            nxt = []
            for mid, v in frontier:
                msg = by_id[mid]
                parents = [p for p in msg.get("msg_refs", []) if p in by_id]
                keep = v if not parents else (1 - beta) * v
                flow[msg["author"]] += keep
                for p in parents:
                    nxt.append((p, (v - keep) / len(parents)))
            frontier, depth = nxt, depth + 1
        for mid, v in frontier:
            flow[by_id[mid]["author"]] += v
    return dict(flow)


def _silent(rec: dict) -> set[int]:
    out = set()
    for f in rec["finals"]:
        final = f.get("final")
        if not f.get("failing") and (not final or (rec["task"] in SET_TASKS and not _pairs(final))):
            out.add(f["agent"])
    return out


def agent_credits(rec: dict, example: dict, mode: str = "shapley_flow", beta: float = 0.5, lam: float = 0.25,
                  commit: str = "top_confidence", silent_floor: bool = False) -> dict[int, float]:
    """Credit of each member: lam * R + (1 - lam) * N * f_i, with f_i its Shapley share (passed along citations)."""
    agents = [a["id"] for a in rec["agents"] if not a.get("failing")]
    if mode == "team":
        R = float(rec["score"].get("score") or 0.0)
        return {a: R for a in agents}
    if mode not in MODES:
        raise ValueError(mode)
    R, shares, conf = game_values(rec, example, commit)
    if mode == "shapley":
        flow = dict(shares)
    else:
        msgs = (rec.get("method_state") or {}).get("messages") or []
        flow = citation_flow({a: v for a, v in shares.items() if v > 0}, rec["finals"], msgs, beta=beta,
                             first_hop=conf)
        for a, v in shares.items():
            if v < 0:
                flow[a] = flow.get(a, 0.0) + v
    n = max(1, len(agents))
    out = {a: lam * R + (1 - lam) * n * flow.get(a, 0.0) for a in agents}
    if silent_floor:
        quiet = _silent(rec) & set(out)
        spoke = [v for a, v in out.items() if a not in quiet]
        if spoke:
            floor = min(spoke)
            out.update({a: min(out[a], floor) for a in quiet})
    return out
