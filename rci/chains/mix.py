"""A round's training file: mainly the newest accepted rows, older rounds as replay.

A row is {"row_id", "ids", "spans", "w"} (rci.train.sft's format), optionally with "skel" (the token spans of a call's
skeleton) and "c0" (its loss tokens inside the call segment), which the twin plan reads.

    rows mix (PlanBench)    S = the rows of the rounds since the current generation was released (rows of flat rounds
                            stay and are trained again); replay P = rho |S| rows of the earlier rounds in a hash order;
                            S above max_tokens loses its oldest rounds first; fewer than min_rows rows in S: starved
    token mix (HiddenBench) a token budget B: the newest slice B (1 - rho), the older slices B rho equally, each slice's
                            rows pass after pass in a hash order until its share is reached, at most max_passes passes
    twin plan               the call skeleton is repeated inside its row until call tokens hold call_share of the loss
"""
from __future__ import annotations

import math
from fractions import Fraction

from rci.chains.util import hkey


class MixError(Exception):
    pass


def n_loss(row: dict) -> int:
    return sum(int(n) for _, n in row["spans"])


def n_seq(row: dict) -> int:
    return len(row["ids"])


def steps_for(n_rows: int, batch: int, epochs: float, clip=None) -> int:
    s = math.ceil(epochs * n_rows / batch) if n_rows else 0
    if clip:
        s = max(int(clip[0]), min(int(clip[1]), s))
    return s


def mix_rows(by_round: dict, r: int, r0: int, rho: float, max_tokens: int, min_rows: int, salt: str) -> tuple:
    """(rows or None when starved, meta). by_round: {round: [row]}."""
    s_rounds = list(range(r0 + 1, r + 1))
    p_rounds = list(range(1, r0 + 1))
    s_rows = {q: sorted(by_round.get(q, []), key=lambda x: hkey(salt, "mix", q, x["row_id"])) for q in s_rounds}
    dropped = []
    while sum(n_seq(x) for v in s_rows.values() for x in v) > max_tokens and len(s_rows) > 1:
        q = min(s_rows)
        dropped.append(q)
        s_rows.pop(q)
    s = [x for q in sorted(s_rows) for x in s_rows[q]]
    if sum(n_seq(x) for x in s) > max_tokens:
        keep, tot = [], 0
        for x in s:
            if tot + n_seq(x) > max_tokens:
                break
            keep.append(x)
            tot += n_seq(x)
        s = keep
    meta = {"S_rounds": sorted(s_rows), "S_rows": len(s), "dropped_rounds": dropped}
    if len(s) < min_rows:
        meta["starved"] = True
        return None, meta
    pool = sorted((x for q in p_rounds for x in by_round.get(q, [])), key=lambda x: hkey(salt, "mix", r, x["row_id"]))
    p = pool[:int(rho * len(s))]
    meta.update(P_rows=len(p), tokens=sum(n_seq(x) for x in s + p), starved=False)
    return s + p, meta


def token_shares(tokens_by_slice: dict, latest: int, replay: list, rho: Fraction, budget: int) -> dict:
    if latest in replay:
        raise MixError(f"slice {latest} is the newest slice and a replay slice")
    if not 0 <= rho < 1:
        raise MixError(f"replay share {rho} outside [0, 1)")
    want = {latest: 1 - rho}
    for g in replay:
        want[g] = rho / len(replay)
    live = {g: f for g, f in want.items() if f > 0 and tokens_by_slice.get(g, 0) > 0}
    if not live:
        raise MixError("the slices hold no row")
    tot = sum(live.values())
    out = {g: int(budget * f / tot) for g, f in live.items()}
    top = max(g for g in live)
    out[top] += budget - sum(out.values())
    return out


def mix_tokens(by_slice: dict, latest: int, replay: list, rho, budget: int, salt: str, max_passes: int) -> tuple:
    """(rows, meta): each slice's rows in the order sha256(salt/mix/s<slice>/p<pass>/row id), pass after pass, until
    its share of sequence tokens is reached (the row that reaches it is the last), at most max_passes passes."""
    rho = Fraction(str(rho))
    targets = token_shares({g: sum(n_seq(x) for x in v) for g, v in by_slice.items()}, latest, replay, rho, budget)
    out, meta = [], {"targets": targets, "slices": {}}
    for g, target in sorted(targets.items()):
        rows, got, passes = by_slice[g], 0, 0
        while got < target and passes < max_passes:
            for x in sorted(rows, key=lambda x: hkey(salt, "mix", f"s{g}", f"p{passes}", x["row_id"])):
                out.append(dict(x, row_id=f"{x['row_id']}@p{passes}" if passes else x["row_id"]))
                got += n_seq(x)
                if got >= target:
                    break
            passes += 1
        meta["slices"][g] = {"target": target, "tokens": got, "passes": passes, "capped": got < target}
    meta["under_budget"] = any(v["capped"] for v in meta["slices"].values())
    meta["tokens"] = sum(n_seq(x) for x in out)
    return out, meta


def span_len(spans) -> int:
    return sum(int(n) for _, n in spans)


def prefix(spans, k: int) -> list:
    out = []
    for s, n in spans:
        if k <= 0:
            break
        out.append([s, min(n, k)])
        k -= min(n, k)
    return out


def twin_tokens(loss: int, call: int, share=(1, 4)) -> int:
    """T with 0 <= p (L + T) - q (C0 + T) < q - p: the call segment holds p/q of the loss tokens in integers."""
    p, q = share
    return max(0, (p * loss - q * call) // (q - p))


def twin_plan(rows: list, salt: str, share=(1, 4)) -> tuple:
    """(rows with repeated skeleton spans, meta). Every row with a skeleton gets m0 = T // K copies of it (K the file's
    skeleton tokens); the remainder goes down the hash order, the last one a prefix of a skeleton."""
    L = sum(n_loss(x) for x in rows)
    C0 = sum(int(x.get("c0") or 0) for x in rows)
    sk = {i: x.get("skel") or [] for i, x in enumerate(rows)}
    K = sum(span_len(s) for s in sk.values())
    T = twin_tokens(L, C0, share) if K else 0
    m0, rem = (T // K, T % K) if K else (0, 0)
    add = {i: [list(s) for _ in range(m0) for s in sp] for i, sp in sk.items() if sp and m0}
    for i in sorted((i for i, sp in sk.items() if sp), key=lambda i: hkey(salt, "twin", rows[i]["row_id"])):
        if rem <= 0:
            break
        take = sk[i] if span_len(sk[i]) <= rem else prefix(sk[i], rem)
        add.setdefault(i, []).extend(list(s) for s in take)
        rem -= span_len(take)
    out = []
    for i, x in enumerate(rows):
        y = {k: v for k, v in x.items() if k not in ("skel", "c0")}
        y["spans"] = [list(s) for s in x["spans"]] + add.get(i, [])
        y.setdefault("w", 1.0)
        out.append(y)
    return out, {"L": L, "C0": C0, "K": K, "T": T, "m0": m0, "call_share": [C0 + T, L + T]}
