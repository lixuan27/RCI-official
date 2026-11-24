"""Structure reader for blackboard records: who completes the facts, through a star, a two-level tree or a chain."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rci.ladder.commit import vote_stake
from rci.ladder.records import records
from rci.paths import data_dir, load_benchmark
from rci.tasks.hiddenbench.facts import holds, keys_of, message_text, words
from rci.tasks.hiddenbench.task import FORWARD_TITLE

CLASSES = ("star", "tree", "chain", "none")
RELAY = ("tree", "chain")


def trace_of(rec: dict, row: dict) -> dict:
    """Messages with the private facts they hold, each member's reads (pushes excluded) and full pushes."""
    n = len(rec["agents"])
    keys = keys_of(row)
    ms = rec["method_state"]
    msgs = {}
    for m in ms.get("messages", []):
        w = words(message_text(m))
        msgs[m["id"]] = {"author": m["author"], "step": m["step"], "forward": m["sub_task"] == FORWARD_TITLE,
                         "facts": frozenset(h for h, k in enumerate(keys) if holds(w, k))}
    last = {a["id"]: max((t["step"] for t in a.get("turns") or []), default=0) for a in rec["agents"]}
    reads: dict = {j: [] for j in last}
    pushes: dict = {j: [] for j in last}
    events = ms.get("events", [])
    for e in events:
        j = e.get("reader")
        if j in last and e["op"] == "push" and e.get("view") == "full":
            pushes[j] += [(e["step"], i) for i in e.get("ids") or [] if i in msgs and msgs[i]["author"] != j]
    pushed_at = {j: {} for j in last}
    for j, ps in pushes.items():
        for s, i in ps:
            pushed_at[j].setdefault(i, s)
    for e in events:
        j = e.get("reader")
        if j not in last or e["step"] >= last[j]:
            continue
        if e["op"] == "search":
            ids = [r["id"] for r in e.get("results") or []]
        elif e["op"] == "visit":
            ids = [int(x) for x in e.get("ids") or [] if str(x).isdigit()]
        else:
            continue
        reads[j] += [(e["step"], i) for i in ids if i in msgs and msgs[i]["author"] != j
                     and not pushed_at[j].get(i, e["step"] + 1) <= e["step"]]
    return {"n": n, "owner": [h % n for h in range(len(keys))], "msgs": msgs, "reads": reads, "pushes": pushes,
            "last": last}


def fact_depths(tr: dict) -> tuple:
    """Depth of every (message, fact) it is credited with: 1 for its owner's, else 1 + the depth of its source."""
    msgs, owner = tr["msgs"], tr["owner"]
    seen = {j: [(s, i, False) for s, i in tr["reads"][j]] + [(s, i, True) for s, i in tr["pushes"].get(j, [])]
            for j in tr["reads"]}
    d, unsourced, pushed = {}, set(), set()
    for mid in sorted(msgs, key=lambda i: (msgs[i]["step"], i)):
        m = msgs[mid]
        a = m["author"]
        for h in m["facts"]:
            if owner[h] == a:
                d[(mid, h)] = 1
                continue
            src = [(d[(i, h)], push) for s, i, push in seen.get(a, [])
                   if i != mid and (s <= m["step"] if push else s < m["step"]) and (i, h) in d
                   and msgs[i]["author"] != a]
            if src:
                d[(mid, h)] = 1 + min(x for x, _ in src)
                if all(push for _, push in src):
                    pushed.add((mid, h))
            else:
                unsourced.add((mid, h))
    return d, unsourced, pushed


def classify(tr: dict) -> dict:
    """The first complete reader and the record's class: star, tree, chain, or none."""
    msgs, owner = tr["msgs"], tr["owner"]
    d, unsourced, _ = fact_depths(tr)
    facts = {i: frozenset(h for h in m["facts"] if (i, h) in d) for i, m in msgs.items()}
    allf = set(range(len(owner)))
    best = None
    for j in sorted(tr["reads"]):
        has = {h for h in allf if owner[h] == j}
        if has == allf:
            continue
        for s, i in sorted(tr["reads"][j]):
            has |= facts[i]
            if has >= allf:
                if best is None or (s, j) < best:
                    best = (s, j)
                break
    out = {"analyst": None, "class": "none", "class_by_fact": "none", "unsourced": len(unsourced), "by_cover": False}
    if best is None:
        return out
    t, a = best
    lack = {h for h in allf if owner[h] != a}
    read = sorted({i for s, i in tr["reads"][a] if s <= t})
    useful = [i for i in read if facts[i] & lack]
    src_depth = {h: min(d[(i, h)] for i in useful if h in facts[i]) for h in lack}
    depth = {i: max((d[(i, h)] for h in facts[i] & lack if d[(i, h)] == src_depth[h]), default=None) for i in useful}

    def covers(ids) -> bool:
        got = set()
        for i in ids:
            got |= facts[i]
        return lack <= got

    necessary = [i for i in useful if not covers([k for k in useful if k != i])]
    deepest = max(src_depth.values())
    by_fact = "star" if deepest == 1 else "tree" if deepest == 2 else "chain"
    if deepest == 1:
        cls = "star"
    elif any(depth[i] >= 3 for i in necessary):
        cls = "chain"
    elif necessary and max(depth[i] for i in necessary) == 2:
        cls = "tree"
    else:
        out["by_cover"] = True
        cls = by_fact
    out.update({"analyst": a, "completed_at": t, "class": cls, "class_by_fact": by_fact, "necessary": necessary,
                "relay_authors": [msgs[i]["author"] for i in necessary if depth[i] >= 2]})
    return out


def relays_of(tr: dict) -> list:
    """Forwards credited with another seat's fact that their author had read before posting."""
    d, _, pushed = fact_depths(tr)
    msgs, owner = tr["msgs"], tr["owner"]
    return [i for i, m in sorted(msgs.items()) if m["forward"]
            and any(owner[h] != m["author"] and (i, h) in d and (i, h) not in pushed for h in m["facts"])]


def budgets(rec: dict, start: int = 1) -> list:
    """Each member's reading budget after the recorded transfers."""
    total = [start] * len(rec["agents"])
    for e in rec["method_state"].get("events", []):
        if e["op"] == "give":
            total[e["from"]] -= e["n"]
            total[e["to"]] += e["n"]
    return total


def simulate(rng: random.Random, cap: int = 0, ko: bool = False, n: int = 4, n_facts: int = 4, budget: int = 1,
             planning_steps: int = 2, max_steps: int = 24) -> dict:
    """One record of the null policy: keep or give the budget at random, forward own facts, then read at random."""
    owner = [h % n for h in range(n_facts)]
    held = {j: {h for h in range(n_facts) if owner[h] == j} for j in range(n)}
    own = {j: frozenset(held[j]) for j in range(n)}
    total = [budget] * n
    order = list(range(n))
    rng.shuffle(order)
    for j in order:
        if rng.random() < 0.5:
            continue
        to = rng.choice([k for k in range(n) if k != j])
        if cap and total[to] + budget > cap:
            continue
        total[j] -= budget
        total[to] += budget
    msgs = {j: {"author": j, "step": 0, "facts": own[j], "forward": True} for j in range(n)}
    reads = {j: [] for j in range(n)}
    seen = {j: set() for j in range(n)}
    post: dict = {}

    def forward(j: int, s: int) -> None:
        facts = frozenset(held[j])
        if not (ko and facts - own[j]):
            msgs[len(msgs)] = {"author": j, "step": s, "facts": facts, "forward": True}
        del post[j]

    for s in range(planning_steps, max_steps):
        for j in sorted(post):
            if post[j] == s:
                forward(j, s)
        any_read = False
        for j in range(n):
            if len(seen[j]) >= total[j]:
                continue
            cand = [i for i, m in msgs.items() if m["author"] != j and m["step"] < s and i not in seen[j]]
            if not cand:
                continue
            i = rng.choice(cand)
            seen[j].add(i)
            reads[j].append((s, i))
            held[j] |= msgs[i]["facts"]
            post[j] = s + 1
            any_read = True
        if not any_read and not post and s > 0:
            break
    return {"n": n, "owner": owner, "msgs": msgs, "reads": reads, "pushes": {j: [] for j in range(n)},
            "last": {j: max_steps for j in range(n)}}


def null_shares(question: str, protocol: str, sims: int, **params) -> dict:
    """Class shares of the null policy on one question, from a seed fixed by the question and the protocol name."""
    rng = random.Random(int(hashlib.sha256(f"{question}/{protocol}".encode()).hexdigest(), 16))
    cls = Counter(classify(simulate(rng, **params))["class"] for _ in range(sims))
    return {k: cls[k] / sims for k in CLASSES}


def clustered(values: dict) -> dict:
    """Mean of per-question means and its question-clustered standard error."""
    qs = [q for q, v in values.items() if v]
    if len(qs) < 2:
        return {"mean": None, "se": None, "questions": len(qs)}
    per = [sum(values[q]) / len(values[q]) for q in qs]
    m = sum(per) / len(per)
    sd = math.sqrt(sum((x - m) ** 2 for x in per) / (len(per) - 1))
    return {"mean": m, "se": sd / math.sqrt(len(per)), "questions": len(qs)}


def protocol_params(rec: dict) -> dict:
    """Null-policy parameters read from a record's settings."""
    cfg = rec.get("config") or {}
    kw = cfg.get("protocol_kwargs") or cfg.get("method_kwargs") or {}
    tk = rec.get("task_kwargs") or {}
    rb = kw.get("read_budget") or [1]
    return {"cap": int(kw.get("budget_cap") or 0), "ko": bool(tk.get("forward_own_only")), "n": len(rec["agents"]),
            "budget": int(rb[0]) if rb and rb[0] is not None and rb[0] >= 0 else 1,
            "planning_steps": int(kw.get("planning_steps") or 0), "max_steps": int(cfg.get("max_steps") or 24)}


def read_cell(recs: dict, bench: dict, hubs: bool) -> dict:
    """Class shares, relays, transfers and hubs of one cell, with question-clustered SEs."""
    ind = defaultdict(lambda: defaultdict(list))
    for q, rr in recs.items():
        row = bench.get(q)
        if row is None:
            continue
        for r in rr:
            tr = trace_of(r, row)
            c = classify(tr)
            for k in CLASSES:
                ind[k][q].append(float(c["class"] == k))
            ind["relay_structure"][q].append(float(c["class"] in RELAY))
            ind["has_relay"][q].append(float(bool(relays_of(tr))))
            ind["answer"][q].append(float(vote_stake(r)))
            if hubs:
                gifts = [e for e in r["method_state"].get("events", []) if e["op"] == "give"]
                ind["transfer"][q].append(float(bool(gifts)))
                ind["hub_formed"][q].append(float(max(budgets(r)) >= len(r["agents"]) - 1))
    return {k: clustered(v) for k, v in ind.items()}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read the structure of blackboard records (first complete reader, star, "
                                             "two-level tree or chain), against a random null policy.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("read")
    r.add_argument("--spec", required=True, help="grading spec (the format of rci.ladder.grade)")
    r.add_argument("--task", required=True)
    r.add_argument("--cells", default=None, help="comma list of cell labels (default: every non-pool cell)")
    r.add_argument("--data", default=None, help="directory of benchmark.json (default: $RCI_DATA/hiddenbench)")
    r.add_argument("--null", type=int, default=0, help="simulations of the null policy per question (e.g. 10000)")
    r.add_argument("--hubs", action="store_true", help="also report transfers and hubs")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tspec = json.loads(Path(a.spec).read_text())[a.task]
    data = Path(a.data) if a.data else data_dir("hiddenbench")
    bench = {f"hb_{int(x['id']):02d}": x for x in load_benchmark(data)}
    pool_keys = {tuple(p) for p in tspec["pool"]}
    wanted = set(a.cells.split(",")) if a.cells else None
    res = {"task": a.task, "cells": {}}
    for c in tspec["cells"]:
        if (c["root"], c["protocol"], c["n"]) in pool_keys or (wanted and c["label"] not in wanted):
            continue
        recs = records(c["root"], c["protocol"], c["n"])
        row = read_cell(recs, bench, a.hubs)
        if a.null and recs:
            params = protocol_params(next(iter(recs.values()))[0])
            name = json.dumps(params, sort_keys=True)
            shares = [null_shares(q, name, a.null, n_facts=len(keys_of(bench[q])), **params) for q in recs if q in bench]
            row["null"] = {k: sum(s[k] for s in shares) / len(shares) for k in CLASSES}
            row["null"]["relay_structure"] = row["null"]["tree"] + row["null"]["chain"]
            row["null"]["params"] = params
        res["cells"][c["label"]] = row
        rs = row.get("relay_structure") or {}
        print(f"[structure] {c['label']}: relay structure {rs.get('mean')} star {(row.get('star') or {}).get('mean')}",
              flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"structure_{a.task}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
