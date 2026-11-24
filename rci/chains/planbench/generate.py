"""New Blocksworld problems in the format of PlanBench, none of them an official problem up to renaming of the blocks.

Every (initial state, goal state) pair of 4 and 5 blocks with the hand empty is enumerated (73 and 501 states) with its
optimal plan length (backward BFS over the exact state graph). The goal keeps only the goal state's `on` facts, as the
official instances do; a pair whose goal already holds is not a problem. A problem is excluded when its key up to a
renaming of the blocks (iso_key) is one of the official instances or one-shot examples (official_iso_digests.json).

A split of size S with block mix {4: w4, 5: w5} gets exactly S w4 / (w4 + w5) four-block problems; the whole split
follows the official plan-length histogram, its four-block part the shares of the free four-block pairs. Problems are
drawn uniformly without replacement inside each (blocks, plan length) cell from a SHA-256 stream, and every problem
belongs to one split only. The one-shot example of a problem comes from a separate pool of its own block count.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from rci.chains.planbench import domain as D
from rci.chains.planbench import text as T
from rci.chains.util import DRng

N_BLOCKS = (4, 5)
OFFICIAL_LENGTH_HIST = {2: 30, 4: 56, 6: 114, 8: 139, 10: 113, 12: 46, 14: 1, 16: 1}
# free pairs (not an official problem up to renaming) per block count and plan length; 4 blocks is recomputed by the
# tests, 5 blocks takes minutes (free_cells(5))
FREE_CELLS = {4: {2: 36, 4: 72, 6: 192, 8: 120, 10: 144, 12: 24},
              5: {2: 4880, 4: 10340, 6: 24240, 8: 46200, 10: 63120, 12: 53760, 14: 29520, 16: 8040}}
EXAMPLE_POOL = {4: 24, 5: 400}
DIGESTS = Path(__file__).resolve().parent / "official_iso_digests.json"


class GenError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def all_states(n: int) -> tuple:
    """Every hand-empty state of n blocks as a sorted tuple of stacks (bottom to top)."""
    blocks = D.LETTERS[:n]
    out = set()
    for perm in itertools.permutations(blocks):
        for mask in range(1 << (n - 1)):
            stacks, cur = [], [perm[0]]
            for i in range(1, n):
                if mask >> (i - 1) & 1:
                    stacks.append(tuple(cur))
                    cur = [perm[i]]
                else:
                    cur.append(perm[i])
            stacks.append(tuple(cur))
            out.add(tuple(sorted(stacks)))
    return tuple(sorted(out))


def state_facts(stacks) -> frozenset:
    facts = {("handempty",)}
    for st in stacks:
        facts.add(("ontable", st[0]))
        facts.add(("clear", st[-1]))
        for lower, upper in zip(st, st[1:]):
            facts.add(("on", upper, lower))
    return frozenset(facts)


def goal_facts(stacks) -> frozenset:
    return frozenset(("on", upper, lower) for st in stacks for lower, upper in zip(st, st[1:]))


@lru_cache(maxsize=None)
def length_table(n: int) -> tuple:
    """cols[g][i]: optimal plan length from state i to the goal of state g (0 when the goal is empty or holds)."""
    states = all_states(n)
    inits = [state_facts(s) for s in states]
    objects = tuple(D.LETTERS[:n])
    nodes = sorted(D.reachable(inits[0]), key=lambda st: sorted(st))
    index = {st: k for k, st in enumerate(nodes)}
    rev = [[] for _ in nodes]
    holds: dict = {}
    for k, st in enumerate(nodes):
        for _, t in D.successors(st, objects):
            rev[index[t]].append(k)
        for f in st:
            if f[0] == "on":
                holds.setdefault(f, set()).add(k)
    init_idx = [index[s] for s in inits]
    cols = []
    for s in states:
        gf = goal_facts(s)
        if not gf:
            cols.append(bytes(len(states)))
            continue
        dist = [-1] * len(nodes)
        frontier = sorted(set.intersection(*[holds[f] for f in gf]))
        for k in frontier:
            dist[k] = 0
        d = 0
        while frontier:
            d += 1
            nxt = []
            for k in frontier:
                for p in rev[k]:
                    if dist[p] < 0:
                        dist[p] = d
                        nxt.append(p)
            frontier = nxt
        cols.append(bytes(dist[i] for i in init_idx))
    return tuple(cols)


@lru_cache(maxsize=None)
def cells(n: int) -> dict:
    """plan length -> [(initial index, goal index)] of the non-trivial pairs of n blocks."""
    out: dict = {}
    for g, col in enumerate(length_table(n)):
        for i, length in enumerate(col):
            if length:
                out.setdefault(length, []).append((i, g))
    return {k: out[k] for k in sorted(out)}


def canonical_key(initial, goal) -> str:
    return (";".join(sorted(" ".join(f) for f in initial)) + "=>" + ";".join(sorted(" ".join(f) for f in goal)))


def iso_key(initial, goal) -> str:
    """The smallest canonical key over every renaming of the blocks."""
    ini, gl = [tuple(f) for f in initial], [tuple(f) for f in goal]
    objs = sorted({a for f in ini + gl for a in f[1:]})
    best = None
    for perm in itertools.permutations(objs):
        m = dict(zip(objs, perm))
        k = canonical_key([(f[0],) + tuple(m[a] for a in f[1:]) for f in ini],
                          [(f[0],) + tuple(m[a] for a in f[1:]) for f in gl])
        if best is None or k < best:
            best = k
    return best


@lru_cache(maxsize=1)
def official_digests() -> frozenset:
    doc = json.loads(DIGESTS.read_text(encoding="utf-8"))
    dig = frozenset(doc["iso_digests"])
    if len(dig) != doc["n"] or not dig:
        raise GenError(f"{DIGESTS.name} is damaged")
    return dig


@lru_cache(maxsize=1 << 18)
def is_official(initial: frozenset, goal: frozenset) -> bool:
    return hashlib.sha256(iso_key(initial, goal).encode()).hexdigest() in official_digests()


def free_cells(n: int) -> dict:
    states = all_states(n)
    return {length: sum(not is_official(state_facts(states[i]), goal_facts(states[g])) for i, g in pairs)
            for length, pairs in cells(n).items()}


def length_quota(size: int, hist) -> dict:
    """plan length -> count, proportional to hist by largest remainders (ties to the shorter plan)."""
    total = sum(hist.values())
    base = {k: size * v // total for k, v in hist.items()}
    for k in sorted(hist, key=lambda k: (-(size * hist[k] % total), k))[:size - sum(base.values())]:
        base[k] += 1
    return {k: base[k] for k in sorted(base) if base[k]}


def block_counts(size: int, mix: dict) -> dict:
    w = {n: Fraction(str(mix.get(n, mix.get(str(n), 0)))) for n in N_BLOCKS}
    share = w[4] / (w[4] + w[5])
    x = size * share
    if x.denominator != 1:
        raise GenError(f"size {size} x four-block share {share} is not a whole number")
    return {4: int(x), 5: size - int(x)}


def quota(size: int, mix: dict) -> dict:
    counts = block_counts(size, mix)
    whole = length_quota(size, OFFICIAL_LENGTH_HIST)
    q4 = length_quota(counts[4], FREE_CELLS[4]) if counts[4] else {}
    q5 = {k: whole.get(k, 0) - q4.get(k, 0) for k in sorted(set(whole) | set(q4))}
    if any(c < 0 for c in q5.values()):
        raise GenError(f"a split of {size} with mix {mix} needs more four-block problems than the histogram holds")
    return {4: q4, 5: {k: c for k, c in q5.items() if c}}


class Generator:
    """Draws problems cell by cell; a problem is used at most once across the example pool and every split."""

    def __init__(self, seed: str, example_pool=None):
        self.seed = str(seed)
        self.cursor: dict = {}
        self.orders: dict = {}
        self.used: set = set()
        self.examples = {n: [] for n in N_BLOCKS}
        for n, size in (example_pool or EXAMPLE_POOL).items():
            n = int(n)
            for length, c in length_quota(int(size), FREE_CELLS[n]).items():
                self.examples[n] += [self.take(n, length, with_plan=True) for _ in range(c)]

    def _order(self, n: int, length: int) -> list:
        key = (n, length)
        if key not in self.orders:
            pairs = list(cells(n)[length])
            DRng(self.seed, "cell", n, length).shuffle(pairs)
            self.orders[key] = pairs
            self.cursor[key] = 0
        return self.orders[key]

    def take(self, n: int, length: int, with_plan: bool = False) -> dict:
        pairs = self._order(n, length)
        states = all_states(n)
        while self.cursor[(n, length)] < len(pairs):
            i, g = pairs[self.cursor[(n, length)]]
            self.cursor[(n, length)] += 1
            ini, gl = state_facts(states[i]), goal_facts(states[g])
            key = canonical_key(ini, gl)
            if key in self.used or is_official(ini, gl):
                continue
            self.used.add(key)
            out = {"n_blocks": n, "plan_length": length, "initial": [list(f) for f in T.sorted_facts(ini)],
                   "goal": [list(f) for f in T.sorted_facts(gl)], "key": key}
            if with_plan:
                plan = D.plan_optimal(ini, gl)
                if plan is None or len(plan) != length:
                    raise GenError(f"planner and table disagree on a {n}-block pair")
                out["plan"] = [list(a) for a in plan]
            return out
        raise GenError(f"no free {n}-block problem of plan length {length} is left")

    def split(self, name: str, size: int, mix: dict) -> list:
        q = quota(int(size), mix)
        rng = DRng(self.seed, "split", name)
        picked = []
        for n in N_BLOCKS:
            for length, c in q[n].items():
                for _ in range(c):
                    pool = self.examples[n]
                    picked.append((self.take(n, length), pool[rng.below(len(pool))]))
        rng.shuffle(picked)
        return [row(inst, ex, f"{name}-{k:05d}", name) for k, (inst, ex) in enumerate(picked)]


def facts(x) -> frozenset:
    return frozenset(tuple(f) for f in x)


def row(inst: dict, ex: dict, rid: str, split: str) -> dict:
    query = T.render_query(facts(ex["initial"]), facts(ex["goal"]), [tuple(a) for a in ex["plan"]],
                           facts(inst["initial"]), facts(inst["goal"]))
    return {"id": rid, "split": split, **{k: inst[k] for k in ("n_blocks", "plan_length", "initial", "goal", "key")},
            "example": {k: ex[k] for k in ("n_blocks", "plan_length", "initial", "goal", "plan", "key")}, "query": query}


def build(spec: list, seed: str, example_pool=None) -> dict:
    """{name: rows} for the requests [(name, size, mix)] in order; prefix-stable for a fixed seed."""
    gen = Generator(seed, example_pool)
    out = {}
    for name, size, mix in spec:
        if name in out:
            raise GenError(f"split {name} twice")
        out[name] = gen.split(name, size, mix)
    return out
