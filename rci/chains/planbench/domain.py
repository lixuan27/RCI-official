"""PlanBench Blocksworld: the four STRIPS actions of the official domain, the plan simulator and a BFS planner.

A fact is a tuple (predicate, *blocks), a state a frozenset of facts, an action a tuple (name, *blocks); blocks are the
PDDL letters a, b, c, ... A plan is correct when every action is applicable in sequence and every goal fact holds at
the end; an empty plan is never correct.
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

LETTERS = tuple("abcdefghijkl")
COLOURS = ("red", "blue", "orange", "yellow", "white", "magenta", "black", "cyan", "green", "violet", "silver", "gold")
OBJECT_NAME = {x: f"{c} block" for x, c in zip(LETTERS, COLOURS)}
ACTION_TEXT = {"pick-up": "pick up the {}", "put-down": "put down the {}", "stack": "stack the {} on top of the {}",
               "unstack": "unstack the {} from on top of the {}"}
PREDICATE_TEXT = {"ontable": "the {} is on the table", "clear": "the {} is clear", "handempty": "the hand is empty",
                  "holding": "the hand is currently holding {}", "on": "the {} is on top of the {}"}
PREDICATE_ARITY = {"clear": 1, "ontable": 1, "handempty": 0, "holding": 1, "on": 2}
MAX_BLOCKS = 6


class Schema(NamedTuple):
    params: tuple
    pre: tuple
    add: tuple
    dele: tuple


SCHEMAS = {
    "pick-up": Schema(("?x",), (("clear", "?x"), ("ontable", "?x"), ("handempty",)), (("holding", "?x"),),
                      (("clear", "?x"), ("ontable", "?x"), ("handempty",))),
    "put-down": Schema(("?x",), (("holding", "?x"),), (("clear", "?x"), ("handempty",), ("ontable", "?x")),
                       (("holding", "?x"),)),
    "stack": Schema(("?x", "?y"), (("clear", "?y"), ("holding", "?x")),
                    (("handempty",), ("clear", "?x"), ("on", "?x", "?y")), (("clear", "?y"), ("holding", "?x"))),
    "unstack": Schema(("?x", "?y"), (("on", "?x", "?y"), ("clear", "?x"), ("handempty",)),
                      (("holding", "?x"), ("clear", "?y")), (("on", "?x", "?y"), ("clear", "?x"), ("handempty",))),
}
ACTIONS = tuple(SCHEMAS)
ARITY = {n: len(s.params) for n, s in SCHEMAS.items()}


class PlanError(ValueError):
    def __init__(self, msg, action=None, missing=()):
        super().__init__(msg)
        self.action = action
        self.missing = tuple(missing)


def fmt(t) -> str:
    return "(" + " ".join(t) + ")"


def ground(action) -> tuple:
    action = tuple(action)
    s = SCHEMAS.get(action[0]) if action else None
    if s is None:
        raise PlanError(f"unknown action {action[:1]!r}", action)
    if len(action) - 1 != len(s.params) or not all(isinstance(a, str) and a for a in action[1:]):
        raise PlanError(f"{fmt(tuple(map(str, action)))}: {action[0]} takes {len(s.params)} argument(s)", action)
    b = dict(zip(s.params, action[1:]))
    g = lambda fs: tuple((f[0],) + tuple(b[x] for x in f[1:]) for f in fs)  # noqa: E731
    return g(s.pre), g(s.add), g(s.dele)


def apply(state, action) -> frozenset:
    pre, add, dele = ground(action)
    missing = tuple(f for f in pre if f not in state)
    if missing:
        raise PlanError(f"{fmt(action)} is not applicable: " + ", ".join(fmt(f) for f in missing) + " not true",
                        tuple(action), missing)
    return frozenset((set(state) - set(dele)) | set(add))


def validate(initial, goal, plan, allow_empty: bool = False) -> tuple:
    """(ok, first error or None, actions executed before the first error)."""
    plan = [tuple(a) for a in plan]
    if not plan and not allow_empty:
        return False, "empty plan", 0
    state = frozenset(initial)
    for i, a in enumerate(plan):
        try:
            state = apply(state, a)
        except PlanError as e:
            return False, f"step {i + 1}: {e}", i
    unmet = sorted(f for f in goal if f not in state)
    if unmet:
        return False, "goal not reached: " + ", ".join(fmt(f) for f in unmet) + " not true", len(plan)
    return True, None, len(plan)


def first_failure(initial, goal, plan) -> tuple:
    """("step", k) for the first inapplicable action (1-based), ("goal", None), ("noplan", None) or ("valid", None)."""
    ok, err, steps = validate(initial, goal, plan)
    if ok:
        return "valid", None
    if not plan:
        return "noplan", None
    if err.startswith("step"):
        return "step", steps + 1
    return "goal", None


def objects_of(*fact_sets) -> tuple:
    return tuple(sorted({a for fs in fact_sets for f in fs for a in f[1:]}))


def ground_actions(objects) -> tuple:
    out = []
    for name, s in SCHEMAS.items():
        out += [(name, x) for x in objects] if len(s.params) == 1 else [(name, x, y) for x in objects for y in objects]
    return tuple(out)


_SUCC: dict = {}


def successors(state, objects) -> tuple:
    key = (objects, state)
    hit = _SUCC.get(key)
    if hit is None:
        hit = []
        for a in ground_actions(objects):
            pre, add, dele = ground(a)
            if all(f in state for f in pre):
                hit.append((a, frozenset((set(state) - set(dele)) | set(add))))
        hit = _SUCC[key] = tuple(hit)
    return hit


def plan_optimal(initial, goal, max_blocks: int = MAX_BLOCKS):
    """A shortest plan by breadth-first search (actions in schema order, then by arguments), or None."""
    initial, goal = frozenset(initial), frozenset(goal)
    objects = objects_of(initial, goal)
    if len(objects) > max_blocks:
        raise ValueError(f"the BFS planner takes at most {max_blocks} blocks")
    if goal <= initial:
        return []
    parent = {initial: None}
    queue = deque([initial])
    while queue:
        s = queue.popleft()
        for a, t in successors(s, objects):
            if t in parent:
                continue
            parent[t] = (s, a)
            if goal <= t:
                plan = []
                while parent[t] is not None:
                    t, a = parent[t]
                    plan.append(a)
                return plan[::-1]
            queue.append(t)
    return None


def reachable(initial) -> set:
    initial = frozenset(initial)
    objects = objects_of(initial)
    seen, queue = {initial}, deque([initial])
    while queue:
        for _, t in successors(queue.popleft(), objects):
            if t not in seen:
                seen.add(t)
                queue.append(t)
    return seen


def wellformed(state) -> str | None:
    """None for a physically meaningful state, else the reason."""
    st = frozenset(state)
    blocks = set(objects_of(st))
    below, above = {}, {}
    for f in st:
        if f[0] == "on":
            x, y = f[1], f[2]
            if x == y or x in below or y in above:
                return f"{fmt(f)}: a block on two blocks, under two blocks or on itself"
            below[x], above[y] = y, x
    held = {f[1] for f in st if f[0] == "holding"}
    table = {f[1] for f in st if f[0] == "ontable"}
    clear = {f[1] for f in st if f[0] == "clear"}
    if len(held) > 1:
        return "more than one block is held"
    if (("handempty",) in st) != (not held):
        return "handempty does not agree with holding"
    for b in blocks:
        if (b in table) + (b in below) + (b in held) != 1:
            return f"block {b} is not in exactly one place"
        if (b in clear) != (b not in above and b not in held):
            return f"clear({b}) does not agree with the stacks"
    for b in blocks:
        seen, x = set(), b
        while x in below:
            if x in seen:
                return "cycle"
            seen.add(x)
            x = below[x]
    return None
