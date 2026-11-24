"""Policies that exercise the engine in tests: a uniformly random legal policy and a simple rule bot (play a card
known playable; else give a hint that makes a teammate's playable card known playable; else discard a card known dead,
then the oldest untouched card; with 8 tokens and nothing else, any hint to the next player)."""
from __future__ import annotations

import random

from rci.chains.hanabi import engine as E
from rci.chains.hanabi import obs as O


def random_code(s: E.State, rng: random.Random) -> str:
    return rng.choice(E.legal_codes(s))


def rule_code(s: E.State) -> str:
    v = O.project(s, s.cur)
    dc = O.dcounts(v.discard)
    codes = [c for c, _ in O.legal_codes(v)]
    for i, (cm, rm, _) in enumerate(v.own, start=1):
        if O.own_mark(cm, rm, v.fw, dc) == "!":
            return f"P{i}"
    if v.info > 0:
        for k in range(1, E.PLAYERS):
            for hc in v.others[k - 1]:
                if not E.playable(hc.card, v.fw) or O.own_mark(hc.cm, hc.rm, v.fw, dc) == "!":
                    continue
                for val in (str(hc.card.r), E.COLORS[hc.card.c]):     # a hint that makes it provably playable
                    cm, rm = (hc.cm, 1 << (hc.card.r - 1)) if val.isdigit() else (1 << hc.card.c, hc.rm)
                    if O.own_mark(cm, rm, v.fw, dc) == "!" and f"H{k}{val}" in codes:
                        return f"H{k}{val}"
    if v.info < E.MAX_TOKENS:
        for i, (cm, rm, _) in enumerate(v.own, start=1):
            if O.own_mark(cm, rm, v.fw, dc) == "x":
                return f"D{i}"
        for i, (_, _, t) in enumerate(v.own, start=1):
            if not t:
                return f"D{i}"
        return "D1"
    return next(c for c in codes if c[0] == "H")


def play(seed: int, policy, rng: random.Random | None = None) -> E.State:
    s = E.new_game(seed)
    while s.end is None:
        s = E.apply_code(s, policy(s) if rng is None else policy(s, rng))
    return s
