"""The Hanabi check: the game score picks one elite game per training deal, and its moves become training rows.

Elite value of a game: v = S_ch + (S_pl + lives_left / 4) / 26, i.e. the order (challenge score, cards played, lives
left); a protocol failure is below every game. For each training deal the group plays G games; the game with the
largest v is the deal's elite game when it is above the deal's mean (ties by a hash). Every move of every elite game,
all four seats, is a candidate row, except
    uncertain plays   a play whose card the player's knowledge (less the identities whose copies it can all see) still
                      allows to be unplayable, successful or not
    known-playable    (from a configured round on) a move of a player holding a card its observation marks '!'
                      that does not play such a card; the '!' plays enter the row cap first
Rows are taken in a hash order up to the cap. A row is the turn's prompt ids (recomputed from the deal and the moves,
held to the sha the driver recorded) plus the returned completion, trailing stop tokens dropped and <|im_end|> appended.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from rci.chains import health as HL
from rci.chains.hanabi import engine as E
from rci.chains.hanabi import obs as O
from rci.chains.util import hkey

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


class RowsError(RuntimeError):
    pass


def load(d) -> dict:
    """(deal, game) -> record of a records directory."""
    out = {}
    for p in sorted(Path(d).glob("*_*.json")):
        if ".tmp" in p.name:
            continue
        rec = json.loads(p.read_text())
        out[(rec["deal"], rec["g"])] = rec
    return out


def elite_value(rec: dict) -> float:
    if rec["end"] == "protocol":
        return -1.0
    return rec["s_ch"] + (rec["s_pl"] + rec["lives_left"] / 4) / 26


def uncertain_plays(rec: dict) -> list:
    """[(uncertain, success)] of every play of a game."""
    s = E.new_game(rec["deal"])
    out = []
    for mv in rec["moves"]:
        if mv["c"][0] == "P":
            hc = s.hands[s.cur][int(mv["c"][1]) - 1]
            vis: dict = {}
            for k in range(1, E.PLAYERS):
                for x in s.hands[(s.cur + k) % E.PLAYERS]:
                    vis[x.card] = vis.get(x.card, 0) + 1
            for c in s.discard:
                vis[c] = vis.get(c, 0) + 1
            for c in range(E.N_COLORS):
                for r in range(1, s.fw[c] + 1):
                    vis[E.Card(c, r)] = vis.get(E.Card(c, r), 0) + 1
            ids = [E.Card(c, r) for c in range(E.N_COLORS) if hc.cm >> c & 1 for r in range(1, E.N_RANKS + 1)
                   if hc.rm >> (r - 1) & 1 and vis.get(E.Card(c, r), 0) < E.COPIES[r]]
            out.append((any(not E.playable(x, s.fw) for x in ids), E.playable(hc.card, s.fw)))
        s = E.apply_code(s, mv["c"])
    return out


def known_playable(rec: dict) -> list:
    """Per move, (the actor holds a card its observation marks '!', the move plays such a card)."""
    s = E.new_game(rec["deal"])
    out = []
    for mv in rec["moves"]:
        if s.cur != mv["s"] or s.turn != mv["t"]:
            raise RowsError(f"deal {rec['deal']} game {rec['g']}: the replay left the record at turn {s.turn}")
        dc = O.dcounts(s.discard)
        km = [O.own_mark(hc.cm, hc.rm, s.fw, dc) == "!" for hc in s.hands[s.cur]]
        code = mv["c"]
        plays = code[0] == "P" and 1 <= int(code[1:]) <= len(km) and km[int(code[1:]) - 1]
        out.append((any(km), bool(plays)))
        s = E.apply_code(s, code)
    return out


def elite(by_deal: dict, salt: str) -> tuple:
    """(elite records, stats). adv_ch: per deal max S_ch - mean S_ch, averaged where positive."""
    out, advs_ch = [], []
    for deal in sorted(by_deal):
        recs = sorted(by_deal[deal], key=lambda x: x["g"])
        vs = [elite_value(x) for x in recs]
        if max(vs) > sum(vs) / len(vs):
            best = max(vs)
            ties = [x for x, v in zip(recs, vs) if v == best]
            out.append(min(ties, key=lambda x: hkey(salt, "elite", deal, x["g"])))
        sch = [x["s_ch"] for x in recs]
        if max(sch) > sum(sch) / len(sch):
            advs_ch.append(max(sch) - sum(sch) / len(sch))
    n = len(by_deal)
    return out, {"deals": n, "elite_games": len(out), "adv_pos_share": len(out) / n if n else None,
                 "elite_adv_mean_ch": statistics.mean(advs_ch) if advs_ch else None}


def candidates(recs: list, drop_uncertain: bool) -> tuple:
    out, dropped = [], 0
    for rec in recs:
        unc = iter(uncertain_plays(rec)) if drop_uncertain else None
        for i, mv in enumerate(rec["moves"]):
            if drop_uncertain and mv["c"][0] == "P":
                u, _ = next(unc)
                if u:
                    dropped += 1
                    continue
            out.append((rec, i))
    return out, dropped


def pick(cands: list, salt: str, cap: int, first=None) -> list:
    cands = sorted(cands, key=lambda x: hkey(salt, "rows", x[0]["deal"], x[0]["g"], x[0]["moves"][x[1]]["t"]))
    if first is not None:
        cands = [x for x in cands if first(x)] + [x for x in cands if not first(x)]
    return cands[:cap]


def stop_ids(tok) -> tuple:
    ids = [t for t in (tok.convert_tokens_to_ids(s) for s in STOP_TOKENS)
           if isinstance(t, int) and t >= 0 and t != getattr(tok, "unk_token_id", None)]
    if not ids:
        raise RowsError("the tokenizer has no <|im_end|>")
    return tuple(ids)


def make_rows(tok, picks: list, tag: str) -> tuple:
    stops = stop_ids(tok)
    by_game: dict = {}
    for rec, i in picks:
        by_game.setdefault((rec["deal"], rec["g"]), (rec, set()))[1].add(i)
    rows, bad = [], 0
    for (deal, g), (rec, want) in sorted(by_game.items()):
        s = E.new_game(deal)
        for i, mv in enumerate(rec["moves"]):
            if i in want:
                ids = O.prompt_ids(tok, O.observe(s))
                if O.ids_sha(ids) != mv["sha"]:
                    bad += 1
                else:
                    comp = list(mv["tok"])
                    while comp and comp[-1] in stops:
                        comp.pop()
                    if not comp:
                        raise RowsError("an empty completion")
                    comp.append(stops[0])
                    rows.append({"row_id": f"{tag}/{deal}/{g}/{mv['t']}", "ids": ids + comp,
                                 "spans": [[len(ids), len(comp)]], "w": 1.0})
            s = E.apply_code(s, mv["c"])
    if bad:
        raise RowsError(f"{bad} rows whose recomputed prompt ids differ from the ones sent (engine, observation or "
                        f"tokenizer changed)")
    return rows, {"rows": len(rows), "tokens": sum(len(x["ids"]) for x in rows)}


def build(records_dir, tok, r: int, c: dict, salt: str) -> tuple:
    games = load(records_dir)
    by_deal: dict = {}
    for (deal, _), rec in games.items():
        by_deal.setdefault(deal, []).append(rec)
    el, meta = elite(by_deal, f"{salt}/r{r}")
    cands, dropped = candidates(el, bool(c.get("drop_uncertain_plays", True)))
    first = None
    km_from = c.get("known_playable_from_round")
    if km_from is not None and r >= int(km_from):
        ann = {(x["deal"], x["g"]): known_playable(x) for x in el}
        cands = [x for x in cands if not (ann[(x[0]["deal"], x[0]["g"])][x[1]][0]
                                          and not ann[(x[0]["deal"], x[0]["g"])][x[1]][1])]
        first = lambda x: ann[(x[0]["deal"], x[0]["g"])][x[1]][1]  # noqa: E731
    picks = pick(cands, f"{salt}/r{r}", int(c["cap"]), first)
    rows, rmeta = make_rows(tok, picks, f"{salt}/r{r}")
    return rows, dict(meta, **rmeta, candidates=len(cands), dropped_uncertain=dropped, known_playable=first is not None)


def summary(games: dict) -> dict:
    recs = list(games.values())
    n = len(recs)
    if not n:
        return {"games": 0}
    calls = sum(max(1, r["n_moves"] + (1 if r["end"] == "protocol" else 0)) for r in recs)
    proto = sum(r["end"] == "protocol" for r in recs)
    return {"games": n, "calls": calls, "protocol_share": proto / calls,
            "ctok_max": max((len(m["tok"]) for r in recs for m in r["moves"]), default=0),
            "s_ch_mean": statistics.mean(r["s_ch"] for r in recs), "s_pl_mean": statistics.mean(r["s_pl"] for r in recs),
            "ev_mean": statistics.mean(elite_value(r) for r in recs),
            "bomb_share": sum(r["end"] == "lives" for r in recs) / n}


def health(s: dict, h: dict) -> list:
    return [HL.at_most("protocol failures / calls", s.get("protocol_share"), h["protocol_share_max"]),
            HL.at_most("completion tokens of a move", s.get("ctok_max"), h["completion_tokens_max"])]
