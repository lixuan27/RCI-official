import asyncio
import json
import random

import pytest

from rci.chains.hanabi import accept as A
from rci.chains.hanabi import bots as B
from rci.chains.hanabi import engine as E
from rci.chains.hanabi import obs as O
from rci.chains.hanabi import play as P


def test_a_deal_is_a_seeded_permutation_of_the_50_cards():
    d = E.deck_order(7)
    assert sorted(d) == sorted(E.canonical_deck()) and len(d) == 50
    assert d == E.deck_order(7) and d != E.deck_order(8)
    with pytest.raises(E.EngineError):
        E.deck_order(-1)


@pytest.mark.parametrize("seed", range(30))
def test_random_games_conserve_cards_and_end_by_the_rules(seed):
    rng = random.Random(seed)
    s = E.new_game(seed)
    while s.end is None:
        assert E.multiset_ok(s)
        assert 0 <= s.info <= E.MAX_TOKENS and 1 <= s.lives <= E.MAX_LIVES
        assert all(len(h) <= E.HAND_SIZE for h in s.hands)
        codes = E.legal_codes(s)
        assert codes and len(set(codes)) == len(codes) and set(codes) <= set(E.all_codes())
        if s.info == E.MAX_TOKENS:
            assert not any(c[0] == "D" for c in codes)
        if s.info == 0:
            assert not any(c[0] == "H" for c in codes)
        s = E.apply_code(s, rng.choice(codes))
    assert E.multiset_ok(s) and s.end in ("lives", "perfect", "deck")
    assert E.score_ch(s) == (0 if s.end == "lives" else sum(s.fw))
    with pytest.raises(E.EngineError):
        E.apply_code(s, "P1")


def test_after_the_last_draw_every_player_moves_once_more():
    s = E.new_game(3)
    while s.end is None and s.deck > 0:
        s = E.apply_code(s, "D1" if s.info < E.MAX_TOKENS else E.legal_codes(s)[-1])
    if s.end is None:
        left = 0
        while s.end is None:
            s = E.apply_code(s, E.legal_codes(s)[-1] if s.info else "D1")
            left += 1
        assert left == E.PLAYERS or s.end != "deck"


def test_a_hint_marks_every_card_is_or_is_not():
    s = E.new_game(11)
    code = next(c for c in E.legal_codes(s) if c.startswith("H1"))
    t = E.apply_code(s, code)
    val, hand = code[2], t.hands[1]
    assert t.info == s.info - 1
    for hc in hand:
        hit = hc.card.c == E.COLORS.index(val) if val in E.COLORS else hc.card.r == int(val)
        assert hc.touched == hit
        if val in E.COLORS:
            assert (hc.cm == 1 << E.COLORS.index(val)) if hit else not hc.cm >> E.COLORS.index(val) & 1
    assert t.hist[-1].touched and t.hist[-1].target == 1


def test_a_failed_play_costs_a_life_and_three_end_the_game_at_zero():
    s = E.new_game(5)
    while s.end is None:
        bad = [c for c in E.legal_codes(s) if c[0] == "P" and not E.playable(s.hands[s.cur][int(c[1]) - 1].card, s.fw)]
        s = E.apply_code(s, bad[0] if bad else "P1")
    if s.end == "lives":
        assert s.lives == 0 and E.score_ch(s) == 0 and E.score_pl(s) == sum(s.fw)


def test_the_rule_bot_finishes_games_and_beats_random_play():
    rule = [E.score_pl(B.play(d, B.rule_code)) for d in range(10)]
    rnd = [E.score_pl(B.play(d, B.random_code, random.Random(d))) for d in range(10)]
    assert sum(rule) > sum(rnd)


def test_the_observation_never_shows_the_observers_own_cards():
    s = E.new_game(21)
    for _ in range(6):
        s = E.apply_code(s, B.rule_code(s))
    me = s.cur
    other = list(s.hands[me])
    swapped = [hc._replace(card=E.Card((hc.card.c + 1) % 5, hc.card.r)) for hc in other]
    hands = list(s.hands)
    hands[me] = tuple(swapped)
    assert O.observe(s) == O.observe(s._replace(hands=tuple(hands)))
    txt = O.observe(s)
    assert "Legal actions:" in txt and txt.endswith("Your action:")
    assert [c for c, _ in O.legal_codes(O.project(s, s.cur))] == E.legal_codes(s)


class Tok:
    """A stand-in tokenizer: the chat template is the characters' code points mod 997, <|im_end|> is 999."""

    def apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tokenize):
        return [ord(ch) % 997 for m in messages for ch in m["content"][-200:]]

    def convert_tokens_to_ids(self, s):
        return {"<|im_end|>": 999, "<|endoftext|>": 998}.get(s, -1)


def fake_server(choose):
    async def post(url, body):
        legal = body["structured_outputs"]["choice"]
        code = choose(legal, body["seed"])
        return {"choices": [{"text": code, "token_ids": [len(code), 999], "finish_reason": "stop",
                             "prompt_token_ids": body["prompt"]}]}
    return post


def test_the_driver_plays_whole_games_and_writes_records(tmp_path):
    post = fake_server(lambda legal, seed: legal[seed % len(legal)])
    st = asyncio.run(P.run(post, Tok(), ["http://x/v1"], "m", "train", 1, [2010000, 2010002], 2, 1.0, tmp_path, 4))
    assert st == {"to_play": 6, "played": 6, "protocol": 0, "infra": 0}
    recs = A.load(tmp_path)
    assert len(recs) == 6
    for rec in recs.values():
        s = E.new_game(rec["deal"])
        for mv in rec["moves"]:
            s = E.apply_code(s, mv["c"])
        assert s.end == rec["end"] and E.score_ch(s) == rec["s_ch"]
    again = asyncio.run(P.run(post, Tok(), ["http://x/v1"], "m", "train", 1, [2010000, 2010002], 2, 1.0, tmp_path, 4))
    assert again["to_play"] == 0


def test_an_illegal_output_is_a_protocol_failure_with_score_zero():
    async def post(url, body):
        return {"choices": [{"text": "P9", "token_ids": [1], "finish_reason": "stop"}]}
    rec = asyncio.run(P.play_game(post, Tok(), "http://x/v1", "m", "select", 1, 3010000, 0, 1.0))
    assert rec["end"] == "protocol" and rec["s_ch"] == 0 and A.elite_value(rec) == -1.0


def test_deal_ranges_are_disjoint_across_roles_and_rounds():
    rs = [P.deal_range(role, r, 600) for role in P.ROLES for r in range(1, 17)]
    rs.sort()
    assert all(a[1] < b[0] for a, b in zip(rs, rs[1:]))
    assert P.samp_seed("confirm", 1, 5, 0, 3) == P.samp_seed("confirm", 1, 5, 0, 3)


def bot_record(deal, g, policy):
    s = E.new_game(deal)
    moves = []
    while s.end is None:
        code = policy(s)
        ids = O.prompt_ids(Tok(), O.observe(s))
        moves.append({"t": s.turn, "s": s.cur, "c": code, "tok": [5, 999], "sha": O.ids_sha(ids), "pl": len(ids)})
        s = E.apply_code(s, code)
    return {"deal": deal, "g": g, "moves": moves, "end": s.end, "s_ch": E.score_ch(s), "s_pl": E.score_pl(s),
            "lives_left": s.lives, "n_moves": len(moves)}


def test_elite_games_and_rows(tmp_path):
    rng = random.Random(0)
    for deal in (11, 12, 13):
        recs = [bot_record(deal, 0, B.rule_code)] + [bot_record(deal, g, lambda s: B.random_code(s, rng))
                                                     for g in (1, 2, 3)]
        for rec in recs:
            (tmp_path / f"{deal}_{rec['g']}.json").write_text(json.dumps(rec))
    cfg = {"cap": 50, "drop_uncertain_plays": True, "known_playable_from_round": 1}
    rows, meta = A.build(tmp_path, Tok(), 1, cfg, "t")
    assert meta["elite_games"] >= 1 and 0 < len(rows) <= 50
    for r in rows:
        (start, n), = r["spans"]
        assert r["ids"][-1] == 999 and n == 2 and start == len(r["ids"]) - 2
    by = {}
    for rec in A.load(tmp_path).values():
        by.setdefault(rec["deal"], []).append(rec)
    el, st = A.elite(by, "t")
    for rec in el:
        vs = [A.elite_value(x) for x in by[rec["deal"]]]
        assert A.elite_value(rec) == max(vs) > sum(vs) / len(vs)


def test_a_drifted_prompt_refuses_the_rows(tmp_path):
    rec = bot_record(11, 0, B.rule_code)
    rec["moves"][0]["sha"] = "0" * 64
    with pytest.raises(A.RowsError):
        A.make_rows(Tok(), [(rec, 0)], "t")


def test_uncertain_and_known_playable_annotations():
    rec = bot_record(31, 0, lambda s: "P1")
    unc = A.uncertain_plays(rec)
    assert len(unc) == sum(m["c"][0] == "P" for m in rec["moves"]) and unc[0][0] is True
    km = A.known_playable(bot_record(31, 0, B.rule_code))
    assert all(plays <= holds for holds, plays in km)


def test_hanabi_health_lines():
    s = {"protocol_share": 0.0, "ctok_max": 2}
    h = {"protocol_share_max": 0.001, "completion_tokens_max": 5}
    assert all(x.ok for x in A.health(s, h))
    assert not all(x.ok for x in A.health({"protocol_share": 0.01, "ctok_max": 2}, h))
