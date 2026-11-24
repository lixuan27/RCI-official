import random

import pytest

from rci.ladder.curves import Member, at, commit, draw_curves, effective, level, runs_worth


def m(score, key):
    return Member(float(score), key, {"answer": key, "confidence": 0.4})


def test_commit_is_the_groups_own_rule():
    ms = [m(1, "a"), m(1, "a"), m(0, "b")]
    assert commit("gpqa", ms) == 1.0
    assert commit("gpqa", [m(0, "b"), m(0, "b"), m(1, "a")]) == 0.0


def test_a_verifier_makes_the_vote_pass_at_n():
    assert commit("fate", [m(0, None), m(1, None), m(0, None)]) == 1.0


def test_curves_on_a_question_half_the_runs_solve():
    pool = [m(1, "a")] * 8 + [m(0, f"w{i}") for i in range(8)]
    c = draw_curves("gpqa", pool, [1, 4, 16], 4000, random.Random(0))
    assert c[1][0] == pytest.approx(0.5, abs=0.03) and c[1][1] == pytest.approx(0.5, abs=0.03)
    assert c[4][1] == pytest.approx(1 - (8 * 7 * 6 * 5) / (16 * 15 * 14 * 13), abs=0.03)
    assert c[16] == (1.0, 1.0)


def test_voting_saturates_where_pass_at_n_does_not():
    pool = [m(1, "a")] * 2 + [m(0, "b")] * 14
    c = draw_curves("gpqa", pool, [1, 8, 16], 2000, random.Random(0))
    assert c[16][0] == 0.0 and c[16][1] == 1.0
    assert c[8][1] > c[8][0]


def test_at_interpolates_and_stays_flat_past_the_ends():
    c = {1: 0.2, 2: 0.4, 4: 0.8}
    assert at(c, 0.5) == 0.2 and at(c, 3) == pytest.approx(0.6) and at(c, 9) == 0.8


def test_effective_size_reads_the_curve_backwards():
    c = {1: 0.2, 2: 0.4, 4: 0.8}
    assert effective(c, 0.6) == pytest.approx(3)
    assert effective(c, 0.2) == 1.0
    assert effective(c, 0.1) == 0.0
    assert effective(c, 0.9) is None


def test_effective_size_uses_the_running_maximum():
    assert effective({1: 0.5, 2: 0.4, 3: 0.6}, 0.55) == pytest.approx(2.5)


def rec(per, keys, tokens):
    return {"task": "gpqa", "example_id": "q", "score": {"score": max(per), "per_agent": per, "preds": keys}, "compute": {"completion_tokens": tokens, "total_tokens": tokens * 2},
            "finals": [{"final": {"answer": k, "confidence": 0.4}} for k in keys]}


def test_runs_worth_reads_pass_at_n_from_zero_runs():
    t = {1: 0.4, 2: 0.6, 4: 0.8}
    assert runs_worth(t, 0.2) == pytest.approx(0.5)
    assert runs_worth(t, 0.7) == pytest.approx(3)
    assert runs_worth(t, 0.9) is None
