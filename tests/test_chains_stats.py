import math

import numpy as np
import pytest

from rci.chains import health as HL
from rci.chains import mix as MX
from rci.chains import stats as ST


def test_paired_gain_and_its_standard_error():
    cand = {"a": 1, "b": 1, "c": 0, "d": 1}
    inc = {"a": 0, "b": 1, "c": 0, "d": 0}
    p = ST.paired(cand, inc)
    d = [1, 0, 0, 1]
    sd = math.sqrt(sum((x - 0.5) ** 2 for x in d) / 3)
    assert p["n"] == 4 and p["d"] == pytest.approx(0.5) and p["se"] == pytest.approx(sd / 2)


def test_paired_pairs_only_shared_tasks_and_refuses_too_few():
    p = ST.paired({"a": 1, "b": 0, "x": 1}, {"a": 0, "b": 0, "y": 1})
    assert p["n"] == 2
    with pytest.raises(ST.StatsError):
        ST.paired({str(i): 1 for i in range(100)}, {str(i): 0 for i in range(90)}, min_share=0.99)
    assert math.isnan(ST.paired({"a": 1}, {"a": 0})["d"])


def test_bootstrap_se_is_close_to_the_analytic_one_and_reproducible():
    rng = np.random.default_rng(1)
    cand = {i: float(x) for i, x in enumerate(rng.integers(0, 6, 600))}
    inc = {i: float(x) for i, x in enumerate(rng.integers(0, 5, 600))}
    b1, b2 = ST.bootstrap_paired(cand, inc, 2000, 7), ST.bootstrap_paired(cand, inc, 2000, 7)
    a = ST.paired(cand, inc)
    assert b1 == b2 and b1["d"] == pytest.approx(a["d"]) and b1["se"] == pytest.approx(a["se"], rel=0.1)
    with pytest.raises(ST.StatsError):
        ST.bootstrap_paired({1: 1.0, 2: 1.0}, {1: 0.0})


def test_planbench_rule_needs_d_at_least_1645_se_and_positive():
    r = ST.PLANBENCH
    assert r.passes(0.021, 0.010)
    assert not r.passes(0.016, 0.010)
    assert r.passes(0.01645, 0.010)
    assert not r.passes(0.0, 0.0)
    assert r.passes(0.001, 0.0)
    assert not r.passes(float("nan"), 0.01)


def test_hiddenbench_rule_is_a_margin_of_0026():
    r = ST.HIDDENBENCH
    assert r.passes(0.044, 0.026) and r.passes(0.027, 1.0)
    assert not r.passes(0.026, 0.0) and not r.passes(0.015, 0.016) and not r.passes(0.001, 0.022)


def test_hanabi_rule_is_d_minus_se_above_zero():
    r = ST.HANABI
    assert r.passes(0.678, 0.045) and r.passes(0.087, 0.074)
    assert not r.passes(0.04, 0.051) and not r.passes(0.05, 0.05) and not r.passes(-0.085, 0.054)


def test_release_rule_from_config_matches_the_constants():
    assert ST.ReleaseRule.of({"k": 1.645, "strict": False, "positive": True}) == ST.PLANBENCH
    assert ST.ReleaseRule.of({"margin": 0.026}) == ST.HIDDENBENCH
    assert ST.ReleaseRule.of({"k": 1.0}) == ST.HANABI
    assert ST.HIDDENBENCH.line(0.5) == pytest.approx(0.026)


def test_planbench_lr_doubles_after_flat_rounds_not_significantly_worse():
    flat = {"label": "flat", "confirm": {"d": 0.01, "se": 0.01}}
    worse = {"label": "flat", "confirm": {"d": -0.05, "se": 0.01}}
    rel = {"label": "released", "confirm": {"d": 0.1, "se": 0.01}}
    assert ST.lr_doubling(1.0, [rel], 2, 1.645) == 1.0
    assert ST.lr_doubling(1.0, [rel, flat], 2, 1.645) == 2.0
    assert ST.lr_doubling(1.0, [flat, worse, flat, flat], 2, 1.645) == 4.0
    assert ST.lr_doubling(1.0, [{"label": "unhealthy"}, {"label": "unhealthy"}], 2, 1.645, 2) == 0.5


def test_hanabi_lr_moves_after_two_releases_at_the_same_extreme_alpha():
    assert ST.lr_alpha_pairs(1e-5, ["2", "2"], (1e-6, 1e-4)) == pytest.approx(2e-5)
    assert ST.lr_alpha_pairs(1e-5, ["1/2", "1/2", "1"], (1e-6, 1e-4)) == pytest.approx(5e-6)
    assert ST.lr_alpha_pairs(1e-5, ["2", "1", "2"], (1e-6, 1e-4)) == pytest.approx(1e-5)
    assert ST.lr_alpha_pairs(8e-5, ["2", "2"], (1e-6, 1e-4)) == pytest.approx(1e-4)


def test_health_lines():
    assert HL.within("x", 1.2, None, 1.25).ok and not HL.within("x", 1.3, None, 1.25).ok
    assert not HL.at_most("x", None, 1.0).ok and HL.at_least("x", 0.5, 0.5).ok
    lines = [HL.at_most("a", 0.0, 0.001), HL.at_most("b", 6, 5)]
    assert not HL.healthy(lines) and HL.failed(lines) == ["b"]


def rows(n, length, prefix):
    return [{"row_id": f"{prefix}{i}", "ids": [1] * length, "spans": [[1, length - 1]]} for i in range(n)]


def test_rows_mix_keeps_rows_since_the_release_and_replays_a_quarter():
    by = {1: rows(40, 10, "a"), 2: rows(300, 10, "b"), 3: rows(100, 10, "c")}
    out, meta = MX.mix_rows(by, 3, 1, 0.25, 10 ** 9, 200, "t")
    assert meta["S_rounds"] == [2, 3] and meta["P_rows"] == 40 and len(out) == 440
    out, meta = MX.mix_rows(by, 3, 1, 0.25, 1500, 50, "t")
    assert meta["dropped_rounds"] == [2] and meta["S_rows"] == 100
    out, meta = MX.mix_rows(by, 3, 2, 0.25, 10 ** 9, 200, "t")
    assert out is None and meta["starved"]


def test_token_mix_gives_the_newest_slice_three_quarters():
    by = {1: rows(50, 100, "a"), 2: rows(50, 100, "b"), 3: rows(50, 100, "c")}
    out, meta = MX.mix_tokens(by, 3, [1, 2], "1/4", 8000, "t", 12)
    assert meta["targets"] == {1: 1000, 2: 1000, 3: 6000}
    assert sum(1 for x in out if x["row_id"].startswith("c")) == 60
    out, meta = MX.mix_tokens({1: rows(2, 100, "a")}, 1, [], "1/4", 10 ** 6, "t", 12)
    assert meta["under_budget"] and len(out) == 24


def test_twin_plan_brings_the_call_share_to_a_quarter_in_integers():
    rs = []
    for i in range(10):
        rs.append({"row_id": f"r{i}", "ids": list(range(100)), "spans": [[10, 90]], "skel": [[80, 5]], "c0": 5})
    out, meta = MX.twin_plan(rs, "t")
    L, C0, T = meta["L"], meta["C0"], meta["T"]
    assert 0 <= (L + T) - 4 * (C0 + T) < 3
    assert sum(n for x in out for _, n in x["spans"]) == L + T
    assert all("skel" not in x and x["w"] == 1.0 for x in out)
    _, meta = MX.twin_plan([{"row_id": "a", "ids": [0] * 10, "spans": [[1, 8]]}], "t")
    assert meta["T"] == 0


def test_steps_for_clips():
    assert MX.steps_for(1000, 32, 2, [20, 400]) == 63
    assert MX.steps_for(100, 32, 2, [20, 400]) == 20
    assert MX.steps_for(0, 32, 1) == 0
