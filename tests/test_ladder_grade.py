import importlib.util
import itertools
import json
import math
import os
import random
import sys

import pytest

import rci.ladder.curves as curves
from rci.ladder.curves import (Member, bh, by_seat, classify, commit, draw_curves, draw_curves_seats, members_of, p_one,
                                  seat_capacity, stakeless, t_drawn, upper)
from rci.ladder.commit import _candidates, vote_stake

HERE = os.path.dirname(__file__)


def table():
    import rci.ladder.grade as g
    return g


def rec(per, keys, tokens=100, conf=0.4, task="gpqa", eid="q", calls=None, n_agents=None, seed=0):
    confs = conf if isinstance(conf, list) else [conf] * len(per)
    r = {"task": task, "example_id": eid, "score": {"score": max(per), "per_agent": per, "preds": keys},
         "compute": {"completion_tokens": tokens, "total_tokens": tokens * 2}, "config": {"seed": seed},
         "finals": [{"final": {"answer": k, "confidence": c}} for k, c in zip(keys, confs)]}
    if n_agents is not None:
        r["config"]["n_agents"] = n_agents
    if calls is not None:
        r["compute"]["n_calls"] = calls
    return r


def put(root, method, n, seed, q, r):
    d = root / method / f"N{n}" / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{q}.json").write_text(json.dumps({**r, "example_id": q, "config": {**r.get("config", {}), "seed": seed}}))


def run_main(t, tmp_path, spec, *flags):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / "out"
    t.main([str(sp), "--out", str(out), "--draws", "60", *flags])
    return out


def test_upper_bound_and_classes():
    assert upper(0.1, 0.02) == pytest.approx(0.1 + 1.645 * 0.02)
    assert classify(-0.1, 0.02) == "below" and classify(0.1, 0.02) == "above" and classify(0.01, 0.02) == "undetermined"
    assert classify(-0.0392, 0.02) == "undetermined"


def flat(v, t, sizes=(1, 2, 4)):
    return {n: (v, t) for n in sizes}


def test_measure_reports_each_contrasts_bound_and_class():
    t = table()
    cs = {f"q{i}": flat(0.2, 0.95) for i in range(50)}
    recs = {f"q{i}": [rec([1.0, 1.0], ["a", "a"], 200)] for i in range(30)}
    recs.update({f"q{i}": [rec([0.0, 0.0], ["c", "c"], 200)] for i in range(30, 50)})
    r = t.measure({"task": "gpqa"}, "x", recs, cs, 100.0, "completion", None)
    assert r["answer"] == pytest.approx(0.6)
    assert r["class_vote"] == "above" and r["class_truth"] == "below"
    for k, e in (("ub_one", "e_one"), ("ub_vote", "e_vote"), ("ub_truth", "e_truth"), ("ub_full", "margin_full")):
        assert r[k] == pytest.approx(r[e][0] + 1.645 * r[e][1])
    assert r["ub_truth"] < 0 < r["ub_vote"]
    cs = {f"q{i}": {1: (0.2, 0.2), 2: (0.58, 0.95), 4: (0.58, 0.95)} for i in range(50)}
    r = t.measure({"task": "gpqa"}, "x", recs, cs, 100.0, "completion", None)
    assert r["class_vote"] == "undetermined" and r["ub_vote"] > 0 and r["level"] == 1


def test_p_one_is_the_normal_tail():
    assert p_one(1.96, 1.0) == pytest.approx(0.025, abs=1e-4)
    assert p_one(0.0, 1.0) == pytest.approx(0.5)
    assert p_one(0.1, 0.0) == 0.0 and p_one(-0.1, 0.0) == 1.0


def test_bh_matches_the_step_up_by_hand():
    q = bh([0.01, 0.04, 0.03, 0.20, None])
    assert q[0] == pytest.approx(0.04) and q[1] == pytest.approx(0.16 / 3) and q[2] == pytest.approx(0.16 / 3)
    assert q[3] == pytest.approx(0.20) and q[4] is None
    assert bh([]) == [] and bh([0.9, 0.95]) == [pytest.approx(0.95), pytest.approx(0.95)]


def two_task_tree(tmp_path, calls=False):
    spec = {}
    for name in ("ta", "tb"):
        root = tmp_path / name
        on = calls is True or (calls and name in calls)
        for i in range(10):
            q = f"q{i}"
            for s in range(3):
                put(root, "independent", 4, s, q, rec([1.0, 0.0, 0.0, 0.0], ["a", "b", "c", "d"], 100, calls=8 if on else None))
                put(root, "strong", 4, s, q, rec([1.0] * 4, ["a"] * 4, 400, calls=40 if on else None))
                good = 1.0 if (i + s) % 3 else 0.0
                put(root, "mid", 4, s, q, rec([good] * 4, ["a" if good else "b"] * 4, 400, calls=40 if on else None))
                put(root, "stop", 4, s, q, rec([1.0] * 4, ["a"] * 4, 400, calls=40 if on else None))
        spec[name] = {"task": "gpqa", "least": 5, "pool": [[str(root), "independent", 4]],
                      "cells": [{"label": "strong", "root": str(root), "protocol": "strong", "n": 4},
                                {"label": "mid", "root": str(root), "protocol": "mid", "n": 4},
                                {"label": "stop", "root": str(root), "protocol": "stop", "n": 4, "stops": True}]}
    return spec


def test_pool_size_and_the_questions_short_of_the_budget():
    t = table()
    cs = {f"q{i}": ({1: (0.2, 0.2), 2: (0.3, 0.4)} if i < 10 else {1: (0.2, 0.2), 2: (0.3, 0.4), 4: (0.4, 0.6)}) for i in range(50)}
    sizes = {f"q{i}": (3 if i < 10 else 8) for i in range(50)}
    recs = {f"q{i}": [rec([1.0, 0.0], ["a", "b"], 200)] for i in range(50)}
    r = t.measure({"task": "gpqa"}, "x", recs, cs, 50.0, "completion", None, sizes=sizes)
    assert r["n_budget"] == 4 and r["pool_n"] == 2
    assert r["pool_size"] == 3 and r["short_q"] == 10 and r["capped"] is False
    assert r["over_pool"] is True and r["min_seat"] is None


def capped_tree(tmp_path):
    root = tmp_path / "cap"
    for i in range(10):
        right = i % 2 == 0
        for s in range(2):
            put(root, "independent", 2, s, f"q{i}", rec([1.0 if right and s == 0 else 0.0, 0.0], ["a" if right and s == 0 else "b", "c"], 100))
            put(root, "big", 4, s, f"q{i}", rec([1.0] * 4, ["a"] * 4, 800))
            put(root, "small", 4, s, f"q{i}", rec([1.0] * 4, ["a"] * 4, 100))
    return {"cap": {"task": "gpqa", "least": 5, "pool": [[str(root), "independent", 2]],
                    "cells": [{"label": "big", "root": str(root), "protocol": "big", "n": 4},
                              {"label": "small", "root": str(root), "protocol": "small", "n": 4}]}}


def test_a_capped_cell_leaves_the_truth_family_and_the_bh_list(tmp_path):
    t = table()
    spec = capped_tree(tmp_path)
    loose = run_main(t, tmp_path, spec, "--bh-q", "0.05")
    rows = {r["cell"]: r for r in json.loads((loose / "cap_completion.json").read_text())["rows"]}
    assert rows["big"]["n_budget"] == 16 and rows["big"]["pool_n"] == 4 and rows["big"]["over_pool"] is True
    assert rows["big"]["bh_pass"]["truth_task"]
    assert "| big |" in "\n".join(x for x in (loose / "bh_completion.md").read_text().splitlines() if "3 (Y > T(n_B))" in x)
    strict = run_main(t, tmp_path, spec, "--bh-q", "0.05", "--strict-pool")
    body = json.loads((strict / "cap_completion_strict.json").read_text())
    rows = {r["cell"]: r for r in body["rows"]}
    big, small = rows["big"], rows["small"]
    assert big["capped"] and big["level"] == 2 and big["class_truth"] == "capped"
    assert big["q_truth_task"] is None and big["q_truth_run"] is None and not any(v for k, v in big["bh_pass"].items() if k.startswith("truth"))
    assert big["q_vote_task"] is not None
    assert small["capped"] is False and small["q_truth_task"] == pytest.approx(small["p_truth"])
    assert body["options"]["bh_m"]["truth_task"] == 1 and body["options"]["bh_m"]["vote_task"] == 2
    level3 = [x for x in (strict / "bh_completion_strict.md").read_text().splitlines() if "3 (Y > T(n_B))" in x]
    assert level3 and not any("| big |" in x for x in level3) and any("| small |" in x for x in level3)
    md = (strict / "cap_completion_strict.md").read_text()
    assert "group, capped" in md and "| 2 capped |" in md


def test_agents_budget_is_the_member_count_whatever_the_tokens():
    t = table()
    cs = {f"q{i}": {1: (0.2, 0.2), 4: (0.5, 0.8), 8: (0.6, 0.9)} for i in range(50)}
    recs = {f"q{i}": [rec([1.0] * 4, ["a"] * 4, 50 + i, n_agents=4)] for i in range(50)}
    r = t.measure({"task": "gpqa"}, "x", recs, cs, 1.0, "agents", None)
    assert r["n_budget"] == 4 and r["v_b"] == pytest.approx(0.5) and r["t_b"] == pytest.approx(0.8)


def test_agents_budget_prices_abstainers_like_tokens(tmp_path):
    t = table()
    root = tmp_path / "ab"
    rev = tmp_path / "rev"
    rev.mkdir()
    for i in range(10):
        q = f"q{i}"
        for s in range(3):
            r = rec([1.0, 0.0, 0.0, 0.0], ["a", "b", "c", "d"], 100)
            r["finals"][3]["final"] = None
            put(root, "independent", 4, s, q, r)
            r = rec([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], list("abcdabcd"), 200)
            r["finals"][3]["final"] = r["finals"][7]["final"] = None
            put(root, "independent", 8, s, q, r)
            put(root, "grp", 4, s, q, rec([1.0] * 4, ["a"] * 4, 400))
            (rev / f"s{s}_{q}.json").write_text(json.dumps({"candidates": [{"label": "A", "score": 1.0}], "chosen": "A",
                                                             "reviewed": s != 2, "actions": 3, "prompt_tokens": 9,
                                                             "completion_tokens": 9}))
    spec = {"ab": {"task": "gpqa", "least": 5, "pool": [[str(root), "independent", 4], [str(root), "independent", 8]],
                   "cells": [{"label": "independent N=4", "root": str(root), "protocol": "independent", "n": 4},
                             {"label": "grp", "root": str(root), "protocol": "grp", "n": 4},
                             {"label": "grp+review", "root": str(root), "protocol": "grp", "n": 4, "review": str(rev),
                              "review_scale": 3.0}]}}
    body = json.loads((run_main(t, tmp_path, spec, "--cost", "agents") / "ab_agents.json").read_text())
    assert body["c1"] == pytest.approx(4 / 3) and body["answer_rate"] == pytest.approx(0.75)
    rows = {r["cell"]: r for r in body["rows"]}
    assert rows["independent N=4"]["n_budget"] == pytest.approx(3.0)
    assert rows["grp"]["n_budget"] == pytest.approx(3.0)
    assert rows["grp+review"]["n_budget"] == pytest.approx((4 + 3.0 * 2 / 3) * 3 / 4)


def seat_pool():
    pool = []
    for run in range(2):
        pool += members_of(rec([1.0, 0.0, 0.0, 0.0], ["a", "b", "b", "b"], task="hiddenbench", eid=f"r{run}"))
    return pool


def test_members_carry_their_seat():
    assert [m.seat for m in members_of(rec([1.0, 0.0, 0.0], ["a", "b", "c"]))] == [0, 1, 2]
    assert sorted(by_seat(seat_pool())) == [0, 1, 2, 3] and all(len(g) == 2 for g in by_seat(seat_pool()).values())


def test_pass_at_n_is_biased_unstratified_and_exact_stratified():
    pool = seat_pool()
    groups = [rec([1.0, 0.0, 0.0, 0.0], ["a", "b", "b", "b"], task="hiddenbench") for _ in range(2)]
    pass4 = sum(max(r["score"]["per_agent"]) for r in groups) / len(groups)
    assert pass4 == 1.0
    mixed = draw_curves("hiddenbench", pool, [2, 4], 4000, random.Random(0))
    assert mixed[4][1] == pytest.approx(1 - 15 / 70, abs=0.03)
    assert mixed[2][1] == pytest.approx(1 - 15 / 28, abs=0.03)
    strat = draw_curves_seats("hiddenbench", pool, [1, 2, 3, 4, 6, 8], 10, random.Random(0))
    assert strat[4] == (0.0, 1.0)
    assert strat[4][1] == pass4
    assert strat[1] == (pytest.approx(0.25), pytest.approx(0.25))
    assert strat[2][1] == pytest.approx(0.5) and strat[3][1] == pytest.approx(0.75)
    assert strat[8] == (0.0, 1.0) and strat[6] == (0.0, 1.0)


def test_closed_form_pass_at_n_matches_enumeration():
    rng = random.Random(3)
    for _ in range(20):
        groups = [([rng.choice([0.0, 0.25, 0.5, 1.0]) for _ in range(rng.randint(1, 5))], rng.randint(0, 3)) for _ in range(3)]
        live = [(xs, min(k, len(xs))) for xs, k in groups if k]
        combos = list(itertools.product(*(itertools.combinations(xs, k) for xs, k in live)))
        want = sum(max(x for part in c for x in part) for c in combos) / len(combos) if live else 0.0
        assert t_drawn(groups) == pytest.approx(want)


def mixed_seats():
    keys = [["a", "a", "b"], ["a", "b", "b"], ["a", "c", "c"], ["b", "c", "a"]]
    confs = [[0.9, 0.2, 0.5], [0.3, 0.8, 0.1], [0.4, 0.4, 0.7], [0.6, 0.1, 0.2]]
    return [Member(1.0 if k == "a" else 0.0, k, {"answer": k, "confidence": c}, s)
            for s in range(4) for k, c in zip(keys[s], confs[s])]


def test_stratified_vote_by_monte_carlo_agrees_with_enumeration(monkeypatch):
    pool = mixed_seats()
    exact = draw_curves_seats("gpqa", pool, [4], 10, random.Random(0))[4]
    want = sum(commit("gpqa", list(c)) for c in itertools.product(*by_seat(pool).values())) / 81
    assert exact[0] == pytest.approx(want)
    calls = []
    real = curves.commit
    monkeypatch.setattr(curves, "ENUM_MAX", 0)
    monkeypatch.setattr(curves, "commit", lambda *a, **k: calls.append(1) or real(*a, **k))
    mc = draw_curves_seats("gpqa", pool, [4], 10, random.Random(0))[4]
    assert len(calls) == 2000
    assert mc[1] == exact[1] and mc[0] == pytest.approx(exact[0], abs=0.03)


def test_sizes_between_multiples_of_the_seats_interpolate():
    pool = mixed_seats()
    c = draw_curves_seats("gpqa", pool, [4, 6, 8, 10, 12], 10, random.Random(0))
    for n, lo, hi in ((6, 4, 8), (10, 8, 12)):
        assert c[n][0] == pytest.approx((c[lo][0] + c[hi][0]) / 2) and c[n][1] == pytest.approx((c[lo][1] + c[hi][1]) / 2)
    assert c[12][1] == 1.0


def test_balanced_split_is_exact_where_interpolation_sits_below():
    keys = [["b", "b", "a"], ["b", "b", "b"], ["c", "c", "a"], ["b", "c", "c"]]
    confs = [[0.9, 0.2, 0.5], [0.3, 0.8, 0.1], [0.4, 0.4, 0.7], [0.6, 0.1, 0.2]]
    pool = [Member(1.0 if k == "a" else 0.0, k, {"answer": k, "confidence": c}, s)
            for s in range(4) for k, c in zip(keys[s], confs[s])]
    groups = list(by_seat(pool).values())

    def brute(n):
        q, r = divmod(n, 4)
        vs, ts = [], []
        for pick in itertools.combinations(range(4), r):
            combos = list(itertools.product(*(itertools.combinations(g, q + (j in pick)) for j, g in enumerate(groups))))
            vs.append(sum(commit("gpqa", [m for part in c for m in part]) for c in combos) / len(combos))
            ts.append(sum(max(m.score for part in c for m in part) for c in combos) / len(combos))
        return sum(vs) / len(vs), sum(ts) / len(ts)

    interp = draw_curves_seats("gpqa", pool, [1, 2, 3, 4, 6, 8, 10, 12], 10, random.Random(0))
    exact = draw_curves_seats("gpqa", pool, [1, 2, 3, 4, 6, 8, 10, 12], 10, random.Random(0), balanced=True)
    for n in (1, 2, 3, 6, 10):
        assert exact[n] == (pytest.approx(brute(n)[0]), pytest.approx(brute(n)[1]))
    for n in (1, 2, 3, 4, 8, 12):
        assert exact[n] == interp[n]
    assert exact[6][1] > interp[6][1] + 0.03 and exact[10][1] > interp[10][1] + 0.03


def test_a_short_seat_caps_the_stratified_curve():
    pool = [Member(1.0 if s == 3 else 0.0, "a" if s == 3 else "b", {"answer": "x", "confidence": 0.5}, s)
            for s in range(4) for _ in range(8 if s < 3 else 3)]
    assert seat_capacity(pool) == (12, 3) and seat_capacity([]) == (0, 0)
    c = draw_curves_seats("gpqa", pool, [1, 2, 3, 4, 6, 8, 12, 16, 24], 10, random.Random(0))
    assert sorted(c) == [1, 2, 3, 4, 6, 8, 12]
    assert c[12][1] == 1.0 and c[4][1] == 1.0 and c[2][1] == pytest.approx(0.5)
    t = table()
    sizes, least = t.pool_sizes({"q": pool, "r": pool[:25]}, True)
    assert sizes == {"q": 12, "r": 4} and least == {"q": 3, "r": 1}
    assert t.pool_sizes({"q": pool}, False) == ({"q": 27}, None)
    with pytest.raises(ValueError):
        curves._seat_plans(by_seat(pool), 16)


def test_vote_free_on_both_sides_end_to_end(tmp_path):
    t = table()
    root = tmp_path / "bcp"
    for i in range(6):
        put(root, "independent", 2, 0, f"q{i}", rec([1.0, 0.0], ["a", "b"], 100, conf=[0.9, 0.1], task="browsecomp"))
        put(root, "blackboard", 2, 0, f"q{i}", rec([1.0, 0.0], ["a", "b"], 200, conf=[0.9, 0.1], task="browsecomp"))
    spec = {"b": {"task": "browsecomp", "least": 5, "pool": [[str(root), "independent", 2]],
                  "cells": [{"label": "blackboard", "root": str(root), "protocol": "blackboard", "n": 2}]}}
    staked = json.loads((run_main(t, tmp_path, spec) / "b_completion.json").read_text())["rows"][0]
    free = json.loads((run_main(t, tmp_path, spec, "--vote-free") / "b_completion_votefree.json").read_text())["rows"][0]
    assert staked["answer"] == 1.0 and staked["v_b"] == 1.0
    assert free["answer"] == 0.5 and free["v_b"] == 0.5
    spec["b"]["vote_free"] = True
    out = run_main(t, tmp_path, spec)
    keyed = json.loads((out / "b_completion_votefree.json").read_text())
    assert keyed["rows"][0]["answer"] == 0.5 and keyed["options"]["vote_free"] is True
    kept = json.loads((out / "b_completion.json").read_text())
    assert kept["rows"][0]["answer"] == 1.0 and kept["options"]["vote_free"] is False


def test_task_rng_makes_a_tasks_draws_its_own(tmp_path):
    t = table()
    spec = two_task_tree(tmp_path)
    both = run_main(t, tmp_path, spec, "--task-rng")
    full = json.loads((both / "tb_completion_taskrng.json").read_text())
    alone = json.loads((run_main(t, tmp_path, spec, "--task-rng", "--only", "tb") / "tb_completion_taskrng.json").read_text())
    assert [r["curve"] for r in alone["rows"]] == [r["curve"] for r in full["rows"]]
    assert full["options"]["task_rng"] is True and not (both / "tb_completion.json").exists()


def golden_tree(root):
    rng = random.Random(2026 * 10000 + 1106)
    spec = {}
    for name in ("ga", "gb"):
        base = root / name
        rev = root / f"{name}_review"
        rev.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            q = f"q{i}"
            for method, n, seeds, lift in (("independent", 4, 3, 0.0), ("independent", 2, 2, 0.0), ("blackboard", 4, 3, 0.15), ("stopper", 4, 2, 0.3)):
                for s in range(seeds):
                    pa = min(0.1 + 0.06 * i + lift, 0.95)
                    keys = []
                    for _ in range(n):
                        u = rng.random()
                        keys.append("a" if u < pa else "b" if u < pa + (1 - pa) / 2 else "c" if u < pa + 3 * (1 - pa) / 4 else "d")
                    per = [1.0 if k == "a" else 0.0 for k in keys]
                    conf = [rng.choice([0.2, 0.5, 0.8]) for _ in keys]
                    r = rec(per, keys, rng.randint(80, 400), conf=conf)
                    r["compute"]["total_tokens"] = r["compute"]["completion_tokens"] + rng.randint(500, 3000)
                    if method == "independent" and rng.random() < 0.2:
                        r["finals"][0]["final"] = None
                        r["score"]["per_agent"][0] = 0.0
                    put(base, method, n, s, q, r)
                    if method == "blackboard":
                        cands = sorted(set(keys))
                        reviewed = len(cands) > 1 and rng.random() < 0.8
                        chosen = rng.choice(cands) if reviewed else (cands[0] if len(cands) == 1 else None)
                        (rev / f"s{s}_{q}.json").write_text(json.dumps({
                            "candidates": [{"label": k, "score": 1.0 if k == "a" else 0.0} for k in cands], "chosen": chosen,
                            "reviewed": reviewed, "actions": rng.randint(1, 9) if reviewed else 0,
                            "prompt_tokens": rng.randint(1000, 4000) if reviewed else 0,
                            "completion_tokens": rng.randint(50, 300) if reviewed else 0}))
        spec[name] = {"task": "gpqa", "least": 5, "pool": [[str(base), "independent", 4], [str(base), "independent", 2]],
                      "cells": [{"label": "independent N=4", "root": str(base), "protocol": "independent", "n": 4},
                                {"label": "blackboard N=4", "root": str(base), "protocol": "blackboard", "n": 4},
                                {"label": "blackboard N=4+review", "root": str(base), "protocol": "blackboard", "n": 4,
                                 "review": str(rev), "review_scale": 2.0},
                                {"label": "stopper", "root": str(base), "protocol": "stopper", "n": 4, "stops": True,
                                 "price": 1.5}]}
    return spec


def sorted_load(load):
    def wrapped(root, method, n):
        got = load(root, method, n)
        return {k: sorted(got[k], key=lambda r: (r.get("config") or {}).get("seed", 0)) for k in sorted(got)}
    return wrapped


def sorted_load(load):
    def wrapped(root, method, n):
        got = load(root, method, n)
        return {k: sorted(got[k], key=lambda r: (r.get("config") or {}).get("seed", 0)) for k in sorted(got)}
    return wrapped


def same(got, want) -> bool:
    if isinstance(want, float) and math.isfinite(want):
        return got == pytest.approx(want, rel=1e-9, abs=1e-12)
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(same(a, b) for a, b in zip(got, want))
    if isinstance(want, dict):
        return all(k in got and same(got[k], v) for k, v in want.items())
    return got == want


GOLDEN_C1 = {'completion': {'ga': 84.91666666666667, 'gb': 77.42307692307692}, 'total': {'ga': 658.3444444444444, 'gb': 684.1868131868132}, 'gpu': {'ga': 84.91666666666667, 'gb': 77.42307692307692}}
GOLDEN_ROWS = {
    'ga': [
        {'cell': 'independent N=4', 'questions': 12, 'answer': 0.5787037037037037, 'pass': 0.8333333333333334, 'member': 0.47222222222222215, 'curve': {'n': [1, 2], 'v': [0.2604166666666667, 0.28125], 't': [0.2604166666666667, 0.4729166666666667]}, 'n_budget': 2.9702322538436374, 'v1': 0.2604166666666667, 'v_b': 0.28125, 't_b': 0.4729166666666667, 't_full': 0.4729166666666667, 'pool_n': 2, 'e_vote': [0.2974537037037037, 0.08371490108091811], 'e_truth': [0.10578703703703703, 0.06762641991739801], 'e_one': [0.31828703703703703, 0.07964366776853882], 'n_t': None, 'eta': None, 'eta_ci': [0.6405744415422402, math.inf], 'level': 2},
        {'cell': 'blackboard N=4', 'questions': 12, 'answer': 0.7569444444444445, 'pass': 0.9444444444444443, 'member': 0.611111111111111, 'curve': {'n': [1, 2, 3, 4, 6, 8, 12], 'v': [0.44583333333333336, 0.48020833333333335, 0.5288194444444445, 0.5414930555555556, 0.5746527777777778, 0.5690972222222223, 0.615625], 't': [0.44583333333333336, 0.6375000000000001, 0.6958333333333333, 0.7833333333333332, 0.8270833333333334, 0.9187499999999998, 0.9520833333333334]}, 'n_budget': 2.8024206738632644, 'v1': 0.44583333333333336, 'v_b': 0.5192148938683532, 't_b': 0.6843078726420238, 't_full': 0.9520833333333334, 'pool_n': 12, 'e_vote': [0.23772955057609127, 0.08047672418475602], 'e_truth': [0.07263657180242065, 0.08046243741517663], 'e_one': [0.3111111111111111, 0.08600107012619511], 'n_t': 3.6984126984127004, 'eta': 1.3197207446069366, 'eta_ci': [0.6395890800890401, 2.4118343195266263], 'level': 2},
        {'cell': 'blackboard N=4+review', 'questions': 12, 'answer': 0.5555555555555556, 'pass': 0.9444444444444443, 'member': 0.611111111111111, 'curve': {'n': [1, 2, 3, 4, 6, 8, 12], 'v': [0.44583333333333336, 0.48020833333333335, 0.5288194444444445, 0.5414930555555556, 0.5746527777777778, 0.5690972222222223, 0.615625], 't': [0.44583333333333336, 0.6375000000000001, 0.6958333333333333, 0.7833333333333332, 0.8270833333333334, 0.9187499999999998, 0.9520833333333334]}, 'n_budget': 5.374877330716388, 'v1': 0.44583333333333336, 'v_b': 0.5642883307436485, 't_b': 0.8134087749427543, 't_full': 0.9520833333333334, 'pool_n': 12, 'e_vote': [-0.008732775188092914, 0.06861371347073734], 'e_truth': [-0.2578532193871988, 0.0643616823554318], 'e_one': [0.10972222222222221, 0.05601295294722074], 'n_t': 1.572463768115942, 'eta': 0.2925580755359038, 'eta_ci': [0.2011768355770101, 0.5126731990343457], 'level': 0},
        {'cell': 'stopper', 'questions': 12, 'answer': 0.8541666666666666, 'pass': 1.0, 'member': 0.6770833333333334, 'curve': {'n': [1, 2, 3, 4, 6, 8, 12], 'v': [0.44583333333333336, 0.48020833333333335, 0.5288194444444445, 0.5414930555555556, 0.5746527777777778, 0.5690972222222223, 0.615625], 't': [0.44583333333333336, 0.6375000000000001, 0.6958333333333333, 0.7833333333333332, 0.8270833333333334, 0.9187499999999998, 0.9520833333333334]}, 'n_budget': 2.8999018645731107, 'v1': 0.44583333333333336, 'v_b': 0.523953562861193, 't_b': 0.6899942754334314, 't_full': 0.9520833333333334, 'pool_n': 12, 'e_vote': [0.33021310380547375, 0.10691188028369163], 'e_truth': [0.16417239123323524, 0.10718515073442497], 'e_one': [0.4083333333333334, 0.09062772113643726], 'n_t': None, 'eta': None, 'eta_ci': [0.8276142131979696, math.inf], 'level': None, 'stops': True},
    ],
    'gb': [
        {'cell': 'independent N=4', 'questions': 12, 'answer': 0.6064814814814815, 'pass': 0.6944444444444443, 'member': 0.42361111111111116, 'curve': {'n': [1, 2, 3], 'v': [0.4604166666666667, 0.4822916666666666, 0.4888888888888889], 't': [0.4604166666666667, 0.6625, 0.78125]}, 'n_budget': 3.15330904675167, 'v1': 0.4604166666666667, 'v_b': 0.4888888888888889, 't_b': 0.78125, 't_full': 0.78125, 'pool_n': 3, 'e_vote': [0.11759259259259258, 0.06014923816414593], 'e_truth': [-0.17476851851851852, 0.06918170696119458], 'e_one': [0.1460648148148148, 0.0599644277553113], 'n_t': 1.722794959908362, 'eta': 0.5463451042589915, 'eta_ci': [0.35334231892429646, 0.7463059593722923], 'level': 1},
        {'cell': 'blackboard N=4', 'questions': 12, 'answer': 0.5833333333333334, 'pass': 0.8333333333333334, 'member': 0.513888888888889, 'curve': {'n': [1, 2, 3, 4, 6, 8, 12], 'v': [0.4583333333333333, 0.46145833333333336, 0.5107638888888888, 0.5447916666666667, 0.5625, 0.615625, 0.63125], 't': [0.4583333333333333, 0.6104166666666667, 0.7166666666666667, 0.75625, 0.7979166666666666, 0.81875, 0.8312499999999999]}, 'n_budget': 3.23224043715847, 'v1': 0.4583333333333333, 'v_b': 0.5186665148755312, 't_b': 0.7258595173041894, 't_full': 0.8312499999999999, 'pool_n': 12, 'e_vote': [0.06466681845780207, 0.057862243369461865], 'e_truth': [-0.14252618397085612, 0.08787119443646753], 'e_one': [0.125, 0.05147242732192426], 'n_t': 1.821917808219178, 'eta': 0.5636702601929157, 'eta_ci': [0.3616729965592372, 1.7452370115091964], 'level': 1},
        {'cell': 'blackboard N=4+review', 'questions': 12, 'answer': 0.4583333333333333, 'pass': 0.8333333333333334, 'member': 0.513888888888889, 'curve': {'n': [1, 2, 3, 4, 6, 8, 12], 'v': [0.4583333333333333, 0.46145833333333336, 0.5107638888888888, 0.5447916666666667, 0.5625, 0.615625, 0.63125], 't': [0.4583333333333333, 0.6104166666666667, 0.7166666666666667, 0.75625, 0.7979166666666666, 0.81875, 0.8312499999999999]}, 'n_budget': 5.247143566815698, 'v1': 0.4583333333333333, 'v_b': 0.555834083664514, 't_b': 0.7822321576419936, 't_full': 0.8312499999999999, 'pool_n': 12, 'e_vote': [-0.09750075033118066, 0.0888656875716044], 'e_truth': [-0.3238988243086604, 0.0988420077931438], 'e_one': [-4.625929269271485e-18, 0.07127388252344845], 'n_t': 1.0, 'eta': 0.19057988165680473, 'eta_ci': [0.13263801287469928, 0.34984382031835987], 'level': 0},
        {'cell': 'stopper', 'questions': 12, 'answer': 0.8541666666666666, 'pass': 1.0, 'member': 0.6875, 'curve': {'n': [1, 2, 3, 4, 6, 8, 12], 'v': [0.4583333333333333, 0.46145833333333336, 0.5107638888888888, 0.5447916666666667, 0.5625, 0.615625, 0.63125], 't': [0.4583333333333333, 0.6104166666666667, 0.7166666666666667, 0.75625, 0.7979166666666666, 0.81875, 0.8312499999999999]}, 'n_budget': 2.623033614836894, 'v1': 0.4583333333333333, 'v_b': 0.4921773518426524, 't_b': 0.6766139882430866, 't_full': 0.8312499999999999, 'pool_n': 12, 'e_vote': [0.36198931482401425, 0.08515264052812797], 'e_truth': [0.17755267842358, 0.09385811457527822], 'e_one': [0.3958333333333333, 0.08244794807279403], 'n_t': None, 'eta': None, 'eta_ci': [1.0743979041065623, math.inf], 'level': None, 'stops': True},
    ],
}
GOLDEN_MD = {
    'completion': {
        'ga': (
            '## ga\n'
            '\n'
            'solo run: 0.1k completion tokens, mean pool size 15\n'
            '\n'
            '| cell | q | n_B | Y | V(n_B) | T(n_B) | Y \u2212 V | Y \u2212 T | n_T | \u03b7 | level |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|\n'
            '| independent N=4 | 12 | 3.0 | 0.579 | 0.281 | 0.473 | +0.297 \u00b1 0.084 | +0.106 \u00b1 0.068 | >2 | >pool [0.64, \u221e] | 2 |\n'
            '| blackboard N=4 | 12 | 2.8 | 0.757 | 0.519 | 0.684 | +0.238 \u00b1 0.080 | +0.073 \u00b1 0.080 | 3.7 | 1.32 [0.64, 2.41] | 2 |\n'
            '| blackboard N=4+review | 12 | 5.4 | 0.556 | 0.564 | 0.813 | -0.009 \u00b1 0.069 | -0.258 \u00b1 0.064 | 1.6 | 0.29 [0.20, 0.51] | 0 |\n'
            '| stopper | 12 | 2.9 | 0.854 | 0.524 | 0.690 | +0.330 \u00b1 0.107 | +0.164 \u00b1 0.107 | \u00b7 | stops early | None |\n'
        ),
        'gb': (
            '## gb\n'
            '\n'
            'solo run: 0.1k completion tokens, mean pool size 15\n'
            '\n'
            '| cell | q | n_B | Y | V(n_B) | T(n_B) | Y \u2212 V | Y \u2212 T | n_T | \u03b7 | level |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|\n'
            '| independent N=4 | 12 | 3.2 | 0.606 | 0.489 | 0.781 | +0.118 \u00b1 0.060 | -0.175 \u00b1 0.069 | 1.7 | 0.55 [0.35, 0.75] | 1 |\n'
            '| blackboard N=4 | 12 | 3.2 | 0.583 | 0.519 | 0.726 | +0.065 \u00b1 0.058 | -0.143 \u00b1 0.088 | 1.8 | 0.56 [0.36, 1.75] | 1 |\n'
            '| blackboard N=4+review | 12 | 5.2 | 0.458 | 0.556 | 0.782 | -0.098 \u00b1 0.089 | -0.324 \u00b1 0.099 | 1.0 | 0.19 [0.13, 0.35] | 0 |\n'
            '| stopper | 12 | 2.6 | 0.854 | 0.492 | 0.677 | +0.362 \u00b1 0.085 | +0.178 \u00b1 0.094 | \u00b7 | stops early | None |\n'
        ),
    },
    'total': {
        'ga': (
            '## ga\n'
            '\n'
            'solo run: 0.7k total tokens, mean pool size 15\n'
            '\n'
            '| cell | q | n_B | Y | V(n_B) | T(n_B) | Y \u2212 V | Y \u2212 T | n_T | \u03b7 | level |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|\n'
            '| independent N=4 | 12 | 3.0 | 0.579 | 0.281 | 0.473 | +0.297 \u00b1 0.084 | +0.106 \u00b1 0.068 | >2 | >pool [0.62, \u221e] | 2 |\n'
            '| blackboard N=4 | 12 | 3.0 | 0.757 | 0.529 | 0.696 | +0.228 \u00b1 0.082 | +0.061 \u00b1 0.081 | 3.7 | 1.23 [0.60, 2.25] | 2 |\n'
            '| blackboard N=4+review | 12 | 7.6 | 0.556 | 0.570 | 0.901 | -0.015 \u00b1 0.071 | -0.346 \u00b1 0.076 | 1.6 | 0.21 [0.14, 0.36] | 0 |\n'
            '| stopper | 12 | 3.3 | 0.854 | 0.533 | 0.723 | +0.321 \u00b1 0.111 | +0.131 \u00b1 0.102 | \u00b7 | stops early | None |\n'
        ),
        'gb': (
            '## gb\n'
            '\n'
            'solo run: 0.7k total tokens, mean pool size 15\n'
            '\n'
            '| cell | q | n_B | Y | V(n_B) | T(n_B) | Y \u2212 V | Y \u2212 T | n_T | \u03b7 | level |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|\n'
            '| independent N=4 | 12 | 3.1 | 0.606 | 0.489 | 0.781 | +0.118 \u00b1 0.060 | -0.175 \u00b1 0.069 | 1.7 | 0.56 [0.36, 0.76] | 1 |\n'
            '| blackboard N=4 | 12 | 3.1 | 0.583 | 0.513 | 0.720 | +0.070 \u00b1 0.057 | -0.136 \u00b1 0.087 | 1.8 | 0.59 [0.38, 1.83] | 1 |\n'
            '| blackboard N=4+review | 12 | 7.2 | 0.458 | 0.593 | 0.810 | -0.135 \u00b1 0.086 | -0.352 \u00b1 0.101 | 1.0 | 0.14 [0.10, 0.26] | 0 |\n'
            '| stopper | 12 | 2.5 | 0.854 | 0.487 | 0.666 | +0.367 \u00b1 0.085 | +0.188 \u00b1 0.093 | \u00b7 | stops early | None |\n'
        ),
    },
    'gpu': {
        'ga': (
            '## ga\n'
            '\n'
            'solo run: 0.1k gpu tokens, mean pool size 15\n'
            '\n'
            '| cell | q | n_B | Y | V(n_B) | T(n_B) | Y \u2212 V | Y \u2212 T | n_T | \u03b7 | level |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|\n'
            '| independent N=4 | 12 | 3.0 | 0.579 | 0.281 | 0.473 | +0.297 \u00b1 0.084 | +0.106 \u00b1 0.068 | >2 | >pool [0.64, \u221e] | 2 |\n'
            '| blackboard N=4 | 12 | 2.8 | 0.757 | 0.519 | 0.684 | +0.238 \u00b1 0.080 | +0.073 \u00b1 0.080 | 3.7 | 1.32 [0.64, 2.41] | 2 |\n'
            '| blackboard N=4+review | 12 | 5.4 | 0.556 | 0.564 | 0.813 | -0.009 \u00b1 0.069 | -0.258 \u00b1 0.064 | 1.6 | 0.29 [0.20, 0.51] | 0 |\n'
            '| stopper | 12 | 4.3 | 0.854 | 0.547 | 0.791 | +0.307 \u00b1 0.117 | +0.063 \u00b1 0.092 | \u00b7 | stops early | None |\n'
        ),
        'gb': (
            '## gb\n'
            '\n'
            'solo run: 0.1k gpu tokens, mean pool size 15\n'
            '\n'
            '| cell | q | n_B | Y | V(n_B) | T(n_B) | Y \u2212 V | Y \u2212 T | n_T | \u03b7 | level |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|\n'
            '| independent N=4 | 12 | 3.2 | 0.606 | 0.489 | 0.781 | +0.118 \u00b1 0.060 | -0.175 \u00b1 0.069 | 1.7 | 0.55 [0.35, 0.75] | 1 |\n'
            '| blackboard N=4 | 12 | 3.2 | 0.583 | 0.519 | 0.726 | +0.065 \u00b1 0.058 | -0.143 \u00b1 0.088 | 1.8 | 0.56 [0.36, 1.75] | 1 |\n'
            '| blackboard N=4+review | 12 | 5.2 | 0.458 | 0.556 | 0.782 | -0.098 \u00b1 0.089 | -0.324 \u00b1 0.099 | 1.0 | 0.19 [0.13, 0.35] | 0 |\n'
            '| stopper | 12 | 3.9 | 0.854 | 0.543 | 0.754 | +0.312 \u00b1 0.096 | +0.101 \u00b1 0.099 | \u00b7 | stops early | None |\n'
        ),
    },
}
