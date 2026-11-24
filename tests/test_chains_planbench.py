import pytest

from rci.chains.planbench import accept as A
from rci.chains.planbench import domain as D
from rci.chains.planbench import generate as G
from rci.chains.planbench import protocol as PR
from rci.chains.planbench import text as T

# a on b, c on the table; goal: c on a
INIT = frozenset({("handempty",), ("on", "a", "b"), ("ontable", "b"), ("ontable", "c"), ("clear", "a"), ("clear", "c")})
GOAL = frozenset({("on", "c", "a")})
PLAN = [("pick-up", "c"), ("stack", "c", "a")]


def test_simulator_accepts_a_correct_plan_and_names_the_first_failure():
    assert D.validate(INIT, GOAL, PLAN) == (True, None, 2)
    ok, err, steps = D.validate(INIT, GOAL, [("pick-up", "b")])
    assert not ok and err.startswith("step 1") and steps == 0
    ok, err, _ = D.validate(INIT, GOAL, [("pick-up", "c"), ("put-down", "c")])
    assert not ok and err.startswith("goal not reached")
    assert D.validate(INIT, GOAL, []) == (False, "empty plan", 0)
    assert not D.validate(INIT, GOAL, [("fly", "c")])[0]
    assert not D.validate(INIT, GOAL, [("stack", "c")])[0]
    assert D.first_failure(INIT, GOAL, [("pick-up", "c"), ("pick-up", "a")]) == ("step", 2)
    assert D.first_failure(INIT, GOAL, PLAN) == ("valid", None)
    assert D.first_failure(INIT, GOAL, []) == ("noplan", None)


def test_bfs_planner_is_optimal_and_states_stay_wellformed():
    plan = D.plan_optimal(INIT, GOAL)
    assert len(plan) == 2 and D.validate(INIT, GOAL, plan)[0]
    for st in D.reachable(INIT):
        assert D.wellformed(st) is None
    assert D.wellformed(INIT | {("holding", "c")}) is not None


def test_length_table_agrees_with_the_planner():
    states = G.all_states(4)
    assert len(states) == 73 and len(G.all_states(5)) == 501
    for i, g in [(0, 5), (3, 40), (70, 2)]:
        ini, gl = G.state_facts(states[i]), G.goal_facts(states[g])
        plan = D.plan_optimal(ini, gl)
        assert len(plan) == G.length_table(4)[g][i]


def test_four_block_free_cells_match_the_table():
    assert G.free_cells(4) == G.FREE_CELLS[4]
    assert len(G.official_digests()) == 262


def test_iso_key_is_invariant_under_renaming():
    ren = {"a": "c", "b": "a", "c": "b"}
    r = lambda fs: frozenset((f[0],) + tuple(ren[x] for x in f[1:]) for f in fs)  # noqa: E731
    assert G.iso_key(INIT, GOAL) == G.iso_key(r(INIT), r(GOAL))
    assert G.canonical_key(INIT, GOAL) != G.canonical_key(r(INIT), r(GOAL))


@pytest.fixture(scope="module")
def sets():
    spec = [("sel", 40, {4: "1/4", 5: "3/4"}), ("conf_r1", 50, {4: "1/50", 5: "49/50"}), ("train_r1", 30, {4: 0, 5: 1})]
    return spec, G.build(spec, "test", {4: 8, 5: 20})


def test_generated_splits_exclude_official_problems_and_are_disjoint(sets):
    spec, built = sets
    keys = [r["key"] for rows in built.values() for r in rows]
    assert len(keys) == len(set(keys)) == 120
    for rows in built.values():
        for r in rows:
            ini, gl = A.facts(r["initial"]), A.facts(r["goal"])
            assert not G.is_official(ini, gl) and r["n_blocks"] in (4, 5)
            assert r["plan_length"] == len(D.plan_optimal(ini, gl))
            ex = r["example"]
            assert ex["n_blocks"] == r["n_blocks"] and D.validate(A.facts(ex["initial"]), A.facts(ex["goal"]),
                                                                  [tuple(a) for a in ex["plan"]])[0]
            assert r["query"].startswith(T.INTRO) and r["query"].endswith("[PLAN]")
    assert sum(r["n_blocks"] == 4 for r in built["sel"]) == 10
    assert sum(r["n_blocks"] == 4 for r in built["conf_r1"]) == 1
    assert all(r["n_blocks"] == 5 for r in built["train_r1"])


def test_splits_are_prefix_stable(sets):
    spec, built = sets
    again = G.build(spec[:2], "test", {4: 8, 5: 20})
    assert [r["key"] for r in again["sel"]] == [r["key"] for r in built["sel"]]
    assert [r["key"] for r in again["conf_r1"]] == [r["key"] for r in built["conf_r1"]]


def test_quota_follows_the_official_histogram():
    q = G.quota(300, {4: "1/4", 5: "3/4"})
    assert sum(q[4].values()) == 75 and sum(q[5].values()) == 225
    whole = G.length_quota(300, G.OFFICIAL_LENGTH_HIST)
    assert {k: q[4].get(k, 0) + q[5].get(k, 0) for k in whole} == whole
    with pytest.raises(G.GenError):
        G.block_counts(301, {4: "1/4", 5: "3/4"})


def test_statement_text_is_the_official_form():
    s = T.statement_text(INIT, GOAL, PLAN)
    assert s.startswith("\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the orange block is clear, the hand is empty")
    assert "the red block is on top of the blue block" in s and "the blue block is on the table" in s
    assert s.endswith("[PLAN]\npick up the orange block\nstack the orange block on top of the red block\n[PLAN END]\n")
    assert "My goal is to have that the orange block is on top of the red block." in s


def test_extraction_follows_the_official_rule():
    text = ("1. Pick up the orange block\n- stack the orange block on top of the red block\n(unstack yellow orange)\n"
            "unstack the red block\n[PLAN END]")
    e = T.extract(text)
    assert e["actions"] == PLAN and e["n_dropped"] == 1 and e["n_other"] == 2
    assert T.extract("pick up and stack the red block")["n_multi"] == 1


def test_judge_needs_the_cut_reading_too():
    good = "pick up the orange block\nstack the orange block on top of the red block\n[PLAN END]"
    v = T.judge(good, INIT, GOAL)
    assert v["official_ok"] and v["strict_ok"] and v["clean"] and not v["tail"] and v["end"]
    v = T.judge("pick up the orange block\n[PLAN END]\nstack the orange block on top of the red block", INIT, GOAL)
    assert v["official_ok"] and not v["strict_ok"] and v["tail"]
    v = T.judge("Sure!\n" + good, INIT, GOAL)
    assert v["strict_ok"] and not v["clean"]
    assert not T.judge("", INIT, GOAL)["official_ok"]
    assert T.judge("<think>x</think>" + good, INIT, GOAL)["think_tag"]


def test_verdict_parsing_and_jury_commitment():
    text = "PLAN A\n1. x | OK | y\nGOAL: MET\nVERDICT A: VALID\nPLAN B\n1. x | FAIL | z\nVERDICT B: INVALID at step 1\n" \
           "PLAN C\n1. x | FAIL | z\nPLAN D\nGOAL: NOT MET"
    v = PR.parse_verdicts(text, "ABCD")
    assert v["A"]["valid"] and v["B"] == {"valid": False, "why": "step", "step": 1, "how": "verdict"}
    assert v["C"]["how"] == "fallback" and v["C"]["step"] == 1 and v["D"]["why"] == "goal"
    ok, bad = {"valid": True}, {"valid": False}
    reviews = {a: {t: (ok if t == 2 else bad) for t in range(4)} for a in range(4)}
    c = PR.commit(reviews, "s", 0, "k")
    assert c["t"] == 2 and c["tie_width"] == 1
    reviews = {a: {t: (ok if t == a else bad) for t in range(4)} for a in range(4)}
    assert PR.commit(reviews, "s", 0, "k")["tie_width"] == 4
    order = PR.jury_order("s", 0, "k", 1)
    assert sorted(order) == [0, 1, 2, 3] and order == PR.jury_order("s", 0, "k", 1)


def call(content, ct=100, fin="stop", prompt=(1, 2, 3), ids=(9, 7)):
    return {"content": content, "fin": fin, "ct": ct, "pt": len(prompt), "error": None, "prompt_ids": list(prompt),
            "ids": list(ids), "think_text": False}


GOOD = "pick up the orange block\nstack the orange block on top of the red block\n[PLAN END]"
BAD = "pick up the red block\n[PLAN END]"


def fake_record(drafts, finals, set_name="train_r1", seed=0, key="p1"):
    reviews = []
    for a in range(4):
        order = PR.jury_order(set_name, seed, key, a)
        lines = [f"VERDICT {lab}: {'VALID' if finals[t] == GOOD else 'INVALID at step 1'}"
                 for lab, t in zip(PR.LABELS, order)]
        reviews.append(call("\n".join(lines), 50))
    rv = PR.reviews_of(set_name, seed, key, [r["content"] for r in reviews])
    return {"key": key, "set": set_name, "seed": seed, "mode": "think", "draft": [call(x) for x in drafts],
            "check": [call("VERDICT A: VALID", 20) for _ in range(4)], "final": [call(x) for x in finals],
            "review": reviews, "commit": PR.commit(rv, set_name, seed, key)}


ROW = {"id": "p1", "initial": [list(f) for f in INIT], "goal": [list(f) for f in GOAL], "plan_length": 2}
CFG = {"r1_sources": ["draft"], "r2_per_q": 1, "r3_per_q": 2, "share_cap": {"R2": 1.0, "R3": 1.0}, "len_hi": 10.0,
       "len_drop_max_share": 0.25, "require_plan_end": False, "row_max_tokens": 16384}


def test_rows_of_a_problem_follow_the_check():
    rec = fake_record([GOOD, BAD, BAD, BAD], [GOOD, GOOD, BAD, BAD])
    rec["_row"], rec["_round"] = ROW, 1
    v = A.judge_record(rec, ROW)
    assert v["commit"]["ok"] and v["oracle"]
    pp = A.plan_problem(rec, v, 1, CFG)
    kinds = [x["kind"] for x in pp["rows"]]
    assert pp["cls"] == "trainable" and kinds.count("R1") == 1 and kinds.count("R2") == 1 and kinds.count("R3") == 2
    rows = A.encode([dict(x, rec=rec) for x in pp["rows"]], 7, 100)
    assert all(r["ids"][-1] == 7 and r["spans"] == [[3, 2]] for r in rows)
    archived = fake_record([GOOD] * 4, [GOOD] * 4)
    archived["_row"] = ROW
    assert A.plan_problem(archived, A.judge_record(archived, ROW), 1, CFG)["cls"] == "archived"
    frontier = fake_record([BAD] * 4, [BAD] * 4)
    frontier["_row"] = ROW
    assert A.plan_problem(frontier, A.judge_record(frontier, ROW), 1, CFG)["cls"] == "frontier"


def test_a_review_with_a_wrong_verdict_is_not_exact():
    rec = fake_record([GOOD, BAD, BAD, BAD], [GOOD, GOOD, BAD, BAD])
    finals = A.judge_record(rec, ROW)["final"]
    assert A.review_exact(rec["review"][0]["content"], finals, ROW, rec, 0)
    flipped = rec["review"][0]["content"].replace("INVALID at step 1", "VALID")
    assert not A.review_exact(flipped, finals, ROW, rec, 0)


def test_truncated_or_tailed_texts_are_not_valid_rows():
    v = A.judge_text(call(GOOD, fin="length"), ROW)
    assert v["strict_ok"] and not A.valid(v)
    v = A.judge_text(call(GOOD + "\nnotes after the plan"), ROW)
    assert not A.valid(v)


def test_health_lines_against_the_base():
    h = {"H1": [None, 1.25], "H2": [None, 1.25], "H3_abs": 0.005, "H3_add": 0.005, "H4_add": 0.02, "H6_sub": 0.05,
         "H7_add": 0.01, "H7_line_repeats": 6, "H7_zr": 0.08, "H7_min_chars": 400, "H8_abs": 0.001, "H8_add": 0.001,
         "runaway_ratio": 1.5}
    rec = fake_record([GOOD, BAD, BAD, BAD], [GOOD, GOOD, BAD, BAD])
    base = A.stats([A.judge_record(rec, ROW)], h)
    assert all(x.ok for x in A.health(base, base, h))
    long = dict(base, draft_ct=base["draft_ct"] * 1.3, runaway=2.0)
    assert [x.name.split()[0] for x in A.health(long, base, h) if not x.ok] == ["H1", "HR"]
