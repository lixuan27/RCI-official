import asyncio
import random
from types import SimpleNamespace

from rci.agents.tooling import AgentCtx, call_tool
from rci.tasks.audit.task import ContradictionAudit
from rci.train.credit import agent_credits


def _example(n=24, files=6):
    sn = [{"idx": i, "file": f"f{i % files}.md", "text": f"f{i % files}.md\nknob{i} timeout is {i} seconds"}
          for i in range(n)]
    con = [{"edited": 0, "partner": 7, "same_file": False}, {"edited": 2, "partner": 8, "same_file": True}]
    return {"id": "toy_x4", "project": "toy", "snippets": sn, "contradictions": con}


def _ctx(agent, n):
    return AgentCtx(agent_id=agent, run=SimpleNamespace(cfg=SimpleNamespace(n_agents=n), step=0))


def _tools(task, ex):
    return {t.name: t for t in task.tools(ex, None)}


def test_shares_are_fixed_balanced_and_cover_the_corpus():
    ex = _example()
    for mode in ("snippet", "file"):
        t = ContradictionAudit(root="nowhere", partition=mode)
        sh = t.shares(ex, 4)
        assert set(sh) == set(range(24)) and set(sh.values()) == {0, 1, 2, 3}
        assert ContradictionAudit(root="nowhere", partition=mode).shares(ex, 4) == sh
        sizes = [sum(1 for v in sh.values() if v == k) for k in range(4)]
        assert max(sizes) - min(sizes) <= 4
        if mode == "file":
            assert all(len({sh[s["idx"]] for s in ex["snippets"] if s["file"] == f}) == 1 for f in {s["file"] for s in ex["snippets"]})
    assert set(ContradictionAudit(root="nowhere", partition="snippet").shares(ex, 1).values()) == {0}


def test_tools_only_reach_the_agents_share():
    ex = _example()
    t = ContradictionAudit(root="nowhere", partition="snippet")
    sh = t.shares(ex, 4)
    tools = _tools(t, ex)
    mine = [i for i, k in sh.items() if k == 1]
    other = next(i for i, k in sh.items() if k != 1)
    ok = asyncio.run(call_tool(tools["fetch_snippet"], _ctx(1, 4), {"index": mine[0]}))
    no = asyncio.run(call_tool(tools["fetch_snippet"], _ctx(1, 4), {"index": other}))
    assert ok.ok and not no.ok and "held by another agent" in no.text
    hits = asyncio.run(call_tool(tools["search_snippet"], _ctx(1, 4), {"pattern": "timeout"})).text
    shown = {int(x.split()[0]) for x in hits.split("Snippet")[1:]}
    assert shown and shown <= set(mine)
    assert asyncio.run(call_tool(tools["fetch_snippet"], _ctx(5, 4), {"index": mine[0]})).ok
    assert asyncio.run(call_tool(tools["fetch_snippet"], _ctx(1, 1), {"index": other})).ok


def test_prompt_lists_the_share_and_plain_ca_is_unchanged():
    ex = _example()
    t = ContradictionAudit(root="nowhere", partition="snippet")
    p = t.task_prompt(ex, 2, 4, random.Random(0))
    mine = sorted(i for i, k in t.shares(ex, 4).items() if k == 2)
    assert f"{len(mine)} snippets" in p and all(f"Snippet{i}" in p for i in mine)
    plain = ContradictionAudit(root="nowhere")
    assert "split among" not in plain.task_prompt(ex, 2, 4, random.Random(0))
    assert asyncio.run(call_tool(_tools(plain, ex)["fetch_snippet"], _ctx(1, 4), {"index": 23})).ok


def test_exact_pairs_score_and_credit():
    ex = _example()
    part, plain = ContradictionAudit(root="nowhere", partition="snippet"), ContradictionAudit(root="nowhere")
    finals = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 5}]}},
              {"agent": 1, "final": {"pairs": [{"a": 8, "b": 2}]}}]
    assert plain.score_group(ex, finals)["score"] == 1.0
    sc = part.score_group(ex, finals)
    assert sc["score"] == 0.5 and sc["per_agent"] == [0.0, 0.5] and sc["strict"]
    rec = {"task": "audit", "score": sc, "finals": finals, "method_state": {"messages": []},
           "agents": [{"id": 0}, {"id": 1}]}
    cred = agent_credits(rec, ex, "shapley", lam=0.0)
    assert cred == {0: 0.0, 1: 1.0}


def test_strict_without_partition_and_precision_fields():
    ex = _example()
    strict, plain = ContradictionAudit(root="nowhere", strict=True), ContradictionAudit(root="nowhere")
    finals = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 5}, {"a": 0, "b": 9}, {"a": 3, "b": 4}]}},
              {"agent": 1, "final": {"pairs": [{"a": 2, "b": 8}]}}]
    p, s = plain.score_group(ex, finals), strict.score_group(ex, finals)
    assert p["score"] == 1.0 and "strict" not in p
    assert s["score"] == 0.5 and s["strict"]
    assert p["n_exact_pairs"] == s["n_exact_pairs"] == 1 and p["precision"] == 0.25
    assert asyncio.run(call_tool(_tools(strict, ex)["fetch_snippet"], _ctx(1, 4), {"index": 23})).ok


def test_share_hint_only_when_asked():
    ex = _example()
    plain = ContradictionAudit(root="nowhere", partition="snippet").task_prompt(ex, 2, 4, random.Random(0))
    hint = ContradictionAudit(root="nowhere", partition="snippet", share_hint=True).task_prompt(ex, 2, 4, random.Random(0))
    assert "tell the other agents what your snippets say" in hint and "tell the other" not in plain
    assert plain.split(" Your starting")[1] == hint.split(" Your starting")[1]
    assert "tell the other" not in ContradictionAudit(root="nowhere", partition="snippet", share_hint=True).task_prompt(
        ex, 0, 1, random.Random(0))


def test_f1_reward_charges_wrong_pairs_and_credit_blames_the_reporter():
    ex = _example()
    f1 = ContradictionAudit(root="nowhere", f1=True)
    assert f1.strict
    one = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 7}]}}]
    spray = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 7}] + [{"a": 0, "b": k} for k in (3, 4, 5, 6)]}}]
    s1, s2 = f1.score_group(ex, one), f1.score_group(ex, spray)
    assert s1["recall"] == s2["recall"] == 0.5
    assert abs(s1["score"] - 2 * 1.0 * 0.5 / 1.5) < 1e-9
    assert abs(s2["score"] - 2 * 0.2 * 0.5 / 0.7) < 1e-9 and s2["score"] < s1["score"]
    assert s1["per_agent"] == [s1["score"]]
    assert f1.score_group(ex, [{"agent": 0, "final": {"pairs": [{"a": 1, "b": 2}]}}])["score"] == 0.0
    finals = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 7}]}},
              {"agent": 1, "final": {"pairs": [{"a": 2, "b": 8}, {"a": 3, "b": 9}]}}]
    sc = f1.score_group(ex, finals)
    rec = {"task": "audit", "score": sc, "finals": finals, "method_state": {"messages": []}, "agents": [{"id": 0}, {"id": 1}]}
    cred = agent_credits(rec, ex, "shapley", lam=0.0)
    assert abs(sum(cred.values()) / 2 - sc["score"]) < 1e-9
    assert abs(cred[0] - 2 * (2 / 3 + 0.8 - 0.5) / 2) < 1e-9 and abs(cred[1] - 2 * (0.5 + 0.8 - 2 / 3) / 2) < 1e-9
    assert cred[0] > cred[1]


def test_f1_shapley_gives_a_pure_sprayer_negative_credit_and_keeps_blame_out_of_the_citation_flow():
    ex = _example()
    f1 = ContradictionAudit(root="nowhere", f1=True)
    msgs = [{"id": 0, "author": 0, "msg_refs": []}]
    finals = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 7}, {"a": 2, "b": 8}], "refs": []}},
              {"agent": 1, "final": {"pairs": [{"a": 2, "b": 8}], "refs": []}},
              {"agent": 2, "final": {"pairs": [{"a": 0, "b": k} for k in (3, 4, 5)], "refs": ["Message0"]}}]
    sc = f1.score_group(ex, finals)
    rec = {"task": "audit", "score": sc, "finals": finals, "method_state": {"messages": msgs},
           "agents": [{"id": i} for i in range(3)]}
    for mode in ("shapley", "shapley_flow"):
        cred = agent_credits(rec, ex, mode, lam=0.0)
        assert abs(sum(cred.values()) / 3 - sc["score"]) < 1e-9
        assert cred[2] < 0 < cred[1] < cred[0]
    assert agent_credits(rec, ex, "shapley_flow", lam=0.0)[0] == agent_credits(rec, ex, "shapley", lam=0.0)[0]


def test_f1_shapley_sampling_matches_exact():
    from rci.train.credit import game_values

    def f1_shapley(ex, finals, max_exact=12, samples=2000):
        import rci.train.credit as cr
        orig = cr._shapley
        cr._shapley = lambda o, c, v, e: orig(o, c, v, e, max_exact=max_exact, samples=samples)
        try:
            rec = {"task": "audit", "score": {"recall": 0.0}, "finals": finals}
            return game_values(rec, ex)[1]
        finally:
            cr._shapley = orig
    rng = random.Random(1)
    ex = {"contradictions": [{"edited": i, "partner": i + 20} for i in range(6)]}
    finals = []
    for i in range(8):
        pairs = []
        for _ in range(rng.randrange(1, 4)):
            j = rng.randrange(8)
            pairs.append({"a": j, "b": j + rng.choice((20, 21))})
        finals.append({"agent": i, "final": {"pairs": pairs}})
    exact, approx = f1_shapley(ex, finals), f1_shapley(ex, finals, max_exact=4, samples=4000)
    assert abs(sum(exact.values()) - sum(approx.values())) < 1e-9
    assert max(abs(exact[a] - approx[a]) for a in exact) < 0.02


def test_priced_claims_charge_each_wrong_pair_and_credit_splits_pair_by_pair():
    ex = _example()
    t = ContradictionAudit(root="nowhere", penalty=0.5)
    assert t.strict
    one = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 7}]}}]
    spray = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 7}] + [{"a": 0, "b": k} for k in (3, 4, 5, 6)]}}]
    s1, s2 = t.score_group(ex, one), t.score_group(ex, spray)
    assert s1["score"] == 0.5 and s2["score"] == (1 - 0.5 * 4) / 2 and s1["recall"] == s2["recall"] == 0.5
    assert s1["penalty"] == 0.5 and abs(s2["f1"] - 2 * 0.2 * 0.5 / 0.7) < 1e-9
    assert t.score_group(ex, [{"agent": 0, "final": {"pairs": []}}])["score"] == 0.0
    finals = [{"agent": 0, "final": {"pairs": [{"a": 0, "b": 7}, {"a": 0, "b": 3}]}},
              {"agent": 1, "final": {"pairs": [{"a": 7, "b": 0}]}}]
    sc = t.score_group(ex, finals)
    rec = {"task": "audit", "score": sc, "finals": finals, "method_state": {"messages": []},
           "agents": [{"id": 0}, {"id": 1}]}
    from rci.train.credit import game_values
    R, sh, _ = game_values(rec, ex)
    assert abs(R - 0.25) < 1e-9 and abs(sh[0] - 0.0) < 1e-9 and abs(sh[1] - 0.25) < 1e-9
    cred = agent_credits(rec, ex, "shapley_flow", lam=0.0)
    assert abs(sum(cred.values()) / 2 - R) < 1e-9 and cred[1] > cred[0]
