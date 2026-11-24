import asyncio

from rci.ladder.reviewer import build_candidates, choose_tool, pick, render, review, submitted
from rci.agents.tooling import ToolResult, ToolSpec
from rci.tasks.base import SUBMIT_NAME
from fake_llm import ScriptedLLM

import rci.ladder.reviewer as ar

GOLD = "THE-REFERENCE-ANSWER"


def rec(answers, per, conf=None, forced=()):
    conf = conf or [0.8] * len(answers)
    finals = [{"agent": i, "failing": False, "forced": i in forced,
               "final": {"answer": a, "explanation": f"why {a}", "rationale": f"because {a}", "refs": [f"Doc{i}"], "confidence": c}}
              for i, (a, c) in enumerate(zip(answers, conf))]
    return {"task": "browsecomp", "example_id": "q1", "finals": finals, "score": {"score": max(per), "per_agent": per, "gold": GOLD}}


class Task:
    def tools(self, example, run):
        return [ToolSpec("browser_open", "open", {"type": "object", "properties": {}}, lambda ctx, args: ToolResult("page"))]

    def system_prompt(self, example):
        return "task system"

    def task_prompt(self, example, agent_id, n, rng):
        return example["question"]


def test_candidates_group_answers_and_drop_forced_members():
    r = rec(["a", "a", "b", "c"], [1.0, 1.0, 0.0, 0.0], forced=(3,))
    assert submitted(r) == [0, 1, 2]
    cs = {c.answer: c for c in build_candidates(r)}
    assert set(cs) == {"a", "b"} and cs["a"].members == [0, 1] and cs["a"].score == 1.0 and cs["b"].score == 0.0


def test_everyone_forced_keeps_everyone():
    r = rec(["a", "b"], [1.0, 0.0], forced=(0, 1))
    assert submitted(r) == [0, 1]


def test_the_reviewer_sees_no_verdict():
    r = rec(["right", "wrong"], [1.0, 0.0])
    text = render(build_candidates(r), 2)
    assert GOLD not in text and "score" not in text.lower() and "1.0" not in text


def test_candidate_order_does_not_follow_support():
    orders = set()
    for ex in range(12):
        r = rec(["a", "a", "a", "b", "b", "c"], [1.0] * 6)
        r["example_id"] = f"q{ex}"
        orders.add(build_candidates(r)[0].answer)
    assert len(orders) > 1


def test_at_most_eight_candidates_and_the_most_supported_stay():
    r = rec(list("abcdefghij") + ["a"], [0.0] * 11)
    cs = build_candidates(r)
    assert len(cs) == 8 and "a" in {c.answer for c in cs}


def test_choose_tool_rejects_anything_but_a_candidate():
    tool = choose_tool(build_candidates(rec(["a", "b"], [1.0, 0.0])))
    assert tool.fn(None, {"candidate": "9"}).final is None
    assert tool.fn(None, {"candidate": "Candidate 2"}).final == {"candidate": "2"}
    assert tool.terminal and tool.name == SUBMIT_NAME


def run_review(r, mode="verify", policy=None):
    llm = ScriptedLLM(policy or (lambda m, t, k, tag: [(SUBMIT_NAME, {"candidate": "1"})]))
    return asyncio.run(review(Task(), {"id": "q1", "question": "who?"}, r, llm, mode)), llm


def test_review_scores_the_choice_by_the_stored_verdict():
    r = rec(["a", "a", "b"], [1.0, 1.0, 0.0])
    res, _ = run_review(r)
    assert res["reviewed"] and res["chosen"] == "1"
    want = next(c["score"] for c in res["candidates"] if c["label"] == "1")
    assert pick(res, fallback=0.5) == (want, True)


def test_one_candidate_needs_no_review():
    res, llm = run_review(rec(["a", "a"], [1.0, 1.0]))
    assert llm.n_calls == 0 and not res["reviewed"] and pick(res, fallback=0.0) == (1.0, True)


def test_no_choice_falls_back_to_the_group_commitment():
    res, _ = run_review(rec(["a", "b"], [1.0, 0.0]), policy=lambda m, t, k, tag: "I cannot tell.")
    assert res["chosen"] is None and pick(res, fallback=0.25) == (0.25, False)


def test_read_mode_has_no_task_tools_and_verify_has():
    seen = {}

    def policy(m, tools, k, tag):
        seen["tools"] = tools
        return [(SUBMIT_NAME, {"candidate": "1"})]

    run_review(rec(["a", "b"], [1.0, 0.0]), "read", policy)
    assert seen["tools"] == [SUBMIT_NAME]
    run_review(rec(["a", "b"], [1.0, 0.0]), "verify", policy)
    assert "browser_open" in seen["tools"]


def test_free_commit_ignores_forced_members():
    r = rec(["right", "wrong", "wrong"], [1.0, 0.0, 0.0], forced=(1, 2))
    assert ar.vote_stake(r) == 0.0 and ar.free_commit(r) == 1.0
