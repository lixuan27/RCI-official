import asyncio
import random

import pytest

from rci.agents.runner import GroupRun, RunConfig
from rci.protocols.blackboard import BlackboardProtocol
from rci.protocols.independent import IndependentAgents
from rci.blackboard.store import BlackboardStore
from rci.tasks.audit.task import ContradictionAudit
from fake_llm import ScriptedLLM


def make_ds(**kw):
    return BlackboardStore(resource_exists=lambda r: r in {"Snippet1", "Snippet2"}, **kw)


def test_publish_rules():
    ds = make_ds()
    r = ds.publish(0, "I've read Snippet1", "it says X", "Snippet1 gave X", ["Snippet1"], step=0)
    assert r.ok and r.message.id == 0
    bad = ds.publish(0, "I've read more", "Y", "no names here", ["Snippet1"], step=0)
    assert not bad.ok and "Snippet1" in bad.error
    missing = ds.publish(0, "x", "y", "Message7 helped", ["Message7"], step=0)
    assert not missing.ok and "Message7" in missing.error
    assert len(ds.messages) == 1


def test_citation_weight_and_decay():
    ds = make_ds(alpha=0.3, gamma=0.98)
    ds.publish(0, "I've found the director", "Name A", "nothing", [], 0)
    ds.publish(1, "I've checked the director", "Name A confirmed", "Message0 gave the name", ["Message0"], 0)
    assert ds.messages[0].weight == pytest.approx(0.3)
    ds.publish(2, "I've extended it", "more", "Message0 gave the name", ["Message0"], 1)
    assert ds.messages[0].weight == pytest.approx(0.3 + 0.3 * 0.7)
    ds.decay()
    assert ds.messages[0].weight == pytest.approx((0.3 + 0.3 * 0.7) * 0.98)
    assert ds.messages[0].cited_by == [1, 2]


def test_weight_reorders_but_never_surfaces_unrelated():
    ds = make_ds(alpha=0.3)
    ds.publish(0, "I've identified the poet Smith", "Smith", "-", [], 0)
    ds.publish(1, "I've identified the poet Jones", "Jones", "-", [], 0)
    ds.publish(2, "I've computed an unrelated integral", "42", "-", [], 0)
    for _ in range(3):
        ds.publish(3, "I've used Jones", "ok", "Message1 gave Jones", ["Message1"], 0)
    hits = ds.search("poet identified", reader=5, step=1)
    ids = [m.id for m in hits]
    assert ids[0] == 1
    assert 2 not in ids
    bm = BlackboardStore(ranking="bm25")
    bm.publish(0, "I've identified the poet Smith", "Smith", "-", [], 0)
    bm.publish(1, "I've identified the poet Jones", "Jones", "-", [], 0)
    assert [m.id for m in bm.search("poet identified", None, 0)][0] == 0


def test_private_partition_and_render_hides_author():
    ds = make_ds(private=True)
    ds.publish(0, "I've found alpha", "a", "-", [], 0)
    ds.publish(1, "I've found alpha too", "b", "-", [], 0)
    assert [m.id for m in ds.search("alpha", reader=0, step=0)] == [0]
    rej = ds.publish(1, "x", "y", "Message0 gave it", ["Message0"], 0)
    assert not rej.ok
    text = ds.render(ds.messages[0])
    assert "author" not in text.lower() and "weight" not in text.lower()


def test_evidence_chain_is_transitive():
    ds = make_ds()
    ds.publish(0, "a", "a", "-", [], 0)
    ds.publish(0, "b", "b", "Message0 gave a", ["Message0"], 0)
    ds.publish(1, "c", "c", "Message1 gave b", ["Message1"], 0)
    assert ds.evidence_chain(["Message2"]) == [0, 1, 2]


def tiny_ca():
    snippets = [{"idx": i, "file": f"f{i // 2}.md", "text": f"f{i // 2}.md\nThe value of knob{i} is {i}."}
                for i in range(8)]
    snippets[5]["text"] = "f2.md\nThe value of knob1 is 99 seconds."
    return {"id": "toy_x1", "project": "toy", "level": 1, "snippets": snippets,
            "contradictions": [{"edited": 5, "partner": 1, "same_file": False, "why": "knob1 differs"}]}


def blackboard_policy(messages, tools, route_key, tag):
    agent = int(route_key.rsplit("a", 1)[1])
    n_tool_msgs = sum(1 for m in messages if m["role"] == "tool")
    if tag == "final":
        return [("submit_answer", {"pairs": [], "refs": [], "rationale": "nothing"})]
    if agent == 0:
        plan = [("fetch_snippet", {"index": 1}),
                ("publish_message", {"sub_task": "I've read Snippet1 about knob1", "artifact": "knob1 is 1",
                                     "rationale": "Snippet1 states it", "refs": ["Snippet1"]}),
                ("submit_answer", {"pairs": [], "refs": ["Message0"], "rationale": "Message0 is my note"})]
    else:
        plan = [("search_datastore", {"query": "knob1"}),
                ("fetch_snippet", {"index": 5}),
                ("report_contradiction", {"snippet_a": 1, "snippet_b": 5, "quote_a": "The value of knob1 is 1",
                                          "quote_b": "The value of knob1 is 99 seconds", "reason": "1 vs 99",
                                          "refs": ["Message0"], "rationale": "Message0 gave knob1"}),
                ("submit_answer", {"pairs": [{"a": 1, "b": 5, "reason": "knob1"}], "refs": ["Message0"],
                                   "rationale": "Message0 gave the other half"})]
    i = min(n_tool_msgs, len(plan) - 1)
    return [plan[i]]


class TinyCA(ContradictionAudit):
    def __init__(self, inst):
        super().__init__()
        self.inst = inst

    def examples(self, limit=None):
        return [self.inst]


def test_end_to_end_blackboard_and_independent():
    task = TinyCA(tiny_ca())
    ex = task.examples()[0]
    cfg = RunConfig(n_agents=2, max_actions=10, max_steps=10, seed=0)
    rec = asyncio.run(GroupRun(task, ex, BlackboardProtocol(), cfg, ScriptedLLM(blackboard_policy)).run())
    assert rec["score"]["score"] == 1.0
    msgs = rec["method_state"]["messages"]
    assert any("contradiction" in m["sub_task"] for m in msgs)
    assert rec["finals"][1]["evidence_chain"] == [0]
    assert rec["compute"]["total_tokens"] > 0

    def react_policy(messages, tools, route_key, tag):
        assert "publish_message" not in tools
        return [("submit_answer", {"pairs": []})]

    rec2 = asyncio.run(GroupRun(task, ex, IndependentAgents(), cfg, ScriptedLLM(react_policy)).run())
    assert rec2["score"]["score"] == 0.0


def test_action_limit_forces_final_and_failing_agent():
    task = TinyCA(tiny_ca())
    ex = task.examples()[0]

    def looping(messages, tools, route_key, tag):
        if tag == "final":
            return [("submit_answer", {"pairs": [{"a": 5, "b": 1}], "refs": [], "rationale": "-"})]
        return [("search_snippet", {"pattern": "knob"})]

    cfg = RunConfig(n_agents=1, n_failing=1, max_actions=4, max_steps=20, seed=1)
    rec = asyncio.run(GroupRun(task, ex, BlackboardProtocol(), cfg, ScriptedLLM(looping)).run())
    a0, a1 = rec["agents"]
    assert a1["actions"] == 4 and a1["final_forced"] and rec["score"]["score"] == 1.0
    assert a0["failing"] and a0["final"] is None and a0["actions"] == 0
    assert rec["score"]["per_agent"][0] is None
    assert all(c["error"] == "injected-failure" for c in a0["calls"])


def test_summarization_keeps_recent_turns():
    task = TinyCA(tiny_ca())
    ex = task.examples()[0]

    def looping(messages, tools, route_key, tag):
        if tag == "final":
            return [("submit_answer", {"pairs": []})]
        return [("search_snippet", {"pattern": "knob"})]

    cfg = RunConfig(n_agents=1, max_actions=12, max_steps=30)
    run = GroupRun(task, ex, IndependentAgents(), cfg, ScriptedLLM(looping))
    asyncio.run(run.run())
    ag = run.agents[0]
    assert ag.summary and ag.summary.startswith("SUMMARY")
    assert len(ag.history) <= 9
    assert any(c.tag == "summary" for c in ag.calls)


def test_dropout_skips_steps():
    task = TinyCA(tiny_ca())
    ex = task.examples()[0]

    def looping(messages, tools, route_key, tag):
        if tag == "final":
            return [("submit_answer", {"pairs": [], "refs": [], "rationale": "-"})]
        return [("search_snippet", {"pattern": "knob"})]

    cfg = RunConfig(n_agents=4, max_actions=1000, max_steps=50, dropout_p=0.8, seed=3)
    rec = asyncio.run(GroupRun(task, ex, BlackboardProtocol(), cfg, ScriptedLLM(looping)).run())
    present = [s["present"] for s in rec["step_log"]]
    assert sum(present) < 4 * 50 * 0.4
