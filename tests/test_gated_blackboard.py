import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from rci.agents.runner import GroupRun, RunConfig
from rci.agents.tooling import AgentCtx
from rci.protocols.blackboard import BlackboardProtocol
from rci.registry import make_task
from rci.blackboard.store import BlackboardStore
from rci.tasks.hiddenbench.task import FORWARD_RATIONALE, FORWARD_TITLE
from fake_llm import ScriptedLLM

ROW = {"id": 1, "description": "Pick a route for the convoy.", "shared_information": ["The western road looks fine for buses today."],
       "hidden_information": ["The western bridge collapsed overnight in the storm.", "Eastern roads were cleared by snowploughs."],
       "possible_answers": ["West", "East"], "correct_answer": "East"}


def task(tmp_path, **kw):
    (tmp_path / "benchmark.json").write_text(json.dumps([ROW]))
    return make_task("hiddenbench", root=str(tmp_path), hint=True, **kw)


def store_run(t, ex, step=0):
    run = SimpleNamespace(cfg=SimpleNamespace(n_agents=2), step=step, state={})
    ds = BlackboardStore(token_limit=512, extra_validator=t.publish_validator(ex, run), seed=0)
    run.state["datastore"] = ds
    return run, ds


def test_forwarded_lines_must_be_facts_word_for_word(tmp_path):
    t = task(tmp_path, conclude="complete")
    ex = t.examples()[0]
    assert t.forward_error(ex, "- The western bridge collapsed overnight in the storm.\n2. Eastern roads were cleared") is None
    assert "word for word" in t.forward_error(ex, "- The bridge is out, so go East.")
    assert "word for word" in t.forward_error(ex, "- East.")
    assert "at least one fact" in t.forward_error(ex, "  \n")
    assert "forward_facts" in t.task_prompt(ex, 0, 2, None) and "forward_facts" not in task(tmp_path).task_prompt(ex, 0, 2, None)
    with pytest.raises(ValueError):
        task(tmp_path, conclude="some")


def test_conclusions_wait_until_every_fact_has_been_read_in_an_earlier_turn(tmp_path):
    t = task(tmp_path, conclude="complete")
    ex = t.examples()[0]
    run, ds = store_run(t, ex)
    fwd = t.blackboard_tools(ex, run, ds)[0]
    assert fwd.name == "forward_facts"
    assert not ds.publish(0, "Weighed the options", "East is best.", "", [], 0).ok
    assert "have not yet read" in ds.events[-1]["error"]
    assert fwd.fn(AgentCtx(agent_id=1, run=run), {"facts": ["Eastern roads were cleared by snowploughs."]}).ok
    assert not fwd.fn(AgentCtx(agent_id=1, run=run), {"facts": ["The east is clearly safer."]}).ok
    assert not ds.publish(0, FORWARD_TITLE, "- Eastern roads were cleared by snowploughs.", "My reading: go East.", [], 0).ok
    ds.search("eastern roads", reader=0, step=0)
    assert not t.holds_every_fact(ex, run, 0)
    run.step = 1
    assert t.holds_every_fact(ex, run, 0) and not t.holds_every_fact(ex, run, 1)
    assert ds.publish(0, "Weighed the options", "East: the bridge is out and the eastern roads are clear.", "", [], 1).ok
    assert not ds.publish(1, "Weighed the options", "West.", "", [], 1).ok


def test_with_none_nobody_concludes_and_the_ordinary_store_is_untouched(tmp_path):
    t = task(tmp_path, conclude="none")
    ex = t.examples()[0]
    run, ds = store_run(t, ex, step=5)
    for i in (0, 1):
        assert ds.publish(i, FORWARD_TITLE, "- " + ROW["hidden_information"][i], FORWARD_RATIONALE, [], 0).ok
    for i in (0, 1):
        ds.search("bridge roads", reader=i, step=0)
    assert t.holds_every_fact(ex, run, 0)
    assert not ds.publish(0, "Weighed the options", "East.", "", [], 5).ok
    assert "have not yet read" not in ds.events[-1]["error"]
    run.step = 0
    assert not ds.publish(1, "Weighed the options", "West.", "", [], 0).ok
    assert "have not yet read" in ds.events[-1]["error"]
    plain = task(tmp_path)
    assert plain.publish_validator(ex, run) is None and plain.blackboard_tools(ex, run, ds) == []


def test_a_store_run_with_the_reader_gate_refuses_then_accepts_a_conclusion(tmp_path):
    t = task(tmp_path, conclude="complete", confidence=True)
    ex = t.examples()[0]
    calls = defaultdict(int)

    def policy(messages, tools, route_key, tag):
        i = int(route_key.rsplit("a", 1)[1])
        k = calls[i]
        calls[i] += 1
        if k == 0:
            return [("forward_facts", {"facts": [ROW["hidden_information"][i]]}),
                    ("publish_message", {"sub_task": "I've weighed the options", "artifact": "West is best.", "rationale": "", "refs": []})]
        if k == 1:
            return [("search_datastore", {"query": "bridge roads forwarded facts"})]
        if k == 2 and i == 0:
            return [("publish_message", {"sub_task": "I've weighed the options", "artifact": "East: the bridge is out.",
                                         "rationale": "", "refs": []})]
        return [("submit_answer", {"answer": "East", "confidence": 0.8, "rationale": "facts", "refs": []})]
    rec = asyncio.run(GroupRun(t, ex, BlackboardProtocol(), RunConfig(n_agents=2, max_actions=8, max_steps=8, seed=0, summarize=False),
                               ScriptedLLM(policy)).run())
    msgs = rec["method_state"]["messages"]
    assert [m["sub_task"] for m in msgs].count(FORWARD_TITLE) == 2
    assert [m["artifact"] for m in msgs if m["sub_task"] != FORWARD_TITLE] == ["East: the bridge is out."]
    assert sum(e["op"] == "publish_rejected" for e in rec["method_state"]["events"]) == 2
    assert rec["score"]["per_agent"] == [1.0, 1.0]
