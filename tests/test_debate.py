import asyncio

from rci.agents.runner import GroupRun, RunConfig
from rci.protocols.debate import DebateProtocol
from fake_llm import ScriptedLLM
from test_blackboard import TinyCA, tiny_ca


def run_mad(n, policy, **cfg):
    task = TinyCA(tiny_ca())
    ex = task.examples()[0]
    run = GroupRun(task, ex, DebateProtocol(), RunConfig(n_agents=n, **cfg), ScriptedLLM(policy))
    rec = asyncio.run(run.run())
    return run, rec


def test_debate_broadcasts_previous_step_records():
    seen = []

    def policy(messages, tools, route_key, tag):
        if tag == "final":
            return [("submit_answer", {"pairs": []})]
        injected = [m["content"] for m in messages if m["role"] == "user" and "work records" in str(m["content"])]
        seen.append((route_key, len(injected), injected[-1] if injected else ""))
        return [("fetch_snippet", {"index": int(route_key[-1])})]

    run, rec = run_mad(3, policy, max_actions=3, max_steps=5)
    later = [s for s in seen if s[1] > 0]
    assert later, "no injection happened"
    last = later[-1][2]
    me = int(later[-1][0][-1])
    others = {0, 1, 2} - {me}
    assert all(f"[agent {j}," in last for j in others) and f"[agent {me}," not in last
    assert rec["method_state"]["n_records"] > 0


def test_debate_single_agent_has_no_injection():
    def policy(messages, tools, route_key, tag):
        assert not any("work records" in str(m.get("content")) for m in messages)
        return [("submit_answer", {"pairs": []})]

    _, rec = run_mad(1, policy, max_actions=3, max_steps=3)
    assert rec["method_state"]["n_records"] == 1


def test_debate_rounds_hold_a_submission_until_the_others_were_read():
    seen = []

    def policy(messages, tools, route_key, tag):
        injected = [m["content"] for m in messages if m["role"] == "user" and "work records" in str(m["content"])]
        seen.append((route_key, len(injected), injected[-1] if injected else ""))
        return [("submit_answer", {"pairs": []})]

    task = TinyCA(tiny_ca())
    method = DebateProtocol(rounds=2)
    assert method.name == "debate_r2"
    run = GroupRun(task, task.examples()[0], method, RunConfig(n_agents=3, max_actions=4, max_steps=4), ScriptedLLM(policy))
    rec = asyncio.run(run.run())
    assert all(a.done and not a.final_forced and len(a.turns) == 2 for a in run.agents)
    assert all("[tentative answer, not submitted yet]" in s[2] and "[submitted]" not in s[2] for s in seen if s[1])
    assert len([s for s in seen if s[1]]) == 3
    assert rec["method_state"]["rounds"] == 2 and rec["method_state"]["inj_tokens_total"] > 0


def test_debate_rounds_one_and_a_lone_member_submit_at_once():
    def policy(messages, tools, route_key, tag):
        return [("submit_answer", {"pairs": []})]

    run, rec = run_mad(3, policy, max_actions=3, max_steps=3)
    assert all(len(a.turns) == 1 for a in run.agents) and rec["method_state"]["inj_tokens_total"] == 0
    task = TinyCA(tiny_ca())
    lone = GroupRun(task, task.examples()[0], DebateProtocol(rounds=3), RunConfig(n_agents=1, max_actions=3, max_steps=3), ScriptedLLM(policy))
    asyncio.run(lone.run())
    assert len(lone.agents[0].turns) == 1 and lone.agents[0].done


def test_debate_rounds_release_a_member_with_no_one_left_to_read():
    def policy(messages, tools, route_key, tag):
        return [("submit_answer", {"pairs": []})]

    task = TinyCA(tiny_ca())
    run = GroupRun(task, task.examples()[0], DebateProtocol(rounds=5), RunConfig(n_agents=1, n_failing=1, max_actions=8, max_steps=8),
                   ScriptedLLM(policy))
    asyncio.run(run.run())
    healthy = [a for a in run.agents if not a.failing]
    assert len(healthy) == 1 and healthy[0].done and not healthy[0].final_forced and len(healthy[0].turns) == 1


def test_debate_failing_agent_broadcasts_error_and_budget_truncates():
    got = []

    def policy(messages, tools, route_key, tag):
        if tag == "final":
            return [("submit_answer", {"pairs": []})]
        for m in messages:
            if m["role"] == "user" and "work records" in str(m["content"]):
                got.append(m["content"])
        return [("search_snippet", {"pattern": "knob"})]

    task = TinyCA(tiny_ca())
    run = GroupRun(task, task.examples()[0], DebateProtocol(inject_budget=200),
                   RunConfig(n_agents=3, n_failing=1, max_actions=4, max_steps=4), ScriptedLLM(policy))
    rec = asyncio.run(run.run())
    assert any("Internal Server Error" in g for g in got)
    assert rec["method_state"]["inj_truncated"] > 0


def test_debate_facts_only_withholds_every_answer_from_the_broadcast():
    seen, holds = [], []

    def policy(messages, tools, route_key, tag):
        injected = [m["content"] for m in messages if m["role"] == "user" and "work records" in str(m["content"])]
        seen.append(injected[-1] if injected else "")
        holds.extend(str(m["content"]) for m in messages if m["role"] == "tool" and "Not submitted" in str(m["content"]))
        return [("submit_answer", {"pairs": [["SECRET-A", "SECRET-B"]]})]

    task = TinyCA(tiny_ca())
    method = DebateProtocol(rounds=2, views=False)
    assert method.name == "debate_r2_facts" and DebateProtocol(rounds=2).name == "debate_r2"
    run = GroupRun(task, task.examples()[0], method, RunConfig(n_agents=3, max_actions=4, max_steps=4), ScriptedLLM(policy))
    rec = asyncio.run(run.run())
    shown = [s for s in seen if s]
    assert shown and all("answer withheld" in s and "SECRET" not in s for s in shown)
    assert holds and all("not which option you favour" in h for h in holds)
    assert all(a.done and len(a.turns) == 2 for a in run.agents) and rec["method_state"]["views"] is False
    assert all("do not say which option you favour" in a.system_prompt for a in run.agents)
    plain = GroupRun(task, task.examples()[0], DebateProtocol(rounds=2), RunConfig(n_agents=3, max_actions=4, max_steps=4), ScriptedLLM(policy))
    asyncio.run(plain.run())
    assert not any("favour" in a.system_prompt for a in plain.agents)


def test_debate_facts_only_needs_rounds_and_strips_an_unparsed_call_from_the_text():
    import pytest
    with pytest.raises(ValueError):
        DebateProtocol(views=False)
    m = DebateProtocol(rounds=2, views=False)

    class A:
        failing, done, submit_name = False, False, "submit_answer"
        last_turn = [{"role": "assistant", "content": 'facts here <tool_call>{"name": "submit_answer", "arguments": {"answer": "SECRET"}}</tool_call>',
                      "tool_calls": []}]
    out = m._parts(None, A())
    assert "SECRET" not in out["content"] and out["content"] == "facts here"
    assert "SECRET" in DebateProtocol(rounds=2)._parts(None, A())["content"]


def test_debate_forward_facts_asks_for_the_given_facts_only_and_withholds_answers():
    seen, holds = [], []

    def policy(messages, tools, route_key, tag):
        injected = [m["content"] for m in messages if m["role"] == "user" and "work records" in str(m["content"])]
        seen.append(injected[-1] if injected else "")
        holds.extend(str(m["content"]) for m in messages if m["role"] == "tool" and "Not submitted" in str(m["content"]))
        return [("submit_answer", {"pairs": [["SECRET-A", "SECRET-B"]]})]

    task = TinyCA(tiny_ca())
    method = DebateProtocol(rounds=2, forward_facts=True)
    assert method.name == "debate_r2_forward_facts" and not method.views
    run = GroupRun(task, task.examples()[0], method, RunConfig(n_agents=3, max_actions=4, max_steps=4), ScriptedLLM(policy))
    rec = asyncio.run(run.run())
    assert all("word for word, and nothing else" in a.system_prompt for a in run.agents)
    assert holds and all("word for word" in h for h in holds)
    assert [s for s in seen if s] and all("SECRET" not in s for s in seen)
    assert rec["method_state"]["forward_facts"] is True


def test_debate_shared_context_appends_the_members_given_facts_to_its_first_broadcast_only():
    from rci.tasks.hiddenbench.task import HiddenBench
    import json as _json
    import tempfile
    from pathlib import Path as _P
    row = {"id": 1, "name": "t", "description": "Pick.", "shared_information": ["S0"], "hidden_information": ["H0", "H1"],
           "possible_answers": ["West", "East"], "correct_answer": "East"}
    d = _P(tempfile.mkdtemp())
    (d / "benchmark.json").write_text(_json.dumps([row]))
    task = HiddenBench(root=str(d))
    seen = []

    def policy(messages, tools, route_key, tag):
        injected = [m["content"] for m in messages if m["role"] == "user" and "work records" in str(m["content"])]
        seen.append((route_key, injected[-1] if injected else ""))
        return [("submit_answer", {"answer": "West"})]

    method = DebateProtocol(rounds=3, shared_context=True)
    assert method.name == "debate_r3_shared_context" and method.views
    run = GroupRun(task, task.examples()[0], method, RunConfig(n_agents=2, max_actions=6, max_steps=6), ScriptedLLM(policy))
    rec = asyncio.run(run.run())
    shown = [t for _, t in seen if t]
    assert "[every fact this member was given]" in shown[0] and ("- H0" in shown[0] or "- H1" in shown[0])
    assert all("[every fact this member was given]" not in t for t in shown[2:])
    assert rec["method_state"]["shared_context"] is True
