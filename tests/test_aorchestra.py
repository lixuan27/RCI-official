import asyncio

from rci.agents.runner import GroupRun, RunConfig
from rci.protocols.aorchestra import AOrchestraProtocol
from fake_llm import ScriptedLLM
from test_blackboard import TinyCA, tiny_ca


def policy(messages, tools, route_key, tag):
    if tag == "trace_summary":
        return "- did things"
    agent = int(route_key.rsplit("a", 1)[1])
    live = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
    if agent == 0:
        if "No subtasks completed yet" in live:
            return [("delegate_task", {"task_instruction": "Find contradictions about knob1", "context": ""}),
                    ("delegate_task", {"task_instruction": "a second call in the same turn"})]
        return [("submit_answer", {"pairs": [{"a": 1, "b": 5}]})]
    n_tool = sum(1 for m in messages if m["role"] == "tool")
    if n_tool < 2:
        return [("fetch_snippet", {"index": 5})]
    return [("finish", {"result": {"pairs": [{"a": 1, "b": 5}]}, "status": "done", "summary": "found it"})]


def test_aorchestra_delegate_wait_report_submit():
    task = TinyCA(tiny_ca())
    llm = ScriptedLLM(policy)
    run = GroupRun(task, task.examples()[0], AOrchestraProtocol(), RunConfig(n_agents=4, max_actions=20, max_steps=20), llm)
    rec = asyncio.run(run.run())
    st = rec["method_state"]
    assert st["n_subagents"] == 1 and st["entries"][0]["status"] == "done"
    assert st["entries"][0]["trace_summary"].startswith("- did")
    assert rec["score"]["score"] == 1.0 and len(rec["finals"]) == 1
    assert any(c["tag"] == "trace_summary" for a in rec["agents"] for c in a["calls"])


def test_aorchestra_timeout_and_attempt_budget():
    def lazy(messages, tools, route_key, tag):
        if tag == "trace_summary":
            return "- nothing"
        agent = int(route_key.rsplit("a", 1)[1])
        if agent == 0:
            return [("delegate_task", {"task_instruction": "look around"})]
        return [("search_snippet", {"pattern": "knob"})]

    task = TinyCA(tiny_ca())
    run = GroupRun(task, task.examples()[0], AOrchestraProtocol(max_attempts=2, sub_max_turns=3),
                   RunConfig(n_agents=1, max_actions=50, max_steps=40), ScriptedLLM(lazy))
    rec = asyncio.run(run.run())
    st = rec["method_state"]
    assert st["attempts"] == 2 and all(e["status"] == "timeout" for e in st["entries"])
    assert rec["score"]["score"] == 0.0
