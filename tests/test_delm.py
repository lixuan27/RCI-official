import asyncio

from rci.agents.runner import GroupRun, RunConfig
from rci.protocols.delm import DeLMProtocol
from fake_llm import ScriptedLLM
from test_blackboard import TinyCA, tiny_ca


def delm_policy(messages, tools, route_key, tag):
    agent = int(route_key.rsplit("a", 1)[1])
    board = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
    n_tool = sum(1 for m in messages if m["role"] == "tool")
    if agent == 0 and "queue not yet created" in board:
        return [("add_tasks", {"items": [{"subject": "scan knob snippets", "detail": "find knob conflicts"},
                                         {"subject": "scan the rest", "detail": "everything else"}]})]
    if "queue not yet created" in board:
        return [("wait_board", {})]
    if f"claimed by a{agent}" not in board and "open   |" in board and n_tool < 6:
        tid = "T1" if agent == 0 else "T2"
        return [("claim_task", {"id": tid})]
    if n_tool < 8:
        return [("fetch_snippet", {"index": 5})]
    if n_tool < 9:
        tid = "T1" if agent == 0 else "T2"
        return [("complete_task", {"id": tid, "summary": "result=knob1 conflict | evidence=Snippet5 | risk=none",
                                   "answer": {"pairs": [{"a": 1, "b": 5}]}})]
    return [("submit_answer", {"pairs": [{"a": 1, "b": 5}]})]


def test_delm_decompose_claim_complete_submit():
    task = TinyCA(tiny_ca())
    run = GroupRun(task, task.examples()[0], DeLMProtocol(), RunConfig(n_agents=2, max_actions=30, max_steps=30),
                   ScriptedLLM(delm_policy))
    rec = asyncio.run(run.run())
    st = rec["method_state"]
    assert st["handoff_done"] and len(st["items"]) == 2
    assert any(n["type"] == "RESULT" for n in st["notes"])
    assert rec["score"]["score"] == 1.0


def test_delm_gates_task_tools_without_claim_and_failing_decomposer_stalls():
    seen_errors = []

    def policy(messages, tools, route_key, tag):
        for m in messages:
            if m["role"] == "tool" and "No claimed subtask" in str(m["content"]):
                seen_errors.append(route_key)
        return [("fetch_snippet", {"index": 1})]

    task = TinyCA(tiny_ca())
    run = GroupRun(task, task.examples()[0], DeLMProtocol(),
                   RunConfig(n_agents=2, n_failing=1, max_actions=5, max_steps=8), ScriptedLLM(policy))
    rec = asyncio.run(run.run())
    assert seen_errors
    assert not rec["method_state"]["handoff_done"]
    assert rec["score"]["score"] == 0.0


def test_delm_finalize_variant_forces_a_submission_only_without_a_recorded_answer():
    task = TinyCA(tiny_ca())

    def idle(messages, tools, route_key, tag):
        if tag == "final":
            return [("submit_answer", {"pairs": [{"a": 1, "b": 5}]})]
        return [("wait_board", {})]

    plain = asyncio.run(GroupRun(task, task.examples()[0], DeLMProtocol(), RunConfig(n_agents=1, max_actions=4, max_steps=4),
                                 ScriptedLLM(idle)).run())
    fin = asyncio.run(GroupRun(task, task.examples()[0], DeLMProtocol(finalize=True),
                               RunConfig(n_agents=1, max_actions=4, max_steps=4), ScriptedLLM(idle)).run())
    assert plain["score"]["score"] == 0.0
    assert fin["method"] == "delm_finalize" and fin["score"]["score"] == 1.0
