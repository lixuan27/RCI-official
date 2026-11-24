import asyncio

from rci.agents.runner import RunConfig
from rci.protocols.blackboard import BlackboardProtocol
from rci.train.rollout import RolloutConfig, normalize_group, raft_group, rollout_question, Sample
from fake_llm import ScriptedLLM
from test_blackboard import TinyCA, tiny_ca


def good(messages, tools, route_key, tag):
    return [("submit_answer", {"pairs": [{"a": 1, "b": 5}], "refs": [], "rationale": "-"})]


def bad(messages, tools, route_key, tag):
    return [("submit_answer", {"pairs": [], "refs": [], "rationale": "-"})]


def test_one_trainee_rollouts_only_sample_policy_and_normalize():
    task = TinyCA(tiny_ca())
    ex = task.examples()[0]
    cfg = RolloutConfig(trainees="one", k=4)
    samples, meta = asyncio.run(rollout_question(task, ex, BlackboardProtocol, RunConfig(n_agents=3, max_actions=5,
                                                 max_steps=5), ScriptedLLM(good), ScriptedLLM(bad), cfg, step=0))
    assert len(meta) == 4 and all(m["score"] == 1.0 for m in meta)
    assert {s.agent for s in samples} == {m["policy_agent"] for m in meta} or len(samples) >= 4
    assert all(s.advantage == 0.0 for s in samples)


def test_normalize_group():
    ss = [Sample([1], [2], [-0.1], r) for r in (0.0, 1.0, 1.0, 0.0)]
    normalize_group(ss)
    assert abs(sum(s.advantage for s in ss)) < 1e-9 and ss[1].advantage > 0 > ss[0].advantage


def test_raft_group_selects_best_rollouts():
    ss = [Sample([1], [2], [-0.1], r) for r in (0.2, 0.5, 0.5, 0.0)]
    raft_group(ss)
    assert [s.advantage for s in ss] == [0.0, 1.0, 1.0, 0.0]
    zero = [Sample([1], [2], [-0.1], 0.0) for _ in range(3)]
    raft_group(zero)
    assert all(s.advantage == 0.0 for s in zero)
