import asyncio
import os
import shutil

from rci.agents.runner import GroupRun, RunConfig
from rci.protocols.blackboard import BlackboardProtocol
from rci.protocols.independent import IndependentAgents
from rci.tasks.swe.task import DIFF_CMD, OUTPUT_CHARS, SWEBenchVerified, _truncate
from fake_llm import ScriptedLLM

EX = {"id": "toy__toy-1", "repo": "toy/toy", "base_commit": "abcdef1234567890", "problem_statement": "x is wrong"}


class FakeBox:
    def __init__(self):
        self.edits, self.cmds = {}, []

    def __call__(self, example, rundir, cmd, timeout):
        os.makedirs(os.path.join(rundir, "upper"), exist_ok=True)
        self.cmds.append((rundir, cmd))
        if cmd == DIFF_CMD:
            return 0, "".join(self.edits.get(rundir, []))
        if cmd.startswith("echo "):
            self.edits.setdefault(rundir, []).append(cmd[5:].split(" >>")[0] + "\n")
            return 0, ""
        if cmd == "sleep 999":
            return 124, "partial"
        return 0, "x" * 30_000 if cmd == "big" else "ok"


def make_task(tmp_path, box, evals):
    def evaluator(example, patch):
        evals.append(patch)
        return {"resolved": patch.strip() == "fix"}

    t = SWEBenchVerified(cache=str(tmp_path), verify=None, executor=box, evaluator=evaluator,
                         remover=lambda d: shutil.rmtree(d, ignore_errors=True))
    t.examples = lambda limit=None: [EX]
    return t


def test_truncate():
    out = _truncate("a" * 30_000)
    assert len(out) < 30_000 and "omitted" in out and out.startswith("a" * (OUTPUT_CHARS // 2))
    assert _truncate("short") == "short"


def test_pass_at_n_with_dedupe_and_unsubmitted_agents(tmp_path):
    box, evals = FakeBox(), []
    task = make_task(tmp_path, box, evals)
    edits = {0: "fix", 1: "fix", 2: "wrong", 3: None}

    def policy(messages, tools, route_key, tag):
        agent = int(route_key.rsplit("a", 1)[1])
        n_tool = sum(1 for m in messages if m["role"] == "tool")
        if tag == "final" or n_tool >= 2:
            return [("submit_answer", {"summary": "done"})]
        if n_tool == 0:
            return [("bash", {"command": "big"})]
        e = edits[agent]
        return [("bash", {"command": f"echo {e} >> f.py" if e else "ls"})]

    rec = asyncio.run(GroupRun(task, EX, IndependentAgents(), RunConfig(n_agents=4, max_actions=5, max_steps=6, seed=0),
                               ScriptedLLM(policy)).run())
    s = rec["score"]
    assert s["score"] == 1.0 and s["per_agent"] == [1.0, 1.0, 0.0, 0.0]
    assert s["n_patches"] == 2 and sorted(evals) == ["fix\n", "wrong\n"]
    rundirs = {d for d, _ in box.cmds}
    assert len(rundirs) == 4
    assert not any(os.path.exists(d) for d in rundirs)


def test_timeout_and_empty_command(tmp_path):
    box, evals = FakeBox(), []
    task = make_task(tmp_path, box, evals)
    (bash,) = task.tools(EX, None)

    class Run:
        run_id, state = "r", {}

    from rci.agents.tooling import AgentCtx, call_tool
    ctx = AgentCtx(agent_id=0, run=Run())
    r = asyncio.run(call_tool(bash, ctx, {"command": "sleep 999"}))
    assert not r.ok and "timed out" in r.text
    r = asyncio.run(call_tool(bash, ctx, {"command": "  "}))
    assert not r.ok


def test_blackboard_on_swe_scores_the_diff(tmp_path):
    box, evals = FakeBox(), []
    task = make_task(tmp_path, box, evals)

    def policy(messages, tools, route_key, tag):
        n_tool = sum(1 for m in messages if m["role"] == "tool")
        if tag == "final" or n_tool >= 1:
            return [("submit_answer", {"summary": "done", "refs": [], "rationale": "-"})]
        return [("bash", {"command": "echo fix >> a.py"})]

    rec = asyncio.run(GroupRun(task, EX, BlackboardProtocol(), RunConfig(n_agents=2, max_actions=4, max_steps=4, seed=0),
                               ScriptedLLM(policy)).run())
    assert rec["score"]["score"] == 1.0 and rec["score"]["n_patches"] == 1


def test_coordinator_takes_the_latest_subagent_diff(tmp_path):
    box, evals = FakeBox(), []
    task = make_task(tmp_path, box, evals)

    class Run:
        run_id, state = "r", {}

    run = Run()
    for aid, edit in ((1, "wrong"), (2, "fix")):
        d = task._rundir(run, aid)
        box(EX, d, f"echo {edit} >> f.py", 10)
    finals = [{"agent": 0, "role": "coordinator", "failing": False, "final": None}]
    out = asyncio.run(task.score_group(EX, finals, run))
    assert out["score"] == 1.0 and evals == ["fix\n"]


def test_layer_cache_evicts_unpinned_layers_lru(tmp_path, monkeypatch):
    import json
    import sys
    import time

    import rci.tasks.swe.sandbox as swe_sandbox

    monkeypatch.setattr(swe_sandbox, "rmtree_ns", lambda p: shutil.rmtree(p, ignore_errors=True))
    store, cache = tmp_path / "store", tmp_path / "cache"
    (store / "images").mkdir(parents=True)
    layers = {"a": ["1" * 64, "2" * 64], "b": ["1" * 64, "3" * 64], "c": ["4" * 64]}
    for key, hs in layers.items():
        rec = {"manifest": {"layers": [{"digest": f"sha256:{h}"} for h in hs]}}
        (store / "images" / f"{key}.json").write_text(json.dumps(rec))
    for h in {h for hs in layers.values() for h in hs}:
        d = cache / "layers" / h
        d.mkdir(parents=True)
        (d / "f").write_bytes(b"x" * 100_000)
    (cache / "layers" / ("5" * 64 + ".tmp1.2")).mkdir()
    one = swe_sandbox.evict(str(store), str(cache), 1e12)
    assert one["removed"] == 0 and len(os.listdir(cache / "sizes")) == 4
    p = swe_sandbox.pin(str(store), str(cache), "b", "run1")
    time.sleep(0.01)
    os.utime(cache / "sizes" / ("4" * 64))
    out = swe_sandbox.evict(str(store), str(cache), 1)
    left = sorted(os.listdir(cache / "layers"))
    assert left == ["1" * 64, "3" * 64, "5" * 64 + ".tmp1.2"] and out["removed"] == 2
    swe_sandbox.unpin(p)
    assert swe_sandbox.evict(str(store), str(cache), 1)["removed"] == 2


def test_training_run_pins_its_image_until_scored(tmp_path, monkeypatch):
    import sys

    import rci.tasks.swe.sandbox as swe_sandbox

    calls = []
    monkeypatch.setattr(swe_sandbox, "pin", lambda store, cache, key, tag: calls.append(("pin", key)) or "P")
    monkeypatch.setattr(swe_sandbox, "evict", lambda store, cache, budget: calls.append(("evict", budget)))
    monkeypatch.setattr(swe_sandbox, "unpin", lambda p: calls.append(("unpin", p)))
    box, evals = FakeBox(), []
    t = make_task(tmp_path, box, evals)
    t.cache_budget = 5e9

    def policy(messages, tools, route_key, tag):
        if tag == "final" or any(m["role"] == "tool" for m in messages):
            return [("submit_answer", {"summary": "done"})]
        return [("bash", {"command": "echo fix >> a.py"})]

    rec = asyncio.run(GroupRun(t, EX, IndependentAgents(), RunConfig(n_agents=2, max_actions=4, max_steps=4, seed=0),
                               ScriptedLLM(policy)).run())
    assert rec["score"]["score"] == 1.0
    assert calls == [("pin", EX["id"]), ("evict", 5e9), ("unpin", "P")]


def test_training_ids_keep_the_instance(tmp_path, monkeypatch):
    import sys

    import rci.tasks.swe.sandbox as swe_sandbox

    from rci.tasks.swe.task import _iid
    keys = []
    monkeypatch.setattr(swe_sandbox, "exec_cmd", lambda store, key, *a, **k: keys.append(key) or (0, ""))
    t = SWEBenchVerified(cache=str(tmp_path), verify=None)
    ex = {**EX, "instance_id": EX["id"], "id": EX["id"] + "@17"}
    assert _iid(ex) == EX["id"] and _iid(EX) == EX["id"]
    t._sandbox_exec(ex, str(tmp_path / "r"), "true", 5)
    assert keys == [EX["id"]]


def test_evict_keeps_the_disk_from_filling(tmp_path, monkeypatch):
    import json
    import shutil as sh
    import sys
    from collections import namedtuple

    import rci.tasks.swe.sandbox as swe_sandbox

    monkeypatch.setattr(swe_sandbox, "rmtree_ns", lambda p: sh.rmtree(p, ignore_errors=True))
    store, cache = tmp_path / "store", tmp_path / "cache"
    (store / "images").mkdir(parents=True)
    for h in ("1" * 64, "2" * 64):
        (cache / "layers" / h).mkdir(parents=True)
        (cache / "layers" / h / "f").write_bytes(b"x" * 100_000)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(sh, "disk_usage", lambda p: Usage(10**12, 10**12, 10))
    out = swe_sandbox.evict(str(store), str(cache), 1e12)
    assert out["removed"] == 2 and out["budget"] == 0.0


def test_evaluation_launch_is_retried_when_the_filesystem_fails_it(tmp_path, monkeypatch):
    import json
    import subprocess

    import rci.tasks.swe.task as swe

    t = SWEBenchVerified.__new__(SWEBenchVerified)
    t.cache, t.eval_python, t.store, t.parquet, t.eval_timeout = str(tmp_path), "py", "s", "p.parquet", 60
    monkeypatch.setattr(swe, "LAUNCH_WAIT_S", 0)
    calls = []

    def flaky(argv, **kw):
        calls.append(argv)
        if len(calls) == 1:
            raise FileNotFoundError(2, "No such file or directory")
        out = argv[argv.index("--out") + 1]
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump({"resolved": True}, f)

    monkeypatch.setattr(subprocess, "run", flaky)
    assert t._sandbox_evaluate({"id": "x"}, "diff") == {"resolved": True} and len(calls) == 2

    def broken(argv, **kw):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "run", broken)
    import pytest
    with pytest.raises(FileNotFoundError):
        t._sandbox_evaluate({"id": "x"}, "diff")
