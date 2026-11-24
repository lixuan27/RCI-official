import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = {
    "ladder.py": [["lines"], ["grade", "--dry-run"]],
    "shared.py": [["lines"], ["grade", "--dry-run"], ["prompts"]],
    "emergence.py": [["lines"], ["grade", "--dry-run"], ["structure", "--dry-run"], ["prompts", "--dry-run"]],
    "channels.py": [["channels", "--dry-run"], ["ledger", "--dry-run"]],
    "prior.py": [["lines"], ["grade", "--dry-run"], ["judge"], ["reviews", "--dry-run"], ["table", "--dry-run"]],
    "recursion.py": [["plan"], ["contrasts", "--dry-run"], ["models", "--replication"], ["train", "--replication"],
                     ["lines", "--replication"], ["rows"]],
    "collective.py": [["lines"], ["solo_team", "--dry-run"], ["own_pool", "--dry-run"]],
    "credit.py": [["arms"], ["train"], ["eval"], ["eval", "--table", "table_stages"], ["heldout", "--dry-run"],
                  ["curves", "--dry-run"], ["example"]],
    "supervisor.py": [["evolve"], ["lines"], ["contrasts", "--dry-run"]],
    "threshold.py": [["--check"]],
    "chains_planbench.py": [["chains"], ["prepare", "--dry-run"], ["plan", "--chain", "planbench"],
                            ["next", "--chain", "planbench", "--tsv"], ["status", "--chain", "planbench"]],
    "chains_hiddenbench.py": [["chains"], ["prepare"], ["plan", "--chain", "t3"], ["plan", "--chain", "t3_solo"],
                              ["plan", "--chain", "t3k"], ["next", "--chain", "t3"]],
    "chains_hanabi.py": [["chains"], ["plan", "--chain", "hanabi"], ["next", "--chain", "hanabi", "--tsv"],
                         ["next", "--chain", "hanabi_control", "--tsv"]],
}


def driver(name, args):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(ROOT / "experiments" / name), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("name,args", [(n, a) for n, cmds in DRIVERS.items() for a in cmds])
def test_every_driver_command_runs_without_a_gpu(name, args):
    out = driver(name, args)
    assert out.returncode == 0, out.stderr[-2000:]


def common():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_common", ROOT / "experiments" / "_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_lines_hold_valid_arguments():
    from rci.registry import PROTOCOLS, TASKS
    out = driver("ladder.py", ["lines", "--model", "q27"])
    lines = [x for x in out.stdout.splitlines() if x.startswith("python -m rci.run")]
    assert len(lines) == 9
    for line in lines:
        argv = shlex.split(line)[3:]
        assert argv[argv.index("--task") + 1] in TASKS and argv[argv.index("--protocol") + 1] in PROTOCOLS
        assert json.loads(argv[argv.index("--task-kwargs") + 1])["profile"] in ("hidden", "full")
        json.loads(argv[argv.index("--protocol-kwargs") + 1])


def test_ladder_spec_grades_every_cell_against_the_pool():
    c = common()
    spec = c.grade_spec(c.load("ladder"))
    t = spec["hiddenbench_q27"]
    assert t["pool"][0][1:] == ["independent", 4]
    dirs = {c["label"]: c["protocol"] for c in t["cells"]}
    assert dirs["Fact forwarding"] == "debate_r2_forward_facts"
    assert dirs["SC+Forwarding"] == "debate_r2_shared_context_forward_facts"
    assert "Blackboard" not in {c["label"] for c in spec["hiddenbench_q35"]["cells"]}


def test_credit_example_matches_the_three_rules():
    out = driver("credit.py", ["example"])
    got = json.loads(out.stdout)
    assert got["R"] == pytest.approx(0.25)
    assert got["own_success"] == pytest.approx([0.5, 0.1875, -0.1875])
    assert got["team_reward"] == pytest.approx([0.25, 0.25, 0.25])
    assert got["shapley"] == pytest.approx([0.375, 0.0625, -0.1875])
    assert sum(got["shapley"]) == pytest.approx(got["R"])


def test_recursion_plan_trains_each_generation_from_its_parent():
    import re
    out = driver("recursion.py", ["plan", "--chain", "step3_9"]).stdout
    assert re.search(r"--model \S*Qwen3\.5-9B --out \S*qwen3\.5-9b-g1-l0 ", out)
    assert re.search(r"--model \S*qwen3\.5-9b-g1-l0/hf --out \S*qwen3\.5-9b-g2-l0 ", out)
    assert re.search(r"--model \S*qwen3\.5-9b-g1-l1/hf --out \S*qwen3\.5-9b-ctrl-l1 ", out)
    assert "--steps 240" in out and "--max-steps 3" in out


def test_credit_arms_load_as_trainer_configs():
    import yaml
    from rci.train.config import load_config
    arms = yaml.safe_load((ROOT / "configs" / "credit" / "arms.yaml").read_text())
    for table in ("table_credit", "table_stages", "table_reward"):
        for name, over in arms[table].items():
            if over is None:
                continue
            sets = [f"{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in over.items()]
            cfg = load_config(str(ROOT / "configs" / "credit" / "base.yaml"), sets + ["policy_urls=[http://x]"])
            cfg.validate()
            assert cfg.lr == pytest.approx(1e-6) and cfg.total_steps == 40


def test_credit_heldout_names_every_run():
    out = driver("credit.py", ["heldout", "--dry-run"]).stdout
    argv = shlex.split(out.strip().splitlines()[-1])
    assert argv[argv.index("--ref") + 1].startswith("untrained=")
    runs = argv[argv.index("--runs") + 1:argv.index("--protocol")]
    assert runs and all("=" in r and r.split("=", 1)[0] for r in runs)
    assert {r.split("=", 1)[0] for r in runs} >= {"shapley", "team_reward", "own_success"}


def test_credit_rejects_an_unknown_arm():
    out = driver("credit.py", ["train", "--arm", "nosuch"])
    assert out.returncode == 2 and "unknown arm" in out.stderr


def test_recursion_control_samples_in_the_contexts_of_the_second_generation():
    out = driver("recursion.py", ["rows", "--chain", "step3_9", "--lineage", "l0"]).stdout.splitlines()
    g1, g2, ctrl = out
    assert "--submit-turns keep --limit-a 0" in g1
    harvest = lambda line: shlex.split(line)[shlex.split(line).index("--harvest") + 1]
    model = lambda line: shlex.split(line)[shlex.split(line).index("--model") + 1]
    assert harvest(g2) == harvest(ctrl) and model(g2) == "qwen3.5-9b-g1-l0" and model(ctrl) == "qwen3.5-9b"


def test_recursion_4b_operator_and_commit_seeds():
    out = driver("recursion.py", ["rows", "--chain", "standard4", "--lineage", "l1"]).stdout
    assert out.count("--protocols a --criterion strict --transplant-f") == 3 and out.count("--weighting rows") == 3
    seeds = [shlex.split(x)[shlex.split(x).index("--seeds") + 1] for x in out.splitlines() if "commit_rows" in x]
    assert seeds == ["30,31", "32,33", "36,37"]
    out9 = driver("recursion.py", ["rows", "--chain", "standard9", "--gen", "control"]).stdout
    assert [shlex.split(x)[shlex.split(x).index("--seeds") + 1] for x in out9.splitlines() if "commit_rows" in x] == ["4", "5"]


def test_prompts_need_an_endpoint_when_they_run():
    env = {k: v for k, v in os.environ.items() if k != "RCI_BASE_URLS"}
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, str(ROOT / "experiments" / "emergence.py"), "prompts"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 2 and "RCI_BASE_URLS" in out.stderr


def test_chain_drivers_point_at_the_paper_and_their_first_step_is_a_gpu_line():
    out = driver("chains_hanabi.py", ["next", "--chain", "hanabi", "--tsv"])
    kind, step, line = out.stdout.strip().split("\t")
    assert (kind, step) == ("gpu", "harvest") and line.startswith("scripts/launch/with_server.sh --model qwen3.5-9b")
    out = driver("chains_hanabi.py", ["next", "--chain", "hanabi_control", "--tsv"])
    assert out.stdout.startswith("wait\t")
    for name in ("chains_planbench", "chains_hiddenbench", "chains_hanabi"):
        doc = (ROOT / "experiments" / f"{name}.py").read_text().split('"""')[1]
        assert any(x in doc for x in ("Table 6", "Table 7", "Figure 11a"))
