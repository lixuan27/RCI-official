"""Table 5, the reward and stages tables, and the credit example: training a 4B group on the contradiction audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from _common import expand, load, run, run_argv, shell, write_json

from rci import paths

TABLES = ("table_credit", "table_stages", "table_reward")


def arms(exp: dict) -> dict:
    with open(expand(exp["arms_file"]), encoding="utf-8") as f:
        return yaml.safe_load(f)


def selected(spec: dict, table: str | None, arm: str | None) -> list:
    out = []
    for t in TABLES:
        if table and t != table:
            continue
        for name, over in (spec.get(t) or {}).items():
            if arm and name != arm:
                continue
            out.append((t, name, over))
    return out


def run_dir(exp: dict, over: dict) -> str:
    from rci.train.config import load_config
    cfg = load_config(expand(exp["base_config"]), [f"run_id={over['run_id']}"])
    return f"{cfg.out_root}/{cfg.run_id}"


def train_commands(exp: dict, spec: dict, table, arm) -> list:
    out = []
    for _, name, over in selected(spec, table, arm):
        if over is None:
            continue
        g = spec["gpus"].get(name, spec["gpus"]["default"])
        argv = ["scripts/launch/train_credit.sh", "--config", expand(exp["base_config"]), "--train-gpus", g["train"],
                "--policy-gpus", g["policy"], "--peer-gpus", g["peer"]]
        for k, v in over.items():
            argv += ["--set", f"{k}={json.dumps(v) if isinstance(v, (bool, list, dict)) else v}"]
        out.append(shell(argv))
    return out


def eval_cells(exp: dict, spec: dict, table, arm, step: int) -> list:
    e = exp["eval"]
    out = []
    tk = {"root": expand(e["data"]), **(exp["stages"]["task_kwargs"] if table == "table_stages" else e["task_kwargs"])}
    sizes = exp["stages"]["n_agents"] if table == "table_stages" else e["n_agents"]
    items = selected(spec, table, arm)
    if table == "table_stages" and arm in (None, "untrained"):
        items = [("table_stages", "untrained", None)] + items
    for t, name, over in items:
        if t == "table_reward":
            continue
        if over is None:
            model, weights, root = exp["model"], None, str(paths.results_dir("credit", t, "untrained"))
        else:
            model = f"{exp['model']}-{name}-s{step}"
            weights = f"{run_dir(exp, over)}/hf/step_{step:06d}"
            root = str(paths.results_dir("credit", t, name, f"s{step}"))
        for n in sizes:
            c = {"task": "audit", "task_kwargs": tk, "protocol": e["protocol"], "protocol_kwargs": {}, "n_agents": n,
                 "served": model, "seeds": e["seeds"], "run": dict(e["run"]), "root": root}
            if over and over.get("trainees") == "one":
                c["run"]["agent-models"] = ",".join([model] + [exp["model"]] * (n - 1))
            out.append((model, weights, c))
    return out


def eval_lines(exp: dict, spec: dict, table, arm, step: int) -> list:
    out = []
    for model, weights, c in eval_cells(exp, spec, table, arm, step):
        if weights:
            out.append(f"# serve {model}: scripts/launch/serve.sh start {model} --model-path {weights}")
        argv = run_argv(c, c["seeds"])
        if c["run"].get("agent-models"):
            argv += ["--pool", "$RCI_POOL", "--agent-models", c["run"]["agent-models"]]
        out.append(shell(argv))
    return out


def heldout(exp: dict, spec: dict, table, step: int, dry: bool) -> int:
    from rci.registry import protocol_dir
    ref = str(paths.results_dir("credit", table, "untrained"))
    runs = sorted({f"{Path(c['root']).parent.name}={c['root']}" for _, w, c in eval_cells(exp, spec, table, None, step) if w})
    sizes = exp["stages"]["n_agents"] if table == "table_stages" else exp["eval"]["n_agents"]
    argv = [sys.executable, "-m", "rci.train.heldout", "--ref", f"untrained={ref}", "--runs", *runs, "--protocol",
            protocol_dir(exp["eval"]["protocol"]), "--n", ",".join(str(n) for n in sizes), "--boot",
            str(exp["eval"]["boot"]), "--out", str(paths.results_dir("credit", f"heldout_{table}_s{step}"))]
    return run(argv, dry)


def curves(exp: dict, spec: dict, table, dry: bool) -> int:
    runs = [run_dir(exp, over) for _, _, over in selected(spec, table, None) if over]
    return run([sys.executable, "-m", "rci.train.curves", "--runs", *runs, "--out",
                str(paths.results_dir("credit", f"curves_{table or 'all'}"))], dry)


def example() -> dict:
    from rci.train.credit import agent_credits, game_values
    ex = {"contradictions": [{"edited": 1, "partner": 2}, {"edited": 3, "partner": 4},
                             {"edited": 5, "partner": 6}, {"edited": 7, "partner": 8}]}
    claims = [[(1, 2), (3, 4)], [(1, 2), (10, 11)], [(12, 13), (14, 15), (16, 17)]]
    finals = [{"agent": i, "final": {"pairs": [{"a": a, "b": b} for a, b in ps]}} for i, ps in enumerate(claims)]
    rec = {"task": "audit", "finals": finals, "agents": [{"id": i} for i in range(3)], "method_state": {"messages": []},
           "score": {"score": 0.25, "recall": 0.5, "penalty": 0.25}}
    true = {frozenset((c["edited"], c["partner"])) for c in ex["contradictions"]}
    own = [(sum(frozenset(p) in true for p in ps) - 0.25 * sum(frozenset(p) not in true for p in ps)) / len(true)
           for ps in claims]
    R, shares, _ = game_values(rec, ex)
    team = agent_credits(rec, ex, "team")
    return {"R": R, "own_success": own, "team_reward": [team[i] for i in range(3)],
            "shapley": [shares[i] for i in range(3)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Shapley credit and its baselines on the contradiction audit (Table 5).")
    ap.add_argument("cmd", choices=["arms", "train", "eval", "heldout", "curves", "example"])
    ap.add_argument("--table", default=None, choices=TABLES)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: the YAML's)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    exp = load("credit")
    spec = arms(exp)
    step = a.step or exp["eval"]["step"]
    if a.arm and a.cmd in ("arms", "train", "eval") and not selected(spec, a.table, a.arm) \
            and not (a.cmd == "eval" and a.arm == "untrained"):
        names = sorted({n for _, n, _ in selected(spec, a.table, None)})
        print(f"unknown arm {a.arm!r}; one of {', '.join(names)}", file=sys.stderr)
        return 2
    if a.cmd == "arms":
        for t, name, over in selected(spec, a.table, a.arm):
            print(f"{t}\t{name}\t{json.dumps(over)}")
        return 0
    if a.cmd == "train":
        print("\n".join(train_commands(exp, spec, a.table, a.arm)))
        return 0
    if a.cmd == "eval":
        print("\n".join(eval_lines(exp, spec, a.table or "table_credit", a.arm, step)))
        return 0
    if a.cmd == "heldout":
        return heldout(exp, spec, a.table or "table_credit", step, a.dry_run)
    if a.cmd == "curves":
        return curves(exp, spec, a.table, a.dry_run)
    got = example()
    write_json(paths.results_dir("credit", "example.json"), got)
    print(json.dumps(got, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
