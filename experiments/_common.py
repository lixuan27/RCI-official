"""Shared helpers of the experiment drivers: YAML loading, cell expansion, run lines and grading specs."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rci import paths

EXPERIMENTS = ROOT / "experiments"
CONFIGS = ROOT / "configs"
RUN_KEYS = ("max-actions", "max-steps", "gen-limit", "temperature", "top-p", "top-k", "presence-penalty", "sampling",
            "reasoning-effort", "concurrency", "llm-concurrency", "limit", "shard", "tag", "order")
RUN_FLAGS = ("no-thinking", "no-summary", "save-transcripts", "capture", "overwrite")
SPEC_KEYS = ("task_kwargs", "set", "per_question", "min_pool", "own_family", "stratify_seats", "least", "vote_free",
             "sizes")


def load(name: str) -> dict:
    with open(EXPERIMENTS / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand(value):
    if isinstance(value, str):
        subs = {"${RCI_DATA}": str(paths.data_dir()), "${RCI_RESULTS}": str(paths.results_dir()),
                "${RCI_CKPT}": str(paths.ckpt_dir()), "${RCI_MODELS}": str(paths.models_dir()),
                "${RCI_LOGS}": str(paths.logs_dir()), "${CONFIGS}": str(CONFIGS)}
        for k, v in subs.items():
            value = value.replace(k, v)
        return value
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def models(exp: dict) -> dict:
    out = {}
    for key, m in (exp.get("models") or {}).items():
        out[key] = {"served": m, "run": {}} if isinstance(m, str) else {"served": m["served"], "run": m.get("run") or {}}
    return out


def example_ids(value) -> list | None:
    if value is None:
        return None
    if isinstance(value, str) and value.startswith("@"):
        with open(expand(value[1:]), encoding="utf-8") as f:
            data = json.load(f)
        return list(data["questions"] if isinstance(data, dict) else data)
    return [str(x) for x in value]


def merged(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def cells(exp: dict, model_keys=None, names=None) -> list:
    ms = models(exp)
    keys = model_keys or list(ms)
    defaults = exp.get("defaults") or {}
    out = []
    for raw in exp.get("cells") or []:
        sizes = raw.get("n_agents", defaults.get("n_agents", 1))
        sizes = sizes if isinstance(sizes, list) else [sizes]
        for n in sizes:
            name = raw["name"] if len(sizes) == 1 else f"{raw['name']}_n{n}"
            if names and name not in names and raw["name"] not in names:
                continue
            for key in keys:
                if key not in ms:
                    raise SystemExit(f"unknown model key {key}")
                if raw.get("models") and key not in raw["models"]:
                    continue
                c = merged({k: v for k, v in defaults.items() if k != "n_agents"},
                           {k: v for k, v in raw.items() if k not in ("models", "n_agents")})
                c["run"] = merged(merged(defaults.get("run") or {}, ms[key]["run"]), raw.get("run") or {})
                c.update(name=name, n_agents=int(n), model=key, served=ms[key]["served"],
                         label=raw.get("label", name) + ("" if len(sizes) == 1 else f" N={n}"))
                c.setdefault("protocol_kwargs", {})
                c.setdefault("task_kwargs", {})
                c["root"] = str(paths.results_dir(exp["experiment"], c["served"], c.get("dir", raw["name"])))
                out.append(expand(c))
    return out


def protocol_dir(cell: dict) -> str:
    from rci.registry import protocol_dir as pd
    return pd(cell["protocol"], **cell["protocol_kwargs"]) + str(cell["run"].get("tag", ""))


def seeds_of(cell: dict) -> list:
    s = cell.get("seeds", [0])
    if isinstance(s, str):
        a, _, b = s.partition("-")
        return list(range(int(a), int(b) + 1)) if b else [int(a)]
    return [int(x) for x in (s if isinstance(s, list) else [s])]


def run_argv(cell: dict, seeds: list, base_urls: str | None = None) -> list:
    argv = ["python", "-m", "rci.run", "--task", cell["task"], "--task-kwargs", json.dumps(cell["task_kwargs"]),
            "--protocol", cell["protocol"], "--protocol-kwargs", json.dumps(cell["protocol_kwargs"]),
            "--n-agents", str(cell["n_agents"]), "--seeds", ",".join(str(s) for s in seeds),
            "--base-urls", base_urls or "$RCI_BASE_URLS", "--model", cell["served"], "--out", cell["root"]]
    ids = example_ids(cell.get("example_ids"))
    if ids:
        argv += ["--example-ids", ",".join(ids)]
    for k in RUN_KEYS:
        if cell["run"].get(k) is not None:
            argv += [f"--{k}", str(cell["run"][k])]
    for k in RUN_FLAGS:
        if cell["run"].get(k):
            argv.append(f"--{k}")
    return argv


def shell(argv: list) -> str:
    return " ".join(f'"{a}"' if a.startswith("$RCI_") else shlex.quote(a) for a in argv)


def lines(exp: dict, model_keys=None, names=None, per_seed: bool = False, base_urls: str | None = None) -> list:
    out = []
    for c in cells(exp, model_keys, names):
        if c.get("skip_lines"):
            continue
        groups = [[s] for s in seeds_of(c)] if per_seed or c.get("per_seed") else [seeds_of(c)]
        out += [shell(run_argv(c, g, base_urls)) for g in groups]
    return out


def resolve(ref: str, exp: dict, model: str) -> list:
    name, other = (ref.split(":", 1)[1], load(ref.split(":", 1)[0])) if ":" in ref else (ref, exp)
    ms = models(other)
    key = model if model in ms else None
    if key is None:
        served = models(exp)[model]["served"]
        key = next((k for k, m in ms.items() if m["served"] == served), None)
    if key is None:
        raise SystemExit(f"{ref}: no model {model} in experiment {other['experiment']}")
    if not any(c["name"] == name for c in other.get("cells") or []):
        raise SystemExit(f"{ref}: no such cell in experiment {other['experiment']}")
    grid = re.compile(re.escape(name) + r"_n\d+$")
    return [c for c in cells(other, [key]) if c["name"] == name or grid.match(c["name"])]


def spec_cell(c: dict, entry: dict) -> dict:
    out = {"label": entry.get("label", c["label"]), "root": c["root"], "protocol": protocol_dir(c), "n": c["n_agents"]}
    for k in ("stops", "bh", "review_scale"):
        if k in entry or k in c:
            out[k] = entry.get(k, c.get(k))
    if entry.get("review"):
        out["review"] = str(Path(c["root"]) / f"review_{entry['review']}")
    return out


def grade_spec(exp: dict, model_keys=None) -> dict:
    spec = {}
    for tname, t in (exp.get("grade") or {}).get("tasks", {}).items():
        for key in t.get("models") or model_keys or list(models(exp)):
            if model_keys and key not in model_keys:
                continue
            entry = {"task": t.get("task", (exp.get("defaults") or {}).get("task"))}
            for k in SPEC_KEYS:
                if k in t:
                    entry[k] = expand(t[k])
            entry["pool"] = [[c["root"], protocol_dir(c), c["n_agents"]] for ref in t["pool"] for c in resolve(ref, exp, key)]
            entry["cells"] = []
            for item in t["cells"]:
                item = {"cell": item} if isinstance(item, str) else item
                for c in resolve(item["cell"], exp, key):
                    entry["cells"].append(spec_cell(c, item))
            spec[f"{tname}_{key}"] = entry
    return spec


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    return path


def run(argv: list, dry: bool) -> int:
    print(shell(argv), flush=True)
    if dry:
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(x for x in (str(ROOT), env.get("PYTHONPATH")) if x)
    return subprocess.call(argv, env=env)


def grade(exp: dict, model_keys=None, dry: bool = False) -> int:
    spec = grade_spec(exp, model_keys)
    if not spec:
        print("no grading tasks", file=sys.stderr)
        return 0
    path = write_json(paths.results_dir(exp["experiment"], "grade_spec.json"), spec)
    opts = [str(x) for x in (exp.get("grade") or {}).get("options", [])]
    out = str(paths.results_dir(exp["experiment"], "grade"))
    return run([sys.executable, "-m", "rci.ladder.grade", str(path), "--out", out] + opts, dry)


def parser(description: str) -> tuple:
    ap = argparse.ArgumentParser(description=description)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=None, help="comma list of model keys (default: every model of the YAML)")
    common.add_argument("--dry-run", action="store_true", help="print commands without running them")
    ln = sub.add_parser("lines", parents=[common], help="print the rci.run command of every cell")
    ln.add_argument("--cells", default=None, help="comma list of cell names")
    ln.add_argument("--per-seed", action="store_true", help="one line per seed")
    ln.add_argument("--base-urls", default=None, help="default: the RCI_BASE_URLS variable at run time")
    sub.add_parser("cells", parents=[common], help="list the cells")
    sub.add_parser("grade", parents=[common], help="write the grading spec and run rci.ladder.grade")
    return ap, sub, common


def keys(a) -> list | None:
    return a.model.split(",") if getattr(a, "model", None) else None


def dispatch(exp: dict, a) -> int | None:
    if a.cmd == "lines":
        names = set(a.cells.split(",")) if a.cells else None
        print("\n".join(lines(exp, keys(a), names, a.per_seed, a.base_urls)))
        return 0
    if a.cmd == "cells":
        for c in cells(exp, keys(a)):
            print(f"{c['model']:>8}  {c['name']:<28} {c['protocol']:<18} N={c['n_agents']:<3} seeds={seeds_of(c)}")
        return 0
    if a.cmd == "grade":
        return grade(exp, keys(a), a.dry_run)
    return None


def env_or(name: str, default: str) -> str:
    return os.environ.get(name) or default
