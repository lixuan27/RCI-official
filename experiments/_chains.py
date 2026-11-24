"""Shared driver of the gated generation chains: one YAML per benchmark, one chain state per chain.

    chains                        the chains of the YAML
    prepare [--dry-run]           build the task sets (PlanBench) or print how to (HiddenBench)
    plan --chain C                every step of round 1 with its GPU line (no state is written)
    next --chain C [--tsv]        the next step: a CPU step to run with `step`, a GPU line to run and mark with `done`
    step --chain C --step S       run a CPU step (accept, mix, candidates, select, verdict)
    done --chain C --step S       mark a GPU step done after its line exited 0
    status --chain C              rounds, labels and released generations
scripts/launch/run_chain.sh runs next / step / done in a loop until the chain stops.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import expand, load, merged

from rci.chains.loop import Chain, ChainError

ADAPTERS = {"planbench": "rci.chains.planbench.adapter:PlanBench",
            "hiddenbench": "rci.chains.hiddenbench.adapter:HiddenBench",
            "hanabi": "rci.chains.hanabi.adapter:Hanabi"}


def adapter_class(name: str):
    import importlib
    mod, attr = ADAPTERS[name].split(":")
    return getattr(importlib.import_module(mod), attr)


def chain_cfg(exp: dict, name: str) -> dict:
    raw = exp["chains"].get(name)
    if raw is None:
        raise SystemExit(f"unknown chain {name}; one of {', '.join(exp['chains'])}")
    if raw.get("inherit"):
        parent = chain_cfg(exp, raw["inherit"])
        raw = merged(parent, {k: v for k, v in raw.items() if k not in ("inherit",)})
    cfg = merged({"base": exp["base"]}, raw)
    return expand(cfg)


def build(exp: dict, name: str) -> Chain:
    cfg = chain_cfg(exp, name)
    parent = build(exp, cfg["parent"]) if cfg.get("parent") else None
    return Chain(name, cfg, adapter_class(cfg["adapter"]), parent)


def plan(ch: Chain) -> list:
    st = ch.state()
    rd = {"r": 1, "gen": 0, "done": [], "lr": ch.adapter.lr(st, 1), "selected": ch.alphas[-1],
          "info": {"mix": {lin: {"steps": 0} for lin in ch.lineages}}}
    out = []
    for r, s in [(None, s) for s in ch.base_steps()] + [(rd, s) for s in ch.plan_round(st, rd)]:
        try:
            out.append(ch.describe(st, r, s))
        except FileNotFoundError as e:
            out.append({"kind": "gpu", "step": s, "round": r["r"] if r else 0, "line": f"(run prepare first: {e})"})
    return out


def main(experiment: str, argv=None, extra=None) -> int:
    exp = load(experiment)
    ap = argparse.ArgumentParser(description=(sys.modules["__main__"].__doc__ or "").strip())
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chains")
    p = sub.add_parser("prepare")
    p.add_argument("--dry-run", action="store_true")
    for name in ("plan", "next", "step", "done", "status"):
        x = sub.add_parser(name)
        x.add_argument("--chain", required=True)
        if name in ("step", "done"):
            x.add_argument("--step", required=True)
        if name == "next":
            x.add_argument("--tsv", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "chains":
        for name, c in exp["chains"].items():
            print(f"{name:<16} {c.get('label', '')}")
        return 0
    if a.cmd == "prepare":
        return (extra or {}).get("prepare", lambda exp, dry: 0)(exp, a.dry_run)
    ch = build(exp, a.chain)
    try:
        if a.cmd == "plan":
            for d in plan(ch):
                print(f"r{d['round']} {d['kind']:<3} {d['step']:<14} {d.get('line', '')}")
            return 0
        if a.cmd == "next":
            d = ch.next()
            if a.tsv:
                print("stop\t" + (ch.state().get("stop") or "") if d is None else
                      f"{d['kind']}\t{d['step']}\t{d.get('line', '')}")
            else:
                print(json.dumps(d if d is not None else {"stop": ch.state().get("stop")}, indent=1))
            return 0
        if a.cmd == "step":
            print(json.dumps(ch.run_cpu(a.step), indent=1, sort_keys=True, default=str)[:4000])
            return 0
        if a.cmd == "done":
            d = ch.next()
            if d is None or d["step"] != a.step or d["kind"] != "gpu":
                raise ChainError(f"{a.step} is not the pending GPU step ({d and d['step']})")
            if a.step.startswith("train:"):
                out = ch.train_dir(ch.state()["rounds"][-1]["r"], a.step.split(":", 1)[1])
                if not (out / "hf").is_dir() or not (out / "master").is_dir():
                    raise ChainError(f"{out} has no hf/ and master/ export")
            ch.mark(a.step)
            return 0
        if a.cmd == "status":
            print(json.dumps(ch.status(), indent=1))
            return 0
    except ChainError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1


def root() -> Path:
    return Path(__file__).resolve().parents[1]
