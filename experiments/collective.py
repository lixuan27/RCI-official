from __future__ import annotations

import argparse
import sys

import recursion as R
from _common import expand, load, run, run_argv, seeds_of, shell, write_json

from rci import paths


def solo_models(exp: dict, rec: dict) -> list:
    s = exp["solo"]
    chain = R.chain_of(rec, s["chain"])
    return [(g, lin, m) for g, lin, _, m in R.models_of(chain) if g in s["generations"]]


def own_models(exp: dict, rec: dict) -> list:
    out = []
    for name, o in exp["own_pool"]["chains"].items():
        chain = R.chain_of(rec, name)
        got = [(g, lin, m) for g, lin, _, m in R.models_of(chain) if g in o["generations"]]
        if o.get("replication") and chain.get("replication"):
            got += [(g, lin, m) for g, lin, seed, m in R.models_of(chain, True)
                    if g in o["generations"] and seed is not None]
        out += [(name, g, lin, m) for g, lin, m in got]
    return out


def solo_root(model: str) -> str:
    return str(paths.results_dir("collective", "solo", model))


def own_root(chain: str, model: str) -> str:
    return str(paths.results_dir("collective", "own_pool", chain, model))


def lines(exp: dict, rec: dict) -> list:
    out = []
    hb = expand(rec["sets"]["hiddenbench"])
    held = expand(rec["sets"]["heldout"])
    for _, _, m in solo_models(exp, rec):
        c = expand({**exp["solo"]["cell"], "served": m, "root": solo_root(m), "protocol_kwargs": {}})
        c["task_kwargs"] = {**c["task_kwargs"], "root": hb}
        out.append(shell(run_argv(c, seeds_of(c))))
    seen = set()
    for chain, _, _, m in own_models(exp, rec):
        if (chain, m) in seen:
            continue
        seen.add((chain, m))
        c = expand({**exp["own_pool"]["cell"], "served": m, "root": own_root(chain, m), "protocol_kwargs": {}})
        c["task_kwargs"] = {**c["task_kwargs"], "root": held}
        out.append(shell(run_argv(c, seeds_of(c))))
    return out


def solo_spec(exp: dict, rec: dict) -> dict:
    s = exp["solo"]
    chain = R.chain_of(rec, s["chain"])
    hb = expand(rec["sets"]["hiddenbench"])
    from rci.registry import protocol_dir
    t = rec["templates"][chain["eval"]["template"]]
    return {"generations": {g: [m for gg, _, m in solo_models(exp, rec) if gg == g] for g in s["generations"]},
            "solo_root": solo_root("{model}"), "solo_protocol": protocol_dir("independent"), "solo_n": 1,
            "team_root": R.eval_root(s["chain"], "{model}", hb),
            "team_protocol": protocol_dir(t["protocol"], **t["protocol_kwargs"]), "team_n": t["n_agents"],
            "team_seeds": seeds_of({"seeds": chain["eval"]["seeds"]}), "bound": s["bound"], "draws": 2000}


def own_spec(exp: dict, rec: dict, name: str) -> dict:
    chain = R.chain_of(rec, name)
    held = expand(rec["sets"]["heldout"])
    from rci.registry import protocol_dir
    base = R.contrasts_spec(rec, name, chain, held)
    gens = {}
    for c, g, _, m in own_models(exp, rec):
        if c == name:
            gens.setdefault(g, [])
            if m not in gens[g]:
                gens[g].append(m)
    cell = exp["own_pool"]["cell"]
    base.update(generations=gens, own_pool={"root": own_root(name, "{model}"), "protocol": protocol_dir("independent"),
                                            "n": cell["n_agents"]}, own_pool_seeds=seeds_of(cell))
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Solo accuracy against team gain and each generation's own pool (Table collective).")
    ap.add_argument("cmd", choices=["lines", "solo_team", "own_pool"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    exp, rec = load("collective"), R.load_rec()
    if a.cmd == "lines":
        print("\n".join(lines(exp, rec)))
        return 0
    if a.cmd == "solo_team":
        path = write_json(paths.results_dir("collective", "solo_team.json"), solo_spec(exp, rec))
        return run([sys.executable, "-m", "rci.recursion.solo_team", str(path), "--out",
                    str(paths.results_dir("collective", "solo_team"))], a.dry_run)
    rc = 0
    for name in exp["own_pool"]["chains"]:
        path = write_json(paths.results_dir("collective", f"own_pool_{name}.json"), own_spec(exp, rec, name))
        rc |= run([sys.executable, "-m", "rci.recursion.own_pool", str(path), "--out",
                   str(paths.results_dir("collective", "own_pool", name))], a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
