"""Table 4, Figure 1b and the recursion appendix: label-free generations trained on verified first turns."""
from __future__ import annotations

import os
import sys

from _common import expand, load, merged, run, run_argv, seeds_of, shell, write_json

from rci import paths

GEN_CODE = {"G1": "g1", "G2": "g2", "control": "ctrl"}
LONG_GENS = ("G0", "G1", "G2", "control")


def load_rec() -> dict:
    return expand(load("recursion"))


def chain_of(exp: dict, name: str) -> dict:
    if name not in exp["chains"]:
        raise SystemExit(f"unknown chain {name}; one of {', '.join(exp['chains'])}")
    return expand(exp["chains"][name])


def served(chain: dict, gen: str, lineage: str | None = None, train_seed=None) -> str:
    if gen == "G0":
        return chain["base"]
    tag = (chain["generations"].get(gen) or {}).get("tag", "")
    name = f"{chain['base']}-{GEN_CODE[gen]}{tag}-{lineage}"
    return name if train_seed is None else f"{name}-t{train_seed}"


def weights(chain: dict, gen: str, lineage: str | None = None) -> str:
    if gen == "G0":
        from rci.agents.families import MODELS
        return str(paths.models_dir(MODELS[chain["base"]][1]))
    return str(paths.ckpt_dir(served(chain, gen, lineage), "hf"))


def set_name(root: str) -> str:
    try:
        rel = os.path.relpath(os.path.abspath(root), os.path.abspath(paths.data_dir()))
    except ValueError:
        rel = os.path.basename(root.rstrip("/"))
    if rel.startswith(".."):
        rel = os.path.basename(root.rstrip("/"))
    return rel.replace(os.sep, "_")


def cell(exp: dict, template: str, served_name: str, root: str, data: str, seeds, extra_run=None, extra_task=None) -> dict:
    c = merged(exp["templates"][template], {"run": extra_run or {}, "task_kwargs": {"root": data, **(extra_task or {})}})
    c.update(served=served_name, root=root, seeds=seeds)
    c.setdefault("protocol_kwargs", {})
    return expand(c)


def harvest_root(name: str, chain: dict, model: str, data: str) -> str:
    return str(paths.results_dir("recursion", name, "harvest", model, set_name(data)))


def harvest_lines(exp: dict, name: str, chain: dict) -> list:
    out = []
    for gen, h in (chain.get("harvest") or {}).items():
        lineages = list(chain["lineages"]) if gen != "G0" else [None]
        for li, lineage in enumerate(lineages):
            model = served(chain, gen, lineage)
            for template, seeds in h["seeds"].items():
                ss = [seeds[li]] if h.get("by_lineage") else seeds
                for data in exp["sets"]["train"]:
                    c = cell(exp, template, model, harvest_root(name, chain, model, data), data, ss,
                             {"max-steps": h["max_steps"], "capture": True, "save-transcripts": True})
                    out.append(shell(run_argv(c, seeds_of(c))))
    return out


def models_of(chain: dict, replication: bool = False) -> list:
    if replication:
        r = chain["replication"]
        got = [(g, None if g == "G0" else lin, None, served(chain, g, None if g == "G0" else lin))
               for g in r.get("with", []) for lin in (chain["lineages"] if g != "G0" else ["l0"])]
        got += [(g, lin, s, served(chain, g, lin, s)) for g in r["generations"] for lin in chain["lineages"]
                for s in r["training_seeds"]]
        return got
    got = [("G0", None, None, chain["base"])]
    gens = ["G1"] + [g for g in ("G2", "control") if g in chain["generations"]]
    got += [(g, lin, None, served(chain, g, lin)) for g in gens for lin in chain["lineages"]]
    return got


def eval_root(name: str, model: str, data: str) -> str:
    return str(paths.results_dir("recursion", name, "eval", model, set_name(data)))


def pool_root(name: str, chain: dict, data: str) -> str:
    owner = (chain.get("reuse") or {}).get("pool", name)
    return str(paths.results_dir("recursion", owner, "pool", chain["base"], set_name(data)))


def eval_lines(exp: dict, name: str, chain: dict, replication: bool = False) -> list:
    out = []
    seeds = chain["replication"]["eval_seeds"] if replication else chain["eval"]["seeds"]
    for data in (exp["sets"]["heldout"], exp["sets"]["hiddenbench"]):
        for _, _, _, model in models_of(chain, replication):
            c = cell(exp, chain["eval"]["template"], model, eval_root(name, model, data), data, seeds,
                     {"save-transcripts": True})
            out.append(shell(run_argv(c, seeds_of(c))))
        if not replication and "pool" not in (chain.get("reuse") or {}):
            c = cell(exp, "pool", chain["base"], pool_root(name, chain, data), data, chain["pool"]["seeds"])
            out.append(shell(run_argv(c, seeds_of(c))))
    return out


def rows_path(name: str, gen: str, lineage: str) -> str:
    return str(paths.results_dir("recursion", name, "rows", f"{gen}_{lineage}.jsonl"))


def harvest_args(exp: dict, name: str, chain: dict, gen: str, lineage: str) -> list:
    owner = (chain.get("reuse") or {}).get("harvest", name)
    model = served(chain, gen, None if gen == "G0" else lineage)
    roots = [harvest_root(owner, chain, model, d) for d in exp["sets"]["train"]]
    return ["--harvest", ",".join(roots), "--data", ",".join(exp["sets"]["train"])]


def commit_seeds(exp: dict, chain: dict, r: dict, src: str, lineage: str) -> list:
    given = r.get("commit_seeds")
    if isinstance(given, dict):
        return list(given[lineage])
    if given:
        return list(given)
    seeds = harvest_seeds(chain, src, lineage, "free3")
    if not seeds and (chain.get("reuse") or {}).get("harvest"):
        seeds = harvest_seeds(chain_of(exp, chain["reuse"]["harvest"]), src, lineage, "free3")
    return seeds


def harvest_seeds(chain: dict, gen: str, lineage: str, template: str) -> list:
    owner_h = (chain.get("harvest") or {}).get(gen) or {}
    seeds = (owner_h.get("seeds") or {}).get(template) or []
    if owner_h.get("by_lineage"):
        return [seeds[list(chain["lineages"]).index(lineage)]]
    return list(seeds)


def rows_commands(exp: dict, name: str, chain: dict, gen: str, lineage: str, base_urls: str | None) -> list:
    g = chain["generations"][gen]
    r = g["rows"]
    src = r["source"]
    ctx = r.get("contexts", src)
    h = harvest_args(exp, name, chain, ctx, lineage)
    out = rows_path(name, gen, lineage)
    sampler = ["--base-urls", base_urls or "$RCI_BASE_URLS", "--model", served(chain, src, None if src == "G0" else lineage)]
    tok = ["--tokenizer", weights(chain, src, lineage)]
    letters = {"forwarding": "a", "free3": "f"}
    cmds = []
    if r["kind"] == "rows":
        protos = ",".join(letters[c] for c in r.get("cells", ["forwarding", "free3"]))
        opts = ["--submit-turns", r["submit_turns"]] if r.get("submit_turns") else []
        if r.get("limit") is not None:
            opts += ["--limit-a", str(r["limit"]), "--limit-f", str(r["limit"])]
        if r.get("submit_turns") == "cut":
            opts += tok
        cmds.append(["python", "-m", "rci.recursion.rows", *h, "--protocols", protos, *opts, "--out", out])
    elif r["kind"] == "context_rows":
        extra = sampler + ["--k", str(r["k"])] + tok if r.get("k") else []
        cmds.append(["python", "-m", "rci.recursion.context_rows", *h, *extra, "--out", out])
    else:
        first = out if not r.get("commit") else out.replace(".jsonl", "_first.jsonl")
        opts = ["--protocols", ",".join(letters[c] for c in r["cells"])] if r.get("cells") else []
        if r.get("criterion"):
            opts += ["--criterion", r["criterion"]]
        if r.get("transplant"):
            opts.append("--transplant-f")
        cmds.append(["python", "-m", "rci.recursion.resample", *h, *sampler, "--k", str(r["k"]), *opts, "--out", first])
        if r.get("commit"):
            hc = harvest_args(exp, name, chain, src, lineage)
            seeds = commit_seeds(exp, chain, r, src, lineage)
            weighting = ["--weighting", r["weighting"]] if r.get("weighting") else []
            cmds.append(["python", "-m", "rci.recursion.commit_rows", "build", *hc, "--seeds",
                         ",".join(str(x) for x in seeds), "--w-rows", first, "--share", str(r["commit"]), *weighting,
                         *tok, "--out", out])
    return cmds


def train_command(exp: dict, name: str, chain: dict, gen: str, lineage: str, train_seed=None) -> list:
    g = chain["generations"][gen]
    seed = chain["lineages"][lineage] if train_seed is None else train_seed
    return ["scripts/launch/train_sft.sh", "--gpus", ",".join(str(i) for i in range(chain["gpus"])),
            "--data", rows_path(name, gen, lineage), "--model", weights(chain, g["parent"], lineage),
            "--out", str(paths.ckpt_dir(served(chain, gen, lineage, train_seed))), "--batch", str(exp["sft"]["batch"]),
            "--lr", str(exp["sft"]["lr"]), "--steps", str(g["steps"]), "--seed", str(seed), "--save-every", "0",
            "--ckpt-min-len", "0"]


def contrasts_spec(exp: dict, name: str, chain: dict, data: str, replication: bool = False) -> dict:
    from rci.registry import protocol_dir
    t = exp["templates"][chain["eval"]["template"]]
    p = exp["templates"]["pool"]
    gens = {}
    if replication:
        r = chain["replication"]
        for g in r.get("with", []):
            gens[g] = [served(chain, g, None if g == "G0" else lin) for lin in (chain["lineages"] if g != "G0" else ["l0"])]
        for g in r["generations"]:
            gens[g] = [[served(chain, g, lin, s) for s in r["training_seeds"]] for lin in chain["lineages"]]
    else:
        for g, _, _, model in models_of(chain):
            gens.setdefault(g, []).append(model)
    return {"chain": name, "label": chain["label"], "set": set_name(data),
            "base_pool": {"root": pool_root(name, chain, data), "protocol": protocol_dir(p["protocol"], **(p.get("protocol_kwargs") or {})),
                          "n": p["n_agents"]},
            "heldout": data, "generations": gens,
            "eval_root": eval_root(name, "{model}", data),
            "protocol": protocol_dir(t["protocol"], **t["protocol_kwargs"]), "n": t["n_agents"],
            "seeds": seeds_of({"seeds": chain["replication"]["eval_seeds"] if replication else chain["eval"]["seeds"]}),
            "multipliers": (chain["replication"].get("multipliers") if replication else None) or chain["multipliers"],
            "draws": 2000}


def plan(exp: dict, name: str, chain: dict, base_urls: str | None) -> list:
    out = [f"# {chain['label']}", "# 1. harvest captured first turns (serve each model with scripts/launch/serve.sh)"]
    out += harvest_lines(exp, name, chain)
    for gen in ("G1", "G2", "control"):
        if gen not in chain["generations"]:
            continue
        out.append(f"# rows and training of {gen}")
        for lineage in chain["lineages"]:
            out += [shell(c) for c in rows_commands(exp, name, chain, gen, lineage, base_urls)]
            out.append(shell(train_command(exp, name, chain, gen, lineage)))
    if chain.get("replication"):
        out.append("# replication over further training seeds (same rows, same parents)")
        for gen in chain["replication"]["generations"]:
            for lineage in chain["lineages"]:
                for s in chain["replication"]["training_seeds"]:
                    out.append(shell(train_command(exp, name, chain, gen, lineage, s)))
    out.append("# evaluation and the frozen base pool")
    out += eval_lines(exp, name, chain)
    if chain.get("replication"):
        out += eval_lines(exp, name, chain, replication=True)
    out.append(f"# contrasts: python experiments/recursion.py contrasts --chain {name}")
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Label-free generations of Table 4 (harvest, rows, SFT, evaluation, contrasts).")
    ap.add_argument("cmd", choices=["plan", "harvest", "rows", "train", "lines", "contrasts", "models"])
    ap.add_argument("--chain", default=None, help="chain name in recursion.yaml (default: every chain)")
    ap.add_argument("--gen", default=None, help="G1, G2 or control (rows, train)")
    ap.add_argument("--lineage", default=None, help="l0 or l1 (rows, train)")
    ap.add_argument("--replication", action="store_true", help="the replication over further training seeds")
    ap.add_argument("--base-urls", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    exp = load_rec()
    names = [a.chain] if a.chain else list(exp["chains"])
    rc = 0
    for name in names:
        chain = chain_of(exp, name)
        if a.cmd == "plan":
            print("\n".join(plan(exp, name, chain, a.base_urls)))
        elif a.cmd == "harvest":
            print("\n".join(harvest_lines(exp, name, chain)))
        elif a.cmd == "models":
            if a.replication and not chain.get("replication"):
                continue
            for g, lin, s, m in models_of(chain, a.replication):
                print(f"{name}\t{g}\t{lin or '-'}\t{s if s is not None else '-'}\t{m}")
        elif a.cmd in ("rows", "train"):
            gens = [a.gen] if a.gen else [g for g in ("G1", "G2", "control") if g in chain["generations"]]
            for gen in gens:
                for lineage in ([a.lineage] if a.lineage else list(chain["lineages"])):
                    if a.replication and not (chain.get("replication")
                                              and gen in chain["replication"]["generations"]):
                        continue
                    if a.cmd == "rows":
                        print("\n".join(shell(c) for c in rows_commands(exp, name, chain, gen, lineage, a.base_urls)))
                    elif a.replication:
                        for s in chain["replication"]["training_seeds"]:
                            print(shell(train_command(exp, name, chain, gen, lineage, s)))
                    else:
                        print(shell(train_command(exp, name, chain, gen, lineage)))
        elif a.cmd == "lines":
            if a.replication and not chain.get("replication"):
                continue
            print("\n".join(eval_lines(exp, name, chain, a.replication)))
        else:
            reps = [False] + ([True] if chain.get("replication") else [])
            for rep in reps:
                for data in (exp["sets"]["heldout"], exp["sets"]["hiddenbench"]):
                    tag = f"{name}_{set_name(data)}" + ("_replication" if rep else "")
                    path = write_json(paths.results_dir("recursion", "contrasts", f"{tag}.json"),
                                      contrasts_spec(exp, name, chain, data, rep))
                    rc |= run([sys.executable, "-m", "rci.recursion.contrasts", str(path), "--out",
                               str(paths.results_dir("recursion", "contrasts", tag))], a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
