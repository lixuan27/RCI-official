from __future__ import annotations

import argparse
import sys

import recursion as R
from _common import expand, load, run, run_argv, seeds_of, shell, write_json

from rci import paths


def team_dir(exp: dict, lineage: str) -> str:
    return str(paths.results_dir("supervisor", f"{exp['generation']}_{lineage}"))


def evolve_commands(exp: dict, rec: dict, base_urls: str | None) -> list:
    chain = R.chain_of(rec, exp["chain"])
    out = []
    for lineage in exp["lineages"]:
        base = ["python", "-m", "rci.organize.supervisor", "--base-urls", base_urls or "$RCI_BASE_URLS", "--model",
                R.served(chain, exp["generation"], lineage), "--train", ",".join(expand(rec["sets"]["train"])),
                "--chain", f"{exp['generation']}_{lineage}", "--versions", str(exp["versions"]), "--dev-tasks",
                str(exp["dev_tasks"]), "--seed", str(exp["dev_seed"] + exp["lineages"].index(lineage)),
                "--out", team_dir(exp, lineage)]
        out.append(shell(base))
        if exp.get("rules_only"):
            out.append(shell(base + ["--rules-only"]))
    return out


def teams(exp: dict) -> list:
    out = [("fixed", None)]
    if exp.get("rules_only"):
        out.append(("rules_only", "rules_only/team.json"))
    out += [(f"v{k}", f"v{k}/team.json") for k in range(1, exp["versions"] + 1)]
    return out


def eval_lines(exp: dict, rec: dict) -> list:
    chain = R.chain_of(rec, exp["chain"])
    held = expand(rec["sets"]["heldout"])
    out = []
    for lineage in exp["lineages"]:
        model = R.served(chain, exp["generation"], lineage)
        for label, rel in teams(exp):
            extra = {"team_file": f"{team_dir(exp, lineage)}/{rel}"} if rel else {}
            c = R.cell(rec, exp["eval"]["template"], model,
                       str(paths.results_dir("supervisor", "eval", model, label)), held, exp["eval"]["seeds"],
                       {"save-transcripts": True})
            c["protocol_kwargs"] = {**c["protocol_kwargs"], **extra}
            out.append(shell(run_argv(c, seeds_of(c))))
    return out


def contrasts(exp: dict, rec: dict, dry: bool) -> int:
    chain = R.chain_of(rec, exp["chain"])
    held = expand(rec["sets"]["heldout"])
    base = R.contrasts_spec(rec, exp["chain"], chain, held)
    labels = [t for t, _ in teams(exp)]
    final = f"v{exp['versions']}"
    base.update(generations={t: [R.served(chain, exp["generation"], lin) + f"/{t}" for lin in exp["lineages"]]
                             for t in labels},
                eval_root=str(paths.results_dir("supervisor", "eval", "{model}")),
                seeds=seeds_of({"seeds": exp["eval"]["seeds"]}),
                contrasts=[[f"{t}_minus_fixed", t, "fixed", "step"] for t in labels if t != "fixed"]
                + ([[f"{final}_minus_rules_only", final, "rules_only", "step"]] if exp.get("rules_only") else []))
    path = write_json(paths.results_dir("supervisor", "contrasts.json"), base)
    return run([sys.executable, "-m", "rci.recursion.contrasts", str(path), "--out",
                str(paths.results_dir("supervisor", "contrasts"))], dry)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Team rewritten by a supervisor from unlabeled records (Appendix E).")
    ap.add_argument("cmd", choices=["evolve", "lines", "contrasts"])
    ap.add_argument("--base-urls", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    exp, rec = load("supervisor"), R.load_rec()
    if a.cmd == "evolve":
        print("\n".join(evolve_commands(exp, rec, a.base_urls)))
        return 0
    if a.cmd == "lines":
        print("\n".join(eval_lines(exp, rec)))
        return 0
    return contrasts(exp, rec, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
