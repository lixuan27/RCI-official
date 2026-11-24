"""DocSweep admission from the 27B model's own behaviour (oracle, single capped member, shortcut probe)."""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os

from rci import paths

ORACLE_SEEDS = (100, 101)
M1_SEEDS = (100, 101, 102, 103)
PER_FAMILY, SMOKE_PER_FAMILY, K8_AGG = 68, 4, 48
FAMILIES = ("agg", "join")


def load_records(root: str, protocol: str, n: int) -> dict[str, list[dict]]:
    """Question -> its scored records under <root>/<protocol>/N<n>/s<seed>/, sorted by seed."""
    out: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(os.path.join(root, protocol, f"N{n}", "s*", "*.json*"))):
        if not f.endswith((".json", ".json.gz")) or f.endswith(".chains.jsonl.gz"):
            continue
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
            rec = json.load(fh)
        sc = rec.get("score") or {}
        if not sc.get("per_agent") or sc.get("score") is None:
            continue
        out.setdefault(os.path.basename(f).split(".json")[0], []).append(rec)
    return {q: sorted(v, key=seed_of) for q, v in sorted(out.items())}


def seed_of(rec: dict) -> int:
    return int((rec.get("config") or {}).get("seed", 0))


def y_of(rec: dict) -> float:
    """The record's committed answer under the commitment rule (most backers, ties by summed confidence)."""
    from rci.ladder.commit import vote_stake
    return float(vote_stake(rec))


def episode_error(rec: dict) -> bool:
    """A host or service error, a failing member, or a context overflow anywhere in the episode."""
    if rec.get("error"):
        return True
    agents = rec.get("agents") or []
    for a in agents:
        if a.get("overflow") or a.get("failing"):
            return True
        if any(c.get("error") for c in a.get("calls") or []):
            return True
    return any(e.get("op") == "overflow" for a in agents for e in (a.get("task_state") or {}).get("events") or [])


def corpus_sha(example: dict) -> str:
    return hashlib.sha256(json.dumps(example, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_corpora(data: str) -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(os.path.join(data, "corpora", "*.json"))):
        ex = json.load(open(f))
        out[str(ex.get("id") or os.path.basename(f)[:-5])] = ex
    return out


def run_failed(rec: dict) -> bool:
    if episode_error(rec):
        return True
    return any(isinstance(e, dict) and e.get("event") == "context_overflow"
               for log in (rec.get("transcripts") or {}).values() for e in (log or []))


def by_seed(recs: dict[str, list[dict]]) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, str]]]:
    scores: dict[str, dict[int, float]] = {}
    errors: dict[str, dict[int, str]] = {}
    for q, rr in recs.items():
        for r in rr:
            if run_failed(r):
                errors.setdefault(q, {})[seed_of(r)] = "run error or context overflow"
            else:
                scores.setdefault(q, {})[seed_of(r)] = y_of(r)
    return scores, errors


def task_api():
    """The DocSweep admission functions, imported lazily."""
    from rci.tasks.docsweep.generate import decoy_ok, generator_sha256
    from rci.tasks.docsweep.normalise import pred_of
    from rci.tasks.docsweep.probe import shortcut_answer
    return {"probe": shortcut_answer, "decoy_ok": decoy_ok, "pred_of": pred_of, "generator_sha256": generator_sha256}


def judge(qid: str, ex: dict, truth: dict, oracle: dict, m1: dict, api: dict, tokenizer, oracle_only: bool = False,
          errors: dict | None = None) -> dict:
    """One candidate's admission row: oracle and m=1 counts, the probe, the decoy, and the verdict."""
    fam = ex.get("family")
    o = oracle.get(qid, {})
    m = m1.get(qid, {})
    err = errors or {}
    e_o, e_m = (err.get("oracle") or {}).get(qid, {}), (err.get("m1") or {}).get(qid, {})
    missing = [f"error oracle s{s}" if s in e_o else f"oracle s{s}" for s in ORACLE_SEEDS if s not in o]
    if not oracle_only:
        missing += [f"error m1 s{s}" if s in e_m else f"m1 s{s}" for s in M1_SEEDS if s not in m]
    oc = sum(1 for s in ORACLE_SEEDS if o.get(s, 0.0) >= 1.0)
    mc = sum(1 for s in M1_SEEDS if m.get(s, 0.0) >= 1.0)
    t = truth.get(qid) or {}
    try:
        probe_pred = api["pred_of"](ex, api["probe"](ex, tokenizer))
        probe_wrong = probe_pred != api["pred_of"](ex, t.get("answer"))
    except Exception as e:
        probe_wrong, missing = False, missing + [f"probe: {e}"]
    decoy = bool(api["decoy_ok"](ex, t)) if fam == "join" else None
    ok = (not missing and oc == len(ORACLE_SEEDS) and (oracle_only or mc <= 1) and probe_wrong and (fam != "join" or decoy))
    return {"family": fam, "sha256": corpus_sha(ex), "oracle_correct": oc, "m1_correct": None if oracle_only else mc,
            "probe_wrong": probe_wrong, "decoy_ok": decoy, "admitted": bool(ok), "role": None, "missing": missing}


def decide(corpora: dict[str, dict], truth: dict, oracle: dict, m1: dict, api: dict, tokenizer,
           per_family: int = PER_FAMILY, smoke: int = SMOKE_PER_FAMILY, errors: dict | None = None) -> dict:
    rows, lists = {}, {"eval": [], "smoke": [], "train": [], "k8": []}
    shortfall = {}
    for fam in FAMILIES:
        cands = sorted(((corpus_sha(ex), q) for q, ex in corpora.items()
                        if ex.get("family") == fam and q.startswith(f"ds_{fam}_")), key=lambda t: t[0])
        taken = []
        for _, q in cands:
            row = judge(q, corpora[q], truth, oracle, m1, api, tokenizer, errors=errors)
            rows[q] = row
            if row["admitted"] and len(taken) < per_family:
                taken.append(q)
            elif row["admitted"]:
                row["role"] = "unused"
        top = sorted(taken, key=lambda q: rows[q]["sha256"])[-smoke:] if smoke else []
        for q in taken:
            rows[q]["role"] = "smoke" if q in top else "eval"
            lists["smoke" if q in top else "eval"].append(q)
        if len(taken) < per_family:
            shortfall[fam] = per_family - len(taken)
    k8 = sorted(((corpus_sha(ex), q) for q, ex in corpora.items() if q.startswith("ds_k8_agg_")), key=lambda t: t[0])
    for _, q in k8:
        if q not in oracle and q not in ((errors or {}).get("oracle") or {}):
            continue
        row = judge(q, corpora[q], truth, oracle, m1, api, tokenizer, oracle_only=True, errors=errors)
        rows[q] = row
        if row["admitted"] and len(lists["k8"]) < K8_AGG:
            row["role"] = "k8"
            lists["k8"].append(q)
    lists["train"] = sorted(q for q in corpora if q.startswith("ds_train_"))
    return {"rows": rows, "shortfall": shortfall, **lists}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.docsweep.admit",
                                 description="Admit DocSweep questions from the admission runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--root", required=True, help="results root holding admit_oracle/ and admit_m1/")
    d.add_argument("--oracle-root", default=None)
    d.add_argument("--m1-root", default=None)
    d.add_argument("--data", default=None, help="default: $RCI_DATA/docsweep")
    d.add_argument("--tokenizer", default="stub")
    d.add_argument("--questions-out", default="configs/docsweep/questions.json")
    d.add_argument("--write-partial", action="store_true")
    d.add_argument("--replace", action="store_true",
                   help="overwrite an existing questions file (the frozen list); prints the old-new differences")
    a = ap.parse_args(argv)
    a.data = a.data or str(paths.data_dir("docsweep"))
    api = task_api()
    corpora = load_corpora(a.data)
    truth_path = os.path.join(a.data, "truth.json")
    truth = json.load(open(truth_path))
    oracle, err_o = by_seed(load_records(a.oracle_root or os.path.join(a.root, "admit_oracle"), "oracle", 1))
    m1, err_m = by_seed(load_records(a.m1_root or os.path.join(a.root, "admit_m1"), "independent", 1))
    try:
        from rci.agents.tokenizer import load_tokenizer
        tok = load_tokenizer(a.tokenizer)
    except Exception:
        tok = a.tokenizer
    res = decide(corpora, truth, oracle, m1, api, tok, errors={"oracle": err_o, "m1": err_m})
    out = {"generator_sha256": api["generator_sha256"](), "truth_sha256": hashlib.sha256(open(truth_path, "rb").read()).hexdigest(),
           "k": 24, "eval": res["eval"], "smoke": res["smoke"], "train": res["train"], "k8": res["k8"],
           "corpus_sha256": {q: corpus_sha(corpora[q]) for q in res["eval"] + res["smoke"] + res["train"] + res["k8"]},
           "admission": {"oracle_seeds": list(ORACLE_SEEDS), "m1_seeds": list(M1_SEEDS),
                         "rows": {q: {k: v for k, v in r.items() if k != "missing"} for q, r in res["rows"].items()}}}
    failed = {f"{kind} {q} s{seed}": why for kind, e in (("oracle", err_o), ("m1", err_m)) for q, by in e.items()
              for seed, why in by.items()}
    report = {"shortfall": res["shortfall"], "eval": len(res["eval"]), "smoke": len(res["smoke"]), "k8": len(res["k8"]),
              "train": len(res["train"]), "not_admitted_missing": {q: r["missing"] for q, r in res["rows"].items() if r["missing"]},
              "failed_runs": failed}
    if failed:
        report["failed_runs_note"] = ("these runs failed and were not scored; delete their records and run them again "
                                      "(rci.run skips an existing record)")
    if res["shortfall"] and not a.write_partial:
        report["written"] = None
        print(json.dumps(report, indent=1, ensure_ascii=False))
        raise SystemExit("admit: shortfall; generate more candidates with the same code and admit again (not written)")
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(a.questions_out)), exist_ok=True)
    if os.path.exists(a.questions_out):
        if not a.replace:
            print(json.dumps(report, indent=1, ensure_ascii=False))
            raise SystemExit(f"admit: {a.questions_out} exists (the list is frozen before the smoke); not overwritten. "
                             "Pass --replace to overwrite it")
        report["replaced"] = replacement_diff(json.load(open(a.questions_out)), out)
        with open(a.questions_out, "w") as fh:
            fh.write(text)
    else:
        with open(a.questions_out, "x") as fh:
            fh.write(text)
    report["written"] = a.questions_out
    print(json.dumps(report, indent=1, ensure_ascii=False))


def replacement_diff(old: dict, new: dict) -> dict:
    """What an explicit --replace changes in the frozen questions file: the eval and smoke lists and the sha256s."""
    out = {}
    for key in ("eval", "smoke", "k8", "train"):
        a, b = set(old.get(key) or []), set(new.get(key) or [])
        if a != b:
            out[key] = {"removed": sorted(a - b), "added": sorted(b - a)}
    for key in ("truth_sha256", "generator_sha256"):
        if old.get(key) != new.get(key):
            out[key] = {"old": old.get(key), "new": new.get(key)}
    return out


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
