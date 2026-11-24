"""The gated round as a state machine.

One round r of a chain whose current generation is g:

    harvest      the harvester (g itself; a fixed generation in a non-recursive control chain) plays fresh training
                 tasks with the chain's protocol                                                         GPU
    accept       an automatic check accepts or rejects each output without answer labels; accepted outputs
                 become the round's training rows                                                        CPU
    mix          the training file: mainly the newest accepted rows                                      CPU
    train        candidates continue from g's (full-precision) weights                                   GPU
    candidates   theta_g + alpha (theta_trained - theta_g) for every alpha != 1                          CPU
    select:<a>   each candidate plays the selection set                                                  GPU
    select       health lines against the base, then the best candidate on the selection set           CPU
    confirm:cand, confirm:inc   the selected candidate and g on confirmation tasks no earlier round used GPU
    verdict      the paired gain d: release generation g + 1 when the rule passes, otherwise a flat round CPU

GPU steps are shell lines (run them with scripts/launch/run_chain.sh or any scheduler, then mark them done); CPU steps
run in this process. Everything a decision read is written next to the state in <results>/chains/<chain>/.
"""
from __future__ import annotations

import json
import shlex
from fractions import Fraction
from pathlib import Path

from rci import paths
from rci.chains import health as HL
from rci.chains import merge as MG
from rci.chains import stats as ST
from rci.chains.util import read_json, write_json

GPU, CPU = "gpu", "cpu"


class ChainError(RuntimeError):
    pass


def alpha_tag(a) -> str:
    f = Fraction(str(a))
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}_{f.denominator}"


class Chain:
    """A chain's state on disk and its next step. `adapter` supplies the benchmark: play lines, the check, rows,
    the selection summary, the health lines and per-task confirmation scores."""

    def __init__(self, name: str, cfg: dict, adapter, parent: "Chain" | None = None):
        self.name, self.cfg, self.parent = name, cfg, parent
        self.root = Path(cfg.get("results") or paths.results_dir("chains", name))
        self.ckpt = Path(cfg.get("ckpt") or paths.ckpt_dir("chains", name))
        self.lineages = list(cfg.get("lineages") or ["a"])
        self.alphas = [str(a) for a in cfg.get("alphas", ["1"])]
        self.adapter = adapter(cfg, self) if isinstance(adapter, type) else adapter
        self.rule = ST.ReleaseRule.of(cfg["release"])

    # ------------------------------------------------------------------------------------------------ state
    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def state(self) -> dict:
        st = read_json(self.state_path)
        if st is None:
            st = {"chain": self.name, "gens": {"0": self.start_gen()}, "rounds": [], "base_done": [], "stop": None}
            if self.cfg.get("fork"):
                st["fork"] = dict(self.cfg["fork"])
        return st

    def save(self, st: dict) -> None:
        write_json(self.state_path, st)

    def start_gen(self) -> dict:
        fork = self.cfg.get("fork")
        if fork:
            if self.parent is None:
                raise ChainError(f"{self.name}: a forked chain needs its parent chain")
            pg = self.parent.state()["gens"].get(str(fork["gen"]))
            if pg is None:
                raise ChainError(f"{self.name}: generation {fork['gen']} of {self.parent.name} is not released yet")
            return dict(pg, origin=f"{self.parent.name}:g{fork['gen']}")
        return self._base_gen(self.cfg["base"])

    def _base_gen(self, base: dict) -> dict:
        out = {lin: {"served": base["served"], "weights": base["weights"], "master": base.get("master") or base["weights"]}
               for lin in self.lineages}
        out["origin"] = "base"
        return out

    def current(self, st: dict) -> int:
        return max(int(k) for k in st["gens"])

    def released(self, st: dict) -> int:
        return sum(1 for x in st["rounds"] if x.get("label") == "released")

    # ------------------------------------------------------------------------------------------------ names
    def rdir(self, r: int) -> Path:
        return self.root / f"r{r}"

    def records(self, role: str, r: int, who: str) -> Path:
        return self.rdir(r) / "records" / role / who

    def base_records(self, role: str) -> Path:
        return self.root / "base" / role

    def rows_path(self, r: int, lin: str) -> Path:
        return self.rdir(r) / f"rows_{lin}.jsonl"

    def mix_path(self, r: int, lin: str) -> Path:
        return self.rdir(r) / f"mix_{lin}.jsonl"

    def train_dir(self, r: int, lin: str) -> Path:
        return self.ckpt / f"r{r}" / f"train_{lin}"

    def cand(self, r: int, alpha) -> dict:
        tag = alpha_tag(alpha)
        out = {}
        for lin in self.lineages:
            if Fraction(str(alpha)) == 1:
                d = self.train_dir(r, lin)
                out[lin] = {"weights": str(d / "hf"), "master": str(d / "master")}
            else:
                d = self.ckpt / f"r{r}" / f"a{tag}_{lin}"
                out[lin] = {"weights": str(d / "hf"), "master": str(d / "master")}
            out[lin]["served"] = f"{self.cfg['base']['served']}-{self.name}-r{r}-a{tag}" + (
                f"-{lin}" if len(self.lineages) > 1 else "")
        return out

    def gen_model(self, st: dict, k: int) -> dict:
        g = st["gens"].get(str(k))
        if g is None:
            raise ChainError(f"{self.name}: no generation {k}")
        return {lin: g[lin] for lin in self.lineages}

    # ------------------------------------------------------------------------------------------------ steps
    def plan_round(self, st: dict, rd: dict) -> list:
        steps = ["harvest", "accept", "mix"] + [f"train:{lin}" for lin in self.lineages]
        if any(Fraction(a) != 1 for a in self.alphas):
            steps.append("candidates")
        steps += [f"select:{alpha_tag(a)}" for a in self.alphas] + ["select"]
        if rd.get("label") == "unhealthy":
            return steps
        return steps + ["confirm:cand", "confirm:inc", "verdict"]

    def stop_reason(self, st: dict) -> str | None:
        if st.get("stop"):
            return st["stop"]
        c = self.cfg.get("stop") or {}
        rounds = st["rounds"]
        decided = [x for x in rounds if x.get("label")]
        if c.get("max_releases") and self.released(st) >= int(c["max_releases"]):
            return f"{self.released(st)} generations released"
        if c.get("max_rounds") and len(decided) >= int(c["max_rounds"]) and (not rounds or rounds[-1].get("label")):
            return f"{len(decided)} rounds decided"
        if c.get("flat_first_two") and len(decided) >= 2 and all(x["label"] != "released" for x in decided[:2]):
            return "rounds 1 and 2 both flat"
        fork = self.cfg.get("fork") or {}
        if fork.get("stop_with_parent_releases") and self.parent is not None:
            pst = self.parent.state()
            rel = [x for x in pst["rounds"] if x.get("label") == "released"]
            n = int(fork["stop_with_parent_releases"])
            if len(rel) >= n and len(decided) >= rel[n - 1]["r"] - self.fork_round():
                return f"the parent chain released generation {n}"
        return None

    def fork_round(self) -> int:
        if not self.parent:
            return 0
        rel = [x for x in self.parent.state()["rounds"] if x.get("label") == "released"]
        k = int(self.cfg["fork"]["gen"])
        return rel[k - 1]["r"] if len(rel) >= k else 0

    def base_steps(self) -> list:
        return [f"base:{role}" for role in self.adapter.base_roles()]

    def next(self) -> dict | None:
        """{"kind", "step", "line"?} of the next step ("wait" while a forked chain's parent has not released its
        fork generation), or None when the chain has stopped (state "stop" says why)."""
        try:
            st = self.state()
        except ChainError as e:
            return {"kind": "wait", "step": str(e), "round": 0}
        for s in self.base_steps():
            if s not in st["base_done"]:
                return self.describe(st, None, s)
        if not st["rounds"] or st["rounds"][-1].get("label"):
            why = self.stop_reason(st)
            if why:
                if st.get("stop") != why:
                    st["stop"] = why
                    self.save(st)
                return None
            r = len(st["rounds"]) + 1
            st["rounds"].append({"r": r, "gen": self.current(st), "done": [], "lr": self.adapter.lr(st, r)})
            self.save(st)
        rd = st["rounds"][-1]
        for s in self.plan_round(st, rd):
            if s not in rd["done"]:
                return self.describe(st, rd, s)
        raise ChainError(f"round {rd['r']}: every step is done but the round has no label")

    def describe(self, st: dict, rd: dict | None, step: str) -> dict:
        kind = CPU if step in ("accept", "mix", "candidates", "select", "verdict") else GPU
        out = {"kind": kind, "step": step, "round": rd["r"] if rd else 0}
        if kind == GPU:
            out["line"] = self.gpu_line(st, rd, step)
        return out

    def harvester(self, st: dict, rd: dict) -> dict:
        h = self.cfg.get("harvester") or "current"
        if h == "current":
            return self.gen_model(st, rd["gen"])
        if h == "start":
            return self.gen_model(st, 0)
        raise ChainError(f"harvester {h!r}")

    def gpu_line(self, st: dict, rd: dict | None, step: str) -> str:
        a = self.adapter
        if step.startswith("base:"):
            role = step.split(":", 1)[1]
            base = self._base_gen(self.cfg["base"])
            return a.play_line({lin: base[lin] for lin in self.lineages}, role, 0, self.base_records(role))
        r = rd["r"]
        if step == "harvest":
            return a.play_line(self.harvester(st, rd), "train", r, self.records("train", r, "harvest"))
        if step.startswith("train:"):
            lin = step.split(":", 1)[1]
            inc = self.gen_model(st, rd["gen"])[lin]
            return a.train_line(self.mix_path(r, lin), inc["master"], self.train_dir(r, lin), rd, lin)
        if step.startswith("select:"):
            tag = step.split(":", 1)[1]
            alpha = next(x for x in self.alphas if alpha_tag(x) == tag)
            return a.play_line(self.cand(r, alpha), "select", r, self.records("select", r, f"a{tag}"))
        if step == "confirm:cand":
            return a.play_line(self.cand(r, rd["selected"]), "confirm", r, self.records("confirm", r, "cand"))
        if step == "confirm:inc":
            return a.play_line(self.gen_model(st, rd["gen"]), "confirm", r, self.records("confirm", r, "inc"))
        raise ChainError(f"no GPU step {step!r}")

    def mark(self, step: str, info: dict | None = None) -> None:
        st = self.state()
        if step.startswith("base:"):
            if step not in st["base_done"]:
                st["base_done"].append(step)
            self.save(st)
            return
        rd = st["rounds"][-1]
        if step not in rd["done"]:
            rd["done"].append(step)
        if info:
            rd.setdefault("info", {})[step] = info
        self.save(st)

    # ------------------------------------------------------------------------------------------------ CPU steps
    def run_cpu(self, step: str) -> dict:
        nxt = self.next()
        if nxt is None or nxt["step"] != step:
            raise ChainError(f"{step} is not the next step ({nxt and nxt['step']})")
        st = self.state()
        rd = st["rounds"][-1]
        r = rd["r"]
        a = self.adapter
        if step == "accept":
            meta = {lin: a.accept(r, self.records("train", r, "harvest"), self.rows_path(r, lin), lin)
                    for lin in self.lineages}
            write_json(self.rdir(r) / "accept.json", meta)
            why = next((m.get("stop") for m in meta.values() if m.get("stop")), None)
            self.mark(step, {"rows": {lin: m.get("rows") for lin, m in meta.items()}})
            if why:
                st = self.state()
                st["rounds"][-1]["label"] = "stopped"
                st["stop"] = why
                self.save(st)
            return meta
        if step == "mix":
            meta = {}
            for lin in self.lineages:
                meta[lin] = a.mix(st, r, lin, self.mix_path(r, lin))
            write_json(self.rdir(r) / "mix.json", meta)
            starved = next((lin for lin, m in meta.items() if m.get("starved")), None)
            self.mark(step, {lin: {"rows": m.get("rows"), "steps": m.get("steps")} for lin, m in meta.items()})
            if starved:
                st = self.state()
                st["rounds"][-1]["label"] = "stopped"
                st["stop"] = f"round {r}: too few accepted rows to train on"
                self.save(st)
            return meta
        if step == "candidates":
            out = {}
            for alpha in self.alphas:
                if Fraction(alpha) == 1:
                    continue
                c = self.cand(r, alpha)
                for lin in self.lineages:
                    inc = self.gen_model(st, rd["gen"])[lin]
                    trained = self.train_dir(r, lin) / "master"
                    MG.merge(trained, inc["master"], c[lin]["master"], alpha, "F32")
                    MG.merge(trained, inc["master"], c[lin]["weights"], alpha, "BF16")
                    out[f"{alpha_tag(alpha)}:{lin}"] = c[lin]
            self.mark(step)
            return out
        if step == "select":
            return self.select(st, rd)
        if step == "verdict":
            return self.verdict(st, rd)
        raise ChainError(f"no CPU step {step!r}")

    def select(self, st: dict, rd: dict) -> dict:
        r, a = rd["r"], self.adapter
        base = {role: a.summary(self.base_records(role), role, 0) for role in a.base_roles()}
        out = {"round": r, "cands": {}}
        ok = []
        for alpha in self.alphas:
            tag = alpha_tag(alpha)
            summ = a.summary(self.records("select", r, f"a{tag}"), "select", r)
            lines = a.health(summ, base)
            good = HL.healthy(lines)
            out["cands"][tag] = {"alpha": alpha, "summary": summ, "health": [x.to_json() for x in lines],
                                 "healthy": good, "rank": a.rank(summ)}
            if good:
                ok.append((a.rank(summ), -Fraction(alpha), alpha))
        st = self.state()
        rd = st["rounds"][-1]
        if ok:
            rd["selected"] = max(ok)[2]
            out["selected"] = rd["selected"]
        else:
            rd["label"] = "unhealthy"
            out["selected"] = None
        write_json(self.rdir(r) / "select.json", out)
        rd["done"].append("select")
        self.save(st)
        return out

    def verdict(self, st: dict, rd: dict) -> dict:
        r, a = rd["r"], self.adapter
        cand = a.scores(self.records("confirm", r, "cand"), "confirm", r)
        inc = a.scores(self.records("confirm", r, "inc"), "confirm", r)
        conf = a.paired(cand, inc)
        gate = a.confirm_gate(self.records("confirm", r, "cand"), self.records("confirm", r, "inc"), r)
        passed = self.rule.passes(conf["d"], conf["se"]) and all(x.ok for x in gate)
        out = {"round": r, "selected": rd["selected"], "confirm": conf, "line": self.rule.line(conf["se"]),
               "gate": [x.to_json() for x in gate], "rule": self.cfg["release"],
               "label": "released" if passed else "flat"}
        st = self.state()
        rd = st["rounds"][-1]
        rd["confirm"] = conf
        rd["label"] = out["label"]
        if passed:
            k = self.current(st) + 1
            st["gens"][str(k)] = dict(self.cand(r, rd["selected"]), origin=f"round {r}, alpha {rd['selected']}")
            rd["released_gen"] = k
            out["gen"] = k
        rd["done"].append("verdict")
        write_json(self.rdir(r) / "verdict.json", out)
        self.save(st)
        return out

    def status(self) -> dict:
        st = self.state()
        return {"chain": self.name, "generations": self.current(st), "stop": st.get("stop"),
                "rounds": [{k: x.get(k) for k in ("r", "gen", "label", "selected", "released_gen")}
                           for x in st["rounds"]]}


def with_server(models: dict, argv: list, gpus: str | None = None) -> str:
    """The shell line that serves every lineage's weights, runs argv (one quoted command, so that "$RCI_BASE_URLS" and
    "$RCI_POOL" expand inside the wrapper), and stops the servers."""
    parts = ["scripts/launch/with_server.sh"]
    if gpus:
        parts += ["--gpus", gpus]
    seen = set()
    for m in models.values():
        if m["served"] in seen:
            continue
        seen.add(m["served"])
        parts += ["--model", m["served"], m["weights"]]
    inner = " ".join(p if p.startswith('"$RCI_') else shlex.quote(p) for p in argv)
    return " ".join(shlex.quote(p) for p in parts) + " -- " + shlex.quote(inner)


def train_line(cfg: dict, data: Path, init: str, out: Path, seed: int, lr: float, steps: int) -> str:
    t = cfg["train"]
    argv = ["scripts/launch/train_sft.sh", "--data", str(data), "--model", str(init), "--out", str(out),
            "--lr", repr(float(lr)), "--batch", str(t["batch"]), "--steps", str(int(steps)), "--seed", str(seed),
            "--warmup", str(t.get("warmup", 0.03)), "--grad-clip", str(t.get("grad_clip", 1.0)), "--master"]
    if t.get("pad_multiple"):
        argv += ["--pad-multiple", str(t["pad_multiple"])]
    return " ".join(shlex.quote(x) for x in argv)


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str)
