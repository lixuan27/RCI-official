"""PlanBench plan generation as a gated chain: four agents, four calls each, the plan simulator as the check."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rci import paths
from rci.chains import stats as ST
from rci.chains.adapter import Adapter
from rci.chains.mix import mix_rows, steps_for
from rci.chains.planbench import accept as A
from rci.chains.planbench import generate as G
from rci.chains.util import write_json, write_jsonl


class PlanBench(Adapter):
    def sets_dir(self) -> Path:
        return Path(self.cfg.get("sets_dir") or paths.data_dir("chains", "planbench"))

    def set_name(self, role: str, r: int) -> str:
        return {"select": "sel", "train": f"train_r{r}", "confirm": f"conf_r{r}"}[role]

    def spec(self) -> list:
        s = self.cfg["sets"]
        out = [("sel", s["select"]["size"], s["select"]["mix"])]
        for r in range(1, int(self.cfg["stop"]["max_rounds"]) + 1):
            out.append((f"conf_r{r}", s["confirm"]["size"], s["confirm"]["mix"]))
            out.append((f"train_r{r}", s["train"]["size"], s["train"]["mix"]))
        return out

    def prepare(self) -> dict:
        built = G.build(self.spec(), self.cfg["sets"]["seed"], self.cfg["sets"].get("example_pool"))
        for name, rows in built.items():
            write_jsonl(A.problems_file(self.sets_dir(), name), rows)
        rep = {name: len(rows) for name, rows in built.items()}
        write_json(self.sets_dir() / "SETS.json", {"seed": self.cfg["sets"]["seed"], "sizes": rep})
        return rep

    def play_argv(self, models: dict, role: str, r: int, out: Path) -> list:
        m = next(iter(models.values()))
        name = self.set_name(role, self.task_round(r))
        return ["python", "-m", "rci.chains.planbench.play", "--problems", str(A.problems_file(self.sets_dir(), name)),
                "--set", name, "--seeds", "0", "--model", m["served"], "--base-urls", '"$RCI_BASE_URLS"',
                "--out", str(out), "--mode", self.cfg.get("mode", "think"),
                "--concurrency", str(self.cfg.get("concurrency", 32))]

    @lru_cache(maxsize=None)
    def problems(self, name: str) -> dict:
        return A.read_problems(A.problems_file(self.sets_dir(), name))

    def eos(self) -> int:
        return int(self.cfg["eos_id"]) if self.cfg.get("eos_id") is not None else A.eos_of(self.cfg["base"]["weights"])

    def accept(self, r: int, records: Path, rows_path: Path, lin: str) -> dict:
        rows, meta = A.build_rows(records, self.problems(self.set_name("train", self.task_round(r))), r,
                                  self.cfg["rows"], self.eos())
        write_jsonl(rows_path, rows)
        return dict(meta, rows=len(rows))

    def mix(self, st: dict, r: int, lin: str, out: Path) -> dict:
        m, t = self.cfg["mix"], self.cfg["train"]
        rows, meta = mix_rows(self.rounds_rows(lin, r), r, self.last_release_round(st), m["replay"], m["max_tokens"],
                              m["min_rows"], "planbench")
        if rows is None:
            return dict(meta, rows=0, steps=0)
        write_jsonl(out, rows)
        return dict(meta, rows=len(rows), steps=steps_for(len(rows), t["batch"], t["epochs"], t.get("steps_clip")))

    def lr(self, st: dict, r: int) -> float:
        t = self.cfg["train"]
        return ST.lr_doubling(float(t["lr"]), st["rounds"], int(t.get("lr_doublings_max", 0)), self.chain.rule.k,
                              int(t.get("lr_halve_after_unhealthy", 0)))

    def lines(self, records: Path, role: str, r: int) -> list:
        probs = self.problems(self.set_name(role, self.task_round(r) if role != "select" else 0))
        return [v for _, v in A.judged(records, probs)]

    def summary(self, records: Path, role: str, r: int) -> dict:
        return A.stats(self.lines(records, role, r), self.cfg["health"])

    def health(self, summ: dict, base: dict) -> list:
        return A.health(summ, base["select"], self.cfg["health"])

    def rank(self, summ: dict) -> float:
        return summ["rate"]

    def scores(self, records: Path, role: str, r: int) -> dict:
        return {v["key"]: float(v["commit"]["ok"]) for v in self.lines(records, role, r)}
