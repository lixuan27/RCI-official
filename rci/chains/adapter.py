"""What a benchmark supplies to the gated round (rci.chains.loop.Chain)."""
from __future__ import annotations

from pathlib import Path

from rci.chains import loop as L
from rci.chains import stats as ST
from rci.chains.util import read_json


class Adapter:
    """Defaults shared by the three benchmarks; a subclass supplies play_argv, accept, summary, health, rank, scores."""

    min_paired_share = 0.99

    def __init__(self, cfg: dict, chain):
        self.cfg, self.chain = cfg, chain

    # GPU lines
    def play_argv(self, models: dict, role: str, r: int, out: Path) -> list:
        raise NotImplementedError

    def play_line(self, models: dict, role: str, r: int, out: Path) -> str:
        return L.with_server(models, self.play_argv(models, role, r, out), self.cfg.get("serve_gpus"))

    def train_seed(self, r: int, lin: str) -> int:
        t = self.cfg["train"]
        return int(t.get("seed", 0)) + 1000 * r + 10 * self.chain.lineages.index(lin) + int(t.get("seed_offset", 0))

    def train_line(self, data: Path, init: str, out: Path, rd: dict, lin: str) -> str:
        steps = ((rd.get("info") or {}).get("mix") or {}).get(lin, {}).get("steps")
        if steps is None:
            steps = (read_json(self.chain.rdir(rd["r"]) / "mix.json", {}) or {}).get(lin, {}).get("steps", 0)
        return L.train_line(self.cfg, data, init, out, self.train_seed(rd["r"], lin), rd["lr"], int(steps or 0))

    def lr(self, st: dict, r: int) -> float:
        return float(self.cfg["train"]["lr"])

    def base_roles(self) -> list:
        return ["select"]

    def task_round(self, r: int) -> int:
        """The round whose fresh tasks round r plays (a forked chain continues the parent's numbering)."""
        return r + self.chain.fork_round()

    def paired(self, cand: dict, inc: dict) -> dict:
        return ST.paired(cand, inc, self.min_paired_share)

    def confirm_gate(self, cand_records: Path, inc_records: Path, r: int) -> list:
        """Health lines the confirmation cells themselves must hold (none by default)."""
        return []

    def rounds_rows(self, lin: str, upto: int) -> dict:
        from rci.chains.util import read_jsonl
        out = {}
        for q in range(1, upto + 1):
            p = self.chain.rows_path(q, lin)
            out[q] = read_jsonl(p) if p.exists() else []
        return out

    def last_release_round(self, st: dict) -> int:
        rel = [x["r"] for x in st["rounds"] if x.get("label") == "released"]
        return max(rel) if rel else 0
