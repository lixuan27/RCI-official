"""Hanabi as a gated chain: four agents, one call per move, the game score as the check.

The recursive chain trains each generation on its own elite games; the control chain branches after the first release
and trains on games of that fixed generation (harvester "start"), with its own weights continuing round by round.
Candidates are theta_g + alpha (theta_trained - theta_g) for alpha in {1/2, 1, 2}; the winner on 300 fresh selection
deals (rank: the mean elite value) meets the current generation on 600 fresh confirmation deals, one game each, the
same sampling seeds; it is released when d - SE > 0 on the challenge score (bootstrap SE over deals).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rci.chains import stats as ST
from rci.chains.adapter import Adapter
from rci.chains.hanabi import accept as A
from rci.chains.hanabi import play as P
from rci.chains.mix import steps_for
from rci.chains.util import write_jsonl


class Hanabi(Adapter):
    def base_roles(self) -> list:
        return []

    def stalled(self, st: dict, upto: int) -> bool:
        """After `stall_held` flat rounds in a row, or a round whose elite advantage is too small, the chain plays
        G_stall games a deal on half the deals (for good)."""
        c, held = self.cfg["harvest"], 0
        for x in st["rounds"][:upto]:
            held = 0 if x.get("label") == "released" else held + 1
            meta = (x.get("info") or {}).get("accept") or {}
            low = meta.get("adv_pos_share") is not None and meta["adv_pos_share"] < c["stall_adv_share"]
            low = low or (meta.get("elite_adv_mean_ch") is not None and meta["elite_adv_mean_ch"] < c["stall_adv_mean"])
            if held >= c["stall_held"] or low:
                return True
        return False

    def harvest_shape(self, st: dict, r: int) -> tuple:
        c = self.cfg["harvest"]
        if self.stalled(st, r - 1):
            return c["deals"] // 2, c["stall_games"]
        return c["deals"], c["games"]

    def play_argv(self, models: dict, role: str, r: int, out: Path) -> list:
        m = next(iter(models.values()))
        rr = self.task_round(r)
        if role == "train":
            n, games = self.harvest_shape(self.chain.state(), r)
        else:
            n, games = self.cfg["sets"][role], 1
        lo, hi = P.deal_range(role, rr, n)
        return ["python", "-m", "rci.chains.hanabi.play", "--role", role, "--round", str(rr), "--deals", str(lo),
                str(hi), "--games", str(games), "--model", m["served"], "--base-urls", '"$RCI_BASE_URLS"',
                "--tokenizer", self.cfg["base"]["weights"], "--out", str(out)]

    @lru_cache(maxsize=1)
    def tok(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.cfg["base"]["weights"])

    def accept(self, r: int, records: Path, rows_path: Path, lin: str) -> dict:
        rows, meta = A.build(records, self.tok(), r, self.cfg["rows"], f"hanabi/{self.chain.name}")
        write_jsonl(rows_path, rows)
        return meta

    def mix(self, st: dict, r: int, lin: str, out: Path) -> dict:
        from rci.chains.util import read_jsonl
        rows = read_jsonl(self.chain.rows_path(r, lin))
        write_jsonl(out, rows)
        t = self.cfg["train"]
        return {"rows": len(rows), "steps": steps_for(len(rows), t["batch"], t["epochs"]), "starved": not rows}

    def lr(self, st: dict, r: int) -> float:
        t = self.cfg["train"]
        winners = [x["selected"] for x in st["rounds"] if x.get("label") == "released"]
        return ST.lr_alpha_pairs(float(t["lr"]), winners, t["lr_bounds"])

    def summary(self, records: Path, role: str, r: int) -> dict:
        return A.summary(A.load(records))

    def health(self, summ: dict, base: dict) -> list:
        return A.health(summ, self.cfg["health"])

    def rank(self, summ: dict) -> float:
        return summ[self.cfg.get("rank", "ev_mean")]

    def scores(self, records: Path, role: str, r: int) -> dict:
        return {deal: float(rec["s_ch"]) for (deal, g), rec in A.load(records).items() if g == 0}

    def confirm_gate(self, cand_records: Path, inc_records: Path, r: int) -> list:
        from rci.chains import health as HL
        h = self.cfg["health"]["protocol_share_max"]
        return [HL.at_most(f"{who} protocol failures / calls", A.summary(A.load(d)).get("protocol_share"), h)
                for who, d in (("candidate", cand_records), ("current generation", inc_records))]

    def paired(self, cand: dict, inc: dict) -> dict:
        b = self.cfg["bootstrap"]
        return ST.bootstrap_paired(cand, inc, int(b["n"]), int(b["seed"]))
