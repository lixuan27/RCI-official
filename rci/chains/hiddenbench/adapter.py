"""HiddenBench as a gated chain: four agents in a three-round debate that commit the answer their stakes back most.

The population has two lineages (seats 0-1 and 2-3), each trained on the rows of its own seats. Training and selection
use hidden-profile tasks the base model wrote (rci.chains.hiddenbench.tasks); a turn is accepted when its committed
answer is the base model's full-information reference. The selection set (200 hidden-profile and 100 control tasks)
runs with two seeds; every round confirms on new seeds of the same tasks, paired by question, and releases when d
exceeds the margin. The single-agent self-training chain (kind "solo") plays the same tasks with four independent
agents and trains each on its own solo turns under the same caps.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from rci import paths
from rci.chains.adapter import Adapter
from rci.chains.hiddenbench import accept as A
from rci.chains.hiddenbench import tasks as TK
from rci.chains.mix import mix_tokens, twin_plan
from rci.chains.util import read_json, read_jsonl, write_json, write_jsonl


class HiddenBench(Adapter):
    def data(self) -> Path:
        return Path(self.cfg.get("sets_dir") or paths.data_dir("chains", "hiddenbench"))

    @property
    def kind(self) -> str:
        return self.cfg.get("kind", "collective")

    def protocol(self) -> tuple:
        p = self.cfg["protocol"][self.kind]
        return p["name"], p.get("kwargs") or {}

    def proto_dir(self) -> str:
        from rci.registry import protocol_dir
        name, kw = self.protocol()
        return protocol_dir(name, **kw)

    def seeds(self, role: str, r: int) -> list:
        s = self.cfg["seeds"]
        if role == "select":
            return list(s["select"])
        if role == "confirm":
            return [x + int(s["confirm_step"]) * (r - 1) for x in s["confirm"]]
        return [int(s["harvest"]) + 10 * r + g for g in range(int(s["harvest_games"]))]

    def frontier(self, r: int) -> dict:
        if r < 1:
            return {}
        return (read_json(self.chain.rdir(r) / "band.json", {}) or {}).get("frontier_ages", {})

    def questions(self, role: str, r: int) -> list | None:
        if role != "train":
            return None
        ids = TK.slice_ids(self.data(), self.task_round(r), int(self.cfg["sets"]["slice"]))
        return ids + sorted(q for q in self.frontier(r - 1) if q not in ids)

    def play_argv(self, models: dict, role: str, r: int, out: Path) -> list:
        name, kw = self.protocol()
        root = self.data() / ("pool" if role == "train" else "select")
        argv = ["python", "-m", "rci.run", "--task", "hiddenbench", "--task-kwargs",
                json.dumps({**self.cfg["task_kwargs"], "root": str(root)}), "--protocol", name,
                "--protocol-kwargs", json.dumps(kw), "--n-agents", "4",
                "--seeds", ",".join(str(s) for s in self.seeds(role, r)), "--out", str(out)]
        served = [models[lin]["served"] for lin in self.chain.lineages]
        if len(set(served)) == 1:
            argv += ["--model", served[0], "--base-urls", '"$RCI_BASE_URLS"']
        else:
            seat = [models[self.lineage_of(j)]["served"] for j in range(4)]
            argv += ["--model", served[0], "--base-urls", '"$RCI_BASE_URLS"', "--pool", '"$RCI_POOL"',
                     "--agent-models", ",".join(seat)]
        for k, v in (self.cfg.get("run") or {}).items():
            argv += [f"--{k}", str(v)]
        argv += ["--capture", "--save-transcripts"]
        q = self.questions(role, r)
        if q is not None:
            argv += ["--example-ids", ",".join(q)]
        return argv

    def lineage_of(self, seat: int) -> str:
        for lin, seats in self.cfg["seats"].items():
            if seat in seats:
                return lin
        raise KeyError(seat)

    @lru_cache(maxsize=None)
    def bench(self, which: str) -> dict:
        rows = json.loads((self.data() / which / "benchmark.json").read_text())
        return {TK.example_id(r): r for r in rows}

    def reference(self, which: str) -> dict:
        return {q: A.norm(r["correct_answer"]) for q, r in self.bench(which).items()}

    @lru_cache(maxsize=1)
    def control(self) -> frozenset:
        return frozenset(json.loads((self.data() / "control.json").read_text()))

    @lru_cache(maxsize=1)
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.cfg["base"]["weights"])

    def piece(self):
        tok, memo = self.tokenizer(), {}

        def f(t: int) -> str:
            if t not in memo:
                memo[t] = tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return memo[t]
        return f

    def im_end(self) -> int:
        return int(self.tokenizer().convert_tokens_to_ids(A.IM_END))

    def switches(self) -> dict:
        t = self.cfg.get("twin") or {}
        return {"answer_value": t.get("answer_value_weight") == "skeleton", "decision_out": bool(t.get("decision_token_out")),
                "answer_value_families": tuple(t.get("answer_value_families", ["D"])),
                "decision_families": tuple(t.get("decision_families", ["D", "R2"]))}

    def keep_path(self, r: int, lin: str) -> Path:
        return self.chain.rdir(r) / f"rows_keep_{lin}.jsonl"

    def accept(self, r: int, records: Path, rows_path: Path, lin: str, chains_of=None, piece=None, im_end=None) -> dict:
        from rci.agents.capture import read_chains

        c, b = self.cfg["rows"], self.cfg["band"]
        recs = A.records(records, self.proto_dir(), 4, set(self.seeds("train", r)))
        bench, ref = self.bench("pool"), self.reference("pool")
        outcomes: dict = {}
        for _, _, rec in recs:
            q = str(rec["example_id"])
            if q in bench:
                outcomes.setdefault(q, []).extend(A.member(rec, j)["final"] for j in range(len(rec["agents"])))
        rep = A.band(outcomes, ref, self.frontier(r - 1), b["rho_hi"], b["f_max"], b["max_age"], f"hb/{self.chain.name}/")
        write_json(self.chain.rdir(r) / "band.json", rep)
        chains_of = chains_of or (lambda p: read_chains(Path(p)))
        piece = piece or self.piece()
        im_end = self.im_end() if im_end is None else im_end
        seats = list(self.cfg["seats"][lin])
        rows, meta = A.build(recs, chains_of, bench, ref, seats, self.kind, piece, im_end, c, r, f"{self.chain.name}/{lin}",
                             self.switches(), set(rep["train"]))
        write_jsonl(rows_path, rows)
        keep = self.cfg.get("keep") or {}
        if keep.get("share"):
            qs = {q for q in rep["archive"] if q in self.control()}
            krows, kmeta = A.build(recs, chains_of, bench, ref, seats, self.kind, piece, im_end,
                                   dict(c, families=keep.get("families", ["D", "R2"])), r,
                                   f"{self.chain.name}/{lin}/keep", self.switches(), qs)
            write_jsonl(self.keep_path(r, lin), krows)
            meta["keep"] = kmeta
        meta["band"] = {k: len(rep[k]) for k in ("train", "archive", "frontier", "expired")}
        meta["band"]["new_solved"] = rep["new_solved"]
        why = A.stagnated(rep, int(b["t_min"]))
        if why:
            meta["stop"] = why
        return meta

    def mix(self, st: dict, r: int, lin: str, out: Path) -> dict:
        m, t = self.cfg["mix"], self.cfg["train"]
        budget = int(m["tokens_per_round"])
        keep = self.cfg.get("keep") or {}
        k_share = float(eval_fraction(keep.get("share", 0)))
        main_b = budget - int(k_share * budget)
        by = {q: read_jsonl(self.chain.rows_path(q, lin)) for q in range(1, r + 1)
              if self.chain.rows_path(q, lin).exists()}
        by = {q: v for q, v in by.items() if v}
        if r not in by:
            return {"rows": 0, "steps": 0, "starved": True}
        lines, meta = mix_tokens(by, r, [q for q in sorted(by) if q != r], m["replay"], main_b,
                                 f"{self.chain.name}/{lin}", int(m["max_passes"]))
        if k_share:
            kb = {q: read_jsonl(self.keep_path(q, lin)) for q in range(1, r + 1) if self.keep_path(q, lin).exists()}
            kb = {q: v for q, v in kb.items() if v}
            if kb:
                latest = max(kb)
                klines, kmeta = mix_tokens(kb, latest, [q for q in sorted(kb) if q != latest], m["replay"],
                                           budget - main_b, f"{self.chain.name}/{lin}/keep", int(m["max_passes"]))
                lines += klines
                meta["keep"] = kmeta
        rows, tmeta = twin_plan(lines, f"{self.chain.name}/{lin}", tuple(m.get("call_share", (1, 4))))
        write_jsonl(out, rows)
        return dict(meta, twin=tmeta, rows=len(rows), steps=math.ceil(len(rows) / int(t["batch"])), starved=False)

    def cell(self, records: Path, role: str, r: int) -> dict:
        recs = A.records(records, self.proto_dir(), 4, set(self.seeds(role, r)))
        return A.cell_stats(recs, self.bench("select"), self.control())

    def summary(self, records: Path, role: str, r: int) -> dict:
        return self.cell(records, role, r)

    def health(self, summ: dict, base: dict) -> list:
        return A.health(summ, base["select"], self.cfg["health"])

    def rank(self, summ: dict) -> float:
        return summ["hidden"] if summ.get("hidden") is not None else -1.0

    def scores(self, records: Path, role: str, r: int) -> dict:
        return self.cell(records, role, r)["per_item"]


def eval_fraction(x) -> float:
    from fractions import Fraction
    return float(Fraction(str(x)))
