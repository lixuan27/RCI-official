"""GPQA-Diamond: graduate-level multiple-choice questions, answered with one letter and scored by exact match."""
from __future__ import annotations

import json
import re
from pathlib import Path

from rci import paths
from rci.tasks.base import Task

LETTERS = "ABCDEFGHIJ"
SYSTEM = """You answer one graduate-level multiple-choice question in biology, chemistry or physics. Exactly one of
the four options is correct. Reason carefully; check units, signs, and whether each option is consistent with the
question. When you are confident, call `submit_answer` with the letter of the correct option (A, B, C or D)."""
SYSTEM_GENERAL = """You answer one multiple-choice question in science, mathematics or engineering. Exactly one of the
options is correct. Reason carefully; check units, signs, and whether each option is consistent with the question.
When you are confident, call `submit_answer` with the letter of the correct option ({letters})."""


def norm_letter(x, letters: str = "ABCD") -> str | None:
    m = re.search(rf"\b([{letters}])\b", str(x or "").strip().upper().strip("()."))
    return m.group(1) if m else None


def _letters(example: dict) -> str:
    return LETTERS[: len(example["choices"])]


class GPQADiamond(Task):
    name = "gpqa"
    msg_token_limit = 512
    max_actions = 16
    max_steps = 24
    summarize = False
    deliberate_first = True
    deliberate_gen_limit = 32768

    def __init__(self, root: str = "", file: str = "samples.jsonl"):
        self.root, self.file = root or str(paths.data_dir("gpqa")), file
        self._rows: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._rows is None:
            p = Path(self.root) / self.file
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run scripts/data/prepare_gpqa.py")
            with open(p) as f:
                self._rows = sorted((json.loads(l) for l in f if l.strip()), key=lambda r: r["idx"])
        return self._rows

    def examples(self, limit: int | None = None) -> list[dict]:
        rows = self._load()[:limit] if limit else self._load()
        return [{**r, "id": r.get("uid") or f"gpqa_{r['idx']:03d}"} for r in rows]

    def system_prompt(self, example: dict) -> str:
        if example.get("source", "gpqa_diamond").startswith("gpqa"):
            return SYSTEM
        letters = _letters(example)
        return SYSTEM_GENERAL.format(letters=", ".join(letters[:-1]) + " or " + letters[-1])

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(example["choices"]))
        return f"Question: {example['question']}\n\n{opts}"

    def answer_properties(self):
        n = max(len(r["choices"]) for r in self._load())
        return {"answer": {"type": "string", "enum": list(LETTERS[:n]),
                           "description": "The letter of the correct option."}}, ["answer"]

    def score_agent(self, example: dict, final: dict | None) -> float:
        return float(norm_letter((final or {}).get("answer"), _letters(example)) == example["answer"])

    def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        out = super().score_group(example, finals, run)
        out["preds"] = [None if f.get("failing") else norm_letter((f.get("final") or {}).get("answer"), _letters(example))
                        for f in finals]
        return out
