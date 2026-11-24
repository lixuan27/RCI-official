"""GPQA-Diamond with the plain single-model protocol: one prompt per sample, no tools, no agent loop."""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
from pathlib import Path

from rci import paths

TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following \
format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{options}"""
LETTERS = "ABCDEFGHIJ"


def prompt(row: dict) -> str:
    """The simple-evals GPQA prompt; with four options it is the original string, more options extend it (A-J)."""
    letters = LETTERS[: len(row["choices"])]
    return TEMPLATE.format(letters=letters, question=row["question"],
                           options="\n".join(f"{L}) {c}" for L, c in zip(letters, row["choices"])))


def extract(text: str, letters: str = "ABCD") -> str | None:
    m = re.findall(rf"(?i)Answer\s*:\s*\$?\(?([{letters}])\)?", text or "")
    return m[-1].upper() if m else None


def summarize(rows: list[dict], k: int) -> dict:
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["idx"]].append(r)
    acc, vote, passk = [], [], []
    for rs in by_q.values():
        acc.append(sum(r["correct"] for r in rs) / len(rs))
        preds = [r["pred"] for r in rs if r["pred"]]
        if preds:
            c = collections.Counter(preds)
            top = max(c.values())
            tied = [p for p, n in c.items() if n == top]
            vote.append(sum(p == rs[0]["answer"] for p in tied) / len(tied))
        else:
            vote.append(0.0)
        passk.append(float(any(r["correct"] for r in rs)))
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    return {"questions": len(by_q), "k": k, "mean_acc": pct(acc), "majority_vote": pct(vote), "pass_at_k": pct(passk)}


async def run(a) -> None:
    from rci.agents.llm import LLMClient

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(Path(a.root) / a.file) if l.strip()]
    rows = rows[: a.limit] if a.limit else rows
    if a.shard:
        i, m = (int(x) for x in a.shard.split("/"))
        rows = rows[i::m]
    done = set()
    if (out / "preds.jsonl").exists():
        for l in open(out / "preds.jsonl"):
            r = json.loads(l)
            done.add((r["idx"], r["sample"]))
    llm = LLMClient([u.strip() for u in a.base_url.split(",")], a.model, max_concurrency=a.concurrency)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    f = open(out / "preds.jsonl", "a")

    async def one(row: dict, j: int):
        if (row["idx"], j) in done:
            return
        async with sem:
            call = await llm.chat([{"role": "user", "content": prompt(row)}], max_tokens=a.max_new, temperature=1.0,
                                  top_p=0.95, top_k=20, presence_penalty=a.presence_penalty, enable_thinking=True,
                                  route_key=f"{row['idx']}_{j}")
            pred = extract(call.content, LETTERS[: len(row["choices"])])
            rec = {"idx": row["idx"], "uid": row.get("uid"), "sample": j, "answer": row["answer"], "pred": pred,
                   "correct": pred == row["answer"],
                   "completion_tokens": call.completion_tokens, "finish_reason": call.finish_reason, "error": call.error}
            async with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()

    await asyncio.gather(*(one(r, j) for r in rows for j in range(a.k)))
    f.close()
    allrows = [json.loads(l) for l in open(out / "preds.jsonl")]
    s = summarize([r for r in allrows if r["sample"] < a.k], a.k)
    toks = sorted(r["completion_tokens"] or 0 for r in allrows)
    s.update({"median_completion_tokens": toks[len(toks) // 2] if toks else None, "presence_penalty": a.presence_penalty,
              "length_cut": sum(r["finish_reason"] == "length" for r in allrows), "model": a.model})
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.gpqa.direct",
                                 description="GPQA-Diamond single-model reference: k samples per question.")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default=None, help="default: $RCI_DATA/gpqa")
    ap.add_argument("--file", default="samples.jsonl")
    ap.add_argument("--shard", default=None, help="i/m: every m-th question starting at i")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    a.root = a.root or str(paths.data_dir("gpqa"))
    asyncio.run(run(a))


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
