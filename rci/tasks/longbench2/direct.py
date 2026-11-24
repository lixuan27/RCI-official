"""LongBench v2 under its official single-model protocol, to anchor this harness to the published numbers."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from rci import paths

TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def build_prompt(row: dict, context: str, tok, max_len: int) -> tuple[str, int]:
    p = TEMPLATE.replace("$DOC$", context.strip()).replace("$Q$", row["question"].strip())
    for L in "ABCD":
        p = p.replace(f"$C_{L}$", row[f"choice_{L}"].strip())
    ids = tok.encode(p, add_special_tokens=False)
    if len(ids) > max_len:
        ids = ids[: max_len // 2] + ids[-max_len // 2:]
        p = tok.decode(ids)
    return p, len(ids)


def extract_answer(text: str) -> str | None:
    text = (text or "").replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", text) or re.search(r"The correct answer is ([A-D])", text)
    return m.group(1) if m else None


def summarize(rows: list[dict]) -> dict:
    def acc(sel):
        sel = list(sel)
        return round(100 * sum(r["correct"] for r in sel) / len(sel), 1) if sel else None
    out = {"n": len(rows), "overall": acc(rows)}
    for k, vals in (("difficulty", ("easy", "hard")), ("length", ("short", "medium", "long"))):
        for v in vals:
            out[v] = acc(r for r in rows if r[k] == v)
    return out


async def run(a) -> None:
    from transformers import AutoTokenizer

    from rci.agents.llm import LLMClient

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(root / a.file) if l.strip()]
    if a.shard:
        i, m = (int(x) for x in a.shard.split("/"))
        rows = rows[i::m]
    rows = rows[: a.limit] if a.limit else rows
    done = {}
    if (out / "preds.jsonl").exists():
        for l in open(out / "preds.jsonl"):
            r = json.loads(l)
            done[(r["_id"], r.get("sample", 0))] = r
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    llm = LLMClient([u.strip() for u in a.base_url.split(",")], a.model, max_concurrency=a.concurrency)
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    f = open(out / "preds.jsonl", "a")

    async def one(row: dict, j: int = 0):
        if (row["_id"], j) in done:
            return
        async with sem:
            ctx = (root / "ctx" / f"{row.get('ctx_id') or row['_id']}.txt").read_text()
            prompt, n_tok = await asyncio.to_thread(build_prompt, row, ctx, tok, a.max_len)
            call = await llm.chat([{"role": "user", "content": prompt}], max_tokens=a.max_new, temperature=1.0,
                                  top_p=0.95, top_k=20, presence_penalty=1.5, enable_thinking=True,
                                  route_key=f"{row['_id']}_{j}")
            pred = extract_answer(call.content)
            rec = {"_id": row["_id"], "sample": j, "idx": row["idx"], "difficulty": row["difficulty"], "length": row["length"],
                   "answer": row["answer"], "pred": pred, "correct": pred == row["answer"], "prompt_tokens": n_tok,
                   "completion_tokens": call.completion_tokens, "finish_reason": call.finish_reason,
                   "error": call.error, "response_tail": (call.content or "")[-300:]}
            async with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(row["_id"], j)] = rec

    await asyncio.gather(*(one(r, j) for r in rows for j in range(a.k)))
    f.close()
    summary = summarize([done[(r["_id"], 0)] for r in rows if (r["_id"], 0) in done])
    summary.update({"max_len": a.max_len, "max_new": a.max_new, "model": a.model})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.longbench2.direct",
                                 description="LongBench v2 single-model reference under the official protocol.")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, help="the served model's directory (its tokenizer truncates)")
    ap.add_argument("--root", default=None, help="default: $RCI_DATA/longbench2")
    ap.add_argument("--file", default="samples.jsonl")
    ap.add_argument("--k", type=int, default=1, help="samples per question (pass rates of training questions)")
    ap.add_argument("--shard", default=None, help="i/m: every m-th question starting at i")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=120000, help="prompt tokens after middle truncation (official "
                    "maxlen for 128k-context models)")
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    a.root = a.root or str(paths.data_dir("longbench2"))
    asyncio.run(run(a))


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
