"""LLM-judge grading of BrowseComp-Plus records with the public grading prompt; see THIRD_PARTY_NOTICES.md."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

from rci import paths
from rci.agents.llm import LLMClient

REGRADE = False

GRADER = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""

_CORRECT = re.compile(r"\**correct\**\s*:\s*\**\s*(yes|no)", re.I)


def response_text(final: dict | None) -> str:
    if not final:
        return ""
    ans, exp = str(final.get("answer") or "").strip(), str(final.get("explanation") or "").strip()
    return f"{exp}\n\nExact Answer: {ans}" if exp else f"Exact Answer: {ans}"


async def grade_root(root: Path, llm: LLMClient, queries: dict, concurrency: int = 256) -> None:
    cache_p = root / "judge_cache.jsonl"
    cache = {}
    if cache_p.exists():
        for l in open(cache_p):
            r = json.loads(l)
            cache[r["key"]] = r["correct"]
    sem = asyncio.Semaphore(concurrency)
    cache_f = open(cache_p, "a")

    async def judge(qid: str, resp: str) -> bool:
        key = hashlib.sha1(f"{qid}\n{resp}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        q = queries[qid]
        async with sem:
            call = await llm.chat([{"role": "user", "content": GRADER.format(
                question=q["question"], response=resp, correct_answer=q["answer"])}],
                max_tokens=2048, temperature=0.0, enable_thinking=False, tag="judge")
        m = _CORRECT.search(call.content or "")
        ok = bool(m and m.group(1).lower() == "yes")
        if m:
            cache[key] = ok
            cache_f.write(json.dumps({"key": key, "qid": qid, "correct": ok, "judge": llm.model}) + "\n")
            cache_f.flush()
        return ok

    async def one(p: Path):
        with gzip.open(p, "rt") as f:
            rec = json.load(f)
        if not rec["score"].get("pending_judge") and not REGRADE:
            return
        qid = rec["example_id"]
        per = []
        for fin in rec["finals"]:
            if fin.get("failing"):
                per.append(None)
            elif not fin.get("final") or not str(fin["final"].get("answer") or "").strip():
                per.append(0.0)
            else:
                per.append(1.0 if await judge(qid, response_text(fin["final"])) else 0.0)
        valid = [x for x in per if x is not None]
        rec["score"] = {"score": max(valid) if valid else 0.0, "per_agent": per, "judge": llm.model}
        tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
        with gzip.open(tmp, "wt") as f:
            json.dump(rec, f, ensure_ascii=False, default=str)
        os.replace(tmp, p)

    files = sorted(root.rglob("*.json.gz"))
    await asyncio.gather(*(one(p) for p in files))
    cache_f.close()
    print(f"[grade] {root}: {len(files)} records checked")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.browsecomp.grade",
                                 description="Judge BrowseComp-Plus records against the gold answers.")
    ap.add_argument("roots", nargs="+", help="record roots written by rci.run")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--queries", default=None, help="default: $RCI_DATA/browsecomp/queries.jsonl")
    ap.add_argument("--regrade", action="store_true", help="re-judge records that already have a score")
    a = ap.parse_args(argv)
    a.queries = a.queries or str(paths.data_dir("browsecomp", "queries.jsonl"))
    global REGRADE
    REGRADE = a.regrade
    queries = {}
    for l in open(a.queries):
        q = json.loads(l)
        queries[f"q{q['query_id']}"] = q
    llm = LLMClient(a.base_url.split(","), a.model)

    async def go():
        for r in a.roots:
            await grade_root(Path(r), llm, queries)
    asyncio.run(go())


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
