"""Post-hoc grading of MMLongBench-Doc run records with the official two-stage pipeline."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path

from rci import paths
from rci.tasks.mmlongdoc.eval_score import breakdown, eval_score

PROMPT_FILE = "prompt_for_answer_extraction.md"
FAILED = "Failed to extract"
_PROMPT: dict = {}


def extract_prompt() -> str:
    """The upstream answer-extraction prompt, fetched into $RCI_DATA/mmlongdoc by the prepare script."""
    if "text" not in _PROMPT:
        p = paths.data_dir("mmlongdoc", PROMPT_FILE)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run scripts/data/prepare_mmlongdoc.py")
        _PROMPT["text"] = p.read_text()
    return _PROMPT["text"]


def extraction_messages(question: str, response: str) -> list[dict]:
    return [{"role": "user", "content": extract_prompt()},
            {"role": "assistant", "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, response)}]


def parse_extraction(text: str) -> str:
    try:
        return text.split("Answer format:")[0].split("Extracted answer:")[1].strip()
    except (IndexError, AttributeError):
        return FAILED


def response_text(final: dict | None) -> str:
    return str((final or {}).get("answer") or "").strip()


def load_samples(data_root: str) -> dict[str, dict]:
    with open(Path(data_root) / "samples.jsonl") as f:
        return {r["id"]: r for r in map(json.loads, (l for l in f if l.strip()))}


def summarize(records: list[dict], samples: dict[str, dict]) -> dict:
    """Pass@N rows use the group's best agent; the single-agent row averages each agent's own score."""
    group, single = [], []
    for rec in records:
        s = samples[rec["example_id"]]
        sc = rec["score"]
        group.append({**s, "score": sc["score"], "pred": sc.get("pred", FAILED)})
        preds = sc.get("preds") or []
        for i, p in enumerate(sc["per_agent"]):
            if p is not None:
                single.append({**s, "score": p, "pred": (preds[i] if i < len(preds) else None) or FAILED})
    return {"n_examples": len(group), "pass_at_n": breakdown(group),
            "single_agent": breakdown(single) if single else None}


async def grade_root(root: Path, llm, samples: dict[str, dict], concurrency: int = 256) -> dict:
    cache_p = root / "extract_cache.jsonl"
    cache: dict[str, str] = {}
    if cache_p.exists():
        for l in open(cache_p):
            r = json.loads(l)
            cache[r["key"]] = r["extracted"]
    sem = asyncio.Semaphore(concurrency)
    cache_f = open(cache_p, "a")

    async def extract(qid: str, resp: str) -> str:
        key = hashlib.sha1(f"{llm.model}\n{qid}\n{resp}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        async with sem:
            call = await llm.chat(extraction_messages(samples[qid]["question"], resp), max_tokens=256,
                                  temperature=0.0, top_p=1.0, enable_thinking=False, tag="judge")
        out = call.content or ""
        if call.error:
            return out
        cache[key] = out
        cache_f.write(json.dumps({"key": key, "qid": qid, "extracted": out, "judge": llm.model}) + "\n")
        cache_f.flush()
        return out

    graded: list[dict] = []

    async def one(p: Path):
        with gzip.open(p, "rt") as f:
            rec = json.load(f)
        if rec.get("task") not in (None, "mmlongdoc"):
            return
        if rec["score"].get("pending_judge"):
            s = samples[rec["example_id"]]
            per, preds, raws = [], [], []
            for fin in rec["finals"]:
                resp = response_text(fin.get("final"))
                if fin.get("failing"):
                    per.append(None), preds.append(None), raws.append(None)
                elif not resp:
                    per.append(0.0), preds.append(None), raws.append(None)
                else:
                    raw = await extract(rec["example_id"], resp)
                    pred = parse_extraction(raw)
                    per.append(0.0 if pred == FAILED else eval_score(s["answer"], pred, s["answer_format"]))
                    preds.append(None if pred == FAILED else pred), raws.append(raw)
            valid = [i for i, x in enumerate(per) if x is not None]
            best = max(valid, key=lambda i: (per[i], -i)) if valid else None
            rec["score"] = {"score": per[best] if best is not None else 0.0, "per_agent": per, "preds": preds,
                            "pred": (preds[best] or FAILED) if best is not None else FAILED, "extracted": raws,
                            "judge": llm.model}
            tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
            with gzip.open(tmp, "wt") as f:
                json.dump(rec, f, ensure_ascii=False, default=str)
            os.replace(tmp, p)
        if rec["score"].get("score") is not None:
            graded.append({"example_id": rec["example_id"], "score": rec["score"]})

    files = sorted(root.rglob("*.json.gz"))
    await asyncio.gather(*(one(p) for p in files))
    cache_f.close()
    summary = summarize(graded, samples)
    (root / "mmlongdoc_summary.json").write_text(json.dumps(summary, indent=1))
    ov = summary["pass_at_n"]["overall"]
    print(f"[grade] {root}: {len(files)} records, {len(graded)} graded; Pass@N acc {ov['acc']:.4f} F1 {ov['f1']:.4f}")
    return summary


def main(argv=None) -> None:
    from rci.agents.llm import LLMClient

    ap = argparse.ArgumentParser(prog="python -m rci.tasks.mmlongdoc.grade",
                                 description="Grade MMLongBench-Doc records with the two-stage official pipeline.")
    ap.add_argument("roots", nargs="+", help="record roots written by rci.run")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="default: $RCI_DATA/mmlongdoc")
    a = ap.parse_args(argv)
    a.data = a.data or str(paths.data_dir("mmlongdoc"))
    samples = load_samples(a.data)
    llm = LLMClient(a.base_url.split(","), a.model)

    async def go():
        for r in a.roots:
            await grade_root(Path(r), llm, samples)
    asyncio.run(go())


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
