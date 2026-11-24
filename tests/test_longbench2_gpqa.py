import asyncio
import json
import random
from types import SimpleNamespace

from rci.agents.tooling import AgentCtx, call_tool
from rci.registry import make_task
from rci.tasks.gpqa.task import norm_letter
from rci.tasks.longbench2.task import SectionDoc, split_sections


def test_sections_cover_the_text_within_the_cap():
    rng = random.Random(0)
    paras = [" ".join(f"w{rng.randrange(500)}" for _ in range(rng.randrange(5, 900))) for _ in range(60)]
    paras.append("x" * 15000)
    text = "\n\n".join(paras)
    secs = split_sections(text, cap=6000)
    assert all(len(s) <= 6000 for s in secs)
    assert "".join(s.replace("\n", "").replace(" ", "") for s in secs) == text.replace("\n", "").replace(" ", "")


def _lb2(tmp_path):
    root = tmp_path / "lb2"
    (root / "ctx").mkdir(parents=True)
    ctx = "\n\n".join(f"Chapter {i}. The keeper of lighthouse {i} was named Ada{i}." for i in range(200))
    (root / "ctx" / "abc.txt").write_text(ctx)
    rows = [{"idx": 0, "_id": "abc", "domain": "Single-Document QA", "sub_domain": "Novel", "difficulty": "hard",
             "length": "short", "question": "Who kept lighthouse 137?", "choice_A": "Ada13", "choice_B": "Ada137",
             "choice_C": "Ada7", "choice_D": "Ada37", "answer": "B", "n_chars": len(ctx)},
            {"idx": 1, "_id": "abc", "domain": "x", "sub_domain": "y", "difficulty": "easy", "length": "long",
             "question": "q", "choice_A": "a", "choice_B": "b", "choice_C": "c", "choice_D": "d", "answer": "A",
             "n_chars": len(ctx)}]
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return make_task("longbench2", root=str(root))


def test_longbench2_tools_and_scoring(tmp_path):
    t = _lb2(tmp_path)
    ex = t.examples()[0]
    assert ex["id"] == "lb2_abc" and len(t.examples()) == 2
    assert [e["idx"] for e in make_task("longbench2", root=t.root, length="long").examples()] == [1]
    prompt = t.task_prompt(ex, 0, 1, random.Random(0))
    assert "B) Ada137" in prompt and "Section1 to Section" in prompt
    tools = {s.name: s for s in t.tools(ex, None)}
    ctx = AgentCtx(agent_id=0, run=SimpleNamespace(cfg=SimpleNamespace(n_agents=1), step=0))
    hit = call_tool(tools["search_document"], ctx, {"query": "lighthouse 137"})
    import asyncio
    hit = asyncio.run(hit) if asyncio.iscoroutine(hit) else hit
    first = int(hit.text.split("\n")[1].split()[0][len("Section"):])
    body = call_tool(tools["read_section"], ctx, {"section": f"Section{first}"})
    body = asyncio.run(body) if asyncio.iscoroutine(body) else body
    assert "Ada137" in body.text
    bad = call_tool(tools["read_section"], ctx, {"section": 10 ** 6})
    bad = asyncio.run(bad) if asyncio.iscoroutine(bad) else bad
    assert not bad.ok and "does not exist" in bad.text
    assert t.resource_exists(ex, None, "Section1") and not t.resource_exists(ex, None, "Page1")
    sc = t.score_group(ex, [{"final": {"answer": "B"}}, {"final": {"answer": "(a)"}}, {"failing": True}, {"final": None}])
    assert sc["score"] == 1.0 and sc["per_agent"] == [1.0, 0.0, None, 0.0] and sc["preds"] == ["B", "A", None, None]


def test_direct_protocol_prompt_truncation_and_answer_regex():
    from rci.tasks.longbench2.direct import build_prompt, extract_answer, summarize

    class Tok:
        def encode(self, s, add_special_tokens=False):
            return list(s)

        def decode(self, ids):
            return "".join(ids)

    row = {"question": "Q?", "choice_A": "a", "choice_B": "b", "choice_C": "c", "choice_D": "d"}
    p, n = build_prompt(row, "x" * 5000, Tok(), max_len=1000)
    assert n == 1000 and p.startswith("Please read the following text") and p.endswith('(insert answer here)".')
    assert extract_answer("... The correct answer is (C).") == "C" and extract_answer("The correct answer is **B**") == "B"
    assert extract_answer("I think A") is None
    rows = [{"correct": True, "difficulty": "easy", "length": "short"}, {"correct": False, "difficulty": "hard",
                                                                        "length": "long"}]
    s = summarize(rows)
    assert s["overall"] == 50.0 and s["easy"] == 100.0 and s["hard"] == 0.0 and s["medium"] is None


def test_gpqa_direct_extraction_and_summary():
    from rci.tasks.gpqa.direct import TEMPLATE, extract, summarize

    assert "Answer: $LETTER" in TEMPLATE and "{question}" in TEMPLATE
    assert extract("... so it is C.\nAnswer: C") == "C" and extract("answer: (b)") == "B" and extract("no idea") is None
    assert extract("Answer: A ... wait, Answer: D") == "D"
    rows = [{"idx": 0, "sample": j, "answer": "B", "pred": p, "correct": p == "B"} for j, p in enumerate("BBC")]
    rows += [{"idx": 1, "sample": j, "answer": "A", "pred": p, "correct": p == "A"} for j, p in enumerate(["C", None, "D"])]
    s = summarize(rows, 3)
    assert s["questions"] == 2 and s["pass_at_k"] == 50.0 and s["majority_vote"] == 50.0
    assert abs(s["mean_acc"] - 100 * (2 / 3 + 0) / 2) < 0.1


def test_mcq_training_rows_with_ten_options(tmp_path):
    from rci.tasks.gpqa.task import SYSTEM
    rows = [{"idx": 0, "uid": "ext_00007", "source": "other_stem", "question": "2+2?",
             "choices": [str(i) for i in range(10)], "answer": "E", "domain": "math"},
            {"idx": 1, "uid": "gpqa_ext_9", "source": "gpqa_extended", "question": "q", "choices": ["a", "b", "c", "d"],
             "answer": "B", "domain": "Physics"}]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    t = make_task("gpqa", root=str(tmp_path), file="train.jsonl", confidence=True)
    a, b = t.examples()
    assert a["id"] == "ext_00007" and b["id"] == "gpqa_ext_9"
    assert "A, B, C, D, E, F, G, H, I or J" in t.system_prompt(a) and t.system_prompt(b) == SYSTEM
    assert "J) 9" in t.task_prompt(a, 0, 1, None)
    props, _ = t.answer_properties()
    assert props["answer"]["enum"] == list("ABCDEFGHIJ")
    assert t.score_agent(a, {"answer": "E"}) == 1.0 and t.score_agent(a, {"answer": "(J)"}) == 0.0
    assert t.score_agent(b, {"answer": "E"}) == 0.0
    d = tmp_path / "diamond"
    d.mkdir()
    (d / "samples.jsonl").write_text(json.dumps({"idx": 0, "question": "q", "choices": ["a", "b", "c", "d"],
                                                 "answer": "C"}) + "\n")
    g = make_task("gpqa", root=str(d))
    assert g.examples()[0]["id"] == "gpqa_000" and g.answer_properties()[0]["answer"]["enum"] == list("ABCD")
    assert g.system_prompt(g.examples()[0]) == SYSTEM


def test_longbench2_whole_document_when_it_fits(tmp_path, monkeypatch):
    import rci.tasks.longbench2.task as lb
    t = _lb2(tmp_path)
    ex = t.examples()[0]
    assert t.deliberates(ex) and t.full(ex) is not None
    assert "<document>" in t.task_prompt(ex, 0, 2, None) and t.system_prompt(ex).startswith(lb.SYSTEM_FULL[:40])
    off = make_task("longbench2", root=t.root, full_tokens=False)
    assert not off.deliberates(ex) and "<document>" not in off.task_prompt(ex, 0, 2, None)
    assert off.system_prompt(ex).startswith(lb.SYSTEM[:40])
    small = make_task("longbench2", root=t.root, full_tokens=5)
    assert not small.deliberates(ex) and t.deliberates(ex)
    monkeypatch.setattr(lb, "FULL_TOKENS", 5)
    lb._FULL.clear()
    assert not t.deliberates(ex) and "<document>" not in t.task_prompt(ex, 0, 2, None)
    lb._FULL.clear()
