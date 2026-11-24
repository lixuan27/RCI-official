import json

from rci.chains.hiddenbench import accept as A
from rci.chains.hiddenbench import tasks as TK

FACTS = ["Alpha has a flooded basement covered in black mould", "Alpha lost its catering licence last winter",
         "Beta offers free parking for every guest", "Beta renovated its acoustics with oak panels",
         "Alpha charges triple overtime after midnight", "Alpha elevators broke during inspection",
         "Beta employs certified sommeliers nightly", "Beta guarantees backup generators onsite"]
ROW = {"id": 1, "description": "Choose the venue for the gala.", "possible_answers": ["Alpha", "Beta"],
       "shared_information": ["Alpha looks cheaper on paper.", "Both venues are downtown."],
       "hidden_information": FACTS, "correct_answer": "Beta"}
PIECES = {1: "x", 2: "</think>", 3: "<tool_call>", 4: "<parameter=answer>", 5: "Beta", 6: "</parameter>",
          7: "<|im_end|>", 8: "</tool_call>", 9: "p"}
GEN = [1] * 300 + [2, 3, 4, 5, 6, 8, 7]
PROMPT = [9] * 10


def piece(t):
    return PIECES[t]


def msg(content, answer, reasoning="thinking"):
    return {"role": "assistant", "content": content, "reasoning_content": reasoning,
            "tool_calls": [{"function": {"name": "submit_answer", "arguments": json.dumps({"answer": answer})}}]}


def record(qid="hb_01", forced=(), tentative="Alpha", final="Beta", extra_calls=()):
    agents, finals, tr = [], [], {}
    for j in range(4):
        mine = [FACTS[h] for h in range(8) if h % 4 == j]
        n = 4 if j in extra_calls else 3
        agents.append({"calls": [{"completion_tokens": 307} for _ in range(n)]})
        finals.append({"agent": j, "final": {"answer": final, "confidence": 0.8}, "forced": j in forced,
                       "failing": False})
        turns = [{"messages": [msg(" ".join(mine), tentative)]}, {"messages": [msg("I read the others.", final)]},
                 {"messages": [msg("Final.", final, " ".join(FACTS))]}]
        if n == 4:
            turns.append({"messages": [msg("Again.", final)]})
        tr[str(j)] = turns
    per = [1.0 if final == "Beta" else 0.0] * 4
    return {"example_id": qid, "agents": agents, "finals": finals, "transcripts": tr,
            "score": {"score": per[0], "per_agent": per, "preds": [final] * 4}}


def chains_for(n_gen=3):
    ids, spans = [], []
    for _ in range(n_gen):
        ids += PROMPT
        spans.append([len(ids), len(GEN)])
        ids += GEN
    return {j: [{"agent": j, "ids": list(ids), "spans": [list(s) for s in spans]}] for j in range(4)}


C = {"families": ["D", "R2", "P"], "min_tokens": 300, "max_len": 16384, "per_question": 4, "p_share": None}


def build(recs, seats=(0, 1, 2, 3), kind="collective", c=C, sw=None, n_gen=3):
    bench = {"hb_01": ROW}
    ref = {"hb_01": "beta"}
    return A.build([("p", 0, r) for r in recs], lambda p: chains_for(n_gen), bench, ref, list(seats), kind, piece, 7, c,
                   1, "t", sw)


def test_accepted_turns_of_a_clean_debate():
    rows, meta = build([record()])
    assert meta["families"] == {"D": 4, "R2": 4, "P": 4}
    d = next(r for r in rows if r["family"] == "D" and r["seat"] == 0)
    assert d["ids"] == chains_for()[0][0]["ids"] and d["spans"] == [[644, 307]] and d["ids"][-1] == 7
    p = next(r for r in rows if r["family"] == "P" and r["seat"] == 0)
    value = 10 + GEN.index(5)
    assert all(not (s <= value < s + n) for s, n in p["spans"]) and p["c0"] == 5
    assert d["skel"] == [[10 + 307 + 10 + 307 + 10 + 301, 2], [10 + 307 + 10 + 307 + 10 + 304, 3]]


def test_lineage_seats_and_the_solo_family():
    rows, _ = build([record()], seats=(2, 3))
    assert {r["seat"] for r in rows} == {2, 3}
    rows, meta = build([record()], kind="solo", c=dict(C, families=["S"]))
    assert meta["families"] == {"S": 4} and all(r["spans"] == [[644, 307]] for r in rows)


def test_a_wrong_final_or_a_forced_member_gives_no_row():
    assert build([record(final="Alpha", tentative="Alpha")])[1]["rows"] == 0
    assert build([record(forced=(2,))])[1]["rows"] == 0
    assert build([record(extra_calls=(1,))])[1]["rows"] == 0


def test_guard_a_needs_the_reasoning_closed_before_the_call():
    gen = [1] * 300 + [3, 4, 5, 6, 8, 2, 7]
    chain = {"ids": PROMPT + gen, "spans": [[10, len(gen)]]}
    assert A.make_row(chain, 10, len(gen), piece, "r", "D", {}, 7, min_tokens=300) is None
    chain = {"ids": PROMPT + GEN, "spans": [[10, len(GEN)]]}
    assert A.make_row(chain, 10, len(GEN), piece, "r", "D", {}, 7, min_tokens=300) is not None
    assert A.make_row(chain, 10, len(GEN), piece, "r", "D", {}, 7, min_tokens=400) is None
    short = {"ids": PROMPT + GEN[:-1], "spans": [[10, len(GEN) - 1]]}
    assert A.make_row(short, 10, len(GEN) - 1, piece, "r", "D", {}, 7) is None


def test_switches_change_only_the_skeleton():
    pieces = [piece(t) for t in GEN]
    base = A.call_parts(pieces, 0)
    assert 303 in base["values"] and 301 in base["skel"]
    av = A.call_parts(pieces, 0, answer_value=True)
    assert 303 in av["skel"] and not av["values"]
    out = A.call_parts(pieces, 0, decision_out=True)
    assert 301 not in out["skel"]
    talk = [piece(t) for t in [1, 2, 1, 3, 4, 5, 6, 8, 7]]
    assert 3 in A.call_parts(talk, 0, decision_out=True)["skel"]
    assert A.call_parts(["x", "y"], 0) is None


def test_p_share_and_per_question_caps():
    rows, meta = build([record()], c=dict(C, p_share=0.2))
    loss = lambda xs: sum(n for x in xs for _, n in x["spans"])  # noqa: E731
    assert loss([r for r in rows if r["family"] == "P"]) <= 0.2 * loss(rows)
    rows, _ = build([record(), record()], c=dict(C, per_question=1))
    assert sorted(r["family"] for r in rows) == ["D", "P", "R2"]


def test_band_frontier_and_stagnation():
    ref = {"q1": "a", "q2": "a", "q3": "a"}
    rep = A.band({"q1": ["a"] * 8, "q2": ["a", "b"] * 4, "q3": ["b"] * 8}, ref, {}, 0.875, 100, 2, "s")
    assert rep["archive"] == ["q1"] and rep["train"] == ["q2"] and rep["frontier"] == ["q3"]
    rep2 = A.band({"q3": ["b"] * 8}, ref, rep["frontier_ages"], 0.875, 100, 2, "s")
    assert rep2["frontier_ages"] == {"q3": 1}
    rep3 = A.band({"q3": ["b"] * 8}, ref, rep2["frontier_ages"], 0.875, 100, 2, "s")
    assert rep3["expired"] == ["q3"] and not rep3["frontier"]
    assert A.stagnated(rep3, 20).startswith("frontier stagnation: no question")
    assert "fewer than 20" in A.stagnated(rep, 20) and A.stagnated(rep, 1) is None


def test_cell_stats_and_health_lines():
    good = A.cell_stats([("p", 51, record())], {"hb_01": ROW}, set())
    assert good["rate"] == 1.0 and good["ledger"]["f"] == 0.0 and good["ledger"]["r"] == 1.0
    assert good["budget_per_game"] == 4 * 3 * 307 and good["ledger"]["z"] == 0.0
    h = {"budget_band": [0.7, 1.5], "budget_low_gates": False, "z_max": 0.05, "f_max": 0.10, "r_min": 0.5,
         "control_floor": 0.05}
    assert all(x.ok for x in A.health(good, good, h))
    long = dict(good, budget_per_game=good["budget_per_game"] * 1.6)
    short = dict(good, budget_per_game=good["budget_per_game"] * 0.5)
    assert not all(x.ok for x in A.health(long, good, h)) and all(x.ok for x in A.health(short, good, h))
    assert not all(x.ok for x in A.health(short, good, dict(h, budget_low_gates=True)))
    ctl = dict(good, control=0.8)
    assert not all(x.ok for x in A.health(ctl, dict(good, control=0.9), h))


def word(n):
    return "".join("abcdefghij"[int(d)] for d in f"{n:06d}")


def task(i, control=False, agree=1.0, answer="Beta"):
    return {"id": i, "description": " ".join(word(10 * i + k) for k in range(8)),
            "possible_answers": ["Alpha", "Beta"], "shared_information": [f"shared{i}"],
            "hidden_information": [f"hidden{i}"], "full_answer": answer, "full_agree": agree,
            "writer_answer": "Alpha", "matches_writer": False}


def test_freeze_keeps_agreed_references_and_deals_blocks():
    hid = [task(i) for i in range(260)] + [task(900, agree=0.5), task(901, answer="Gamma")]
    ctl = [task(1000 + i, control=True) for i in range(160)]
    fr = TK.freeze(hid, ctl, 0.75)
    assert fr["report"]["dropped"] == {"agreement below the reference line": 1, "answer is not exactly one option": 1}
    assert len(fr["select"]) == 300 and sum(r["control"] for r in fr["select"]) == 100
    assert len(fr["pool"]) == 100 and all(r["correct_answer"] == "Beta" and "writer_answer" not in r for r in fr["pool"])
    assert sum(r["control"] for r in fr["pool"][:20]) == 9
    assert not {r["id"] for r in fr["select"]} & {r["id"] for r in fr["pool"]}


def test_freeze_writes_the_sets(tmp_path):
    fr = TK.freeze([task(i) for i in range(230)], [task(1000 + i, control=True) for i in range(120)], 0.75)
    TK.write(tmp_path, fr)
    assert len(json.loads((tmp_path / "select" / "benchmark.json").read_text())) == 300
    assert TK.slice_ids(tmp_path, 1, 10) == json.loads((tmp_path / "pool_order.json").read_text())[:10]
