import json
import random

from rci.tasks.splitbench.writer import novelty, parse, prompt, to_benchmark


def reply(draw, **over):
    opts = ["North Pier", "East Dock", "West Quay"]
    t = {"name": "Port", "description": "Pick a port for the convoy among North Pier, East Dock and West Quay.",
         "possible_answers": opts, "correct_answer": opts["ABC".index(draw["right"])],
         "shared_information": [f"Shared fact {i} about the docks." for i in range(draw["n_shared"])],
         "hidden_information": [f"Private fact {i} about the cranes." for i in range(draw["n_hidden"])],
         "explanation": "Because of the cranes."}
    t.update(over)
    return "<think>planning</think>\n" + json.dumps(t)


def test_draws_are_deterministic_and_put_right_and_decoy_apart():
    a, da = prompt(random.Random("0:7"))
    b, db = prompt(random.Random("0:7"))
    assert a == b and da == db and da["right"] != da["decoy"] and f"option {da['right']} exactly" in a


def test_parse_keeps_a_reply_with_the_shape_and_records_the_decoy():
    _, d = prompt(random.Random("0:1"))
    t = parse(reply(d), d)
    assert t is not None and t["correct_answer"] == t["possible_answers"]["ABC".index(d["right"])]
    assert t["decoy"] == t["possible_answers"]["ABC".index(d["decoy"])] and len(t["hidden_information"]) == d["n_hidden"]


def test_parse_refuses_wrong_key_wrong_counts_and_verdicts_in_the_facts():
    _, d = prompt(random.Random("0:1"))
    wrong = [o for o in ["North Pier", "East Dock", "West Quay"] if o != parse(reply(d), d)["correct_answer"]][0]
    assert parse(reply(d, correct_answer=wrong), d) is None
    assert parse(reply(d, hidden_information=["one fact"]), d) is None
    assert parse(reply(d, shared_information=["So the correct answer is East Dock."] * 3), d) is None
    assert parse(reply(d, possible_answers=["A", "A", "B"]), d) is None
    assert parse("no json here", d) is None


def test_novelty_and_benchmark_form():
    _, d = prompt(random.Random("0:1"))
    t = parse(reply(d), d)
    assert novelty(t, [t]) == 1.0 and novelty(t, []) == 0.0
    rows = to_benchmark([t, t], start=1000)
    assert [r["id"] for r in rows] == [1000, 1001] and rows[0]["writer_answer"] == t["correct_answer"]


def test_n_hidden_fixes_the_fact_count_and_keeps_every_other_draw():
    for key in ("0:1", "8:5", "8:77"):
        _, free = prompt(random.Random(key))
        text, fixed = prompt(random.Random(key), n_hidden=8)
        assert fixed["n_hidden"] == 8 and {k: v for k, v in fixed.items() if k != "n_hidden"} == \
            {k: v for k, v in free.items() if k != "n_hidden"}
        assert '"hidden_information": 8 facts' in text
    assert prompt(random.Random("0:1"), n_hidden=None) == prompt(random.Random("0:1"))
