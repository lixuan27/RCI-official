import gzip
import json

from rci.tasks.hiddenbench.facts import keys_of, members, written_facts
from rci.registry import make_task

ROW = {"id": 1, "name": "t", "description": "Pick a route for the convoy.", "shared_information": ["West looks fine."],
       "hidden_information": ["The western bridge collapsed overnight.", "Eastern roads were cleared by snowploughs.",
                              "Fuel depot sits beside the eastern highway.", "Bandits ambushed travellers near Westgate."],
       "possible_answers": ["West", "East"], "correct_answer": "East"}


def msg(i, author, step, text):
    return {"id": i, "author": author, "step": step, "sub_task": "shared facts", "artifact": text, "rationale": ""}


def record():
    msgs = [msg(0, 0, 0, "The western bridge collapsed overnight."), msg(1, 1, 0, "Eastern roads were cleared by snowploughs."),
            msg(2, 2, 1, "Fuel depot sits beside the eastern highway."), msg(3, 3, 2, "I think West.")]
    events = [{"op": "search", "step": 1, "reader": 3, "results": [{"id": 0}, {"id": 1}]},
              {"op": "visit", "step": 2, "reader": 3, "ids": [2]},
              {"op": "search", "step": 3, "reader": 3, "results": [{"id": 3}]},
              {"op": "search", "step": 2, "reader": 0, "results": [{"id": 1}, {"id": 2}]},
              {"op": "search", "step": 3, "reader": 0, "results": [{"id": 3}]}]
    agents = [{"id": 0, "turns": [{"step": 0}, {"step": 2}, {"step": 3}]}, {"id": 1, "turns": [{"step": 0}]},
              {"id": 2, "turns": [{"step": 1}]}, {"id": 3, "turns": [{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}]}]
    return {"example_id": "hb_01", "agents": agents, "method_state": {"messages": msgs, "events": events},
            "score": {"per_agent": [1.0, 0.0, 0.0, 0.0], "preds": ["east", "west", "west", "west"]}}


def test_keys_drop_words_every_member_already_sees():
    k = keys_of(ROW)
    assert "west" not in k[0] and {"bridge", "collapsed", "overnight"} <= k[0]


def test_members_count_reads_before_the_submitting_turn_only():
    m = {x["agent"]: x for x in members(record(), ROW)}
    assert m[3]["shown"] == [0, 1, 2] and m[3]["complete"] and m[3]["completed_at"] == 2 and m[3]["written"] == [3]
    assert m[0]["shown"] == [1, 2] and m[0]["has"] == {0, 1, 2} and not m[0]["complete"]
    assert m[1]["shown"] == [] and m[1]["has"] == {1}
    assert written_facts(record(), ROW) == [True, True, True, False]


def test_fact_sentences_keep_the_stated_facts_and_drop_the_reading():
    from rci.tasks.hiddenbench.facts import fact_sentences
    text = ("Facts held: The western bridge collapsed overnight. So West is clearly unsafe and we must avoid it. "
            "West looks fine. | I recommend East.")
    out = fact_sentences(text, ROW)
    assert out == "Facts held: The western bridge collapsed overnight. West looks fine."
