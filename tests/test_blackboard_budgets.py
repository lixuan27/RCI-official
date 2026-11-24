import json
from types import SimpleNamespace

from rci.tasks.hiddenbench.facts import answer_view, members
from rci.protocols.blackboard import BlackboardProtocol
from rci.registry import make_task
from rci.blackboard.store import BlackboardStore
from rci.tasks.hiddenbench.task import FORWARD_TITLE

ROW = {"id": 1, "description": "Pick a route for the convoy.", "shared_information": ["The western road looks fine for buses today."],
       "hidden_information": ["The western bridge collapsed overnight in the storm.", "Eastern roads were cleared by snowploughs.",
                              "Southern tunnel reopened to all convoy traffic.", "Northern pass is closed by an avalanche."],
       "possible_answers": ["West", "East"], "correct_answer": "East"}


def task(tmp_path, **kw):
    (tmp_path / "benchmark.json").write_text(json.dumps([ROW]))
    return make_task("hiddenbench", root=str(tmp_path), hint=True, confidence=True, conclude="complete", **kw)


def test_choice_shows_headers_free_and_reads_only_by_visit():
    ds = BlackboardStore(token_limit=512, read_budget={1: 1}, seed=0, choice=True, describe=lambda m: f"kind:{m.sub_task}")
    ds.publish(2, FORWARD_TITLE, "Eastern roads were cleared by snowploughs.", "", [], 0)
    ds.publish(0, "Recommendation", "East: the bridge is out.", "", [], 0)
    hits = ds.search("east roads bridge", reader=1, step=1)
    assert len(hits) == 2 and ds.budget_left(1) == 1
    assert ds.events[-1]["results"] == [] and sorted(ds.events[-1]["headers"]) == [0, 1]
    text = ds.render_many(hits, reader=1)
    assert "kind:Recommendation" in text and "the bridge is out" not in text and "not read" in text
    found, _ = ds.visit(["Message1"], reader=1, step=2)
    assert found and ds.budget_left(1) == 0
    assert "the bridge is out" in ds.render(ds.messages[1], reader=1)
    again = ds.search("east roads bridge", reader=1, step=3)
    assert [r["id"] for r in ds.events[-1]["results"]] == [1] and ds.events[-1]["headers"] == [0] and len(again) == 2
    assert "Eastern roads" in ds.render(ds.messages[0], reader=0)


def test_choice_tells_every_seat_the_structure():
    m = BlackboardProtocol(read_budget=[0, 1], choice=True)
    s0, s1 = m.budget_prompt(0, 4), m.budget_prompt(2, 4)
    assert "you can read every message" in s0 and "3 of the other participants can read at most 1" in s0
    assert "you can read at most 1 message" in s1 and "1 of the other participants can read every message" in s1
    assert "choose what to read" in s1 and "choose what to read" not in s0
    sym = BlackboardProtocol(read_budget=[2], choice=True).budget_prompt(1, 4)
    assert "3 of the other participants can read at most 2" in sym


def test_describe_message_says_kind_and_line_count_without_content(tmp_path):
    t = task(tmp_path)
    assert t.describe_message(FORWARD_TITLE, "fact one here\nfact two here") == "facts copied word for word (2 lines)"
    assert "read every participant's facts" in t.describe_message("Analysis", "East because ...")


def test_answer_view_keeps_the_option_and_drops_the_facts():
    text = "The western bridge collapsed overnight in the storm. I recommend East."
    assert answer_view(text, ROW) == "It recommends: East."
    assert answer_view("The western bridge collapsed overnight in the storm.", ROW) is None


def run_with(t, m, n=4):
    run = SimpleNamespace(task=t, example=t.examples()[0], cfg=SimpleNamespace(n_agents=n, n_failing=0, seed=0, join_route="shared"),
                          state={}, step=0)
    try:
        m.build(run)
    except Exception:
        pass
    return run


def test_full_broadcast_delivers_certified_messages_once_and_counts_as_read(tmp_path):
    t = task(tmp_path)
    m = BlackboardProtocol(read_budget=[0, 1], broadcast="full")
    run = run_with(t, m)
    ds = run.state["datastore"]
    ds.extra_validator = None
    ds.publish(1, FORWARD_TITLE, ROW["hidden_information"][1], "", [], 0)
    ds.publish(0, "Recommendation", " ".join(ROW["hidden_information"]) + " I recommend East.", "", [], 1)
    agent = SimpleNamespace(id=2)
    text = m.injection(run, agent, 2)
    assert "Delivered by the store" in text and "I recommend East" in text and "Eastern roads" in text
    assert m.injection(run, agent, 3) is None
    assert ds.events[-1] == {"op": "push", "step": 2, "reader": 2, "ids": [1], "view": "full"}
    assert ds.known(2, 1) and ds.budget_left(2) == 1
    run.step = 2
    assert t.holds_every_fact(run.example, run, 2)
    assert m.injection(run, SimpleNamespace(id=0), 2) is None
    assert "delivered to every participant" in m.budget_prompt(0) and "delivered to every participant" in m.budget_prompt(3)


def test_answer_broadcast_shows_only_the_conclusion_and_does_not_count_as_read(tmp_path):
    t = task(tmp_path)
    m = BlackboardProtocol(read_budget=[0, 1], broadcast="answer")
    run = run_with(t, m)
    ds = run.state["datastore"]
    ds.extra_validator = None
    ds.publish(0, "Recommendation", " ".join(ROW["hidden_information"]) + " I recommend East.", "", [], 1)
    text = m.injection(run, SimpleNamespace(id=3), 2)
    assert "It recommends: East." in text and "snowploughs" not in text
    assert ds.events[-1]["view"] == "answer" and not ds.known(3, 0)
    run.step = 3
    assert not t.holds_every_fact(run.example, run, 3)


def test_members_count_full_pushes_up_to_the_last_turn_and_ignore_answer_pushes():
    rec = {"agents": [{"id": i, "turns": [{"step": 0}, {"step": 2}]} for i in range(4)],
           "method_state": {"messages": [{"id": 0, "author": 0, "sub_task": "Recommendation", "artifact": " ".join(ROW["hidden_information"]),
                                          "rationale": "", "step": 1}],
                            "events": [{"op": "push", "step": 2, "reader": 1, "ids": [0], "view": "full"},
                                       {"op": "push", "step": 2, "reader": 2, "ids": [0], "view": "answer"}]}}
    mem = {x["agent"]: x for x in members(rec, ROW)}
    assert mem[1]["complete"] and mem[1]["shown"] == [0]
    assert not mem[2]["complete"] and mem[2]["shown"] == []
