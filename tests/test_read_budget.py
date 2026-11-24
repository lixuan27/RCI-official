import json

from rci.protocols.blackboard import BlackboardProtocol
from rci.registry import make_task
from rci.blackboard.store import BlackboardStore
from types import SimpleNamespace


def store(budget):
    ds = BlackboardStore(token_limit=512, read_budget=budget, seed=0)
    ds.publish(1, "Facts of seat one", "The western bridge collapsed.", "", [], 0)
    ds.publish(2, "Facts of seat two", "Eastern roads were cleared.", "", [], 0)
    ds.publish(0, "My own facts", "Buses run today.", "", [], 0)
    ds.publish(3, "A consolidation", "Bridge out and roads clear: go East.", "Message0 gave the bridge", ["Message0"], 0)
    return ds


def test_a_budget_withholds_what_goes_over_and_logs_only_what_was_shown():
    ds = store({0: 1})
    first = ds.search("bridge roads facts", reader=0, step=0)
    foreign = [m.id for m in first if m.author != 0]
    assert len(foreign) == 1 and ds.budget_left(0) == 0
    assert ds.events[-1]["withheld"] >= 1 and all(r["id"] in [m.id for m in first] for r in ds.events[-1]["results"])
    again = ds.search("bridge roads facts", reader=0, step=1)
    assert {m.id for m in again if m.author != 0} == set(foreign)
    assert any(m.author == 0 for m in ds.search("own facts buses", reader=0, step=1))
    other = next(i for i in (0, 1, 3) if i not in foreign)
    found, errors = ds.visit([f"Message{other}"], reader=0, step=2)
    assert not found and "withheld" in errors[0]
    assert ds.budget_left(1) is None and len(ds.search("bridge roads facts", reader=1, step=0)) == 4


def test_builds_on_lines_hide_unread_messages():
    ds = store({0: 1})
    ds.visit(["Message3"], reader=0, step=0)
    assert "builds on" not in ds.render(ds.messages[3], reader=0)
    assert "builds on" in ds.render(ds.messages[3], reader=1)


def test_the_method_gives_each_seat_its_budget_and_says_so(tmp_path):
    m = BlackboardProtocol(read_budget=[0, 1])
    assert m.budget_of(0) == 0 and m.budget_of(3) == 1
    assert m.budget_prompt(0) == "" and "at most 1 message" in m.budget_prompt(2)
    (tmp_path / "benchmark.json").write_text(json.dumps([{"id": 1, "description": "d", "shared_information": ["s"],
                                                          "hidden_information": ["h0", "h1"], "possible_answers": ["A", "B"],
                                                          "correct_answer": "A"}]))
    t = make_task("hiddenbench", root=str(tmp_path))
    run = SimpleNamespace(task=t, example=t.examples()[0], cfg=SimpleNamespace(n_agents=4, n_failing=0, seed=0, join_route="shared"),
                          state={})
    m.build = BlackboardProtocol.build.__get__(m)
    try:
        m.build(run)
    except Exception:
        pass
    ds = run.state["datastore"]
    assert ds.read_budget == {1: 1, 2: 1, 3: 1}


def test_the_reader_gate_with_one_unbounded_seat_lets_only_that_seat_conclude(tmp_path):
    import asyncio
    from collections import defaultdict
    from rci.agents.runner import GroupRun, RunConfig
    from rci.tasks.hiddenbench.facts import members
    from fake_llm import ScriptedLLM
    row = {"id": 1, "description": "Pick a route for the convoy.", "shared_information": ["The western road looks fine for buses today."],
           "hidden_information": ["The western bridge collapsed overnight in the storm.", "Eastern roads were cleared by snowploughs.",
                                  "Southern tunnel reopened to all convoy traffic."],
           "possible_answers": ["West", "East"], "correct_answer": "East"}
    (tmp_path / "benchmark.json").write_text(json.dumps([row]))
    t = make_task("hiddenbench", root=str(tmp_path), hint=True, confidence=True, conclude="complete")
    calls = defaultdict(int)

    def policy(messages, tools, route_key, tag):
        i = int(route_key.rsplit("a", 1)[1])
        calls[i] += 1
        if calls[i] == 1:
            return [("forward_facts", {"facts": [row["hidden_information"][i]]})]
        if calls[i] == 2:
            return [("search_datastore", {"query": "bridge roads tunnel forwarded facts"})]
        if calls[i] == 3:
            return [("publish_message", {"sub_task": "I've weighed it", "artifact": "East.", "rationale": "", "refs": []})]
        return [("submit_answer", {"answer": "East", "confidence": 0.8, "rationale": "facts", "refs": []})]
    rec = asyncio.run(GroupRun(t, t.examples()[0], BlackboardProtocol(read_budget=[0, 1]),
                               RunConfig(n_agents=3, max_actions=8, max_steps=8, seed=0, summarize=False), ScriptedLLM(policy)).run())
    comp = {m["agent"]: m["complete"] for m in members(rec, row)}
    assert comp == {0: True, 1: False, 2: False}
    concl = [m["author"] for m in rec["method_state"]["messages"] if m["sub_task"] == "I've weighed it"]
    assert concl == [0]


def test_certified_voice_is_free_to_read_and_other_messages_still_count():
    ds = BlackboardStore(token_limit=512, read_budget={1: 1}, seed=0, free_read=lambda m: m.sub_task == "Verdict")
    ds.publish(2, "Facts of seat two", "Eastern roads were cleared.", "", [], 0)
    ds.publish(3, "Facts of seat three", "The western bridge collapsed.", "", [], 0)
    ds.publish(0, "Verdict", "Bridge out and roads clear: go East.", "", [], 1)
    first, _ = ds.visit(["Message0"], reader=1, step=2)
    assert first and ds.budget_left(1) == 0
    found, errors = ds.visit(["Message1"], reader=1, step=3)
    assert not found and "withheld" in errors[0]
    found, errors = ds.visit(["Message2"], reader=1, step=3)
    assert found and not errors and ds.budget_left(1) == 0
    assert any(m.id == 2 for m in ds.search("verdict east bridge roads", reader=1, step=4))


def test_voice_wires_the_tasks_certification_and_says_so(tmp_path):
    from rci.tasks.hiddenbench.task import FORWARD_TITLE
    (tmp_path / "benchmark.json").write_text(json.dumps([{"id": 1, "description": "d", "shared_information": ["s"],
                                                          "hidden_information": ["h0", "h1"], "possible_answers": ["A", "B"],
                                                          "correct_answer": "A"}]))
    t8 = make_task("hiddenbench", root=str(tmp_path), conclude="complete")
    tall = make_task("hiddenbench", root=str(tmp_path), conclude="all")
    assert t8.certified("Recommendation: A") and not t8.certified(FORWARD_TITLE) and not tall.certified("Recommendation: A")
    m = BlackboardProtocol(read_budget=[0, 1], voice=True)
    assert "free" in m.budget_prompt(1) and m.budget_prompt(0) == ""
    assert "free" not in BlackboardProtocol(read_budget=[0, 1]).budget_prompt(1)
    run = SimpleNamespace(task=t8, example=t8.examples()[0], cfg=SimpleNamespace(n_agents=4, n_failing=0, seed=0, join_route="shared"),
                          state={})
    try:
        m.build(run)
    except Exception:
        pass
    ds = run.state["datastore"]
    assert ds.free_read is not None
    assert not ds.free_read(SimpleNamespace(sub_task=FORWARD_TITLE)) and ds.free_read(SimpleNamespace(sub_task="Recommendation"))
