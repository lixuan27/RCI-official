from types import SimpleNamespace

from rci.protocols.base import Protocol
from rci.protocols.blackboard import BlackboardProtocol
from rci.blackboard.store import BlackboardStore
from rci.tasks.hiddenbench.task import FORWARD_TITLE
from test_blackboard_budgets import ROW, run_with, task


def test_negative_budget_means_none_and_give_moves_unused_budget():
    ds = BlackboardStore(token_limit=512, read_budget={0: 1, 1: 1, 2: -1, 3: 0}, seed=0, choice=True)
    assert ds.budget_left(2) == 0 and ds.budget_left(3) is None and ds.budget_left(0) == 1
    ds.publish(1, FORWARD_TITLE, "Eastern roads were cleared by snowploughs.", "", [], 0)
    assert ds.visit(["Message0"], reader=2, step=1)[0] == []
    assert ds.give(0, 2, 1, step=2) is None
    assert ds.budget_left(0) == 0 and ds.budget_left(2) == 1
    assert ds.events[-1] == {"op": "give", "step": 2, "from": 0, "to": 2, "n": 1}
    assert ds.visit(["Message0"], reader=2, step=3)[0]
    assert "between 1 and 0" in ds.give(0, 1, 1, step=4)
    assert "already read every message" in ds.give(1, 3, 1, step=4)
    assert "not limited" in ds.give(3, 1, 1, step=4) and "another participant" in ds.give(1, 1, 1, step=4)
    refused = [e for e in ds.events if e["op"] == "give_refused"]
    assert len(refused) == 4 and refused[0]["from"] == 0 and "between 1 and 0" in refused[0]["error"]


def test_hold_is_free_and_lifts_on_a_certified_analysis_or_the_deadline(tmp_path):
    t = task(tmp_path)
    m = BlackboardProtocol(read_budget=[0, 1], broadcast="full", wait_until=16)
    run = run_with(t, m)
    ds = run.state["datastore"]
    ds.extra_validator = None
    scout = SimpleNamespace(id=2, submit_name="submit_answer")
    run.step = 3
    hold = m.gate(run, scout, "submit_answer")
    assert isinstance(hold, Protocol.Hold) and getattr(hold, "free", False) and "Wait" in hold
    assert m.gate(run, scout, "search_datastore") is None
    assert run.state["holds"] == [{"step": 3, "agent": 2, "kind": "submit"}]
    waiting = m.gate(run, scout, "wait")
    assert isinstance(waiting, Protocol.Hold) and "no action used" in waiting
    assert m.defer_final(run, scout) and m.defer_final(run, scout)
    assert [h["kind"] for h in run.state["holds"]] == ["submit", "wait", "frozen"] and run.state["frozen"] == {2: 3}
    ds.publish(1, FORWARD_TITLE, ROW["hidden_information"][1], "", [], 3)
    assert m.gate(run, scout, "submit_answer") is not None
    ds.publish(0, "Recommendation", " ".join(ROW["hidden_information"]) + " I recommend East.", "", [], 4)
    assert m.gate(run, scout, "submit_answer") is None
    assert m.gate(run, scout, "wait") is None and not m.defer_final(run, scout)
    run2 = run_with(t, m)
    run2.step = 16
    assert m.gate(run2, scout, "submit_answer") is None
    assert BlackboardProtocol(read_budget=[0, 1]).gate(run2, scout, "submit_answer") is None


def test_hold_spares_a_member_holding_every_fact(tmp_path):
    t = task(tmp_path)
    m = BlackboardProtocol(read_budget=[0, 1], wait_until=16)
    run = run_with(t, m)
    ds = run.state["datastore"]
    ds.extra_validator = None
    for j in (1, 2, 3):
        ds.publish(j, FORWARD_TITLE, ROW["hidden_information"][j], "", [], 0)
    ds.search("bridge roads tunnel pass storm snowploughs avalanche", reader=0, step=1)
    run.step = 2
    assert m.gate(run, SimpleNamespace(id=0, submit_name="submit_answer"), "submit_answer") is None
    assert m.gate(run, SimpleNamespace(id=1, submit_name="submit_answer"), "submit_answer") is not None


def test_give_names_seats_and_shows_live_budgets(tmp_path):
    m = BlackboardProtocol(read_budget=[1], choice=True, broadcast="full", wait_until=16, transfers=True)
    p = m.budget_prompt(2, 4)
    assert "You are Participant 2" in p and "Participant 0 can read at most 1" in p and "give_reading_budget" in p
    assert "Timing rule" in p and "Participant 2 can read" not in p
    t = task(tmp_path)
    run = run_with(t, m)
    ds = run.state["datastore"]
    tools = {s.name: s for s in m.extra_tools(run, 1)}
    ctx = SimpleNamespace(agent_id=1, run=run, step=0)
    res = tools["give_reading_budget"].fn(ctx, {"to": 3, "n": 1})
    assert res.ok and ds.budget_left(3) == 2 and ds.budget_left(1) == 0
    assert "Participant 3: 2" in res.text and "Participant 1 (you): 0" in res.text
    assert not tools["give_reading_budget"].fn(ctx, {"to": 9, "n": 1}).ok
    live = m.live_block(run, SimpleNamespace(id=0, turns=[]), 1)
    assert "Participant 0 (you): 1" in live and "Participant 3: 2" in live
    star = BlackboardProtocol(read_budget=[4, -1], choice=True)
    assert "can read no messages written by others" in star.budget_prompt(0, 4)
    assert "cannot read messages" in BlackboardProtocol(read_budget=[4, -1]).budget_prompt(1, 4)
    assert "give_reading_budget" not in {s.name for s in BlackboardProtocol(read_budget=[1]).extra_tools(run_with(t, BlackboardProtocol(read_budget=[1])), 1)}


def test_frozen_member_makes_no_call_keeps_deliveries_and_is_forced_when_the_hold_lifts():
    import asyncio
    from rci.agents.agent import Agent

    class M:
        hold = True

        def defer_final(self, run, agent):
            return self.hold
    m = M()
    a = Agent.__new__(Agent)
    forced = []

    async def force_final(step):
        forced.append(step)
        a.done = True
    a.done, a.frozen, a.history, a.turns, a.actions = False, True, [], [], 16
    a.ctx = SimpleNamespace(step=0)
    a.cfg = SimpleNamespace(stateless=False)
    a.run = SimpleNamespace(method=m)
    a.force_final = force_final
    a._llm = None
    asyncio.run(a.turn(5, injection="Delivered by the store: ..."))
    assert not a.done and a.frozen and a.history == [[{"role": "user", "content": "Delivered by the store: ..."}]]
    m.hold = False
    asyncio.run(a.turn(6))
    assert forced == [6] and a.done and not a.frozen and a.turns[-1].step == 6 and a.turns[-1].tools == ["forced_final"]


def test_wait_tool_exists_only_under_the_timing_rule(tmp_path):
    t = task(tmp_path)
    on = BlackboardProtocol(read_budget=[0, 1], wait_until=16)
    names = {s.name for s in on.extra_tools(run_with(t, on), 1)}
    off = BlackboardProtocol(read_budget=[0, 1])
    assert "wait" in names and "wait" not in {s.name for s in off.extra_tools(run_with(t, off), 1)}
    assert "call `wait`" in on.budget_prompt(1, 4)


def test_plan_phase_holds_visits_for_free_and_is_told(tmp_path):
    t = task(tmp_path)
    m = BlackboardProtocol(read_budget=[1], choice=True, broadcast="full", wait_until=16, transfers=True, planning_steps=2)
    assert "Reading opens at round 3" in m.budget_prompt(1, 4)
    run = run_with(t, m)
    a = SimpleNamespace(id=1, submit_name="submit_answer")
    run.step = 1
    h = m.gate(run, a, "visit_messages")
    assert isinstance(h, Protocol.Hold) and "round 3" in h and run.state["holds"][-1]["kind"] == "plan"
    assert m.gate(run, a, "give_reading_budget") is None and m.gate(run, a, "search_datastore") is None
    run.step = 2
    assert m.gate(run, a, "visit_messages") is None
    assert "Reading opens" not in BlackboardProtocol(read_budget=[1], choice=True, transfers=True).budget_prompt(1, 4)


def test_hold_can_exempt_unbounded_seats(tmp_path):
    t = task(tmp_path)
    m = BlackboardProtocol(read_budget=[0, 1], broadcast="full", wait_until=16, hold_unbounded=False)
    run = run_with(t, m)
    run.state["datastore"].extra_validator = None
    run.step = 3
    assert m.gate(run, SimpleNamespace(id=0, submit_name="submit_answer"), "submit_answer") is None
    assert m.gate(run, SimpleNamespace(id=2, submit_name="submit_answer"), "submit_answer") is not None
    assert "Timing rule" not in m.budget_prompt(0, 4) and "Timing rule" in m.budget_prompt(2, 4)
    assert "Timing rule" in BlackboardProtocol(read_budget=[0, 1], broadcast="full", wait_until=16).budget_prompt(0, 4)


def test_parse_briefs_accepts_the_smoke_format():
    from rci.protocols.blackboard import parse_briefs
    smoke = ('<alloc>[3,1,0,0]</alloc> <briefs> "Read all others\' messages, compile all facts into the correct analysis, and post it '
             'once ready." "Read one message, but wait for the analysis before making your decision." "Wait for the complete analysis '
             'from others before choosing your answer." "Wait for the analysis from participants who read all facts." </briefs>')
    b = parse_briefs(smoke, 4)
    assert b and len(b) == 4 and b[0].startswith("Read all others") and b[3].endswith("read all facts.")
    assert parse_briefs('<briefs>["a", "b", "c", "d"]</briefs>', 4) == ["a", "b", "c", "d"]
    assert parse_briefs('<briefs>"a" "b" "c"</briefs>', 4) is None
    assert parse_briefs('<briefs>["a", "", "c", "d"]</briefs>', 4) is None
    assert parse_briefs("no tags", 4) is None
    assert parse_briefs('<briefs>"say \\"hi\\"" "b" "c" "d"</briefs>', 4)[0] == 'say \\"hi\\"'
