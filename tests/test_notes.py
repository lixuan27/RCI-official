import json

from rci.organize.self_prompt import FWD, episode_summary, parse, rules_prompt, shared_prompt
from rci.protocols.blackboard import BlackboardProtocol

REC = {
    "agents": [{"id": j} for j in range(4)],
    "score": {"preds": ["east", "east", "west", "east"], "truth_never_here": True},
    "method_state": {
        "messages": [{"id": 0, "author": 1, "sub_task": FWD, "step": 0}, {"id": 1, "author": 2, "sub_task": FWD, "step": 0},
                     {"id": 2, "author": 0, "sub_task": "Analysis", "step": 5}],
        "events": [{"op": "give", "step": 1, "from": 1, "to": 0, "n": 1},
                   {"op": "give_refused", "step": 7, "from": 3, "to": 0, "n": 1,
                    "error": "Refused: you can give between 1 and 0 message(s) of reading budget."}],
        "holds": [{"step": 3, "agent": 2, "kind": "wait"}, {"step": 4, "agent": 2, "kind": "submit"}, {"step": 1, "agent": 1, "kind": "plan"}],
    },
}
MEM = [{"agent": 0, "shown": [0, 1], "complete": True, "last": 6}, {"agent": 1, "shown": [], "complete": False, "last": 7},
       {"agent": 2, "shown": [0], "complete": False, "last": 8}, {"agent": 3, "shown": [1], "complete": False, "last": 4}]


def test_episode_summary_reports_how_the_group_worked_and_nothing_about_correctness():
    s = episode_summary(REC, MEM, "1")
    assert "round 1: Participant 1 gave 1 to Participant 0" in s and "Participant 3 to 0" in s
    assert "first analysis: Participant 0 at round 5" in s
    assert "Participant 2: forwarded 1 message(s), read 1 by others, did not hold every fact, waited 2 time(s)" in s
    assert "3 of 4 gave the same answer" in s
    low = s.lower()
    assert "correct" not in low and "right" not in low and "truth" not in low
    none = episode_summary({**REC, "method_state": {"messages": [], "events": []}}, MEM, "2")
    assert "nobody gave reading budget" in none and "no analysis was posted" in none


def test_prompts_state_rules_without_the_solution_and_parse_keeps_old_text():
    for msgs in (shared_prompt(4, 1, "", "Task 1: ..."), rules_prompt(4, 1)):
        sysmsg = msgs[0]["content"]
        assert "only 1 message(s)" in sysmsg and "Nobody is told whether an answer was right" in sysmsg
        assert "give your budget to" not in sysmsg.lower() and "integrat" not in sysmsg.lower()
    assert parse("x <notes> a  b </notes> y <notes>c d</notes>", "notes", "old") == "c d"
    assert parse("no tags", "notes", "old") == "old" and len(parse("<card>" + "w " * 400 + "</card>", "card", "").split()) == 150
