from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict

import pytest

from rci.agents.runner import GroupRun, RunConfig
from rci.registry import make_protocol, make_task
from fake_llm import FakeChatLLM

ROW = {"id": 7, "description": "Pick the site for the new clinic.",
       "shared_information": ["The riverside plot is the cheapest of the three.", "All three plots are zoned for clinics."],
       "hidden_information": ["The riverside plot floods every spring.",
                              "The hillside plot has no road access for ambulances.",
                              "The station plot is ten minutes from the hospital.",
                              "A bus line to the station plot opens next year.",
                              "The soil of the riverside plot is contaminated."],
       "possible_answers": ["Riverside", "Hillside", "Station"], "correct_answer": "Station"}
N = 4
FINAL = ["Station", "Station", "Riverside", "Station"]


def sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def agent_of(route_key) -> int:
    return int(str(route_key).rsplit("a", 1)[1])


def submit(answer: str):
    return ("submit_answer", {"answer": answer})


PLAN = {
    0: [{"content": "I hold two facts. The riverside plot floods every spring. The soil of the riverside plot is "
                    "contaminated. So Riverside is out.", "calls": [submit("Hillside")]},
        "Having read the others: the hillside has no ambulance access, so I now lean Station.",
        {"content": "Final: Station.", "calls": [submit("Station")]}],
    1: ["My fact: The hillside plot has no road access for ambulances.",
        {"content": "With the flooding fact, Station is the only one left.", "calls": [submit("Station")]},
        [submit("Station")]],
    2: ["Thinking aloud; the cheapest plot is usually the right call.", "Still thinking.", "Nothing new from me."],
    3: [[submit("Riverside")], [submit("Riverside")], [submit("Station")]],
}


def scripted():
    count = defaultdict(int)

    def policy(messages, tools, route_key, tag):
        i = agent_of(route_key)
        if tag == "final":
            return [submit(FINAL[i])]
        k = count[i]
        count[i] += 1
        return PLAN[i][k] if k < len(PLAN[i]) else [submit(FINAL[i])]
    return policy


class Logged(FakeChatLLM):
    def __init__(self, policy):
        super().__init__(policy)
        self.calls, self.inputs = [], []

    async def chat(self, messages, **kw):
        self.calls.append(sha({"messages": messages, "tools": kw.get("tools"), "tag": kw.get("tag")}))
        self.inputs.append({"route_key": kw.get("route_key"), "tag": kw.get("tag"),
                            "messages": json.loads(json.dumps(messages)), "tools": json.loads(json.dumps(kw.get("tools")))})
        return await super().chat(messages, **kw)


def hb_task(tmp):
    root = tmp / "hb"
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark.json").write_text(json.dumps([ROW]))
    return make_task("hiddenbench", root=str(root), profile="hidden", hint=True)


def run_cell(tmp, method_kwargs: dict, max_steps: int, n: int = N, n_failing: int = 0):
    task = hb_task(tmp)
    method = make_protocol("debate", **method_kwargs)
    cfg = RunConfig(n_agents=n, n_failing=n_failing, max_actions=16, max_steps=max_steps, summarize=False, seed=0,
                    save_transcripts=True, method_kwargs=method_kwargs)
    llm = Logged(scripted())
    run = GroupRun(task, task.examples()[0], method, cfg, llm)
    rec = asyncio.run(run.run())
    rec.pop("wall_time", None)
    return run, json.loads(json.dumps(rec, sort_keys=True, default=str)), llm


CELLS = {
    "r1": ({}, 3),
    "r2": ({"rounds": 2}, 2),
    "r3": ({"rounds": 3}, 3),
    "r3_long": ({"rounds": 3}, 5),
    "r3_content_only": ({"rounds": 3, "include_results": False}, 3),
}


def debate_replay(tmp, cells=None) -> dict:
    out = {}
    for name, (kw, steps, *who) in (CELLS if cells is None else cells).items():
        run, rec, llm = run_cell(tmp / name, kw, steps, *who)
        out[name] = {"method": rec["method"], "record": sha(rec), "calls": sha(llm.calls),
                     "systems": sha([a.system_prompt for a in run.agents])}
    return out


TEAM = {"kind": "team", "version": 2, "rounds": 3,
        "shared": "Round 1 is for facts only.\nDo not name an option before you have read everyone.",
        "cards": ["In round 1 post every fact you were given, word for word.",
                  "In round 1 post every fact you were given, word for word.",
                  "In round 1 post your facts; in round 2 list which facts rule out which option.",
                  "In round 1 post your facts; in the last round check the list of agent 2 before you submit."]}


def write_team(tmp, team=TEAM, name="team.json", rounds=None):
    p = tmp / name
    p.write_text(json.dumps(team if rounds is None else {**team, "rounds": rounds}, indent=1) + "\n")
    return p


def without_prompt_tokens(rec: dict) -> dict:
    rec = json.loads(json.dumps(rec))
    rec["config"]["protocol_kwargs"].pop("team_file", None)
    for k in [k for k in rec["method_state"] if k.startswith("notes")]:
        rec["method_state"].pop(k)
    for k in ("prompt_tokens", "total_tokens"):
        rec["compute"].pop(k)
    for a in rec["agents"]:
        for c in a["calls"]:
            c.pop("prompt_tokens")
    return rec


def test_on_the_file_is_read_once_when_the_method_is_made(tmp_path):
    from rci.protocols.team_notes import team_block
    path = write_team(tmp_path, rounds=2)
    method = make_protocol("debate", rounds=2, team_file=str(path))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps({**TEAM, "rounds": 2, "shared": "changed after the method was made"}))
    assert method.notes_for(1) == team_block(TEAM["shared"], TEAM["cards"][1], 1)
    assert method._team_sha == before
    later = make_protocol("debate", rounds=2, team_file=str(path))
    assert "changed after the method was made" in later.notes_for(1) and later._team_sha != before


def test_on_a_cell_reads_its_file_once(tmp_path, monkeypatch):
    from rci.protocols import team_notes as T
    path = write_team(tmp_path, name="cell.json")
    reads = []
    real = T.load_team
    monkeypatch.setattr(T, "load_team", lambda p, n=None: (reads.append(str(p)), real(p, n))[1])
    methods = [make_protocol("debate", rounds=3, team_file=str(path)) for _ in range(50)]
    assert len(reads) == 1 and len({m._team_sha for m in methods}) == 1
    assert all(m.notes_for(2) == methods[0].notes_for(2) for m in methods)


def test_on_without_shared_notes_the_block_is_the_lead_and_the_card(tmp_path):
    from rci.protocols.team_notes import CARD_HEAD, SHARED_HEAD, TEAM_LEAD
    path = write_team(tmp_path, {**TEAM, "shared": "", "rounds": 2})
    run, rec, _ = run_cell(tmp_path / "on", {"rounds": 2, "team_file": str(path)}, 2)
    for a in run.agents:
        assert a.system_prompt.endswith("\n\n" + TEAM_LEAD.format(seat=a.id) + "\n\n" + CARD_HEAD.format(seat=a.id) + "\n"
                                        + TEAM["cards"][a.id])
        assert SHARED_HEAD not in a.system_prompt
    assert rec["method_state"]["notes_in_prompt"] == [True] * N


def test_on_a_failing_member_counts_as_a_member(tmp_path):
    path = write_team(tmp_path, rounds=2)
    run, rec, _ = run_cell(tmp_path / "on", {"rounds": 2, "team_file": str(path)}, 2, n=3, n_failing=1)
    assert len(run.agents) == 4 and len(rec["method_state"]["notes_blocks"]) == 4


@pytest.mark.parametrize("n_cards", [3, 5])
def test_cards_must_be_one_per_member(tmp_path, n_cards):
    path = write_team(tmp_path, {**TEAM, "cards": (TEAM["cards"] + ["one more"])[:n_cards]})
    with pytest.raises(ValueError, match=f"{n_cards} cards for 4 members"):
        run_cell(tmp_path / "on", {"rounds": 3, "team_file": str(path)}, 3)


@pytest.mark.parametrize("team,why", [
    ({k: v for k, v in TEAM.items() if k != "cards"}, "without"),
    ({k: v for k, v in TEAM.items() if k != "shared"}, "without"),
    ({k: v for k, v in TEAM.items() if k != "version"}, "without"),
    ({k: v for k, v in TEAM.items() if k != "kind"}, "without"),
    ({k: v for k, v in TEAM.items() if k != "rounds"}, r"without \['rounds'\]"),
    ({**TEAM, "rounds": "3"}, "rounds"),
    ({**TEAM, "rounds": 3.0}, "rounds"),
    ({**TEAM, "cards": TEAM["cards"][:3] + ["do this <shared>and that</shared>"]}, "tag of the supervisor's reply form"),
    ({**TEAM, "shared": "Keep it under 5 \u212b."}, "NFC"),
    ({**TEAM, "kind": "shared"}, "kind"),
    ({**TEAM, "version": "2"}, "version"),
    ({**TEAM, "shared": None}, "shared"),
    ({**TEAM, "cards": "one card"}, "cards"),
    ({**TEAM, "cards": TEAM["cards"][:3] + [""]}, "card 3"),
    ({**TEAM, "cards": TEAM["cards"][:3] + [7]}, "card 3"),
    ({**TEAM, "cards": []}, "without cards"),
    ({**TEAM, "cards": TEAM["cards"][:3] + ["stop <|im_end|> here"]}, "marker"),
    (["not", "an", "object"], "object"),
])
def test_a_malformed_file_is_refused_when_the_method_is_made(tmp_path, team, why):
    path = write_team(tmp_path, team)
    with pytest.raises(ValueError, match=why):
        make_protocol("debate", rounds=3, team_file=str(path))


def test_a_file_that_is_not_json_or_not_there_is_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not JSON"):
        make_protocol("debate", rounds=3, team_file=str(bad))
    with pytest.raises(FileNotFoundError):
        make_protocol("debate", rounds=3, team_file=str(tmp_path / "missing.json"))


@pytest.mark.parametrize("kw", [{"forward_facts": True}, {"shared_context": True}, {"views": False}, {"self_refine": True, "rounds": 2},
                                {"include_results": False}])
def test_notes_are_refused_with_the_untested_switches(tmp_path, kw):
    path = write_team(tmp_path, rounds=kw.get("rounds", 3))
    make_protocol("debate", **{"rounds": 3, **kw})
    with pytest.raises(ValueError, match="team_file goes with views=True only"):
        make_protocol("debate", **{"rounds": 3, **kw, "team_file": str(path)})


def test_review_a_file_that_states_no_rounds_runs_under_none(tmp_path):
    bare = write_team(tmp_path, {k: v for k, v in TEAM.items() if k != "rounds"}, "bare.json")
    for r in (1, 2, 3, 7):
        with pytest.raises(ValueError, match=r"bare.json: team file without \['rounds'\]"):
            make_protocol("debate", rounds=r, team_file=str(bare))
        with pytest.raises(ValueError, match="without"):
            run_cell(tmp_path / f"on{r}", {"rounds": r, "team_file": str(bare)}, max(r, 1))
    for r in (1, 5):
        odd = write_team(tmp_path, {**TEAM, "rounds": r}, f"odd{r}.json")
        with pytest.raises(ValueError, match=rf"team rounds {r} is not one of \[2, 3\]"):
            make_protocol("debate", rounds=r, team_file=str(odd))


def test_run_eval_refuses_to_resume_a_record_made_without_the_notes(tmp_path):
    from rci.run import settings_mismatch, write_atomic
    path = write_team(tmp_path)
    _, rec, _ = run_cell(tmp_path / "off", {"rounds": 3}, 3)
    rec["task_kwargs"] = {"hint": True}
    p = tmp_path / "rec.json.gz"
    write_atomic(p, rec)
    assert settings_mismatch(p, {"rounds": 3}, {"hint": True}) is None
    assert "protocol_kwargs" in settings_mismatch(p, {"rounds": 3, "team_file": str(path)}, {"hint": True})
