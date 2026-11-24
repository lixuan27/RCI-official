"""HiddenBench and SplitBench hidden-profile tasks: shared facts mislead, private facts must be pooled."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from rci import paths
from rci.agents.tooling import ToolSpec
from rci.tasks.hiddenbench.facts import answer_view, holds, keys_of, words
from rci.tasks.hiddenbench.facts import holders as holders_of
from rci.tasks.base import STR_LIST, Task, ToolResult

HINT = "\n\nBefore you decide, tell the others every fact you hold that they may not have, and read what they have told you."
QUOTE_MIN = 12
CONCLUDE = ("all", "complete", "none")
FORWARD_TITLE = "I've forwarded facts I hold, word for word"
FORWARD_RATIONALE = "Copied word for word from the information I was given."
RULE = {"complete": ("\n\nRule of this store: until you have read the facts every other participant holds, it takes only "
                     "facts copied word for word, posted with `forward_facts`, and `publish_message` is refused. Once you "
                     "have read every participant's facts, `publish_message` also takes your analysis and recommendation."),
        "none": ("\n\nRule of this store: it takes only facts copied word for word, posted with `forward_facts`; "
                 "`publish_message` is refused, so nobody posts analysis or recommendations.")}
NOT_YET = " You have not yet read the facts every other participant holds; read what the others have posted first."
REFUSED = {"complete": "Refused: until you have read every participant's facts, the store takes only facts copied word for word (use forward_facts).",
           "none": "Refused: this store takes only facts copied word for word (use forward_facts)."}
FORWARD_DOC = ("Post facts to the shared store, each copied word for word from the information you were given (one fact per "
               "item).")
RELAY_DOC = ("Post facts to the shared store, each copied word for word from the information you were given or from messages "
             "by others you have read (one fact per item).")
OWN_ONLY_DOC = ("Post facts to the shared store, each copied word for word from the information you were given (facts read in "
                "others' messages are refused; one fact per item).")
RELAY_RULE = (" `forward_facts` takes facts copied word for word from the information you were given and also from messages by "
              "others you have read: you may pass on other participants' facts exactly as written, alone or together with your "
              "own.")
OWN_ONLY_RULE = (" `forward_facts` takes only facts copied word for word from the information you were given yourself: facts "
                 "you read in others' messages are refused.")
OWN_ONLY_REFUSED = "Refused: in this store you can forward only facts you were given yourself, word for word."
BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
SYSTEM = """You take part in a group decision. Other participants may hold information you do not, and you may hold information they do not. Work out which option is correct from everything available to you, then call `submit_answer` with exactly one of the options."""
PROFILES = ("hidden", "full", "shared", "relayed", "subset")
RELAYED = "\n\nWhat the other participants have told you:\n{told}"


def norm(x) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().lower().strip(" .,:;!?\"'`")


def share(items: list[str], seat: int, n: int) -> list[str]:
    """The private facts of a seat among n."""
    return [x for i, x in enumerate(items) if i % n == seat % n]


class HiddenBench(Task):
    name = "hiddenbench"
    msg_token_limit = 512
    max_actions = 16
    max_steps = 24
    summarize = False

    def __init__(self, root: str = "", profile: str = "hidden", hint: bool = False, conclude: str = "all",
                 provenance: bool = False, relay_doc: bool = False, forward_own_only: bool = False, subset_k: int = 0,
                 ids_file: str = ""):
        if profile not in PROFILES:
            raise ValueError(profile)
        if (profile == "subset") != (int(subset_k) > 0):
            raise ValueError("the subset profile needs subset_k > 0, and subset_k needs the subset profile")
        if relay_doc and forward_own_only:
            raise ValueError("relay_doc and forward_own_only contradict each other")
        if (relay_doc or forward_own_only) and conclude == "all":
            raise ValueError("relay_doc and forward_own_only need a store that takes forwards (conclude != 'all')")
        if conclude not in CONCLUDE:
            raise ValueError(f"conclude must be one of {CONCLUDE}")
        self.root = os.path.expandvars(root) if root else str(paths.data_dir("hiddenbench"))
        self.profile, self.hint, self.conclude = profile, hint, conclude
        self.ids_file = os.path.expandvars(ids_file) if ids_file else ""
        self.provenance, self.relay_doc, self.forward_own_only = provenance, relay_doc, forward_own_only
        self.subset_k = int(subset_k)
        self._rows: list[dict] | None = None

    def examples(self, limit: int | None = None) -> list[dict]:
        if self._rows is None:
            self._rows = sorted(paths.load_benchmark(self.root), key=lambda r: r["id"])
        rows = [{**r, "id": f"hb_{r['id']:02d}"} for r in self._rows]
        if self.ids_file:
            keep = json.loads(Path(self.ids_file).read_text())
            keep = set(keep["questions"] if isinstance(keep, dict) else keep)
            rows = [r for r in rows if r["id"] in keep]
        return rows[:limit] if limit else rows

    def system_prompt(self, example: dict) -> str:
        return SYSTEM

    def facts(self, example: dict, seat: int, n: int) -> list[str]:
        hidden = example["hidden_information"]
        if self.profile == "subset":
            k = min(self.subset_k, len(hidden))
            return list(example["shared_information"]) + random.Random(f"{example['id']}/{seat}/{self.subset_k}").sample(hidden, k)
        own = {"hidden": share(hidden, seat, n), "relayed": share(hidden, seat, n), "full": hidden, "shared": []}[self.profile]
        return list(example["shared_information"]) + own

    def told(self, example: dict, seat: int, n: int) -> list[str]:
        """Under "relayed", what each other seat holds, as that participant's words; empty otherwise."""
        if self.profile != "relayed":
            return []
        return [f"Participant {k}: {x}" for k in range(n) if k != seat % n
                for x in share(example["hidden_information"], k, n)]

    def task_prompt(self, example: dict, agent_id: int, n_agents: int, rng) -> str:
        facts = "\n".join(f"- {x}" for x in self.facts(example, agent_id, n_agents))
        options = "\n".join(f"- {x}" for x in example["possible_answers"])
        told = self.told(example, agent_id, n_agents)
        told = RELAYED.format(told="\n".join(f"- {x}" for x in told)) if told else ""
        hint = (HINT if self.hint else "") + RULE.get(self.conclude, "")
        hint += (RELAY_RULE if self.relay_doc else "") + (OWN_ONLY_RULE if self.forward_own_only else "")
        return f"{example['description']}\n\nInformation available to you:\n{facts}{told}\n\nOptions:\n{options}{hint}"

    def forward_error(self, example: dict, artifact: str, allowed: list[str] | None = None) -> str | None:
        pool = list(example["shared_information"]) + list(example["hidden_information"]) if allowed is None else list(allowed)
        facts = [norm(x) for x in pool]
        items = [BULLET.sub("", x).strip().strip("\"'") for x in str(artifact or "").splitlines()]
        items = [x for x in items if x]
        if not items:
            return "Refused: forward at least one fact, copied word for word."
        for x in items:
            nx = norm(x)
            if len(nx) < QUOTE_MIN or not any(nx in f for f in facts):
                return f"Refused: not a fact copied word for word: {x[:120]!r}. Forward only facts, exactly as given."
        return None

    def holders(self, example: dict, text: str, n: int) -> tuple[list[int], bool]:
        """The members among n whose private facts `text` holds, and whether it holds a shared fact (by content)."""
        return holders_of(example, text, n)

    def describe_message(self, sub_task: str, artifact: str, holders: list[int] | None = None, shared: bool = False) -> str:
        """An unread message's header."""
        if sub_task == FORWARD_TITLE:
            k = len([x for x in str(artifact or "").splitlines() if x.strip()])
            lines = f"{k} line{'s' if k != 1 else ''}"
            if holders:
                who = (f"Participant {holders[0]}" if len(holders) == 1 else
                       "Participants " + ", ".join(str(h) for h in holders[:-1]) + f" and {holders[-1]}")
                return f"facts held by {who}{' and shared facts' if shared else ''} ({lines})"
            if holders is not None and shared:
                return f"shared facts only ({lines})"
            return f"facts copied word for word ({lines})"
        if self.certified(sub_task):
            return "analysis or recommendation (posted by a participant who has read every participant's facts)"
        return "analysis or recommendation"

    def answer_view(self, example: dict, text: str) -> str | None:
        return answer_view(text, example)

    def certified(self, sub_task: str) -> bool:
        return self.conclude == "complete" and sub_task != FORWARD_TITLE

    def holds_every_fact(self, example: dict, run, author: int) -> bool:
        ds = run.state.get("datastore")
        if ds is None:
            ds = run.state["datastores"][author]
        keys, n = keys_of(example), run.cfg.n_agents
        has = {h for h in range(len(keys)) if h % n == author % n}
        shown: set[int] = set()
        for e in ds.events:
            if e.get("reader") != author:
                continue
            if e["op"] == "push" and e.get("view") == "full" and e["step"] <= run.step:
                shown |= set(e.get("ids") or [])
            if e["step"] >= run.step:
                continue
            if e["op"] == "search":
                shown |= {r["id"] for r in e.get("results") or []}
            elif e["op"] == "visit":
                shown |= set(e.get("ids") or [])
        for i in shown:
            m = ds.messages[i]
            if m.author != author:
                w = words(" | ".join((m.sub_task, m.artifact, m.rationale)))
                has |= {h for h, k in enumerate(keys) if holds(w, k)}
        return len(has) == len(keys)

    def publish_validator(self, example: dict, run):
        if self.conclude == "all":
            return None

        def check(author, sub_task, artifact, rationale, refs):
            if sub_task == FORWARD_TITLE:
                if refs or rationale not in ("", FORWARD_RATIONALE):
                    return "Refused: a forwarded message carries the facts only, with no rationale or references."
                err = self.forward_error(example, artifact)
                if err or not self.forward_own_only:
                    return err
                n = run.cfg.n_agents + getattr(run.cfg, "n_failing", 0)
                own = list(example["shared_information"]) + share(example["hidden_information"], author, n)
                return OWN_ONLY_REFUSED if self.forward_error(example, artifact, own) else None
            complete = self.holds_every_fact(example, run, author)
            if self.conclude == "complete" and complete:
                return None
            return REFUSED[self.conclude] + ("" if complete else NOT_YET)
        return check

    def blackboard_tools(self, example: dict, run, datastore) -> list[ToolSpec]:
        if self.conclude == "all":
            return []

        def forward(ctx, args: dict) -> ToolResult:
            facts = args.get("facts") or []
            if isinstance(facts, str):
                facts = facts.splitlines()
            artifact = "\n".join(f"- {str(x).strip()}" for x in facts if str(x).strip())
            res = datastore.publish(ctx.agent_id, FORWARD_TITLE, artifact, FORWARD_RATIONALE, [], ctx.run.step)
            if not res.ok:
                return ToolResult(res.error, ok=False)
            return ToolResult(f"Forwarded as {res.message.name}." + (f" Note: {res.note}" if res.note else ""))
        doc = RELAY_DOC if self.relay_doc else OWN_ONLY_DOC if self.forward_own_only else FORWARD_DOC
        return [ToolSpec("forward_facts", doc, {
                             "type": "object", "properties": {"facts": {**STR_LIST, "description": "Facts, each word for word."}},
                             "required": ["facts"]}, forward)]

    def score_agent(self, example: dict, final: dict | None) -> float:
        return float(norm((final or {}).get("answer")) == norm(example["correct_answer"]))

    def score_group(self, example: dict, finals: list[dict], run=None) -> dict:
        out = super().score_group(example, finals, run)
        out["preds"] = [None if f.get("failing") else norm((f.get("final") or {}).get("answer")) or None for f in finals]
        return out
