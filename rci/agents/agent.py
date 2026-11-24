"""A tool-using ReAct agent with periodic self-summarisation."""
from __future__ import annotations

import json
from dataclasses import dataclass

from rci.agents import events
from rci.agents.llm import ContextTooLong, LLMCall, LLMClient, PromptMeter
from rci.agents.tooling import AgentCtx, ToolResult, ToolSpec, call_tool
from rci.agents.tokens import truncate_to_tokens
from rci.blackboard.prompts import DELIBERATE_NEXT, FORCED_FINAL_PROMPT, NO_TOOL_NUDGE, SUMMARY_PROMPT

FAILING_OUTPUT = "Error code: 500 - {'error': {'message': 'Internal Server Error', 'type': 'server_error', 'code': 500}}"
IGNORED_DONE = "Ignored: you have already submitted your answer."
ACTION_LIMIT = "Error: action limit reached; this call was not executed."
PER_TURN_REFUSED = "Refused: at most {k} `{name}` call(s) per turn. This call was not executed and used no action."
STATUS_LINE = "\n\n(Actions left: {a} of {A}. Steps left: {s} of {S}.)"
COMPACT_HEAD = ("Your history was compacted. You keep the blocks below and, after them, the calls and results of your "
                "last turn.")


@dataclass
class AgentConfig:
    max_actions: int = 300
    gen_limit: int = 16384
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    enable_thinking: bool | None = True
    keep_reasoning: bool = True
    summarize: bool = True
    summarize_after: int = 10
    keep_recent: int = 3
    summary_limit: int = 1024
    capture: bool = False
    max_final_attempts: int = 3
    tool_result_char_limit: int = 60000
    max_turns: int | None = None
    stateless: bool = False
    compact_at: int = 0
    status_line: bool = False
    meter_prompts: bool = False
    deliver_on_final: bool = False


@dataclass
class TurnMeta:
    step: int
    n_tool_calls: int
    tools: list[str]
    actions_after: int
    errors: int
    text_chars: int


class Agent:
    def __init__(self, agent_id: int, system_prompt: str, task_prompt: str, tools: list[ToolSpec],
                 llm: LLMClient, cfg: AgentConfig, run, *, failing: bool = False, role: str | None = None,
                 route_key: str | None = None):
        self.id = agent_id
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        self.tools = {t.name: t for t in tools}
        terminals = [t.name for t in tools if t.terminal]
        if len(terminals) != 1:
            raise ValueError(f"agent needs exactly one terminal tool, got {terminals}")
        self.submit_name = terminals[0]
        self.llm, self.cfg, self.run = llm, cfg, run
        self.failing, self.role = failing, role
        self.route_key = route_key or f"{getattr(run, 'run_id', 'run')}/a{agent_id}"
        self.ctx = AgentCtx(agent_id=agent_id, run=run)
        self.history: list[list[dict]] = []
        self.summary: str | None = None
        self.draft: str | None = None
        self.actions = 0
        self.done = False
        self.final: dict | None = None
        self.final_forced = False
        self.frozen = False
        self.final_errors: list[str] = []
        self.calls: list[LLMCall] = []
        self.turns: list[TurnMeta] = []
        self.full_log: list[dict] = []
        self.last_output: str = ""
        self.last_turn: list[dict] = []
        self.wait_pred = None
        self.wait_until_step: int | None = None
        self.done_notified = False
        self.memory: str | None = None
        self.compactions = 0
        self._compacted_at: tuple | None = None
        self.overflow = False
        self._rebuilt_unsent = False
        self._last_turn: tuple | None = None
        self._meter_obj: PromptMeter | None = None

    def wait(self, pred, until_step: int) -> None:
        self.wait_pred, self.wait_until_step = pred, until_step

    def is_waiting(self, step: int) -> bool:
        if self.wait_pred is None:
            return False
        if step >= (self.wait_until_step or 0) or self.wait_pred():
            self.wait_pred, self.wait_until_step = None, None
            return False
        return True

    def messages(self, extra: list[dict] | None = None, live: str | None = None, system: str | None = None) -> list[dict]:
        user = self.task_prompt
        if self.draft:
            user += "\n\n## Your own analysis before you started acting\n" + self.draft
        if self.summary:
            user += "\n\n## Summary of your earlier work on this task\n" + self.summary
        msgs = [{"role": "system", "content": system or self.system_prompt}, {"role": "user", "content": user}]
        if self.memory:
            msgs.append({"role": "user", "content": self.memory})
        for turn in self.history:
            msgs.extend(turn)
        if extra:
            msgs.extend(extra)
        if live:
            msgs.append({"role": "user", "content": live})
        return msgs

    def tool_schemas(self, only: str | None = None) -> list[dict]:
        return [t.schema() for n, t in self.tools.items() if only is None or n == only]

    def _assistant_msg(self, call: LLMCall) -> dict:
        msg: dict = {"role": "assistant", "content": call.content or ""}
        if call.tool_calls:
            msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                  "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                 for tc in call.tool_calls]
        if self.cfg.keep_reasoning and call.reasoning:
            msg["reasoning_content"] = call.reasoning
        return msg

    async def _llm(self, msgs: list[dict], *, tools: list[dict] | None, tool_choice="auto", tag: str,
                   max_tokens: int | None = None, enable_thinking: bool | None = None,
                   want_token_ids: bool = False) -> LLMCall:
        if self.failing:
            call = LLMCall(content=FAILING_OUTPUT, error="injected-failure", tag=tag)
        else:
            kw = dict(tools=tools, tool_choice=tool_choice, max_tokens=max_tokens or self.cfg.gen_limit,
                      temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k,
                      presence_penalty=self.cfg.presence_penalty,
                      enable_thinking=self.cfg.enable_thinking if enable_thinking is None else enable_thinking,
                      route_key=self.route_key)
            if want_token_ids:
                kw["want_token_ids"] = True
            call = await self.llm.chat(msgs, capture=self.cfg.capture, tag=tag, **kw)
        self.calls.append(call)
        return call

    def _event(self, op: str, **fields) -> dict | None:
        return events.log(self.run, self.id, op, **fields)

    def _why(self, reason: str) -> dict:
        return {"reason": reason} if events.enabled(self.run) else {}

    def _thinking(self, enable_thinking: bool | None = None) -> bool | None:
        return self.cfg.enable_thinking if enable_thinking is None else enable_thinking

    def _render(self, msgs: list[dict], tools: list[dict] | None, enable_thinking: bool | None = None) -> list[int]:
        tok = self.run.task.chat_tokenizer()
        return list(tok.render(msgs, tools=tools, enable_thinking=self._thinking(enable_thinking)))

    def _meter(self) -> PromptMeter:
        if self._meter_obj is None:
            self._meter_obj = PromptMeter(self.run.task.chat_tokenizer())
        return self._meter_obj

    def _status(self, step: int) -> str:
        A = self.cfg.max_actions
        if self.cfg.max_turns is not None:
            S, s = self.cfg.max_turns, max(0, self.cfg.max_turns - (len(self.turns) + 1))
        else:
            S = self.run.cfg.max_steps
            s = max(0, S - (step + 1))
        return STATUS_LINE.format(a=max(0, A - self.actions), A=A, s=s, S=S)

    async def _send(self, msgs: list[dict], *, tools: list[dict] | None, tag: str, tool_choice="auto",
                    max_tokens: int | None = None, injected: bool = False, ids: list[int] | None = None) -> LLMCall:
        method = getattr(self.run, "method", None)
        hook = getattr(method, "on_prompt", None)
        rebuilt = self._rebuilt_unsent
        if hook is not None:
            if rebuilt:
                hook(self.run, self, len(self.calls), "rebuilt")
            elif injected:
                hook(self.run, self, len(self.calls), "injection")
        self._rebuilt_unsent = False
        meter = self.cfg.meter_prompts
        new = None
        if meter:
            if ids is None:
                ids = self._render(msgs, tools)
            new = self._meter().new_tokens(self.route_key, ids)
        self._event("call", call=len(self.calls), tag=tag, rendered_prompt_tokens=len(ids) if ids is not None else None,
                    new_prompt_tokens=new, compacted=rebuilt, injected=bool(injected))
        call = await self._llm(msgs, tools=tools, tool_choice=tool_choice, tag=tag, max_tokens=max_tokens,
                               want_token_ids=meter)
        if meter:
            call.rendered_prompt_tokens, call.new_prompt_tokens = len(ids), new
            call.out_ids = self._meter().commit(self.route_key, ids, call)
            if not self.cfg.capture:
                call.prompt_token_ids, call.token_ids = None, None
        return call

    def compact(self, step: int, *, in_final: bool = False, tokens_before: int | None = None,
                folded: int = 0) -> dict | None:
        task, method = self.run.task, getattr(self.run, "method", None)
        blocks: list = []
        tb = getattr(task, "memory_blocks", None)
        if tb is not None:
            blocks += list(tb(self.run.example, self.run, self) or [])
        mb = getattr(method, "memory_blocks", None)
        if mb is not None:
            blocks += list(mb(self.run, self) or [])
        self.memory = COMPACT_HEAD + "\n\n" + "\n\n".join(text for _, text in blocks)
        last = None
        for i in range(len(self.history) - 1, -1, -1):
            if any(m.get("role") == "assistant" for m in self.history[i]):
                last = i
                break
        folded += len(self.history) - (last + 1 if last is not None else 0)
        kept_entry, kept_names, dropped_effect, unread = None, [], [], 0
        if last is not None:
            entry = self.history[last]
            a_pos = next(k for k, m in enumerate(entry) if m.get("role") == "assistant")
            tcs = list(entry[a_pos].get("tool_calls") or [])
            after = entry[a_pos + 1:]
            results = [m for m in after if m.get("role") == "tool"]
            host = [m for m in after if m.get("role") != "tool"]
            n_results = len(results)
            if len(results) != len(tcs):
                by_id = {m.get("tool_call_id"): m for m in results}
                results = [by_id.get(tc.get("id")) for tc in tcs]
            oks = self._last_turn[1] if self._last_turn is not None and self._last_turn[0] is entry else None
            keep: list[int] = []
            for k, tc in enumerate(tcs):
                name = (tc.get("function") or {}).get("name", tc.get("name"))
                spec = self.tools.get(name)
                ok = bool(oks[k]) if oks is not None and k < len(oks) else False
                if spec is not None and spec.on_compact == "effect" and ok:
                    dropped_effect.append(name)
                else:
                    keep.append(k)
                    kept_names.append(name)
            unread = n_results - sum(1 for k in keep if results[k] is not None) - sum(
                1 for k in range(len(tcs)) if k not in keep and results[k] is not None)
            if keep or host:
                asst: dict = {"role": "assistant", "content": ""}
                if keep:
                    asst["tool_calls"] = [tcs[k] for k in keep]
                kept_entry = [asst] + [results[k] for k in keep if results[k] is not None] + host
                self._last_turn = (kept_entry, [bool(oks[k]) if oks is not None and k < len(oks) else False
                                                for k in keep])
        self.history = [kept_entry] if kept_entry is not None else []
        if kept_entry is None:
            self._last_turn = None
        where = (step, len(self.calls))
        if self._compacted_at != where:
            self.compactions += 1
        self._compacted_at = where
        self._rebuilt_unsent = True
        self.full_log.append({"step": step, "event": "compact", "in_final": in_final, "memory": self.memory,
                              "kept": kept_entry or []})
        return self._event("compact", call=len(self.calls), tokens_before=tokens_before, tokens_after=None,
                           blocks=[name for name, _ in blocks], kept_calls=kept_names,
                           dropped_effect_calls=dropped_effect, dropped_unread_results=unread,
                           folded_injections=folded, in_final=in_final)

    def _overflow(self, step: int, stage: str) -> None:
        self.overflow = True
        self.full_log.append({"step": step, "event": "overflow", "stage": stage})
        self._event("overflow", call=len(self.calls), stage=stage)

    async def _compacting_call(self, step: int, turn_msgs: list[dict], live, plain, tools, tag: str,
                               max_tokens) -> LLMCall | None:
        injected = bool(turn_msgs and turn_msgs[0].get("role") == "user")
        compacted = False

        def build():
            return self.messages(turn_msgs, live, system=plain)

        def do_compact(ids_before: list[int]) -> list[int]:
            nonlocal injected
            folded = 0
            if injected:
                del turn_msgs[0]
                injected, folded = False, 1
            ev = self.compact(step, tokens_before=len(ids_before), folded=folded)
            ids_after = self._render(build(), tools)
            if ev is not None:
                ev["tokens_after"] = len(ids_after)
            return ids_after

        ids = self._render(build(), tools)
        if len(ids) > self.cfg.compact_at:
            ids, compacted = do_compact(ids), True
            if len(ids) > self.cfg.compact_at:
                self._overflow(step, "over_after_compaction")
                return None
        try:
            return await self._send(build(), tools=tools, tag=tag, max_tokens=max_tokens, injected=injected, ids=ids)
        except ContextTooLong:
            if compacted:
                self._overflow(step, "context_too_long_after_compaction")
                return None
        ids = do_compact(ids)
        if len(ids) > self.cfg.compact_at:
            self._overflow(step, "over_after_compaction")
            return None
        try:
            return await self._send(build(), tools=tools, tag=tag, max_tokens=max_tokens, injected=injected, ids=ids)
        except ContextTooLong:
            self._overflow(step, "context_too_long_after_compaction")
            return None

    async def turn(self, step: int, injection: str | None = None, live: str | None = None) -> None:
        if self.done:
            return
        self.ctx.step = step
        if self.frozen:
            if injection and not self.cfg.stateless:
                self.history.append([{"role": "user", "content": injection}])
                hook = getattr(getattr(self.run, "method", None), "on_prompt", None)
                if hook is not None:
                    hook(self.run, self, len(self.calls), "frozen_history")
            method = getattr(self.run, "method", None)
            if method is not None and method.defer_final(self.run, self):
                return
            self.frozen = False
            self.turns.append(TurnMeta(step=step, n_tool_calls=0, tools=["forced_final"], actions_after=self.actions,
                                       errors=0, text_chars=0))
            await self.force_final(step, **self._why("hold_lifted"))
            return
        if self.failing:
            call = await self._llm([], tools=None, tag="turn")
            self.last_output = call.content
            self.turns.append(TurnMeta(step=step, n_tool_calls=0, tools=[], actions_after=self.actions,
                                       errors=1, text_chars=len(call.content)))
            self.full_log.append({"step": step, "messages": [{"role": "assistant", "content": call.content}]})
            return
        turn_msgs: list[dict] = []
        if injection:
            turn_msgs.append({"role": "user", "content": injection})
        task = getattr(self.run, "task", None)
        deliberate = not self.turns and task is not None and (
            task.deliberates(self.run.example) if hasattr(task, "deliberates") else bool(getattr(task, "deliberate_first", False)))
        plain = task.system_prompt(self.run.example) if deliberate else None
        if self.cfg.compact_at > 0:
            call = await self._compacting_call(
                step, turn_msgs, live, plain, None if deliberate else self.tool_schemas(),
                "deliberate" if deliberate else "turn",
                getattr(task, "deliberate_gen_limit", None) if deliberate else None)
            if call is None:
                self.turns.append(TurnMeta(step=step, n_tool_calls=0, tools=["forced_final"],
                                           actions_after=self.actions, errors=0, text_chars=0))
                await self.force_final(step, **self._why("overflow"))
                return
        else:
            try:
                call = await self._send(self.messages(turn_msgs, live, system=plain),
                                        tools=None if deliberate else self.tool_schemas(),
                                        tag="deliberate" if deliberate else "turn",
                                        max_tokens=getattr(task, "deliberate_gen_limit", None) if deliberate else None,
                                        injected=bool(injection))
            except ContextTooLong:
                await self.summarize(force=True)
                try:
                    call = await self._send(self.messages(turn_msgs, live), tools=self.tool_schemas(), tag="turn",
                                            injected=bool(injection))
                except ContextTooLong:
                    self.done = True
                    self.full_log.append({"step": step, "event": "context_overflow"})
                    return
        self.last_output = call.content or ""
        turn_msgs.append(self._assistant_msg(call))
        errors = 0
        follow_ups: list[dict] = []
        method = getattr(self.run, "method", None)
        if deliberate:
            if self.cfg.stateless:
                self.draft = truncate_to_tokens(call.content or "", 2000)
            turn_msgs.append({"role": "user", "content": DELIBERATE_NEXT.format(submit=self.submit_name)})
        elif not call.tool_calls:
            turn_msgs.append({"role": "user", "content": NO_TOOL_NUDGE.format(submit=self.submit_name)})
        call_idx = len(self.calls) - 1
        per_turn: dict[str, int] = {}
        oks: list[bool] = []
        for tc in call.tool_calls:
            name = tc["name"]
            per_turn[name] = per_turn.get(name, 0) + 1
            spec = self.tools.get(name)
            free = spec is not None and spec.free
            counted, refused, res = False, None, None
            if self.done:
                raw, ok, refused = IGNORED_DONE, False, "done"
            elif spec is not None and spec.per_turn and per_turn[name] > spec.per_turn:
                res = ToolResult(PER_TURN_REFUSED.format(k=spec.per_turn, name=name), ok=False, free=True)
                refused = "per_turn"
            elif not free and self.actions >= self.cfg.max_actions:
                raw, ok, refused = ACTION_LIMIT, False, "action_cap"
            else:
                if not free:
                    self.actions += 1
                    counted = True
                gate = method.gate(self.run, self, name) if method is not None else None
                if spec is None:
                    res = ToolResult(f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}.", ok=False)
                    refused = "unknown"
                elif gate:
                    res = ToolResult(gate, ok=False)
                    refused = "gate"
                    if getattr(gate, "free", False) and counted:
                        self.actions -= 1
                        counted = False
                else:
                    res = await call_tool(spec, self.ctx, tc["arguments"])
                    if res.free and counted:
                        self.actions -= 1
                        counted = False
            if res is not None:
                raw, ok = res.text, res.ok
                if not res.ok:
                    errors += 1
                text = raw if len(raw) <= self.cfg.tool_result_char_limit else (
                    raw[: self.cfg.tool_result_char_limit] + "\n[output truncated]")
            else:
                text = raw
            if self.cfg.status_line:
                text += self._status(step)
            oks.append(bool(ok))
            turn_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
            if res is not None and res.content:
                follow_ups.append({"role": "user", "content": res.content})
            if events.enabled(self.run):
                args = tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"], ensure_ascii=False)
                fields = dict(call=call_idx, tool=name, args=args[:400], counted=counted, refused=refused, ok=bool(ok),
                              reason=None if ok else raw[:200])
                for k, v in ((res.meta or {}) if res is not None else {}).items():
                    fields.setdefault(k, v)
                self._event("tool", **fields)
            if spec is not None and spec.terminal and res is not None and res.final is not None:
                self.done = True
                self.final = res.final
        turn_msgs.extend(follow_ups)
        self.last_turn = turn_msgs
        self._last_turn = (turn_msgs, oks)
        if not self.cfg.stateless:
            self.history.append(turn_msgs)
        self.turns.append(TurnMeta(step=step, n_tool_calls=len(call.tool_calls),
                                   tools=[tc["name"] for tc in call.tool_calls], actions_after=self.actions,
                                   errors=errors, text_chars=len(call.content or "")))
        self.full_log.append({"step": step, "messages": turn_msgs})
        if self.cfg.summarize and not self.done and len(self.history) >= self.cfg.summarize_after:
            await self.summarize()
        limit_hit = self.actions >= self.cfg.max_actions or (
            self.cfg.max_turns is not None and len(self.turns) >= self.cfg.max_turns)
        if not self.done and limit_hit:
            if method is not None and method.defer_final(self.run, self):
                self.frozen = True
                self.full_log.append({"step": step, "event": "frozen"})
            elif method is not None and not method.needs_final(self.run, self):
                self.done = True
            else:
                why = "action_cap" if self.actions >= self.cfg.max_actions else "turn_cap"
                await self.force_final(step, **self._why(why))

    def _render_turns(self, turns: list[list[dict]]) -> str:
        out = []
        for turn in turns:
            for m in turn:
                role = m["role"]
                content = m.get("content")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "[image]") for p in content if isinstance(p, dict))
                if role == "assistant":
                    calls = "; ".join(f"{tc['function']['name']}({tc['function']['arguments']})"
                                      for tc in m.get("tool_calls", []))
                    out.append(f"[you] {content or ''}" + (f"\n[you called] {calls}" if calls else ""))
                elif role == "tool":
                    out.append(f"[tool result] {content}")
                else:
                    out.append(f"[{role}] {content}")
        return "\n".join(out)

    async def summarize(self, force: bool = False) -> None:
        keep = 0 if force else self.cfg.keep_recent
        if len(self.history) <= keep:
            return
        old, recent = self.history[: len(self.history) - keep], self.history[len(self.history) - keep:]
        body = SUMMARY_PROMPT.format(limit=self.cfg.summary_limit)
        body += "\n\n## The task\n" + truncate_to_tokens(self.task_prompt, 4000)
        if self.summary:
            body += "\n\n## Earlier summary\n" + self.summary
        body += "\n\n## History to summarize\n" + self._render_turns(old)
        msgs = [{"role": "system", "content": "You write concise, faithful working summaries for an agent."},
                {"role": "user", "content": body}]
        try:
            call = await self._llm(msgs, tools=None, tag="summary", max_tokens=self.cfg.summary_limit,
                                   enable_thinking=False)
            text = (call.content or "").strip()
        except ContextTooLong:
            text = ""
        if not text:
            text = truncate_to_tokens(self._render_turns(old), self.cfg.summary_limit)
        self.summary = truncate_to_tokens(text, self.cfg.summary_limit)
        self.history = recent
        self.full_log.append({"event": "summary", "summary": self.summary})

    async def force_final(self, step: int, reason: str | None = None) -> None:
        """Force the member's submission (up to `max_final_attempts` calls with only the submit tool, then the salvage)."""
        if self.done:
            return
        if self.failing:
            self.done = True
            return
        self.final_forced = True
        self._event("force_final", reason=reason, call=len(self.calls))
        compacting = self.cfg.compact_at > 0
        render = compacting or self.cfg.meter_prompts
        extra: list[dict] = []
        inj_msg = None
        if compacting and self.cfg.deliver_on_final:
            method = getattr(self.run, "method", None)
            inj = method.injection(self.run, self, step) if method is not None else None
            if inj:
                inj_msg = {"role": "user", "content": inj}
                extra.append(inj_msg)
        extra.append({"role": "user", "content": FORCED_FINAL_PROMPT.format(submit=self.submit_name)})
        compacted_here = False
        choice = {"type": "function", "function": {"name": self.submit_name}}
        spec = self.tools[self.submit_name]
        last_args = None

        def fold_and_compact(ids_before: list[int]) -> list[int]:
            nonlocal inj_msg, compacted_here
            folded = 0
            if inj_msg is not None and any(m is inj_msg for m in extra):
                extra[:] = [m for m in extra if m is not inj_msg]
                inj_msg, folded = None, 1
            ev = self.compact(step, in_final=True, tokens_before=len(ids_before), folded=folded)
            compacted_here = True
            ids_after = self._render(self.messages(extra), tools)
            if ev is not None:
                ev["tokens_after"] = len(ids_after)
            return ids_after

        tools = self.tool_schemas(only=self.submit_name)
        for attempt in range(self.cfg.max_final_attempts):
            self.ctx.state["final_last_attempt"] = attempt == self.cfg.max_final_attempts - 1
            ids = self._render(self.messages(extra), tools) if render else None
            if compacting and not compacted_here and len(ids) > self.cfg.compact_at:
                ids = fold_and_compact(ids)
            try:
                call = await self._send(self.messages(extra), tools=tools,
                                        tool_choice=choice if attempt == 0 else "auto", tag="final",
                                        injected=inj_msg is not None and any(m is inj_msg for m in extra), ids=ids)
            except ContextTooLong:
                if not compacting:
                    await self.summarize(force=True)
                    continue
                if not compacted_here:
                    fold_and_compact(ids)
                    continue
                self._overflow(step, "context_too_long_in_final")
                break
            tcs = [tc for tc in call.tool_calls if tc["name"] == self.submit_name]
            if not tcs:
                self.final_errors.append(f"no {self.submit_name} call: {(call.content or '')[:200]}")
                extra = extra + [self._assistant_msg(call),
                                 {"role": "user", "content": FORCED_FINAL_PROMPT.format(submit=self.submit_name)}]
                continue
            last_args = tcs[0]["arguments"]
            res = await call_tool(spec, self.ctx, last_args)
            if res.final is not None:
                self.final = res.final
                break
            self.final_errors.append(res.text[:300])
            extra = extra + [self._assistant_msg(call),
                             {"role": "tool", "tool_call_id": tcs[0]["id"], "content": res.text}]
            for tc in call.tool_calls[1:]:
                extra.append({"role": "tool", "tool_call_id": tc["id"], "content": "Ignored."})
        if self.final is None and last_args is not None:
            res = await call_tool(spec, self.ctx, sanitize_submission(last_args, spec.parameters))
            if res.final is not None:
                self.final = {**res.final, "salvaged": True}
            else:
                self.final_errors.append("salvage: " + res.text[:300])
        self.done = True
        self.full_log.append({"step": step, "event": "forced_final", "final": self.final})

    def record(self) -> dict:
        summ = getattr(getattr(self.run, "task", None), "agent_state_summary", None)
        out = {
            "task_state": summ(self.ctx.state) if summ else None,
            "id": self.id, "role": self.role, "failing": self.failing, "actions": self.actions,
            "n_turns": len(self.turns), "done": self.done, "final": self.final, "final_forced": self.final_forced,
            "final_errors": self.final_errors,
            "calls": [c.summary() for c in self.calls],
            "turns": [t.__dict__ for t in self.turns],
        }
        if self.compactions:
            out["compactions"] = self.compactions
        if self.overflow:
            out["overflow"] = True
        return out

    def transcript(self) -> list[dict]:
        return self.full_log


def sanitize_submission(raw, parameters: dict) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    props = parameters.get("properties") or {}
    out = {}
    for k, sch in props.items():
        if k in ("refs",):
            out[k] = []
        elif k == "rationale":
            out[k] = ""
        elif k in raw:
            v = raw[k]
            if sch.get("type") == "string" and not isinstance(v, str) and v is not None:
                v = v if not isinstance(v, (list, dict)) else json.dumps(v, ensure_ascii=False)
                v = str(v)
            out[k] = v
    return out


def dumps_args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False)
