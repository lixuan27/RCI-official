"""FATE proof checker: static gate + REPL compile + statement fingerprint + axiom audit."""
from __future__ import annotations

import json
import queue
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rci.tasks.fate.lean_service.repl import ReplDied, ReplProcess, ReplTimeout, lake_env, stage_lean_path

STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

FORBIDDEN = [
    r"\baxiom\b", r"\bopaque\b", r"\bunsafe\b", r"\bimplemented_by\b", r"@\[\s*extern\b",
    r"\bmacro\b", r"\bmacro_rules\b", r"\bsyntax\b", r"\belab\b", r"\belab_rules\b",
    r"\bdeclare_syntax_cat\b", r"\binitialize\b", r"\bbuiltin_initialize\b",
    r"#eval\b", r"#exit\b", r"\brun_cmd\b", r"\brun_elab\b", r"\brun_meta\b", r"\brun_tac\b",
    r"\bby_elab\b", r"\bmodifyEnv\b", r"\baddDecl\b", r"\baddAndCompile\b",
    r"\bdebug\.", r"\bRCIFate\b", r"#fate_fingerprint",
]
_FORBIDDEN_RE = [(p, re.compile(p)) for p in FORBIDDEN]
_IMPORT_RE = re.compile(r"^\s*import\s+\S+.*$")
_FP_RE = re.compile(r"FATE_FP (\{.*\})", re.S)
_SORRY_RE = re.compile(r"declaration uses ['`]sorry['`]")


def strip_comments(code: str) -> str:
    """Remove Lean comments (nested /- -/ and --) keeping line structure."""
    out, i, depth, n = [], 0, 0, len(code)
    while i < n:
        if code.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth and code.startswith("-/", i):
            depth -= 1
            i += 2
            continue
        if depth:
            if code[i] == "\n":
                out.append("\n")
            i += 1
            continue
        if code.startswith("--", i):
            j = code.find("\n", i)
            i = n if j == -1 else j
            continue
        out.append(code[i])
        i += 1
    return "".join(out)


def blank_imports(code: str) -> tuple[str, list[str]]:
    """Replace `import` lines by empty lines (keeps line numbers for messages)."""
    imports, lines = [], []
    for line in code.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _IMPORT_RE.match(line):
            imports.append(line.strip())
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines), imports


def static_gate(code: str) -> list[str]:
    body = strip_comments(code)
    return [p for p, rx in _FORBIDDEN_RE if rx.search(body)]


def format_messages(msgs: list[dict], limit_chars: int) -> str:
    parts = []
    for m in msgs:
        pos = m.get("pos") or {}
        parts.append(f"{pos.get('line', '?')}:{pos.get('column', '?')}: "
                     f"{m.get('severity', '?')}: {m.get('data', '')}")
    s = "\n\n".join(parts)
    if len(s) > limit_chars:
        s = s[:limit_chars] + f"\n... [truncated {len(s) - limit_chars} chars]"
    return s


@dataclass
class Problem:
    pid: str
    theorem: str
    formal_statement: str


def load_problems(path: str) -> dict[str, Problem]:
    probs = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            probs[d["pid"]] = Problem(d["pid"], d["theorem"], d["formal_statement"])
    return probs


class LeanChecker:
    def __init__(self, lean_env_dir: str, repl_bin: str, lake: str, header_lean: str,
                 n_workers: int, max_uses: int = 400, max_rss_gb: float = 24.0,
                 problems: dict[str, Problem] | None = None, ref_cache: str | None = None,
                 msg_limit_chars: int = 12000, stage_dir: str | None = None, log=print):
        self.lean_env_dir, self.repl_bin = lean_env_dir, repl_bin
        self.env = lake_env(lean_env_dir, lake)
        self.stage_dir, self._lake_env, self._stage_lock = stage_dir, self.env, threading.Lock()
        if stage_dir:
            self.env = stage_lean_path(self.env, stage_dir, log)
        self.header = "import Mathlib\n" + Path(header_lean).read_text()
        self.n_workers, self.max_uses, self.max_rss_gb = n_workers, max_uses, max_rss_gb
        self.problems = problems or {}
        self.ref_cache_path = ref_cache
        self.ref_fp: dict[str, dict] = {}
        if ref_cache and Path(ref_cache).exists():
            self.ref_fp = json.loads(Path(ref_cache).read_text())
        self.msg_limit = msg_limit_chars
        self.log = log
        self.idle: "queue.Queue[ReplProcess]" = queue.Queue()
        self.lock = threading.Lock()
        self.stats = {"checks": 0, "accepted": 0, "timeouts": 0, "restarts": 0,
                      "busy": 0, "queued": 0, "check_s_sum": 0.0, "wait_s_sum": 0.0}
        self.ready = threading.Event()

    def _ensure_stage(self) -> None:
        if not self.stage_dir:
            return
        with self._stage_lock:
            if all((Path(p) / ".staged").exists() for p in self.env["LEAN_PATH"].split(":") if p):
                return
            self.log("[stage] staged LEAN_PATH missing on this node; restaging")
            self.env = stage_lean_path(self._lake_env, self.stage_dir, self.log)

    def _new_worker(self) -> ReplProcess:
        self._ensure_stage()
        w = ReplProcess(self.repl_bin, self.lean_env_dir, self.env, self.header,
                        self.max_uses, self.max_rss_gb)
        w.start()
        return w

    def start(self, parallel_start: int = 8) -> None:
        sem = threading.Semaphore(parallel_start)

        def boot(i):
            with sem:
                for attempt in range(3):
                    try:
                        w = self._new_worker()
                        self.log(f"[pool] worker {i} up in {w.init_s:.1f}s rss={w.rss_gb():.2f}GB")
                        self.idle.put(w)
                        self.ready.set()
                        return
                    except Exception as e:
                        self.log(f"[pool] worker {i} boot failed ({attempt}): {e}")
        ts = [threading.Thread(target=boot, args=(i,), daemon=True) for i in range(self.n_workers)]
        for t in ts:
            t.start()

    def _acquire(self) -> ReplProcess:
        with self.lock:
            self.stats["queued"] += 1
        w = self.idle.get()
        with self.lock:
            self.stats["queued"] -= 1
            self.stats["busy"] += 1
        if w.needs_recycle():
            w.kill()
            w = self._new_worker()
            with self.lock:
                self.stats["restarts"] += 1
        return w

    def _release(self, w: ReplProcess, broken: bool) -> None:
        with self.lock:
            self.stats["busy"] -= 1
        if broken:
            w.kill()
            with self.lock:
                self.stats["restarts"] += 1

            def respawn():
                for _ in range(3):
                    try:
                        self.idle.put(self._new_worker())
                        return
                    except Exception as e:
                        self.log(f"[pool] respawn failed: {e}")
            threading.Thread(target=respawn, daemon=True).start()
        else:
            self.idle.put(w)

    def _run(self, w: ReplProcess, code: str, theorem: str, timeout: float) -> dict:
        t0 = time.time()
        r = w.send({"cmd": code, "env": w.base_env}, timeout=timeout)
        elab_s = time.time() - t0
        if "env" not in r:
            return {"repl_error": r.get("message", json.dumps(r)[:500]), "elab_s": elab_s,
                    "messages": []}
        nonce = secrets.token_hex(8)
        r2 = w.send({"cmd": f'#fate_fingerprint {theorem} "{nonce}"', "env": r["env"]},
                    timeout=max(30.0, timeout - elab_s))
        fp = None
        for m in r2.get("messages", []):
            mm = _FP_RE.search(m.get("data", ""))
            if mm:
                d = json.loads(mm.group(1))
                if d.get("nonce") == nonce:
                    fp = d
        return {"messages": r.get("messages", []), "sorries": len(r.get("sorries", [])),
                "fp": fp, "fp_cmd_messages": r2.get("messages", []) if fp is None else [],
                "elab_s": elab_s}

    def reference_fp(self, prob: Problem, timeout: float = 900.0) -> dict:
        if prob.pid in self.ref_fp:
            return self.ref_fp[prob.pid]
        code, _ = blank_imports(prob.formal_statement)
        w = self._acquire()
        broken = False
        try:
            res = self._run(w, code, prob.theorem, timeout)
        except (ReplTimeout, ReplDied):
            broken = True
            raise
        finally:
            self._release(w, broken)
        errs = [m for m in res["messages"] if m.get("severity") == "error"]
        fp = res.get("fp")
        if errs or not fp or not fp.get("found"):
            raise RuntimeError(f"reference for {prob.pid} does not elaborate: "
                               f"{format_messages(errs, 2000)} fp={fp}")
        rec = {"fp": fp["fp"], "kind": fp["kind"], "deps": fp["deps"], "axioms": fp["axioms"]}
        with self.lock:
            self.ref_fp[prob.pid] = rec
            if self.ref_cache_path:
                tmp = self.ref_cache_path + ".tmp"
                Path(tmp).write_text(json.dumps(self.ref_fp, indent=0, sort_keys=True))
                Path(tmp).replace(self.ref_cache_path)
        return rec

    def check(self, code: str, pid: str | None = None, theorem: str | None = None,
              formal_statement: str | None = None, timeout: float = 900.0) -> dict:
        """Full check."""
        t_req = time.time()
        if pid is not None:
            prob = self.problems[pid]
        else:
            prob = Problem(f"adhoc:{theorem}", theorem, formal_statement)
        out = {"pid": prob.pid, "theorem": prob.theorem, "accepted": False}
        bad = static_gate(code)
        if bad:
            out.update(status="forbidden", time_s=round(time.time() - t_req, 3),
                       messages=f"rejected before compilation: forbidden construct(s) {bad}")
            return out
        try:
            ref = self.reference_fp(prob)
        except Exception as e:
            out.update(status="internal_error", messages=f"reference failed: {e}",
                       time_s=round(time.time() - t_req, 3))
            return out
        body, imports = blank_imports(code)
        t_wait = time.time()
        w = self._acquire()
        wait_s = time.time() - t_wait
        broken = False
        try:
            res = self._run(w, body, prob.theorem, timeout)
        except ReplTimeout:
            broken = True
            out.update(status="timeout", messages=f"Lean did not finish within {timeout:.0f}s",
                       wait_s=round(wait_s, 3), time_s=round(time.time() - t_req, 3))
            with self.lock:
                self.stats["timeouts"] += 1
                self.stats["checks"] += 1
            return out
        except ReplDied as e:
            broken = True
            out.update(status="internal_error", messages=f"Lean process died: {e}",
                       wait_s=round(wait_s, 3), time_s=round(time.time() - t_req, 3))
            return out
        finally:
            self._release(w, broken)
        msgs = res["messages"]
        errors = [m for m in msgs if m.get("severity") == "error"]
        sorry = any(_SORRY_RE.search(m.get("data", "")) for m in msgs) or res.get("sorries", 0) > 0
        fp = res.get("fp") or {}
        axioms = fp.get("axioms", [])
        if "repl_error" in res:
            status = "error"
            msgs = [{"severity": "error", "pos": {"line": 1, "column": 0},
                     "data": res["repl_error"]}]
        elif errors:
            status = "error"
        elif sorry:
            status = "sorry"
        elif not fp.get("found"):
            status = "missing_theorem"
        elif fp.get("kind") != "theorem":
            status = "not_a_theorem"
        elif fp.get("fp") != ref["fp"]:
            status = "statement_mismatch"
        elif fp.get("unsafe"):
            status = "unsafe"
        elif not set(axioms) <= STANDARD_AXIOMS:
            status = "nonstandard_axioms"
        else:
            status = "accepted"
        verdict = {
            "accepted": "The proof is complete and accepted.",
            "error": "Compilation failed.",
            "sorry": "The proof still contains `sorry`.",
            "missing_theorem": f"No declaration named `{prob.theorem}` was produced "
                               "(keep the given theorem name, outside any namespace).",
            "not_a_theorem": f"`{prob.theorem}` must be declared with `theorem`.",
            "statement_mismatch": "The theorem statement differs from the given one "
                                  "(it must be proved exactly as stated).",
            "unsafe": "Unsafe declarations are not allowed.",
            "nonstandard_axioms": f"The proof depends on non-standard axioms {sorted(set(axioms) - STANDARD_AXIOMS)}.",
        }[status]
        shown = format_messages(msgs, self.msg_limit)
        out.update(status=status, accepted=status == "accepted",
                   messages=(verdict + ("\n\n" + shown if shown else "")),
                   n_errors=len(errors), axioms=axioms, imports=imports,
                   elab_s=round(res["elab_s"], 3), wait_s=round(wait_s, 3),
                   time_s=round(time.time() - t_req, 3))
        with self.lock:
            self.stats["checks"] += 1
            self.stats["accepted"] += int(status == "accepted")
            self.stats["check_s_sum"] += res["elab_s"]
            self.stats["wait_s_sum"] += wait_s
        return out

    def health(self) -> dict:
        with self.lock:
            s = dict(self.stats)
        s.update(workers=self.n_workers, idle=self.idle.qsize(), refs=len(self.ref_fp),
                 problems=len(self.problems))
        return s
