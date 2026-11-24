"""One long-lived leanprover-community/repl process with `import Mathlib` preloaded."""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


class ReplTimeout(Exception):
    pass


class ReplDied(Exception):
    pass


def lake_env(lean_env_dir: str, lake: str) -> dict:
    """Environment that `lake env` would give (LEAN_PATH etc.), computed once."""
    out = subprocess.run([lake, "env", "env", "-0"], cwd=lean_env_dir,
                         capture_output=True, check=True).stdout
    env = {}
    for item in out.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode()] = v.decode()
    return env


def stage_lean_path(env: dict, stage_dir: str, log=print) -> dict:
    """Copy every LEAN_PATH entry (toolchain + package oleans) to node-local storage and point LEAN_PATH there."""
    t0 = time.time()
    root = Path(stage_dir)
    root.mkdir(parents=True, exist_ok=True)
    new = []
    for i, p in enumerate(env["LEAN_PATH"].split(":")):
        if not p or not Path(p).is_dir():
            continue
        dst = root / f"{i:02d}"
        if not (dst / ".staged").exists():
            subprocess.run(["rm", "-rf", str(dst)], check=True)
            subprocess.run(["cp", "-r", p, str(dst)], check=True)
            (dst / ".staged").write_text(p)
        new.append(str(dst))
    env = dict(env)
    env["LEAN_PATH"] = ":".join(new)
    log(f"[stage] {len(new)} LEAN_PATH entries -> {root} in {time.time() - t0:.1f}s")
    return env


@dataclass
class ReplProcess:
    repl_bin: str
    cwd: str
    env: dict
    header: str
    max_uses: int = 400
    max_rss_gb: float = 24.0
    proc: subprocess.Popen | None = None
    base_env: int | None = None
    uses: int = 0
    init_s: float = 0.0
    _buf: bytes = field(default=b"", repr=False)

    def start(self, timeout: float = 600.0) -> None:
        self.kill()
        t0 = time.time()
        self.proc = subprocess.Popen(
            [self.repl_bin], cwd=self.cwd, env=self.env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._buf = b""
        self.uses = 0
        r = self.send({"cmd": self.header}, timeout=timeout)
        errs = [m for m in r.get("messages", []) if m.get("severity") == "error"]
        if "env" not in r or errs:
            raise RuntimeError(f"REPL header failed: {json.dumps(r)[:2000]}")
        self.base_env = r["env"]
        self.init_s = time.time() - t0

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None
        self.base_env = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def rss_gb(self) -> float:
        if not self.alive():
            return 0.0
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
        except OSError:
            pass
        return 0.0

    def needs_recycle(self) -> bool:
        return (not self.alive()) or self.uses >= self.max_uses or self.rss_gb() > self.max_rss_gb

    def send(self, obj: dict, timeout: float) -> dict:
        if not self.alive():
            raise ReplDied("repl not running")
        data = (json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise ReplDied("stdin closed") from e
        self.uses += 1
        deadline = time.time() + timeout
        fd = self.proc.stdout.fileno()
        while True:
            idx = self._buf.find(b"\n\n")
            while idx != -1:
                chunk, rest = self._buf[:idx], self._buf[idx + 2:]
                s = chunk.strip()
                if s:
                    try:
                        val = json.loads(s)
                        self._buf = rest
                        return val
                    except json.JSONDecodeError:
                        pass
                idx = self._buf.find(b"\n\n", idx + 2)
            left = deadline - time.time()
            if left <= 0:
                raise ReplTimeout(f"no response within {timeout:.0f}s")
            r, _, _ = select.select([fd], [], [], min(left, 5.0))
            if r:
                b = os.read(fd, 1 << 20)
                if not b:
                    raise ReplDied("stdout EOF")
                self._buf += b
            elif self.proc.poll() is not None:
                raise ReplDied(f"exited rc={self.proc.returncode}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "stage":
        e = stage_lean_path(lake_env(sys.argv[2], sys.argv[3]), sys.argv[4],
                            log=lambda m: print(m, file=sys.stderr))
        print(e["LEAN_PATH"])
    else:
        sys.exit("usage: python -m rci.tasks.fate.lean_service.repl stage LEAN_ENV_DIR LAKE STAGE_DIR")
