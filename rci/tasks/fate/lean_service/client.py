"""Client for the FATE Lean service (`server.py`)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

FAILOVER_WAIT_S = 3600.0


class LeanClient:
    def __init__(self, url: str, connect_timeout: float = 10.0, addr_file: str | None = None,
                 failover_wait_s: float = FAILOVER_WAIT_S, failover_poll_s: float = 15.0):
        self.url = url.rstrip("/")
        self.connect_timeout = connect_timeout
        import requests
        self.session = requests.Session()
        self.addr_file, self.failover_wait_s, self.failover_poll_s = addr_file, failover_wait_s, failover_poll_s
        self._failover_lock = threading.Lock()

    @classmethod
    def from_addr_file(cls, path: str, wait_s: float = 0.0) -> "LeanClient":
        deadline = time.time() + wait_s
        while True:
            p = Path(path)
            if p.exists():
                return cls(json.loads(p.read_text())["url"], addr_file=path)
            if time.time() >= deadline:
                raise FileNotFoundError(f"Lean service address file {path} not found")
            time.sleep(5)

    def _alive(self, url: str) -> bool:
        import requests
        try:
            self.session.get(url + "/health", timeout=(self.connect_timeout, 30.0)).raise_for_status()
            return True
        except (requests.RequestException, ValueError):
            return False

    def _failover(self, failed_url: str) -> bool:
        if not self.addr_file:
            return False
        with self._failover_lock:
            if self.url != failed_url:
                return True
            deadline = time.time() + self.failover_wait_s
            while True:
                try:
                    url = json.loads(Path(self.addr_file).read_text())["url"].rstrip("/")
                except (OSError, ValueError, KeyError):
                    url = None
                if url and self._alive(url):
                    self.url = url
                    return True
                if time.time() >= deadline:
                    return False
                time.sleep(self.failover_poll_s)

    def _post(self, route: str, payload: dict, read_timeout: float) -> dict:
        import requests
        for attempt in range(3):
            url = self.url
            try:
                r = self.session.post(url + route, json=payload, timeout=(self.connect_timeout, read_timeout))
                break
            except requests.ConnectionError:
                if attempt == 2 or not self._failover(url):
                    raise
        if r.status_code != 200:
            raise RuntimeError(f"{route} HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def check(self, code: str, pid: str | None = None, theorem: str | None = None,
              formal_statement: str | None = None, timeout: float = 300.0) -> dict:
        payload = {"code": code, "timeout": timeout}
        if pid is not None:
            payload["pid"] = pid
        else:
            payload.update(theorem=theorem, formal_statement=formal_statement)
        return self._post("/check", payload, read_timeout=timeout + 1800.0)

    def search(self, query: str, k: int = 8) -> dict:
        return self._post("/search", {"query": query, "k": k}, read_timeout=60.0)

    def health(self) -> dict:
        r = self.session.get(self.url + "/health", timeout=(self.connect_timeout, 30.0))
        r.raise_for_status()
        return r.json()
