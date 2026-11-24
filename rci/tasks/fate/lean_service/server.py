"""HTTP front end of the FATE Lean service (one process per CPU node)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rci.tasks.fate.lean_service.checker import LeanChecker, load_problems

MAX_TIMEOUT = 900.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_app(checker: LeanChecker, index, pool: ThreadPoolExecutor) -> web.Application:
    from aiohttp import web
    lat: list[float] = []

    async def check(req: web.Request) -> web.Response:
        body = await req.json()
        code = body.get("code")
        if not isinstance(code, str):
            return web.json_response({"error": "missing 'code'"}, status=400)
        timeout = min(float(body.get("timeout", 300.0)), MAX_TIMEOUT)
        pid = body.get("pid")
        if pid is not None and pid not in checker.problems:
            return web.json_response({"error": f"unknown pid {pid!r}"}, status=400)
        if pid is None and not (body.get("theorem") and body.get("formal_statement")):
            return web.json_response({"error": "give 'pid' or 'theorem'+'formal_statement'"},
                                     status=400)
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(pool, lambda: checker.check(
            code, pid=pid, theorem=body.get("theorem"),
            formal_statement=body.get("formal_statement"), timeout=timeout))
        if "time_s" in res:
            lat.append(res["time_s"])
        return web.json_response(res)

    async def search(req: web.Request) -> web.Response:
        if index is None:
            return web.json_response({"error": "search index not loaded"}, status=503)
        body = await req.json()
        from .search import format_results
        t0 = time.time()
        res = index.search(str(body.get("query", "")), int(body.get("k", 8)))
        return web.json_response({"results": res, "text": format_results(res),
                                  "time_s": round(time.time() - t0, 4)})

    async def health(_req: web.Request) -> web.Response:
        h = checker.health()
        if lat:
            xs = sorted(lat)
            h["latency_s"] = {"n": len(xs), "p50": xs[len(xs) // 2],
                              "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "max": xs[-1]}
        h["search"] = index is not None
        return web.json_response(h)

    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.add_routes([web.post("/check", check), web.post("/search", search),
                    web.get("/health", health)])
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rci.tasks.fate.lean_service.server",
                                 description="Serve Lean proof checking and Mathlib search over HTTP.")
    ap.add_argument("--lean-env", required=True)
    ap.add_argument("--repl-bin", required=True)
    ap.add_argument("--lake", required=True)
    ap.add_argument("--header", required=True, help="FateCheck.lean")
    ap.add_argument("--problems", nargs="*", default=[])
    ap.add_argument("--ref-cache", default=None)
    ap.add_argument("--index", default=None, help="search_mathlib index dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-uses", type=int, default=400)
    ap.add_argument("--max-rss-gb", type=float, default=24.0)
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--addr-file", default=None)
    a = ap.parse_args(argv)

    problems = {}
    for f in a.problems:
        problems.update(load_problems(f))
    checker = LeanChecker(a.lean_env, a.repl_bin, a.lake, a.header, a.workers,
                          max_uses=a.max_uses, max_rss_gb=a.max_rss_gb, problems=problems,
                          ref_cache=a.ref_cache, stage_dir=a.stage_dir, log=log)
    index = None
    if a.index and Path(a.index).exists():
        from .search import MathlibIndex
        t0 = time.time()
        index = MathlibIndex(a.index)
        log(f"[search] {len(index.decls)} decls loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    checker.start()
    checker.ready.wait()
    log(f"[pool] first worker ready after {time.time() - t0:.1f}s; {len(problems)} problems")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", a.port))
    port = sock.getsockname()[1]
    host = socket.gethostname()
    if a.addr_file:
        info = {"url": f"http://{host}:{port}", "host": host, "port": port,
                "jobid": os.environ.get("RCI_JOB_ID"), "workers": a.workers,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = a.addr_file + f".tmp{os.getpid()}"
        Path(tmp).write_text(json.dumps(info) + "\n")
        os.replace(tmp, a.addr_file)
    log(f"[http] serving on http://{host}:{port}")
    pool = ThreadPoolExecutor(max_workers=max(64, 4 * a.workers))
    from aiohttp import web
    web.run_app(make_app(checker, index, pool), sock=sock, print=None,
                access_log=None, handle_signals=True)
    return 0


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)
