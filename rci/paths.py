"""Filesystem locations, read from RCI_* environment variables at call time."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

DEFAULTS = {
    "RCI_DATA": "./data",
    "RCI_RESULTS": "./results",
    "RCI_CKPT": "./checkpoints",
    "RCI_MODELS": "./models",
    "RCI_LOGS": "./logs",
}


def env(name: str, default: str | None = None) -> str | None:
    """The value of an environment variable, or `default` when it is unset or empty."""
    v = os.environ.get(name)
    return v if v else default


def _root(name: str) -> Path:
    return Path(env(name, DEFAULTS[name])).expanduser()


def data_dir(*parts: str) -> Path:
    return _root("RCI_DATA").joinpath(*parts)


def results_dir(*parts: str) -> Path:
    return _root("RCI_RESULTS").joinpath(*parts)


def ckpt_dir(*parts: str) -> Path:
    return _root("RCI_CKPT").joinpath(*parts)


def models_dir(*parts: str) -> Path:
    return _root("RCI_MODELS").joinpath(*parts)


def logs_dir(*parts: str) -> Path:
    return _root("RCI_LOGS").joinpath(*parts)


def cache_dir(*parts: str) -> Path:
    base = env("RCI_CACHE") or os.path.join(env("TMPDIR") or tempfile.gettempdir(), "rci")
    return Path(base).expanduser().joinpath(*parts)


def job_id() -> str:
    """The owner tag written into result lock files: RCI_JOB_ID, else the process id."""
    return env("RCI_JOB_ID") or f"pid{os.getpid()}"


def load_benchmark(directory) -> list:
    """The rows of `<directory>/benchmark.json`, with a hint to the preparation script when the file is absent."""
    p = Path(directory) / "benchmark.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/data/prepare_hiddenbench.py for HiddenBench or "
                                "scripts/data/prepare_splitbench.sh for SplitBench (see README, Data)")
    return json.loads(p.read_text())


def cli(main) -> None:
    """Run a command-line `main` and exit with its code; a missing input file becomes one line and exit code 2."""
    try:
        code = main()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        code = 2
    sys.exit(code)
