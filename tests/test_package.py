import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rci import registry

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "vllm", "transformers", "fla", "pyarrow", "fitz", "bm25s", "wandb", "matplotlib")


def run(args, **kw):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, **kw)


@pytest.mark.parametrize("path", sorted(registry.TASKS.values()) + sorted(p for p, _ in registry.PROTOCOLS.values()))
def test_every_registry_entry_imports(path):
    mod, attr = path.split(":")
    assert hasattr(importlib.import_module(mod), attr)


def test_core_modules_import_without_heavy_dependencies():
    code = ("import sys, rci.run, rci.ladder.grade, rci.recursion.contrasts, rci.train.credit, rci.organize.structure; "
            f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))")
    out = run(["-c", code])
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize("module", ["rci.run", "rci.ladder.grade", "rci.recursion.contrasts", "rci.agents.families"])
def test_command_lines_print_help(module):
    out = run(["-m", module, "--help"])
    assert out.returncode == 0, out.stderr
    assert "usage" in out.stdout.lower()


def test_the_package_never_edits_sys_path_or_imports_scripts():
    bad = []
    for p in (ROOT / "rci").rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"^\s*(from|import) (scripts|experiments|src)\b|sys\.path", line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad


def test_protocol_directories_follow_the_paper_names():
    assert registry.protocol_dir("blackboard") == "blackboard"
    assert registry.protocol_dir("independent") == "independent"
    assert registry.protocol_dir("debate", rounds=2, forward_facts=True) == "debate_r2_forward_facts"
    assert registry.protocol_dir("debate", rounds=2, shared_context=True) == "debate_r2_shared_context"


def test_shipped_notes_and_task_lists_load():
    from rci.protocols.notes import load_notes
    for p in sorted((ROOT / "configs" / "notes").glob("*.json")):
        d = json.loads(p.read_text())
        assert d["kind"] in ("rules", "shared", "roles")
        got = load_notes(str(p))
        assert got
    t56 = json.loads((ROOT / "configs" / "hiddenbench" / "tasks56.json").read_text())
    assert len(t56["questions"]) == 56 and len(set(t56["questions"])) == 56
    ds = json.loads((ROOT / "configs" / "docsweep" / "questions.json").read_text())
    assert len(ds["eval"]) == 128 and "corpus_sha256" not in ds


def test_the_tree_holds_no_cjk_text_or_bytecode():
    cjk = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x2E80, 0x9FFF), (0xAC00, 0xD7AF), (0xFF00, 0xFFEF))) + "]")
    for p in ROOT.rglob("*"):
        if ".git" in p.parts or "__pycache__" in p.parts or not p.is_file():
            continue
        assert not p.name.endswith((".pyc", ".DS_Store")), p
        if p.suffix in (".py", ".md", ".yaml", ".json", ".sh", ".sbatch", ".toml", ".txt", ".lean", ".example"):
            assert not cjk.search(p.read_text(encoding="utf-8")), p
