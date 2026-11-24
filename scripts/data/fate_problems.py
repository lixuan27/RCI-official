#!/usr/bin/env python3
"""Build fate_m.jsonl (and fate_h.jsonl when FATE-H is present) from pinned FATE checkouts."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

DECL_RE = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:theorem|lemma)\s+([^\s({\[:]+)", re.M)
NS_RE = re.compile(r"^\s*namespace\b", re.M)
EVAL_N = {"FATE-M": 100, "FATE-H": 100}


def strip_comments(s: str) -> str:
    s = re.sub(r"/-.*?-/", "", s, flags=re.S)
    return re.sub(r"--[^\n]*", "", s)


def commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print("usage: scripts/data/fate_problems.py SRC_DIR OUT_DIR\n"
              "SRC_DIR holds the FATE-M (and optionally FATE-H) checkouts; scripts/data/prepare_fate.sh calls this.")
        return 2
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split, fname in [("FATE-M", "fate_m.jsonl"), ("FATE-H", "fate_h.jsonl")]:
        if not (src / split / f"{split}.json").exists():
            continue
        items = json.loads((src / split / f"{split}.json").read_text())
        rows, seen = [], set()
        for it in sorted(items, key=lambda x: x["id"]):
            fs = it["formal_statement"]
            body = strip_comments(fs)
            names = DECL_RE.findall(body)
            assert len(names) == 1, (split, it["id"], names)
            assert not NS_RE.search(body), (split, it["id"], "namespace")
            assert body.rstrip().endswith("sorry"), (split, it["id"])
            lf = src / split / split.replace("-", "") / f"{it['id']}.lean"
            if lf.exists():
                assert lf.read_text().strip() == fs.strip(), (split, it["id"], "json != .lean")
            pid = f"{split}/{it['id']}"
            assert pid not in seen
            seen.add(pid)
            rows.append({"pid": pid, "split": split, "id": it["id"], "theorem": names[0],
                         "formal_statement": fs, "informal_statement": it["informal_statement"],
                         "tag": it.get("tag"), "version": it.get("version"),
                         "eval": it["id"] <= EVAL_N[split]})
        tmp = out / (fname + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.replace(out / fname)
        manifest[split] = {"n": len(rows), "n_eval": sum(r["eval"] for r in rows),
                           "commit": commit(src / split), "file": fname}
        print(split, manifest[split])
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
