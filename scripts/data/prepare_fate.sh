#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/data/prepare_fate.sh [--shards N]"
  echo "Clones FATE-M at its pinned commit, writes \$RCI_DATA/fate/fate_m.jsonl, builds a Lean project pinned to"
  echo "FATE-M's Mathlib, builds the Lean REPL, exports Mathlib declarations and builds the search index."
  echo "Needs elan/lake on PATH (scripts/data/install_lean.sh)."
  exit 2
}

SHARDS=16
while [ $# -gt 0 ]; do
  case $1 in
    --shards) SHARDS=$2; shift 2 ;;
    *) usage ;;
  esac
done
command -v lake > /dev/null || { echo "lake not found: run scripts/data/install_lean.sh"; exit 2; }

DATA=${RCI_DATA:-./data}
FATE=$DATA/fate
SRC=$FATE/src
ENV_DIR=$FATE/lean_env
REPL_DIR=$FATE/repl
FATE_M_COMMIT=4eb33c8ccd0ff058b461cd763cc406509129743f
REPL_COMMIT=527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a
mkdir -p "$SRC"

clone_at() {
  local url=$1 rev=$2 dest=$3
  if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD)" = "$rev" ]; then return 0; fi
  rm -rf "$dest"
  git init -q "$dest"
  git -C "$dest" remote add origin "$url"
  git -C "$dest" fetch -q --depth 1 origin "$rev"
  git -C "$dest" checkout -q FETCH_HEAD
  [ "$(git -C "$dest" rev-parse HEAD)" = "$rev" ] || { echo "$url is not at $rev"; exit 1; }
}

clone_at https://github.com/frenzymath/FATE-M.git "$FATE_M_COMMIT" "$SRC/FATE-M"
python scripts/data/fate_problems.py "$SRC" "$FATE"

mkdir -p "$ENV_DIR"
python - "$SRC/FATE-M" "$ENV_DIR" <<'PY'
import json
import sys
from pathlib import Path

src, env = Path(sys.argv[1]), Path(sys.argv[2])
tc = (src / "lean-toolchain").read_text().strip()
assert tc == "leanprover/lean4:v4.28.0", tc
man = json.loads((src / "lake-manifest.json").read_text())
man["name"] = "lean_env"
(env / "lake-manifest.json").write_text(json.dumps(man, indent=1) + "\n")
(env / "lean-toolchain").write_text(tc + "\n")
(env / "lakefile.lean").write_text("""import Lake
open Lake DSL

package lean_env where
  leanOptions := #[
    \u27e8`pp.unicode.fun, true\u27e9,
    \u27e8`pp.proofs.withType, false\u27e9
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.28.0"
""")
PY
(cd "$ENV_DIR" && lake exe cache get)

clone_at https://github.com/leanprover-community/repl.git "$REPL_COMMIT" "$REPL_DIR"
(cd "$REPL_DIR" && lake build)

OUT=$FATE/mathlib_decls
STAGE=${RCI_CACHE:-${TMPDIR:-/tmp}/rci}/lean_stage
mkdir -p "$OUT" "$STAGE"
LEAN=$(cd "$ENV_DIR" && lake env which lean)
LP=$(python -m rci.tasks.fate.lean_service.repl stage "$ENV_DIR" "$(command -v lake)" "$STAGE")
EXPORT=$(python -c 'import rci.tasks.fate.lean_service as m, os; print(os.path.join(os.path.dirname(m.__file__), "lean", "ExportDecls.lean"))')
for i in $(seq 0 $((SHARDS - 1))); do
  LEAN_PATH=$LP FATE_EXPORT_OUT=$OUT/shard_$i.jsonl.tmp FATE_EXPORT_SHARD=$i FATE_EXPORT_NSHARDS=$SHARDS \
    "$LEAN" "$EXPORT" > "$OUT/shard_$i.log" 2>&1 &
done
wait
for i in $(seq 0 $((SHARDS - 1))); do
  grep -q EXPORT_DONE "$OUT/shard_$i.log" || { echo "export shard $i failed, see $OUT/shard_$i.log"; exit 1; }
  mv "$OUT/shard_$i.jsonl.tmp" "$OUT/shard_$i.jsonl"
done
rm -rf "$STAGE"
python -m rci.tasks.fate.lean_service.search build "$OUT"/shard_*.jsonl --out "$FATE/mathlib_index"
echo "FATE-M ready under $FATE; start the checker with scripts/launch/serve_lean.sh"
