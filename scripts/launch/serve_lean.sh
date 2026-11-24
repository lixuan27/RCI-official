#!/usr/bin/env bash
set -uo pipefail

WORKERS=${WORKERS:-16}
DATA=${RCI_DATA:-./data}
FATE=$DATA/fate
STAGE=${RCI_CACHE:-${TMPDIR:-/tmp}/rci}/lean_service_$$
SERVICE=$FATE/service
mkdir -p "$SERVICE" "$STAGE"
command -v lake > /dev/null || { echo "lake not found: run scripts/data/install_lean.sh"; exit 2; }
HEADER=$(python -c 'import rci.tasks.fate.lean_service as m, os; print(os.path.join(os.path.dirname(m.__file__), "lean", "FateCheck.lean"))')
ADDR=$SERVICE/server_$$.json
trap 'kill ${SRV:-} 2>/dev/null; rm -rf "$STAGE"; rm -f "$SERVICE/current.json" "$ADDR"' EXIT TERM INT
ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || true

python -u -m rci.tasks.fate.lean_service.server --lean-env "$FATE/lean_env" \
  --repl-bin "$FATE/repl/.lake/build/bin/repl" --lake "$(command -v lake)" --header "$HEADER" \
  --problems "$FATE/fate_m.jsonl" --ref-cache "$FATE/ref_fp.json" --index "$FATE/mathlib_index" \
  --workers "$WORKERS" --stage-dir "$STAGE" --addr-file "$ADDR" &
SRV=$!
for _ in $(seq 1 360); do
  [ -s "$ADDR" ] && break
  kill -0 $SRV 2>/dev/null || { echo "the Lean service exited during startup"; exit 1; }
  sleep 5
done
[ -s "$ADDR" ] || { echo "the Lean service did not start"; exit 1; }
cp "$ADDR" "$SERVICE/current.json.tmp" && mv "$SERVICE/current.json.tmp" "$SERVICE/current.json"
echo "Lean service ready: $(cat "$SERVICE/current.json")"
wait $SRV
