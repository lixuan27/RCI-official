#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/data/prepare_splitbench.sh [--writer SERVED] [--n 1500] [--ids-file SPLIT.json (main set only)]"
  echo "Writes SplitBench and its control set with the writer model, admits tasks by the behaviour of independent"
  echo "members (full / shared / hidden facts), splits them by a hash of the id into train and held-out sets and"
  echo "seals the writer's answer key into writer_key.json. Needs RCI_BASE_URLS (or a serve.sh server for the writer)."
  exit 2
}

WRITER=qwen3.8-27b; N=1500; IDS=""
while [ $# -gt 0 ]; do
  case $1 in
    --writer) WRITER=$2; shift 2 ;;
    --n) N=$2; shift 2 ;;
    --ids-file) IDS=$2; shift 2 ;;
    *) usage ;;
  esac
done
DATA=${RCI_DATA:-./data}
RESULTS=${RCI_RESULTS:-./results}
if [ -z "${RCI_BASE_URLS:-}" ]; then
  F=$RESULTS/.urls/$WRITER.txt
  [ -f "$F" ] || { echo "start the writer first: scripts/launch/serve.sh start $WRITER"; exit 2; }
  RCI_BASE_URLS=$(cat "$F")
fi
[ -f "$DATA/hiddenbench/benchmark.json" ] || python scripts/data/prepare_hiddenbench.py

for SET in splitbench splitbench_control; do
  FLAG=""; PROFILES="full shared hidden"
  if [ $SET = splitbench_control ]; then FLAG=--control; PROFILES="full shared"; fi
  RAW=$DATA/$SET/raw
  [ -f "$RAW/benchmark.json" ] || python -m rci.tasks.splitbench.generate --base-url "$RCI_BASE_URLS" --model "$WRITER" \
    --n "$N" --out "$RAW" $FLAG
  for P in $PROFILES; do
    python -m rci.run --task hiddenbench --task-kwargs "{\"root\": \"$RAW\", \"profile\": \"$P\", \"confidence\": true}" \
      --protocol independent --n-agents 4 --seeds 0 --base-urls "$RCI_BASE_URLS" --model "$WRITER" \
      --out "$RESULTS/splitbench_admission/$SET/$P" --max-actions 16 --max-steps 24 --concurrency 64
  done
  python -m rci.tasks.splitbench.admit --round "$RAW" --results "$RESULTS/splitbench_admission/$SET" \
    --out "$DATA/$SET/admission" $FLAG
  if [ $SET = splitbench ] && [ -n "$IDS" ]; then
    python -m rci.tasks.splitbench.split split --root "$DATA/$SET" --ids-file "$IDS"
  else
    python -m rci.tasks.splitbench.split split --root "$DATA/$SET"
  fi
done
echo "SplitBench ready: $DATA/splitbench/{train,heldout} and $DATA/splitbench_control/{train,heldout}"
