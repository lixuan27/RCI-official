#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/data/prepare_docsweep.sh [--model SERVED] [--agg-seed S] [--join-seed S] [--n-agg 100] [--n-join 110]"
  echo "Generates DocSweep corpora (24 documents each) with the 27B tokenizer and admits questions by the model's own"
  echo "behaviour: an oracle that opens every relevant document answers, a single capped member answers at most one of"
  echo "four seeds, and the scripted shortcut probe fails. Writes \$RCI_RESULTS/docsweep_admission/questions.json; the"
  echo "question list of the paper is configs/docsweep/questions.json, whose reference hashes tell whether a"
  echo "regenerated corpus is identical. Needs RCI_BASE_URLS (or a serve.sh server for the model)."
  exit 2
}

MODEL=qwen3.8-27b; AGG_SEED=0; JOIN_SEED=1; N_AGG=100; N_JOIN=110
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --agg-seed) AGG_SEED=$2; shift 2 ;;
    --join-seed) JOIN_SEED=$2; shift 2 ;;
    --n-agg) N_AGG=$2; shift 2 ;;
    --n-join) N_JOIN=$2; shift 2 ;;
    *) usage ;;
  esac
done
DATA=${RCI_DATA:-./data}
RESULTS=${RCI_RESULTS:-./results}
TOK=${RCI_MODELS:-./models}/Qwen3.8-27B
OUT=$DATA/docsweep
[ -d "$TOK" ] || { echo "tokenizer not found at $TOK: run scripts/data/download_models.py qwen3.8-27b"; exit 2; }

python -m rci.tasks.docsweep.generate --out "$OUT" --family agg --n "$N_AGG" --seed "$AGG_SEED" --tokenizer "$TOK"
python -m rci.tasks.docsweep.generate --out "$OUT" --family join --n "$N_JOIN" --seed "$JOIN_SEED" --tokenizer "$TOK"

if [ -z "${RCI_BASE_URLS:-}" ]; then
  F=$RESULTS/.urls/$MODEL.txt
  [ -f "$F" ] || { echo "corpora generated; start the model (scripts/launch/serve.sh start $MODEL) and rerun to admit"; exit 2; }
  RCI_BASE_URLS=$(cat "$F")
fi
ADM=$RESULTS/docsweep_admission
TK="{\"tokenizer\": \"$TOK\", \"confidence\": true, \"split\": \"all\"}"
python -m rci.run --task docsweep --task-kwargs "$TK" --protocol oracle --n-agents 1 --seeds 100,101 \
  --base-urls "$RCI_BASE_URLS" --model "$MODEL" --out "$ADM/oracle" --max-actions 3 --max-steps 4 --gen-limit 12288 \
  --concurrency 24 --save-transcripts
python -m rci.run --task docsweep --task-kwargs "$TK" --protocol independent --n-agents 1 --seeds 100,101,102,103 \
  --base-urls "$RCI_BASE_URLS" --model "$MODEL" --out "$ADM/m1" --max-actions 12 --max-steps 24 --gen-limit 16384 \
  --concurrency 24 --save-transcripts
python -m rci.tasks.docsweep.admit decide --root "$ADM" --oracle-root "$ADM/oracle" --m1-root "$ADM/m1" \
  --data "$OUT" --tokenizer "$TOK" --questions-out "$ADM/questions.json"
