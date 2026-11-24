#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/data/prepare_audit.sh [--writer SERVED] [--workers N]"
  echo "Builds the contradiction audit: fetches the documentation of 16 evaluation and 35 training projects at pinned"
  echo "commits, packs snippets, finds candidate pairs, plants contradictions with the writer model (four judges each)"
  echo "and writes the levels x1 ... x32. Needs RCI_BASE_URLS (or a serve.sh server for the writer)."
  exit 2
}

WRITER=qwen3.8-27b; WORKERS=8
while [ $# -gt 0 ]; do
  case $1 in
    --writer) WRITER=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    *) usage ;;
  esac
done
RESULTS=${RCI_RESULTS:-./results}

python -m rci.tasks.audit.build.fetch
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
python -m rci.tasks.audit.build.selftest
python -m rci.tasks.audit.build.build --workers "$WORKERS" --threads 2

if [ -z "${RCI_BASE_URLS:-}" ]; then
  F=$RESULTS/.urls/$WRITER.txt
  [ -f "$F" ] || { echo "corpora built; start the writer (scripts/launch/serve.sh start $WRITER) and rerun to plant"; exit 2; }
  RCI_BASE_URLS=$(cat "$F")
fi
for SPLIT in eval train; do
  python -m rci.tasks.audit.inject inject --split $SPLIT --base-url "$RCI_BASE_URLS" --model "$WRITER" \
    --parallel-projects 3 --batch 32
  python -m rci.tasks.audit.inject inject --split $SPLIT --base-url "$RCI_BASE_URLS" --model "$WRITER" \
    --parallel-projects 3 --batch 32 --second-pass
  python -m rci.tasks.audit.inject levels --split $SPLIT
done
echo "contradiction audit ready under ${RCI_DATA:-./data}/audit"
