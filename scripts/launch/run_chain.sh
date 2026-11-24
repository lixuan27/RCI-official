#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: scripts/launch/run_chain.sh DRIVER CHAIN [--gpus 0,1,2,3] [--steps N]"
  echo "Runs a gated generation chain step by step: asks the driver (experiments/chains_<benchmark>.py) for the next"
  echo "step, runs a CPU step in the driver, runs a GPU step's line on the GPUs and marks it done, until the chain"
  echo "stops (or after N steps). Any scheduler can run the same loop: every step is resumable."
  exit 2
}

[ $# -ge 2 ] || usage
DRIVER=$1; CHAIN=$2; shift 2
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}; MAXSTEPS=0
while [ $# -gt 0 ]; do
  case $1 in
    --gpus) GPUS=$2; shift 2 ;;
    --steps) MAXSTEPS=$2; shift 2 ;;
    *) usage ;;
  esac
done
PY=${PYTHON:-python}
LOGS=${RCI_LOGS:-./logs}/chains/$CHAIN
mkdir -p "$LOGS"
n=0
while :; do
  NEXT=$("$PY" "$DRIVER" next --chain "$CHAIN" --tsv) || { echo "[run_chain] the driver failed"; exit 1; }
  IFS=$'\t' read -r KIND STEP LINE <<< "$NEXT"
  case $KIND in
    stop) echo "[run_chain] $CHAIN stopped: $STEP"; exit 0 ;;
    wait) echo "[run_chain] $CHAIN waits: $STEP"; exit 0 ;;
    cpu) "$PY" "$DRIVER" step --chain "$CHAIN" --step "$STEP" || { echo "[run_chain] $STEP failed"; exit 1; } ;;
    gpu)
      LOG=$LOGS/$(date +%s)_${STEP//[:\/]/_}.log
      echo "[run_chain] $STEP (log $LOG)"
      CUDA_VISIBLE_DEVICES=$GPUS bash -c "$LINE" > "$LOG" 2>&1 || { echo "[run_chain] $STEP failed, see $LOG"; exit 1; }
      "$PY" "$DRIVER" done --chain "$CHAIN" --step "$STEP" || exit 1
      ;;
    *) echo "[run_chain] unexpected answer: $NEXT"; exit 1 ;;
  esac
  n=$((n + 1))
  [ "$MAXSTEPS" -gt 0 ] && [ $n -ge "$MAXSTEPS" ] && exit 0
done
