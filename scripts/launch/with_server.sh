#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: scripts/launch/with_server.sh [--gpus 0,1,2,3] --model SERVED DIR [--model SERVED2 DIR2] -- 'COMMAND'"
  echo "Serves each model on an equal share of the GPUs (scripts/launch/serve.sh), runs COMMAND with RCI_BASE_URLS"
  echo "(the first model's endpoints) and RCI_POOL (JSON {served name: endpoints}) set, then stops the servers."
  exit 2
}

GPUS=${CUDA_VISIBLE_DEVICES:-0}; NAMES=(); DIRS=()
while [ $# -gt 0 ]; do
  case $1 in
    --gpus) GPUS=$2; shift 2 ;;
    --model) [ $# -ge 3 ] || usage; NAMES+=("$2"); DIRS+=("$3"); shift 3 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[ ${#NAMES[@]} -ge 1 ] && [ $# -ge 1 ] || usage
CMD="$*"
PY=${PYTHON:-python}
IFS=',' read -ra DEVS <<< "$GPUS"
PER=$(( ${#DEVS[@]} / ${#NAMES[@]} ))
[ "$PER" -ge 1 ] || { echo "[with_server] fewer GPUs than models"; exit 2; }
URLS_DIR=${RCI_RESULTS:-./results}/.urls

stop_all() { for n in "${NAMES[@]}"; do scripts/launch/serve.sh stop "$n" > /dev/null; done; }
trap 'stop_all; exit 143' TERM INT

PAIRS=()
for i in "${!NAMES[@]}"; do
  G=$(IFS=,; echo "${DEVS[*]:$((i * PER)):$PER}")
  scripts/launch/serve.sh start "${NAMES[$i]}" --gpus "$G" --port $((20000 + 200 * i)) --model-path "${DIRS[$i]}" \
    > /dev/null || { echo "[with_server] could not serve ${NAMES[$i]}"; stop_all; exit 1; }
  PAIRS+=("${NAMES[$i]}=$(cat "$URLS_DIR/${NAMES[$i]}.txt")")
done
export RCI_BASE_URLS=${PAIRS[0]#*=}
export RCI_POOL=$("$PY" -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:])))' "${PAIRS[@]}")
bash -c "$CMD"
RC=$?
stop_all
exit $RC
