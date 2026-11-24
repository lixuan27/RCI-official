#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: scripts/launch/serve.sh start SERVED [--gpus 0,1,2,3] [--tp N] [--port BASE] [--model-path DIR]"
  echo "       scripts/launch/serve.sh stop SERVED"
  echo "Starts one vLLM server per group of TP GPUs for a served model name, waits until every server is healthy,"
  echo "and writes the comma-separated base URLs to \$RCI_RESULTS/.urls/SERVED.txt (read by run_lines.sh)."
  exit 2
}

[ $# -ge 2 ] || usage
CMD=$1; SERVED=$2; shift 2
GPUS=${CUDA_VISIBLE_DEVICES:-0}; TP=""; PORT=20000; MODEL_PATH=""
while [ $# -gt 0 ]; do
  case $1 in
    --gpus) GPUS=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --model-path) MODEL_PATH=$2; shift 2 ;;
    *) usage ;;
  esac
done

RESULTS=${RCI_RESULTS:-./results}
LOGS=${RCI_LOGS:-./logs}/serve
CACHE=${RCI_CACHE:-${TMPDIR:-/tmp}/rci}
URLS_DIR=$RESULTS/.urls
mkdir -p "$URLS_DIR" "$LOGS" "$CACHE"
PIDFILE=$URLS_DIR/$SERVED.pids

if [ "$CMD" = stop ]; then
  [ -f "$PIDFILE" ] && kill -TERM $(cat "$PIDFILE") 2>/dev/null
  rm -f "$PIDFILE" "$URLS_DIR/$SERVED.txt"
  echo "stopped $SERVED"
  exit 0
fi
[ "$CMD" = start ] || usage

PY=${PYTHON:-python}
command -v "$PY" > /dev/null || { echo "$PY not found: activate the environment or set PYTHON"; exit 2; }
command -v vllm > /dev/null || { echo "vllm not found: pip install -e '.[serve]'"; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
GPU_Q=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)
GPU_NAME=${GPU_Q%%,*}; GPU_MEM=${GPU_Q##*,}; GPU_MEM=${GPU_MEM// /}
FAM=$("$PY" -m rci.agents.families shell "$SERVED" --gpu "$GPU_NAME" --gpu-mem "$GPU_MEM" --model-path "$MODEL_PATH") \
  || { echo "could not resolve the serving flags of $SERVED (run from the repository root after pip install -e .)"; exit 2; }
eval "$FAM"
[ -d "$MODEL_PATH" ] || { echo "weights of $SERVED not found at $MODEL_PATH: run scripts/data/download_models.py or pass --model-path"; exit 2; }
[ -n "$TP" ] || TP=${FAMILY_TP:-1}
[ -n "$TP" ] || TP=1
if [ -n "${FAMILY_ENV:-}" ]; then
  for kv in $FAMILY_ENV; do export "$kv"; done
fi
if [ -n "${FAMILY_ENCODINGS:-}" ]; then
  ENC=$CACHE/tiktoken_enc
  mkdir -p "$ENC"
  for e in $FAMILY_ENCODINGS; do
    [ -f "$ENC/${e%%:*}" ] || cp "${TIKTOKEN_CACHE_DIR:-$CACHE/tiktoken}/${e#*:}" "$ENC/${e%%:*}" \
      || { echo "tokenizer vocabulary ${e%%:*} missing (set TIKTOKEN_CACHE_DIR)"; exit 2; }
  done
  export TIKTOKEN_ENCODINGS_BASE=$ENC
fi

IFS=',' read -ra DEVS <<< "$GPUS"
NGPU=${#DEVS[@]}
[ $((NGPU % TP)) = 0 ] || { echo "$NGPU GPUs are not a multiple of TP=$TP"; exit 2; }
BASE=$("$PY" scripts/launch/free_port.py "$PORT" $((NGPU / TP))) || exit 2
PIDS=""; URLS=""
for g in $(seq 0 $((NGPU / TP - 1))); do
  P=$((BASE + g))
  DV=$(IFS=,; echo "${DEVS[*]:$((g * TP)):$TP}")
  CUDA_VISIBLE_DEVICES=$DV VLLM_CACHE_ROOT=$CACHE/vllm_$P TRITON_CACHE_DIR=$CACHE/triton_$P \
    vllm serve "$MODEL_PATH" --served-model-name "$SERVED" --host 127.0.0.1 --port $P --tensor-parallel-size $TP \
    --max-model-len ${FAMILY_MAX_MODEL_LEN:-131072} --gpu-memory-utilization 0.90 --max-num-seqs 256 \
    --enable-prefix-caching $FAMILY_FLAGS ${FAMILY_SERVE_EXTRA:-} ${VLLM_EXTRA:-} > "$LOGS/${SERVED}_$P.log" 2>&1 &
  PIDS="$PIDS $!"
  URLS="${URLS:+$URLS,}http://127.0.0.1:$P/v1"
done
echo "$PIDS" > "$PIDFILE"

for u in ${URLS//,/ }; do
  for _ in $(seq 1 180); do
    curl -sf "${u%/v1}/health" > /dev/null && break
    for p in $PIDS; do
      kill -0 $p 2>/dev/null || { echo "a server of $SERVED exited, see $LOGS"; kill -TERM $PIDS 2>/dev/null; rm -f "$PIDFILE"; exit 1; }
    done
    sleep 10
  done
  curl -sf "${u%/v1}/health" > /dev/null || { echo "server $u not healthy after 30 min, see $LOGS"; kill -TERM $PIDS; exit 1; }
done
echo "$URLS" > "$URLS_DIR/$SERVED.txt"
echo "export RCI_BASE_URLS=$URLS"
