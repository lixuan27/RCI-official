#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: scripts/launch/train_credit.sh --config configs/credit/base.yaml [--train-gpus 0,1] [--policy-gpus 2,3]"
  echo "                                      [--peer-gpus ''] [--set key=value ...]"
  echo "Starts the rollout servers (policy servers receive weight updates; peer servers hold the frozen members of"
  echo "the one-trainee arm), waits until they are healthy and runs rci.train.loop with one process per training GPU."
  echo "A rerun with the same run_id resumes from its last checkpoint."
  exit 2
}

CONFIG=""; TRAIN_GPUS=0,1; POLICY_GPUS=2,3; PEER_GPUS=""; SETS=()
while [ $# -gt 0 ]; do
  case $1 in
    --config) CONFIG=$2; shift 2 ;;
    --train-gpus) TRAIN_GPUS=$2; shift 2 ;;
    --policy-gpus) POLICY_GPUS=$2; shift 2 ;;
    --peer-gpus) PEER_GPUS=$2; shift 2 ;;
    --set) SETS+=(--set "$2"); shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$CONFIG" ] || usage
PY=${PYTHON:-python}
command -v "$PY" > /dev/null || { echo "$PY not found: activate the environment or set PYTHON"; exit 2; }
command -v vllm > /dev/null || { echo "vllm not found: pip install -e '.[serve,train]'"; exit 2; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
CACHE=${RCI_CACHE:-${TMPDIR:-/tmp}/rci}
LOGDIR=${RCI_LOGS:-./logs}/credit/$(date +%s)_$$
mkdir -p "$LOGDIR" "$CACHE"
FP="$PY scripts/launch/free_port.py"
export MASTER_PORT=$($FP 29000 1) || exit 1
WT_PORT=$($FP 10000 8) || exit 1
PORT=$($FP 20000 8) || exit 1
MODEL_PATH=$("$PY" -c 'import sys; from rci.train.config import load_config; print(load_config(sys.argv[1]).model_path)' "$CONFIG") || exit 1
[ -d "$MODEL_PATH" ] || { echo "weights not found at $MODEL_PATH: run scripts/data/download_models.py"; exit 2; }

PIDS=(); POLICY_URLS=(); PEER_URLS=()
serve() {
  local wt=()
  [ "$3" = 1 ] && wt=(--weight-transfer-config '{"backend":"nccl"}')
  ( [ "$3" = 1 ] && export VLLM_SERVER_DEV_MODE=1
    export CUDA_VISIBLE_DEVICES=$1 VLLM_CACHE_ROOT=$CACHE/vllm_$PORT TRITON_CACHE_DIR=$CACHE/triton_$PORT
    exec vllm serve "$MODEL_PATH" --served-model-name "$2" --host 127.0.0.1 --port $PORT \
      --max-model-len ${MAX_MODEL_LEN:-40960} --gpu-memory-utilization ${SERVER_UTIL:-0.90} --max-num-seqs 256 \
      --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --generation-config vllm --language-model-only "${wt[@]}" ) > "$LOGDIR/vllm_${2}_gpu$1.log" 2>&1 &
  PIDS+=($!)
  if [ "$2" = policy ]; then POLICY_URLS+=("http://127.0.0.1:$PORT"); else PEER_URLS+=("http://127.0.0.1:$PORT"); fi
  PORT=$((PORT + 1))
}
for g in ${POLICY_GPUS//,/ }; do serve $g policy 1; done
for g in ${PEER_GPUS//,/ }; do serve $g peer 0; done
[ ${#POLICY_URLS[@]} -gt 0 ] || { echo "[train_credit] no policy server"; exit 2; }
TPIDS=()
trap 'kill -TERM ${TPIDS[@]} 2>/dev/null; sleep 30; kill ${PIDS[@]} 2>/dev/null; exit 143' TERM INT

for u in "${POLICY_URLS[@]}" ${PEER_URLS[@]+"${PEER_URLS[@]}"}; do
  for _ in $(seq 1 240); do
    curl -sf "$u/health" > /dev/null && break
    for p in ${PIDS[@]}; do
      kill -0 $p 2>/dev/null || { echo "[train_credit] a rollout server exited, see $LOGDIR"; kill ${PIDS[@]} 2>/dev/null; exit 1; }
    done
    sleep 10
  done
  curl -sf "$u/health" > /dev/null || { echo "[train_credit] $u not healthy, see $LOGDIR"; kill ${PIDS[@]}; exit 1; }
done

lst() { local IFS=,; echo "[$*]"; }
SETS+=(--set "policy_urls=$(lst "${POLICY_URLS[@]}")" --set "peer_urls=$(lst ${PEER_URLS[@]+"${PEER_URLS[@]}"})"
       --set "wt_master_port=$WT_PORT")
NPROC=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
for ((r = 0; r < NPROC; r++)); do
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS TRITON_CACHE_DIR=$CACHE/triton_train_$r RANK=$r LOCAL_RANK=$r WORLD_SIZE=$NPROC \
    LOCAL_WORLD_SIZE=$NPROC "$PY" -m rci.train.loop --config "$CONFIG" "${SETS[@]}" > "$LOGDIR/train_rank$r.log" 2>&1 &
  TPIDS+=($!)
done
RC=0
while :; do
  left=0
  for i in ${!TPIDS[@]}; do
    p=${TPIDS[$i]}; [ -z "$p" ] && continue
    if kill -0 $p 2>/dev/null; then left=1; continue; fi
    wait $p; rc=$?; TPIDS[$i]=""
    if [ $rc -ne 0 ]; then RC=$rc; echo "[train_credit] rank $i exited $rc"; kill -TERM ${TPIDS[@]} 2>/dev/null; fi
  done
  [ $left = 0 ] && break
  sleep 5
done
kill ${PIDS[@]} 2>/dev/null
wait 2>/dev/null
echo "[train_credit] exit $RC, logs in $LOGDIR"
exit $RC
