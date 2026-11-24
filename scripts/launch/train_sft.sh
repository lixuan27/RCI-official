#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: scripts/launch/train_sft.sh --data ROWS.jsonl --model PARENT_DIR --out CKPT_DIR [--gpus 0,1,2,3] [rci.train.sft options]"
  echo "Full-parameter supervised fine-tuning on one node, one process per GPU; the final weights are written to"
  echo "CKPT_DIR/hf, where rci.agents.families finds a served name <name> as \$RCI_CKPT/<name>/hf."
  exit 2
}

[ $# -ge 1 ] || usage
GPUS=${CUDA_VISIBLE_DEVICES:-0}
ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --gpus) GPUS=$2; shift 2 ;;
    -h|--help) usage ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

PY=${PYTHON:-python}
command -v "$PY" > /dev/null || { echo "$PY not found: activate the environment or set PYTHON"; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$("$PY" scripts/launch/free_port.py 29500 1) || exit 1
CACHE=${RCI_CACHE:-${TMPDIR:-/tmp}/rci}
LOGDIR=${RCI_LOGS:-./logs}/sft/$(date +%s)_$$
mkdir -p "$LOGDIR" "$CACHE"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
echo "[train_sft] $NPROC processes on GPUs $GPUS, logs in $LOGDIR"

PIDS=()
for ((r = 0; r < NPROC; r++)); do
  log="$LOGDIR/sft_rank$r.log"
  CUDA_VISIBLE_DEVICES=$GPUS TRITON_CACHE_DIR=$CACHE/triton_sft_$r RANK=$r LOCAL_RANK=$r WORLD_SIZE=$NPROC \
    LOCAL_WORLD_SIZE=$NPROC "$PY" -m rci.train.sft "${ARGS[@]}" > "$log" 2>&1 &
  PIDS+=($!)
done
trap 'kill -TERM ${PIDS[@]} 2>/dev/null; exit 143' TERM INT
RC=0
while :; do
  left=0
  for i in ${!PIDS[@]}; do
    p=${PIDS[$i]}; [ -z "$p" ] && continue
    if kill -0 $p 2>/dev/null; then left=1; continue; fi
    wait $p; rc=$?; PIDS[$i]=""
    if [ $rc -ne 0 ]; then RC=$rc; echo "[train_sft] rank $i exited $rc"; kill -TERM ${PIDS[@]} 2>/dev/null; fi
  done
  [ $left = 0 ] && break
  sleep 10
done
echo "[train_sft] exit $RC"
exit $RC
