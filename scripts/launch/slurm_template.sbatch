#!/bin/bash
#SBATCH --job-name=rci
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -uo pipefail
mkdir -p logs/slurm
export RCI_JOB_ID=${SLURM_JOB_ID:-$$}
export RCI_CACHE=${RCI_CACHE:-${TMPDIR:-/tmp}/rci_$RCI_JOB_ID}
mkdir -p "$RCI_CACHE"
trap 'rm -rf "$RCI_CACHE"' EXIT

MODE=${MODE:?set MODE to eval, sft or credit}
case $MODE in
  eval)
    scripts/launch/serve.sh start "${SERVED:?}" --gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" ${TP:+--tp $TP} || exit 1
    scripts/launch/run_lines.sh "${LINES:?}" --model "$SERVED" --jobs "${JOBS:-4}"
    RC=$?
    scripts/launch/serve.sh stop "$SERVED"
    exit $RC
    ;;
  sft)
    exec scripts/launch/train_sft.sh --gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" ${SFT_ARGS:?}
    ;;
  credit)
    exec scripts/launch/train_credit.sh --config "${CONFIG:-configs/credit/base.yaml}" \
      --train-gpus "${TRAIN_GPUS:-0,1}" --policy-gpus "${POLICY_GPUS:-2,3}" --peer-gpus "${PEER_GPUS:-}" ${CREDIT_ARGS:-}
    ;;
  *)
    echo "unknown MODE $MODE"
    exit 2
    ;;
esac
