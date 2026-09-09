#!/usr/bin/env bash
# Train the three visual stages on the Replica-based echoloc_dataset and
# evaluate each one, in dependency order (comp needs mono and mv).
#
# mono is retrained on both collections rather than reusing a replica_f-only
# checkpoint: comp freezes mono and mv, so a mono that never saw replica_g's
# in-place rotations would bias the selector it is meant to teach.
#
#   bash scripts/train_echoloc_pipeline.sh [gpu]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/opt/conda/envs/f3loc/bin/python}
GPU=${1:-0}
ROOT=${ECHOLOC_ROOT:-/root/storage/echoloc_dataset}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p outputs/logs

run() {
  local name="$1"; shift
  echo "[chain] === $name start $(date -Is)"
  "$@" > "outputs/logs/${name}.log" 2>&1
  local rc=$?
  echo "[chain] === $name rc=$rc $(date -Is)"
  return $rc
}

evaluate() {
  local name="$1" net="$2"
  $PY scripts/eval_job.py --run-name "$name" --net "$net" \
      --datasets replica_f replica_g --bn-modes eval upstream \
      --gpu "$GPU" --dataset-root "$ROOT"
}

run echoloc_mono_fg $PY scripts/train_visual.py --net mono --config configs/echoloc_mono.yaml \
    --gpus "$GPU" --datasets replica_f replica_g --run-name echoloc_mono_fg || exit 1
evaluate echoloc_mono_fg mono

run echoloc_mv_fg $PY scripts/train_visual.py --net mv --config configs/echoloc_mv.yaml \
    --gpus "$GPU" --datasets replica_f replica_g --run-name echoloc_mv_fg || exit 1
evaluate echoloc_mv_fg mv

run echoloc_comp_fg $PY scripts/train_visual.py --net comp --config configs/echoloc_comp.yaml \
    --gpus "$GPU" --mono-ckpt outputs/echoloc_mono_fg/mono.ckpt \
    --mv-ckpt outputs/echoloc_mv_fg/mv.ckpt || exit 1
evaluate echoloc_comp_fg comp

echo "[chain] ALL DONE $(date -Is)"
