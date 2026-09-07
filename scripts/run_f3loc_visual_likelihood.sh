#!/usr/bin/env bash
set -euo pipefail

: "${F3LOC_ROOT:?Set F3LOC_ROOT to the pinned external F3Loc clone}"
: "${F3LOC_DATASET_ROOT:?Set F3LOC_DATASET_ROOT to the full F3Loc dataset root}"
: "${F3LOC_CHECKPOINT_ROOT:?Set F3LOC_CHECKPOINT_ROOT to the directory containing comp.ckpt}"

python "$(dirname "$0")/verify_f3loc_baseline.py" \
  --f3loc-root "$F3LOC_ROOT" --dataset-root "$F3LOC_DATASET_ROOT" --checkpoint-root "$F3LOC_CHECKPOINT_ROOT"

cd "$F3LOC_ROOT"
python eval_observation.py --net_type comp --dataset gibson_g --dataset_path "$F3LOC_DATASET_ROOT" \
  --ckpt_path "$F3LOC_CHECKPOINT_ROOT" --viz_dir "${AVFPLOC_DATA_ROOT:-/file2/jeongeon/AV-FPLoc}/reports/f3loc_visual" --viz_limit 20 --save_records
