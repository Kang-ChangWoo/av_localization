#!/usr/bin/env bash
# The two s3d runs, restarted after the flat image-naming fix.
set -u
cd "$(dirname "$0")/.."
P=/opt/conda/envs/f3loc/bin/python
run () { echo "[queue] gpu $1 $3 $(date +%H:%M)"
  $P scripts/train_visual.py --net "$2" --config "configs/datasets/$3.yaml" \
     --gpus "$1" --num-workers 6 > "logs/visual_$3.log" 2>&1
  echo "[queue] done $3 rc=$? $(date +%H:%M)"; }
( run 4 mono s3d_mono_lr3e4 ) &
( run 5 mono s3d_mono_lr1e3 ) &
wait
