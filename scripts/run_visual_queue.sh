#!/usr/bin/env bash
# Vision training across the three datasets, one GPU per run.
#
# Lives in the repository rather than a temporary directory: the previous
# attempt put the launcher in a scratch path that was cleaned between calls, so
# the queue reported success and started nothing.
#
# Ten runs over six GPUs. mp3d goes first on each shared GPU because it is the
# dataset with ten times the frames and therefore the one whose result matters
# most; replica follows on the same device.
set -u
cd "$(dirname "$0")/.."
P=/opt/conda/envs/f3loc/bin/python
mkdir -p logs

run () {   # run <gpu> <net> <config-stem>
  echo "[queue] gpu $1  $2  $3  $(date +%H:%M)"
  $P scripts/train_visual.py --net "$2" --config "configs/datasets/$3.yaml" \
     --gpus "$1" --num-workers 6 > "logs/visual_$3.log" 2>&1
  echo "[queue] done $3 rc=$? $(date +%H:%M)"
}

( run 0 mono mp3d_mono_lr3e4 ; run 0 mono replica_mono_lr3e4 ) &
( run 1 mono mp3d_mono_lr1e3 ; run 1 mono replica_mono_lr1e3 ) &
( run 2 mv   mp3d_mv_lr3e4   ; run 2 mv   replica_mv_lr3e4   ) &
( run 3 mv   mp3d_mv_lr1e3   ; run 3 mv   replica_mv_lr1e3   ) &
( run 4 mono s3d_mono_lr3e4 ) &
( run 5 mono s3d_mono_lr1e3 ) &
wait
echo "[queue] all finished $(date +%H:%M)"
