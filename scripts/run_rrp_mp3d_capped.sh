#!/usr/bin/env bash
# The two Matterport3D ray-predictor runs, restarted under the eight-epoch
# budget the configs now carry.
#
# They were stopped at epoch three and epoch zero of sixty, which at two hours
# an epoch would have been four days. Restarting rather than resuming keeps the
# cosine schedule consistent with the epoch budget, so each run can be described
# as what it is instead of as a truncation of a longer one; the reason for the
# budget is recorded in the configs themselves.
set -u
cd "$(dirname "$0")/.."
DC=/root/storage/implementation/shared_audio/av_localization/DisCo-FLoc
P=/opt/conda/envs/f3loc/bin/python
mkdir -p logs

rrp () {   # rrp <gpu> <config> <tag>
  echo "[rrp] gpu $1 $3 start $(date +%H:%M)"
  ( cd "$DC" && CUDA_VISIBLE_DEVICES=$1 $P training/train_rrp_model.py \
      -c "$2" --exp_name "$3" ) > "logs/rrp_$3.log" 2>&1
  echo "[rrp] $3 rc=$? $(date +%H:%M)"
}

( rrp 6 configs/newdata/rrp_mp3d.yaml      rrp_mp3d      ) &
( rrp 4 configs/newdata/rrp_mp3d_lr1e4.yaml rrp_mp3d_lr1e4 ) &
wait
echo "[rrp] both Matterport3D runs finished $(date +%H:%M)"
