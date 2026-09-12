#!/usr/bin/env bash
# UnLoc and DisCo-FLoc on the two new datasets, alongside the F3Loc runs that
# are already queued elsewhere.
#
# Three things this script has to respect and that are easy to get wrong.
# UnLoc grabs every visible GPU by default, so each run is pinned with
# CUDA_VISIBLE_DEVICES rather than a flag. Its loader only accepts the name
# `gibson_f`, so each dataset is reached through a symlink of that name.
# And Structured3D is 640x360 at 80 degrees where the others are 640x480 at
# 106, which changes F_W at localisation time but not training.
set -u
cd "$(dirname "$0")/.."
UN=/root/storage/implementation/shared_audio/av_localization/UnLoc
DC=/root/storage/implementation/shared_audio/av_localization/DisCo-FLoc
mkdir -p logs

unloc () {  # unloc <gpu> <data_dir> <tag> <epochs>
  echo "[queue] gpu $1 unloc $3 $(date +%H:%M)"
  ( cd "$UN" && CUDA_VISIBLE_DEVICES=$1 /opt/conda/envs/unloc/bin/python train.py \
      --dataset_path "$2" --dataset gibson_f --batch_size 16 --max_epochs "$4" \
    ) > "logs/unloc_$3.log" 2>&1
  echo "[queue] done unloc $3 rc=$? $(date +%H:%M)"
}
rrp () {    # rrp <gpu> <config> <tag>
  echo "[queue] gpu $1 rrp $3 $(date +%H:%M)"
  ( cd "$DC" && CUDA_VISIBLE_DEVICES=$1 /opt/conda/envs/f3loc/bin/python \
      training/train_rrp_model.py -c "$2" --exp_name "$3" \
    ) > "logs/rrp_$3.log" 2>&1
  echo "[queue] done rrp $3 rc=$? $(date +%H:%M)"
}

( unloc 0 data_mp3d mp3d 30 ) &
( unloc 1 data_s3d  s3d  30 ) &
( rrp   6 configs/newdata/rrp_mp3d.yaml rrp_mp3d ) &
( rrp   7 configs/newdata/rrp_s3d.yaml  rrp_s3d  ) &
wait
echo "[queue] new-dataset queue finished $(date +%H:%M)"
