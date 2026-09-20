#!/usr/bin/env bash
# DisCo-FLoc RRP retrained with the validation split as the checkpoint monitor.
#
# The four RRP runs behind the tables were trained with `val_split: test`, so the
# top-3 checkpoints (and the one the tables used) were chosen by the test-split
# action loss. That is checkpoint selection on the test set, and the DisCo rows
# cannot be described as test-once until it is undone. These runs are identical
# in every setting (epochs, lr, batch, schedule) except `val_split: val`; the
# configs are the *_val.yaml copies next to the originals. Sequential on one
# card, shortest first, so usable rows appear as early as possible.
set -u
cd "$(dirname "$0")/.."
DC=/root/storage/implementation/shared_audio/av_localization/DisCo-FLoc
GPU=${GPU:-1}
mkdir -p logs
rrp () {    # rrp <config> <tag>
  echo "[queue] gpu $GPU rrp $2 start $(date '+%m-%d %H:%M')"
  ( cd "$DC" && CUDA_VISIBLE_DEVICES=$GPU /opt/conda/envs/f3loc/bin/python \
      training/train_rrp_model.py -c "$1" --exp_name "$2" \
    ) > "logs/rrp_$2.log" 2>&1
  echo "[queue] done rrp $2 rc=$? $(date '+%m-%d %H:%M')"
}
# RRP uses about 2 GB, so the 400-epoch Replica run (about two days) goes in
# its own lane and the three shorter runs queue behind each other next to it:
#   LANE=short scripts/run_rrp_val_queue.sh   # s3d -> gibson -> mp3d
#   LANE=long  scripts/run_rrp_val_queue.sh   # replica
LANE=${LANE:-short}
if [ "$LANE" = long ]; then
  rrp configs/replica_avfploc/rrp_replica_f_val.yaml rrp_replica_f_val
else
  rrp configs/newdata/rrp_s3d_val.yaml               rrp_s3d_val
  rrp configs/newdata/rrp_gibson_val.yaml            rrp_gibson_val
  rrp configs/newdata/rrp_mp3d_lr1e4_val.yaml        rrp_mp3d_lr1e4_val
fi
echo "[queue] val-split RRP queue ($LANE) finished $(date '+%m-%d %H:%M')"
