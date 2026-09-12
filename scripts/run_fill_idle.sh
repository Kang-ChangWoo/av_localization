#!/usr/bin/env bash
# Fill whatever GPUs are idle. Everything here either has no run yet or is a
# second learning rate for a model that only has one.
set -u
cd "$(dirname "$0")/.."
P=/opt/conda/envs/f3loc/bin/python
DC=/root/storage/implementation/shared_audio/av_localization/DisCo-FLoc
mkdir -p logs
comp () {
  local g=$1 ds=$2 lr=$3
  local m="outputs/visual/$ds/mono_$lr/mono.ckpt" v="outputs/visual/$ds/mv_$lr/mv.ckpt"
  [ -f "$m" ] && [ -f "$v" ] || { echo "[skip] $ds comp $lr"; return; }
  [ -f "outputs/visual/$ds/comp_$lr/comp.ckpt" ] && { echo "[skip] $ds comp $lr exists"; return; }
  echo "[queue] gpu $g comp $ds $lr $(date +%H:%M)"
  $P scripts/train_visual.py --net comp --config "configs/datasets/${ds}_mono_${lr}.yaml" \
     --gpus "$g" --num-workers 6 --mono-ckpt "$m" --mv-ckpt "$v" \
     --run-name "$ds/comp_$lr" --epochs 10 > "logs/visual_${ds}_comp_${lr}.log" 2>&1
  echo "[queue] done $ds comp $lr rc=$? $(date +%H:%M)"
}
rrp () {
  echo "[queue] gpu $1 rrp $3 $(date +%H:%M)"
  ( cd "$DC" && CUDA_VISIBLE_DEVICES=$1 $P training/train_rrp_model.py \
      -c "$2" --exp_name "$3" ) > "logs/rrp_$3.log" 2>&1
  echo "[queue] done rrp $3 rc=$? $(date +%H:%M)"
}
( comp 2 replica lr3e4 ) &
( comp 3 replica lr1e3 ) &
( rrp  4 configs/newdata/rrp_mp3d_lr1e4.yaml rrp_mp3d_lr1e4 ) &
( rrp  5 configs/newdata/rrp_s3d_lr1e4.yaml  rrp_s3d_lr1e4  ) &
wait
echo "[queue] idle-fill finished $(date +%H:%M)"
