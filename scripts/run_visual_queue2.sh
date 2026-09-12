#!/usr/bin/env bash
# Everything left to train once mono and mv exist on each dataset.
#
# comp fuses a trained mono and mv, so it can only run after them and is fed the
# matching learning-rate pair. s3d has no mv, and therefore no comp, because it
# was built with L=0 and has no source views.
set -u
cd "$(dirname "$0")/.."
P=/opt/conda/envs/f3loc/bin/python
mkdir -p logs
comp () {  # comp <gpu> <dataset> <lrtag>
  local g=$1 ds=$2 lr=$3
  local m="outputs/visual/$ds/mono_$lr/mono.ckpt" v="outputs/visual/$ds/mv_$lr/mv.ckpt"
  if [ ! -f "$m" ] || [ ! -f "$v" ]; then echo "[skip] $ds comp $lr: missing mono or mv"; return; fi
  echo "[queue] gpu $g comp $ds $lr $(date +%H:%M)"
  $P scripts/train_visual.py --net comp --config "configs/datasets/${ds}_mono_${lr}.yaml" \
     --gpus "$g" --num-workers 6 --mono-ckpt "$m" --mv-ckpt "$v" \
     --run-name "$ds/comp_$lr" --epochs 10 \
     > "logs/visual_${ds}_comp_${lr}.log" 2>&1
  echo "[queue] done $ds comp $lr rc=$? $(date +%H:%M)"
}
( comp 4 mp3d lr3e4 ; comp 4 replica lr3e4 ) &
( comp 5 mp3d lr1e3 ; comp 5 replica lr1e3 ) &
wait
echo "[queue] comp finished $(date +%H:%M)"
