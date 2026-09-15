#!/usr/bin/env bash
# Candidate grids for validation rooms, so that every selection the paper makes
# (feature, structure, rule, scalars, projection) can be made on rooms the
# test set never shares. Same engine and settings as the test grids; the DESDF
# for each room is built by scripts/build_desdf.py, which reproduces the
# shipped test-room caches to 0.2 px.
#
#   scripts/render_val_grids.sh replica replica_f "apartment_1 frl_apartment_4 office_3"
#   scripts/render_val_grids.sh mp3d mp3d_f "EU6Fwq7SyZv_f2 1LXtFkjw3qL_f2 ..."
#   scripts/render_val_grids.sh s3d s3d "scene_03242 ..."
set -u
cd "$(dirname "$0")/.."
DS=$1; COLL=$2; SCENES=$3
SHARDS=${SHARDS:-8}
ROOT=/root/storage/echoloc_dataset/$DS
OUT=outputs/acoustic_grid_val/$DS
DESDF=outputs/desdf_val/$DS
mkdir -p "$OUT" "logs/render_val/$DS"
source .envs/ss_env.sh

for s in $SCENES; do
  [ -f "$OUT/$s.npz" ] && { echo "[skip] $s already merged"; continue; }
  [ -f "$DESDF/$s/desdf.npy" ] || { echo "[missing desdf] $s"; continue; }
  for i in $(seq 0 $((SHARDS-1))); do
    [ -f "$OUT/${s}_shard${i}.npz" ] && continue
    $SS_ENV/bin/python scripts/soundspaces_render.py \
      --dataset-root "$ROOT" --collection "$COLL" --scene "$s" --desdf-dir "$DESDF" \
      --sample-rate 48000 --indirect-rays 20000 --ray-depth 50 \
      --diffraction --max-diffraction-order 10 \
      --shard "$i" --shards "$SHARDS" --progress-every 500 \
      --out "$OUT/${s}_shard${i}.npz" > "logs/render_val/$DS/${s}_${i}.log" 2>&1 &
  done
  wait
  /opt/conda/envs/f3loc/bin/python scripts/merge_grid_shards.py \
     --shard-dir "$OUT" --out-dir "$OUT" --scenes "$s" >> "logs/render_val/$DS/merge.log" 2>&1
  [ -f "$OUT/$s.npz" ] && rm -f "$OUT/${s}_shard"*.npz
  echo "[done] $DS/$s $(date +%H:%M)"
done
echo "[render] $DS validation grids finished $(date +%H:%M)"
