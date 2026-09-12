#!/usr/bin/env bash
# Acoustic candidate grids for the Matterport3D test scenes.
#
# Candidates come from floorplan_proxy/<scene>/floorplan.glb, the watertight
# extrusion of the same wall mask as map.png. That is the whole point: a
# deployment has a 2D floorplan and nothing else, so the candidate side must be
# derivable from it even when a furnished mesh happens to exist.
#
# The twelve scenes are the whole Matterport3D test split, listed in
# outputs/mp3d_render_scenes.json, so nothing is selected by size.
set -u
cd "$(dirname "$0")/.."
source .envs/ss_env.sh
ROOT=/root/storage/echoloc_dataset/mp3d
OUT=outputs/acoustic_grid_mp3d
SHARDS=${SHARDS:-8}
mkdir -p "$OUT" logs/render_mp3d

SCENES=$($SS_ENV/bin/python -c "import json;print(' '.join(json.load(open('outputs/mp3d_render_scenes.json'))))")
for s in $SCENES; do
  [ -f "$OUT/$s.npz" ] && { echo "[skip] $s already merged"; continue; }
  for i in $(seq 0 $((SHARDS-1))); do
    [ -f "$OUT/${s}_shard${i}.npz" ] && continue
    $SS_ENV/bin/python scripts/soundspaces_render.py \
      --dataset-root "$ROOT" --collection mp3d_f --scene "$s" \
      --sample-rate 48000 --indirect-rays 20000 --ray-depth 50 \
      --diffraction --max-diffraction-order 10 \
      --shard "$i" --shards "$SHARDS" --progress-every 500 \
      --out "$OUT/${s}_shard${i}.npz" > "logs/render_mp3d/${s}_${i}.log" 2>&1 &
  done
  wait
  /opt/conda/envs/f3loc/bin/python scripts/merge_grid_shards.py \
     --shard-dir "$OUT" --out-dir "$OUT" --scenes "$s" >> logs/render_mp3d/merge.log 2>&1
  [ -f "$OUT/$s.npz" ] && rm -f "$OUT/${s}_shard"*.npz
  echo "[done] $s $(date +%H:%M)"
done
echo "[render] all scenes finished $(date +%H:%M)"
