#!/usr/bin/env bash
# Binaural candidate grids for the Replica test scenes.
#
# Why this exists: the shipped grids use six omnidirectional receivers on a 5 cm
# ring, which by construction carries no directivity, and the method therefore
# has to take every heading from vision. A binaural pair does carry heading,
# through the engine's HRTF, so the candidate has to be rendered once per head
# orientation for that to mean anything.
#
# It is also cheaper than it sounds. The ring costs six engine calls per cell
# because the receiver is moved to each microphone; binaural costs one call per
# (cell, orientation). Thirty-six orientations, matching the DESDF's yaw bins,
# come to about 68 core-hours for the three test scenes.
#
# Geometry is floorplan_proxy/<scene>/floorplan.glb exactly as before: the
# watertight extrusion of the same wall mask as map.png, so the candidate side
# stays derivable from a 2D floorplan alone.
set -u
cd "$(dirname "$0")/.."
source .envs/ss_env.sh
ROOT=/root/storage/echoloc_dataset/replica
OUT=outputs/acoustic_grid_binaural
YAW=${YAW:-36}
SHARDS=${SHARDS:-24}
mkdir -p "$OUT" logs/render_binaural
for s in office_4 apartment_2 frl_apartment_5; do
  [ -f "$OUT/$s.npz" ] && { echo "[skip] $s"; continue; }
  for i in $(seq 0 $((SHARDS-1))); do
    [ -f "$OUT/${s}_shard${i}.npz" ] && continue
    $SS_ENV/bin/python scripts/soundspaces_render.py \
      --dataset-root "$ROOT" --collection replica_f --scene "$s" \
      --layout binaural --yaw-bins "$YAW" \
      --sample-rate 48000 --indirect-rays 20000 --ray-depth 50 \
      --diffraction --max-diffraction-order 10 \
      --shard "$i" --shards "$SHARDS" --progress-every 200 \
      --out "$OUT/${s}_shard${i}.npz" > "logs/render_binaural/${s}_${i}.log" 2>&1 &
  done
  wait
  /opt/conda/envs/f3loc/bin/python scripts/merge_grid_shards.py \
     --shard-dir "$OUT" --out-dir "$OUT" --scenes "$s" >> logs/render_binaural/merge.log 2>&1
  [ -f "$OUT/$s.npz" ] && rm -f "$OUT/${s}_shard"*.npz
  echo "[done] $s $(date +%H:%M)"
done
echo "[binaural] finished $(date +%H:%M)"
