#!/usr/bin/env bash
# Acoustic candidate grids for the Gibson test scenes (the whole test split, 69
# floors), smallest first so that results accumulate while the large floors
# render. Same engine and settings as the Matterport3D grids; candidates come
# from floorplan_proxy/<scene>/floorplan.glb, the extrusion of map.png.
#
#   SHARDS=24 scripts/render_gibson_grids.sh
set -u
cd "$(dirname "$0")/.."
source .envs/ss_env.sh
ROOT=/root/storage/echoloc_dataset/gibson
OUT=outputs/acoustic_grid_gibson
SHARDS=${SHARDS:-24}
mkdir -p "$OUT" logs/render_gibson

SCENES=$(/opt/conda/envs/f3loc/bin/python - <<'EOF'
import json, numpy as np
m = json.load(open("/root/storage/echoloc_dataset/gibson/dataset_meta.json"))
sz = []
for s in m["split"]["test"]:
    d = np.load(f"/root/storage/echoloc_dataset/gibson/desdf/{s}/desdf.npy", allow_pickle=True).item()["desdf"]
    sz.append((int((d.max(-1) > 0).sum()), s))
print(" ".join(s for _, s in sorted(sz)))
EOF
)
for s in $SCENES; do
  [ -f "$OUT/$s.npz" ] && { echo "[skip] $s already merged"; continue; }
  for i in $(seq 0 $((SHARDS-1))); do
    [ -f "$OUT/${s}_shard${i}.npz" ] && continue
    $SS_ENV/bin/python scripts/soundspaces_render.py \
      --dataset-root "$ROOT" --collection gibson_f --scene "$s" \
      --sample-rate 48000 --indirect-rays 20000 --ray-depth 50 \
      --diffraction --max-diffraction-order 10 \
      --shard "$i" --shards "$SHARDS" --progress-every 500 \
      --out "$OUT/${s}_shard${i}.npz" > "logs/render_gibson/${s}_${i}.log" 2>&1 &
  done
  wait
  /opt/conda/envs/f3loc/bin/python scripts/merge_grid_shards.py \
     --shard-dir "$OUT" --out-dir "$OUT" --scenes "$s" >> logs/render_gibson/merge.log 2>&1
  [ -f "$OUT/$s.npz" ] && rm -f "$OUT/${s}_shard"*.npz
  echo "[done] $s $(date +%H:%M)"
done
echo "[render] all Gibson test scenes finished $(date +%H:%M)"
