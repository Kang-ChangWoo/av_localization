#!/usr/bin/env bash
# Image-source candidate grids for every DESDF scene (no simulator needed).
set -u
cd "$(dirname "$0")/.."
for s in $(ls /root/storage/echoloc_dataset/desdf); do
  echo "=== $s $(date -Is)"
  PYTHONDONTWRITEBYTECODE=1 /opt/conda/envs/f3loc/bin/python scripts/pyroom_grid.py \
      --scene "$s" --max-order 3 --progress-every 500
done
echo "ALL PRA GRIDS DONE $(date -Is)"
