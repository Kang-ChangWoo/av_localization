#!/usr/bin/env bash
# Render the SoundSpaces candidate grid for every DESDF (test) scene.
set -u
cd "$(dirname "$0")/.."
source .envs/ss_env.sh
for s in $(ls /root/storage/echoloc_dataset/desdf); do
  echo "=== $s $(date -Is)"
  $SS_ENV/bin/python scripts/soundspaces_render.py --scene "$s" --progress-every 500
done
echo "ALL GRIDS DONE $(date -Is)"
