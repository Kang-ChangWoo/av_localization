#!/usr/bin/env bash
# Finish the Matterport3D grids at a worker count the machine can actually feed.
#
# The first launch used twenty-four shards per scene, which left more than half
# of the ninety-six cores idle and put the twelve scenes at twelve hours. The
# work is embarrassingly parallel and the only cost of more shards is more
# engine startups, so the rest of the scenes run forty-eight wide. That plus the
# thirty-two binaural workers and the dataloaders of the five training runs fits
# the ninety-six cores without oversubscribing them.
#
# The first scene is already in flight under the twenty-four-way partition. A
# shard index means nothing across partitions, so that scene is allowed to
# finish and is merged on its own terms before anything is relaunched, and any
# shard left over from the old partition is deleted rather than inherited.
set -u
cd "$(dirname "$0")/.."
OUT=outputs/acoustic_grid_mp3d

while pgrep -f "soundspaces_render.py.*collection mp3d" > /dev/null; do sleep 60; done
echo "[wide] the 24-way scene has drained $(date +%H:%M)"

for s in $(ls "$OUT"/*_shard*.npz 2>/dev/null \
           | sed 's#.*/##; s/_shard[0-9]*\.npz$//' | sort -u); do
  /opt/conda/envs/f3loc/bin/python scripts/merge_grid_shards.py \
     --shard-dir "$OUT" --out-dir "$OUT" --scenes "$s" >> logs/render_mp3d/merge.log 2>&1
  if [ -f "$OUT/$s.npz" ]; then
    echo "[wide] merged $s"
  else
    # an incomplete 24-way partition cannot be reused by a 48-way run, and
    # keeping it would make the new run skip shards it never rendered
    echo "[wide] $s did not merge; discarding its old-partition shards"
  fi
  rm -f "$OUT/${s}_shard"*.npz
done

SHARDS=48 exec bash scripts/render_mp3d_grids.sh
