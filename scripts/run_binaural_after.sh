#!/usr/bin/env bash
# Wait for every ring render to drain, then do the Replica binaural grids.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "soundspaces_render.py.*--layout ring" > /dev/null \
   || pgrep -f "soundspaces_render.py.*collection s3d" > /dev/null \
   || pgrep -f "soundspaces_render.py.*collection mp3d" > /dev/null; do sleep 120; done
YAW=36 SHARDS=24 bash scripts/render_binaural_grids.sh
