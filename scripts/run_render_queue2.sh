#!/usr/bin/env bash
# mp3d ring grids first, then the Replica binaural grids. mp3d is the paper's
# second dataset and blocks its main table; binaural is an extension.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "soundspaces_render.py" > /dev/null; do sleep 60; done
SHARDS=24 bash scripts/render_mp3d_grids.sh
YAW=36 SHARDS=24 bash scripts/render_binaural_grids.sh
