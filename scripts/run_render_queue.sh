#!/usr/bin/env bash
# s3d first (already under way), then mp3d. mp3d is the one that matters for the
# paper: it is the only new dataset with a furnished-scan query, so it is the
# only one where the furnished-to-floorplan gap this method addresses exists.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "soundspaces_render.py.*collection s3d" > /dev/null; do sleep 60; done
SHARDS=24 bash scripts/render_mp3d_grids.sh
