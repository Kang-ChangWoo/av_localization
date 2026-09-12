#!/usr/bin/env bash
# Structured3D mode-table extraction.
#
# Two dataset differences from Replica are load-bearing and are passed
# explicitly rather than left to the defaults:
#   --f-w 0.5959   S3D renders at 80 deg horizontal FOV, not 106.26, so the ray
#                  fan differs; this matches configs/datasets/s3d_*.yaml.
#   flat RGB names S3D collections are rendered with L=0, one image per pose;
#                  the extractor falls back to that naming on its own.
# S3D ships no furnished mesh, so query and candidates are both floorplan_closed.
# That is the matched-geometry condition, not the deployment condition Replica
# reports, and the tables must say so.
set -u
cd "$(dirname "$0")/.."
SC=$(ls outputs/acoustic_grid_s3d/*.npz | grep -v shard | xargs -n1 basename | sed 's/.npz//')
exec /opt/conda/envs/f3loc/bin/python scripts/extract_mode_table.py --gpu 2 \
  --dataset-root /root/storage/echoloc_dataset/s3d \
  --collections s3d --scenes $SC --condition floorplan_closed \
  --grid-dir outputs/acoustic_grid_s3d --backbone f3loc_mono \
  --checkpoint outputs/visual/s3d/mono_lr3e4/mono.ckpt \
  --feature stft_band --nfft 256 --hop 64 --f-w 0.5959 \
  --tag _s3d --n-poses 20
