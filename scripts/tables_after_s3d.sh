#!/usr/bin/env bash
# Wait for the Structured3D re-extraction, then produce every table that reads
# its mode tables: the per-rule fusion comparison, and the LaTeX and Markdown
# the paper uses.
#
# Structured3D is held out by room rather than by pose collection, because it
# ships one collection over many rooms where Replica ships two collections over
# the same three. That is the stricter of the two splits and the flag says so.
set -u
cd "$(dirname "$0")/.."
P=/opt/conda/envs/f3loc/bin/python

while pgrep -f "extract_mode_table.py.*--tag _s3d" > /dev/null; do sleep 30; done
echo "[tables] extraction done $(date +%H:%M)"
wc -l outputs/analysis/*_floorplan_closed_f3loc_mono_s3d.csv

$P scripts/mode_fusion_eval.py --backbone f3loc_mono_s3d \
   --primary-condition floorplan_closed --split-by scene \
   --out outputs/analysis/fusion_s3d.md \
   --json-out outputs/metrics/mode_fusion_eval_s3d.json
echo "[tables] fusion comparison written $(date +%H:%M)"

$P scripts/paper_artifacts.py
echo "[tables] paper tables written $(date +%H:%M)"
