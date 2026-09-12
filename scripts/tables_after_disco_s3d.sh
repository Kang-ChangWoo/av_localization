#!/usr/bin/env bash
# Rebuild every Structured3D table once the DisCo-FLoc extraction lands.
#
# The wait matches the extractor by its checkpoint path rather than by its tag,
# because a pattern containing the tag also matches the shell that is waiting on
# it and the loop would fall straight through. That has cost a set of tables
# once already: they were generated from the pre-fix extraction.
set -u
cd "$(dirname "$0")/.."
P=/opt/conda/envs/f3loc/bin/python

while pgrep -f "extract_mode_table.py.*--backbone disco_rrp" > /dev/null; do sleep 30; done
echo "[tables] DisCo extraction done $(date +%H:%M)"
ls -la --time-style=+%H:%M outputs/analysis/queries_floorplan_closed_*_s3d.csv

for bb in f3loc_mono_s3d disco_rrp_s3d; do
  [ -f "outputs/analysis/queries_floorplan_closed_${bb}.csv" ] || continue
  $P scripts/mode_fusion_eval.py --backbone "$bb" \
     --primary-condition floorplan_closed --split-by scene \
     --out "outputs/analysis/fusion_${bb}.md" \
     --json-out "outputs/metrics/mode_fusion_eval_${bb}.json"
  echo "[tables] fusion comparison for $bb written $(date +%H:%M)"
done

$P scripts/paper_artifacts.py
echo "[tables] paper tables written $(date +%H:%M)"
