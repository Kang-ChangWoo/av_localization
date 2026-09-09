#!/usr/bin/env bash
# End-to-end reproduction of the F3Loc visual observation models.
#
#   bash scripts/reproduce_visual.sh [--skip-warm]
#
# Runs the three training stages in dependency order (comp needs mono and mv),
# then evaluates every network variant on both Gibson collections. Logs and
# checkpoints land under outputs/. GPU 0 is left free throughout.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/opt/conda/envs/f3loc/bin/python}"
GPUS="${GPUS:-1,2,3,4,5,6,7}"
EVAL_GPU="${EVAL_GPU:-1}"
OUT=outputs
mkdir -p "$OUT/logs"

run() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@" 2>&1 | tee "$OUT/logs/${name}.log" | grep -vE "^(Epoch|Training|Validation|Sanity)" || true
}

if [[ "${1:-}" != "--skip-warm" ]]; then
  run warm "$PYTHON" scripts/warm_dataset_cache.py --datasets gibson_f gibson_g --splits train val
fi

run train_mono "$PYTHON" scripts/train_visual.py --net mono --config configs/visual_mono.yaml --gpus "$GPUS"
run train_mv   "$PYTHON" scripts/train_visual.py --net mv   --config configs/visual_mv.yaml   --gpus "$GPUS"
run train_comp "$PYTHON" scripts/train_visual.py --net comp --config configs/visual_comp.yaml --gpus "$GPUS" \
  --mono-ckpt "$OUT/mono/mono.ckpt" --mv-ckpt "$OUT/mv/mv.ckpt"

for dataset in gibson_f gibson_g; do
  for net in mono mv comp; do
    run "eval_${net}_${dataset}" "$PYTHON" scripts/eval_visual.py --net "$net" --dataset "$dataset" \
      --ckpt "$OUT/${net}/${net}.ckpt" --gpu "$EVAL_GPU" --out "$OUT/metrics/${net}_${dataset}.json"
  done
  run "eval_comp_s_${dataset}" "$PYTHON" scripts/eval_visual.py --net comp_s --dataset "$dataset" \
    --mono-ckpt "$OUT/mono/mono.ckpt" --mv-ckpt "$OUT/mv/mv.ckpt" \
    --gpu "$EVAL_GPU" --out "$OUT/metrics/comp_s_${dataset}.json"
done

echo "=== metrics ==="
for f in "$OUT"/metrics/*.json; do echo "--- $f"; cat "$f"; done
