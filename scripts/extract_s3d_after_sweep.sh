#!/usr/bin/env bash
# Wait for the ray-fan sweep to finish, take the setting it selects, and
# re-extract every Structured3D scene with it.
#
# Why the sweep exists. The matcher compares V rays spaced ten degrees apart
# against consecutive orientation bins of the DESDF, so V rays span (V-1)*10
# degrees and that span has to fit inside the camera. Replica and Matterport3D
# render at 106 degrees, where eleven rays span 100 and fit. Structured3D
# renders at 80, where the outer rays of an eleven-ray fan land outside the
# image, the depth interpolation returns NaN for them, and localisation
# collapses: recall at one metre read 1.3 per cent. Seven rays span sixty
# degrees and fit with room to spare.
#
# The alternative of keeping eleven rays and pretending the camera is wider
# (F_W = 0.375) also avoids the NaN, but it reads each ray from the wrong image
# column and so mislabels the ray angles against the DESDF bins. The sweep is
# what decides between them on measured recall rather than on argument, and the
# winning setting is written into the filename of the tables it produces.
set -u
cd "$(dirname "$0")/.."
SWEEP=${1:?path to the sweep output}

while pgrep -f "extract_mode_table.py.*--tag _sweep" > /dev/null; do sleep 30; done
echo "[s3d] sweep done $(date +%H:%M)"
cat "$SWEEP"

read -r V FW < <(/opt/conda/envs/f3loc/bin/python - "$SWEEP" <<'PY'
import re, sys
best, bs = None, -1.0
for ln in open(sys.argv[1]):
    m = re.search(r"V=\s*(\d+) F_W=([\d.]+)\s+n=\s*\d+\s+.*?@1m\s+([\d.]+)%", ln)
    if m and float(m.group(3)) > bs:
        best, bs = (m.group(1), m.group(2)), float(m.group(3))
print(best[0], best[1])
PY
)
echo "[s3d] selected V=$V F_W=$FW"

SC=$(ls outputs/acoustic_grid_s3d/*.npz | grep -v shard | xargs -n1 basename | sed 's/.npz//')
exec /opt/conda/envs/f3loc/bin/python scripts/extract_mode_table.py --gpu 3 \
  --dataset-root /root/storage/echoloc_dataset/s3d \
  --collections s3d --scenes $SC --condition floorplan_closed \
  --grid-dir outputs/acoustic_grid_s3d --backbone f3loc_mono \
  --checkpoint outputs/visual/s3d/mono_lr3e4/mono.ckpt \
  --feature stft_band --nfft 256 --hop 64 \
  --n-rays "$V" --f-w "$FW" --tag _s3d --n-poses 20
