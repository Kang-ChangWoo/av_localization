#!/usr/bin/env bash
# Fetch the F3Loc authors' released checkpoints.
#
# Upstream publishes mono.ckpt, mv.ckpt and comp.ckpt in the Google Drive folder
# linked from its README at the pinned revision. They are ~420 MB together and
# are not committed here. The project's Lightning wrappers name their submodules
# after upstream's, so these load with load_from_checkpoint unchanged.
#
#   bash scripts/fetch_official_checkpoints.sh [dest]
set -euo pipefail
DEST="${1:-outputs/official_ckpt}"
FOLDER="https://drive.google.com/drive/folders/1-TDlM9hjeODizeebgfPx7zWez0XPYCKm"
mkdir -p "$DEST"
python -m gdown --folder "$FOLDER" -O "$DEST"
ls -la "$DEST"
echo
echo "Evaluate against the published table with upstream's own BatchNorm handling:"
echo "  python scripts/eval_visual.py --net mono --dataset gibson_f \\"
echo "      --ckpt $DEST/mono.ckpt --bn-mode upstream"
