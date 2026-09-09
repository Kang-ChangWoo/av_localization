#!/usr/bin/env python3
"""Cross-check our evaluation against upstream's, unmodified.

``scripts/eval_visual.py`` re-implements the evaluation loop so inference can be
batched. That rewrite is the one place where a reproduction could flatter itself
without anyone noticing, so this runs the vendored ``eval_observation.py`` --
byte-identical to the pinned upstream revision -- against the same checkpoint
and prints both sets of numbers.

Upstream loads checkpoints through ``depth_net_pl`` / ``mv_depth_net_pl``, whose
constructors do not accept the extra hyperparameters our modules save. The saved
tensors are identical (``tests/test_visual_training.py`` pins the state_dict key
sets), so this writes a translated checkpoint carrying only the arguments
upstream declares.

    python scripts/verify_against_upstream_eval.py --net mono --dataset gibson_f \
        --ckpt outputs/mono_primary/mono.ckpt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "third_party" / "f3loc"

# Constructor arguments each upstream Lightning wrapper actually accepts.
UPSTREAM_HPARAMS = {
    "mono": ("shape_loss_weight", "lr", "d_min", "d_max", "d_hyp", "D", "F_W"),
    "mv": ("D", "d_min", "d_max", "d_hyp", "shape_loss_weight", "F_W"),
}
UPSTREAM_CKPT_NAME = {"mono": "mono.ckpt", "mv": "mv.ckpt"}
UPSTREAM_NET_TYPE = {"mono": "d", "mv": "mvd"}


def translate_checkpoint(src: Path, dst: Path, net: str) -> dict:
    ckpt = torch.load(src, map_location="cpu")
    hparams = dict(ckpt.get("hyper_parameters", {}))
    kept = {k: v for k, v in hparams.items() if k in UPSTREAM_HPARAMS[net]}
    ckpt["hyper_parameters"] = kept
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", default="mono", choices=["mono", "mv"])
    parser.add_argument("--dataset", default="gibson_f", choices=["gibson_f", "gibson_g"])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data" / "f3loc"))
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    workdir = Path(tempfile.mkdtemp(prefix="upstream_eval_"))
    try:
        logs = workdir / "logs"
        kept = translate_checkpoint(args.ckpt, logs / UPSTREAM_CKPT_NAME[args.net], args.net)
        print(f"[verify] translated checkpoint hyperparameters: {kept}")
        print(f"[verify] running vendored eval_observation.py (pinned upstream, unmodified)")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["PYTHONPATH"] = str(VENDOR)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [
            sys.executable, str(VENDOR / "eval_observation.py"),
            "--net_type", UPSTREAM_NET_TYPE[args.net],
            "--dataset", args.dataset,
            "--dataset_path", args.dataset_root,
            "--ckpt_path", str(logs),
        ]
        result = subprocess.run(cmd, cwd=VENDOR, env=env, capture_output=True, text=True)
        sys.stdout.write(result.stdout[-4000:])
        if result.returncode != 0:
            sys.stderr.write(result.stderr[-4000:])
            return result.returncode

        upstream = {
            m.group(1): float(m.group(2))
            for m in re.finditer(r"([\d.]+m(?: \d+ deg)?) recall =\s*([\d.]+)", result.stdout)
        }
        print("\n[verify] upstream eval_observation.py:")
        for key, value in upstream.items():
            print(f"           {key:14s} {value*100:.2f}%")
        print("[verify] compare against outputs/metrics/*.json from scripts/eval_visual.py;"
              " the two should agree to within rounding.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
