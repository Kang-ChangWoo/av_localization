#!/usr/bin/env python3
"""Run every evaluation a finished training job owes, in one process.

Split out of ``run_queue.py`` so evaluation is a schedulable unit of work rather
than something the scheduler blocks on: a job's four evaluations (two
collections x two BatchNorm protocols) then run on one GPU while other jobs keep
training on the others.

    python scripts/eval_job.py --run-name mono_faithful --net mono \
        --datasets gibson_f gibson_g --bn-modes upstream eval --gpu 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def metrics_filename(run_name: str, dataset: str, bn_mode: str) -> str:
    suffix = "" if bn_mode == "eval" else f"_{bn_mode}"
    return f"{run_name}_{dataset}{suffix}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--net", required=True, choices=["mono", "mv", "comp", "comp_s"])
    parser.add_argument("--datasets", nargs="+", default=["gibson_f", "gibson_g"])
    parser.add_argument("--bn-modes", nargs="+", default=["upstream"])
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--dataset-root", default=None,
                        help="collection root; defaults to eval_visual.py's own default")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    ckpt = args.output_dir / args.run_name / f"{args.net}.ckpt"
    if not ckpt.exists():
        print(f"[eval_job] missing checkpoint {ckpt}")
        return 1

    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for dataset in args.datasets:
        for bn_mode in args.bn_modes:
            out_json = metrics_dir / metrics_filename(args.run_name, dataset, bn_mode)
            if args.skip_existing and out_json.exists():
                print(f"[eval_job] {out_json.name} present, skipping")
                continue
            cmd = [
                sys.executable, "scripts/eval_visual.py",
                "--net", args.net,
                "--dataset", dataset,
                "--ckpt", str(ckpt),
                "--gpu", args.gpu,
                "--bn-mode", bn_mode,
                "--out", str(out_json),
            ]
            # upstream's protocol is defined at batch size 1
            if bn_mode == "upstream":
                cmd += ["--batch-size", "1"]
            if args.dataset_root:
                cmd += ["--dataset-root", args.dataset_root]
            print(f"[eval_job] {args.run_name} {dataset} bn={bn_mode}")
            result = subprocess.run(cmd, cwd=REPO_ROOT)
            if result.returncode != 0:
                failures += 1
                print(f"[eval_job] FAILED {dataset} bn={bn_mode} (rc={result.returncode})")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
