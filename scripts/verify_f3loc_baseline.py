#!/usr/bin/env python3
"""Validate an external F3Loc baseline without copying it into AV-FPLoc."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PINNED = "9e8027d9219ca505078283ebfd925580f476ab97"


def revision(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3loc-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()
    head = revision(args.f3loc_root)
    required = {
        "eval_observation.py": args.f3loc_root / "eval_observation.py",
        "Gibson data": args.dataset_root / "gibson_g",
        "DESDF": args.dataset_root / "desdf",
        "F3Loc checkpoint": args.checkpoint_root / "comp.ckpt",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    report = {"f3loc_root": str(args.f3loc_root), "head": head, "pinned_head": PINNED,
              "revision_matches": head == PINNED, "required": {name: str(path) for name, path in required.items()},
              "missing": missing, "ready_for_visual_likelihood": head == PINNED and not missing}
    print(json.dumps(report, indent=2))
    return 0 if report["ready_for_visual_likelihood"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
