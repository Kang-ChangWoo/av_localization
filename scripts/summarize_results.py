#!/usr/bin/env python3
"""Summarise queue results against the published F3Loc numbers.

Reads every metrics JSON under ``outputs/metrics`` plus ``queue_status.json``
and prints one table per Gibson collection, with the paper's recall in the same
row so deviations are visible at a glance.

    python scripts/summarize_results.py
    python scripts/summarize_results.py --markdown > outputs/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Observation-module recalls reported in the CVPR 2024 paper, as fractions.
PAPER = {
    ("mono", "gibson_f"): (0.047, 0.286, 0.366, 0.351),
    ("mv", "gibson_f"): (0.132, 0.409, 0.452, 0.437),
    ("mono", "gibson_g"): (0.043, 0.267, 0.337, 0.323),
    ("mv", "gibson_g"): (0.093, 0.270, 0.310, 0.292),
    ("comp_s", "gibson_g"): (0.105, 0.343, 0.396, 0.380),
    ("comp", "gibson_g"): (0.122, 0.394, 0.445, 0.432),
}
COLUMNS = ("recall_0.1m", "recall_0.5m", "recall_1m", "recall_1m_30deg")
HEADERS = ("0.1m", "0.5m", "1m", "1m/30deg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown report")
    args = parser.parse_args()

    metrics_dir = args.output_dir / "metrics"
    results = []
    for path in sorted(metrics_dir.glob("*.json")):
        data = json.loads(path.read_text())
        stem = path.stem
        if stem.endswith("_upstream"):
            stem = stem[: -len("_upstream")]
        job = stem.rsplit(f"_{data['dataset']}", 1)[0]
        data.setdefault("bn_mode", "eval")
        results.append((job, data))

    status_path = args.output_dir / "queue_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": []}
    by_name = {j["name"]: j for j in status["jobs"]}

    out: list[str] = []
    emit = out.append

    counts: dict[str, int] = {}
    for job in status["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    emit(f"# Visual reproduction results\n" if args.markdown else "VISUAL REPRODUCTION RESULTS")
    emit(f"jobs: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    emit("")

    for dataset in ("gibson_f", "gibson_g"):
        rows = [(job, d) for job, d in results if d["dataset"] == dataset]
        if not rows:
            continue
        emit(f"## {dataset}" if args.markdown else f"--- {dataset} ---")
        if args.markdown:
            emit("| run | net | protocol | " + " | ".join(HEADERS) + " | vs paper 1m |")
            emit("| --- | --- | --- | " + " | ".join("---" for _ in HEADERS) + " | --- |")
        else:
            emit(f"{'run':26s} {'net':6s} {'protocol':9s} " + " ".join(f"{h:>9s}" for h in HEADERS) + "   vs paper 1m")

        # paper reference rows first
        for (net, ds), values in PAPER.items():
            if ds != dataset:
                continue
            label = f"PAPER ({net})"
            cells = [f"{v*100:.1f}%" for v in values]
            if args.markdown:
                emit(f"| _{label}_ | {net} | published | " + " | ".join(cells) + " | — |")
            else:
                emit(f"{label:26s} {net:6s} {'published':9s} " + " ".join(f"{c:>9s}" for c in cells) + "   —")

        for job, data in sorted(rows, key=lambda r: (r[0], r[1].get("bn_mode", "eval"))):
            net = data["net"]
            cells = [f"{data[c]*100:.1f}%" for c in COLUMNS]
            ref = PAPER.get((net, dataset))
            delta = f"{(data['recall_1m'] - ref[2]) * 100:+.1f}pp" if ref else "—"
            state = by_name.get(job, {}).get("status", "?")
            name = job if state in ("done", "?") else f"{job} [{state}]"
            mode = data.get("bn_mode", "eval")
            if args.markdown:
                emit(f"| {name} | {net} | {mode} | " + " | ".join(cells) + f" | {delta} |")
            else:
                emit(f"{name:26s} {net:6s} {mode:9s} " + " ".join(f"{c:>9s}" for c in cells) + f"   {delta}")
        emit("")

    failed = [j["name"] for j in status["jobs"] if j["status"] in ("failed", "skipped")]
    if failed:
        emit(("**Not completed:** " if args.markdown else "NOT COMPLETED: ") + ", ".join(failed))

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
