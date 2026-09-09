#!/usr/bin/env python3
"""Draw the audio-visual localization pipeline, with measured numbers on it.

A flowchart of boxes is not worth much on its own. This one carries the result
at each stage, so where the information is lost is visible in the same picture
as the data flow: the two branches, what each produces, and what the measured
rank of the true pose is at every point they meet.

    python scripts/plot_pipeline_flowchart.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT = "#e3f2fd"
VISION = "#e8f5e9"
AUDIO = "#fff3e0"
FUSE = "#f3e5f5"
DEAD = "#ffebee"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, title, body="", fc="#ffffff", ec="#37474f", fs=9, lw=1.2, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top",
                fontsize=fs, fontweight="bold", zorder=3)
        if body:
            ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top",
                    fontsize=fs - 1.4, color="#37474f", zorder=3, linespacing=1.45)

    def arrow(x1, y1, x2, y2, label="", color="#455a64", ls="-", rad=0.0, fs=7.5):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                     lw=1.4, color=color, ls=ls, zorder=1,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.0, label, ha="center", va="bottom",
                    fontsize=fs, color=color, style="italic", zorder=3)

    ax.text(50, 98, "Audio-visual floorplan localization: where the information goes",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 94.6, "boxes carry the measured result at that stage   |   "
                      "median rank of the true pose among ~700 valid candidates",
            ha="center", fontsize=9, color="#546e7a")

    # ---- inputs
    box(2, 80, 20, 11, "RGB image", "one frame, 106 deg FOV\nGibson / Replica render", fc=INPUT)
    box(26, 80, 20, 11, "2D floorplan", "map.png, 0.01 m/px\nfree space is exactly 255", fc=INPUT)
    box(50, 80, 22, 11, "active acoustic obs.", "6-mic ring, r = 5 cm\n8 kHz, co-located source",
        fc=INPUT)
    box(76, 80, 22, 11, "shared pose grid", "PoseGrid 0.1 m/cell, 36 yaw\nvalid_pose_mask",
        fc=INPUT, ec="#0277bd")

    # ---- vision branch
    box(2, 62, 22, 13, "F3Loc depth net", "mono / mv / comp\n40 or 160 rays, z-depth", fc=VISION)
    box(2, 44, 22, 13, "DESDF matching", "exp(-L1 / 40) over\n(H, W, 36) pose volume", fc=VISION)
    box(2, 26, 22, 13, "visual posterior", "GT rank median 278\nrecall at 1 m 39.3%", fc=VISION,
        ec="#2e7d32", lw=2.0)
    arrow(13, 80, 13, 75.5)
    arrow(13, 62, 13, 57.5)
    arrow(13, 44, 13, 39.5)
    arrow(36, 80, 22, 57.5, "range field", rad=-0.15)

    # ---- audio branch, prediction side
    box(28, 62, 22, 13, "ray-cast wall returns", "72 rays, first hit\nverified vs DESDF: r = 0.9999",
        fc=AUDIO)
    box(28, 44, 22, 13, "predicted events", "tau = 2d/c - (r/c + guard/fs)\n17 merged arrivals per cell",
        fc=AUDIO)

    # ---- audio branch, observation side
    box(54, 62, 22, 13, "reflection window", "peak + 16 samples, 1024 long\n0.25 ms bins", fc=AUDIO)
    box(54, 44, 22, 13, "observed events", "local maxima, 0.5 ms apart\nwithin 10 dB: ~14 arrivals",
        fc=AUDIO)
    arrow(39, 80, 39, 75.5)
    arrow(39, 62, 39, 57.5)
    arrow(63, 80, 63, 75.5)
    arrow(63, 62, 63, 57.5)

    # ---- matching
    box(26, 23, 48, 14, "one-sided event matching",
        "K = exp(-0.5 ((tau_i - tau_j)/sigma)^2) . conf_j,   sigma = 1 ms\n"
        "m_i = max_j K   after a uniqueness constraint;   score = weighted mean of top q\n"
        "only predicted events are scored, so unexplainable echoes cost nothing",
        fc=AUDIO, ec="#ef6c00", lw=2.0)
    arrow(39, 44, 39, 38.5)
    arrow(63, 44, 63, 38.5)

    box(26, 6, 22, 13, "acoustic 2D score", "GT rank 342 of ~700\nvs 411 for symmetric L1",
        fc=AUDIO, ec="#ef6c00", lw=2.0)
    arrow(37, 23, 37, 19.5)

    # ---- fusion
    box(2, 6, 22, 13, "vision + acoustic", "rank-normalised, log space\nrecall 39.3% -> 22.7%",
        fc=DEAD, ec="#c62828", lw=2.0)
    arrow(13, 26, 13, 19.5)
    arrow(26, 12.5, 24.5, 12.5, "", color="#c62828")

    # ---- the ceiling
    box(52, 6, 22, 13, "same matching, rendered grid",
        "GT rank 2 to 6,  percentile 95.9\nneeds a per-scene render:\na diagnostic, not a method",
        fc="#e0f2f1", ec="#00695c", lw=2.0, ls="--")
    arrow(63, 23, 63, 19.5, color="#00695c", ls="--")

    # ---- the verdict strip
    box(80, 6, 18, 69, "what this says",
        "\nThe comparison was\nthe wrong shape and\nis now fixed:\n"
        "  L1        rank 411\n  one-sided rank 342\n\n"
        "The tracer is right\nto 6 mm against DESDF.\n\n"
        "The clock was wrong\nby 2.15 ms, more than\ntwice the tolerance.\n"
        "Fixed and tested.\n\n"
        "Yet with the room\nrendered from the same\nwall mask, no clutter\n"
        "at all, the truth is\nstill at rank 199.\n\n"
        "So the limit is the\n2D delay-only\nrepresentation, not\n"
        "the clutter and not\nthe geometry.",
        fc="#fafafa", ec="#37474f", fs=9)

    ax.text(50, 3.5,
            "Dashed box is a diagnostic, not a method.  Red path is measured and negative.  "
            "Every acoustic parameter is in physical units and was fixed before evaluation.",
            ha="center", fontsize=8.5, color="#546e7a", style="italic")

    args = parse_args()
    out = args.out or REPO_ROOT / "outputs" / "figures" / "pipeline_flowchart.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
