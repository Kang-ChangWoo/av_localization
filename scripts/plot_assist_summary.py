#!/usr/bin/env python3
"""One figure for the feasibility question, across every scene.

Vision is the reference everywhere: the black line is what it achieves alone,
and each acoustic model is only allowed to reorder vision's own shortlist. The
gap between the coloured curves and the grey oracle is the part of the headroom
the acoustic model fails to use -- which is the number that says what to fix.

    python scripts/plot_assist_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STYLE = {"A": ("#8d99ae", "A  2D walls"),
         "A+": ("#457b9d", "A+  floor/ceiling"),
         "B": ("#f4a261", "B  image sources"),
         "C": ("#e63946", "C  SoundSpaces")}
SCENE_LABEL = {"office_4": "office_4  (bare single room)",
               "apartment_2": "apartment_2  (furnished, scan voids)",
               "frl_apartment_5": "frl_apartment_5  (furnished, cluttered)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["office_4", "apartment_2", "frl_apartment_5"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    M = REPO_ROOT / "outputs" / "metrics"
    data = {}
    for s in args.scenes:
        f = M / f"vision_assist_{s}.json"
        if f.exists():
            data[s] = json.loads(f.read_text())
    if not data:
        print("no vision_assist_*.json found")
        return 1

    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 9),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, (scene, d) in enumerate(data.items()):
        K = np.array(d["K"])
        base = d["vision_recall_1m"]

        ax = axes[0, j]
        ax.fill_between(K, 100 * base, 100 * np.array(d["oracle_recall_1m"]),
                        color="grey", alpha=0.12)
        ax.plot(K, 100 * np.array(d["oracle_recall_1m"]), "--", color="grey", lw=1.3,
                label="oracle: best candidate in the shortlist")
        ax.axhline(100 * base, color="black", lw=2.5,
                   label=f"vision alone  {100*base:.0f}%")
        for k, v in d["models"].items():
            c, lab = STYLE[k]
            ax.plot(K, 100 * np.array(v), "-o", ms=3.5, color=c, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("shortlist size K  (vision's top-K)")
        if j == 0:
            ax.set_ylabel("recall @ 1 m  [%]")
        ax.set_ylim(0, 102); ax.grid(alpha=0.3)
        ax.set_title(f"{SCENE_LABEL.get(scene, scene)}\n{d['poses']} poses, "
                     f"median {d['vision_median_m']:.2f} m", fontsize=10)
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper left")

        # how much of the available headroom each model actually uses
        ax = axes[1, j]
        names, gains, used = [], [], []
        for k, v in d["models"].items():
            v = np.array(v)
            i = int(np.argmax(v))
            head = d["oracle_recall_1m"][i] - base
            names.append(k)
            gains.append(100 * (v[i] - base))
            used.append(100 * (v[i] - base) / head if head > 1e-9 else 0.0)
        x = np.arange(len(names))
        bars = ax.bar(x, gains, color=[STYLE[k][0] for k in names])
        for i, (b, g, u) in enumerate(zip(bars, gains, used)):
            ax.text(i, g, f"{g:+.0f}pp\n({u:.0f}% of headroom)", ha="center",
                    va="bottom" if g >= 0 else "top", fontsize=8)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        if j == 0:
            ax.set_ylabel("best gain over vision alone  [pp]")
        ax.grid(alpha=0.3, axis="y")
        lim = max(6, max(gains) * 1.5 if gains else 6)
        ax.set_ylim(min(0, min(gains) * 1.4 - 1), lim)

    fig.suptitle("Can sound narrow vision's hypotheses?  Recording made in the real scanned room; "
                 "every candidate model knows only the 2D floorplan.\n"
                 "Acoustics may only reorder vision's shortlist, so it can never make the answer worse "
                 "than vision alone.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = args.out or REPO_ROOT / "outputs" / "viz" / "vision_assist_all_scenes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")

    # a compact text table alongside the figure
    print(f"\n{'scene':22s} {'vision':>7s} " + " ".join(f"{k:>12s}" for k in STYLE))
    for scene, d in data.items():
        base = d["vision_recall_1m"]
        cells = []
        for k in STYLE:
            if k not in d["models"]:
                cells.append(f"{'-':>12s}"); continue
            v = np.array(d["models"][k]); i = int(np.argmax(v))
            cells.append(f"{100*v[i]:5.0f}% ({100*(v[i]-base):+3.0f})")
        print(f"{scene:22s} {100*base:6.0f}% " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
