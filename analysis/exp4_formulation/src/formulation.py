#!/usr/bin/env python3
"""Why verification rather than acoustic localization? Candidate-set size and oracle.

For every benchmark and backbone, the acoustic score is asked to choose among
the top-K visual hypotheses on its own (argmax of alpha over the first K),
next to the gated fusion restricted to the same K and the oracle over the
same K (a correct hypothesis exists among them). K runs over 1, 2, 3, 5, 10,
and the last row is the acoustic score over every candidate cell of the
grid, which is acoustic localization with no visual shortlist at all.

If sound could search, acoustic-alone would improve with K and beat its
whole-grid number; if it can only help choose, acoustic-alone degrades with K
while the gated fusion improves, and the oracle says how much of the
shortlist is still left on the table.

    python src/formulation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import DS, BB, NAME, C, load_rows, boot_mean, pct, write_md  # noqa: E402

KS = (1, 2, 3, 5, 10)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from track1_core.likelihood.mode_fusion import choose
    J, L = {}, []
    W = L.append
    W("## Table 1. Recall at 1 m as the candidate set grows\n")
    W("| benchmark | backbone | K | acoustic alone among K | gated fusion over K | oracle over K | headroom (oracle − fused) |")
    W("|---|---|---|---|---|---|---|")
    for ds, _ in DS:
        for bb, _ in BB:
            rows, cfg = load_rows(ds, bb)
            if rows is None:
                continue
            out = []
            for K in KS:
                alone = np.array([r["d"][:K][int(np.argmax(r["a"][:K]))] < 1 for r in rows], float)
                gated = np.array([r["d"][:K][choose(r["v"][:K], r["a"][:K], cfg)[0]] < 1 for r in rows], float)
                orc = np.array([(r["d"][:K] < 1).any() for r in rows], float)
                out.append(dict(K=K, alone=float(alone.mean()), alone_ci=boot_mean(alone)[1], gated=float(gated.mean()),
                                gated_ci=boot_mean(gated)[1], oracle=float(orc.mean())))
                W(f"| {NAME[ds]} | {NAME[bb]} | {K} | {pct(alone.mean())} | {pct(gated.mean())} | {pct(orc.mean())} | {100*(orc.mean()-gated.mean()):.1f} |")
            grid = float(np.mean([r["e_ac"] < 1 for r in rows]))
            cells = int(np.median([r["n_cells"] for r in rows]))
            out.append(dict(K="grid", alone=grid, cells=cells))
            W(f"| {NAME[ds]} | {NAME[bb]} | all cells (median {cells}) | {pct(grid)} | – | – | – |")
            J[f"{ds}/{bb}"] = out
            print(f"[{ds}/{bb}] K=1 {pct(out[0]['alone'])}  K=10 alone {pct(out[4]['alone'])} gated {pct(out[4]['gated'])} oracle {pct(out[4]['oracle'])}  grid {pct(grid)}", flush=True)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "formulation.json").write_text(json.dumps(J, indent=1))
    write_md(HERE / "md" / "formulation.md", "Why verification rather than acoustic localization?", L)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.5), sharex=True)
    for col, (ds, name) in enumerate(DS):
        for rowi, (bb, lab) in enumerate(BB):
            ax = axes[rowi, col]; k = f"{ds}/{bb}"
            if k not in J:
                ax.axis("off"); continue
            cs = [c for c in J[k] if c["K"] != "grid"]; x = [c["K"] for c in cs]
            ax.plot(x, [100 * c["alone"] for c in cs], "^-", color=C["acoustic"], label="acoustic alone among K")
            ax.plot(x, [100 * c["gated"] for c in cs], "s-", color=C["fused"], label="gated fusion over K")
            ax.plot(x, [100 * c["oracle"] for c in cs], ":", color=C["oracle"], label="oracle over K")
            ax.axhline(100 * J[k][-1]["alone"], color=C["acoustic"], lw=0.7, ls="--", label="acoustic alone, whole grid")
            ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.set_title(f"{name}, {lab}", fontsize=8)
            if col == 0:
                ax.set_ylabel("recall @1 m (%)")
            if rowi == 2:
                ax.set_xlabel("K hypotheses")
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); (HERE / "figs").mkdir(exist_ok=True); fig.savefig(HERE / "figs" / "formulation.png", dpi=170)
    print("wrote md/formulation.md, data/, figs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
