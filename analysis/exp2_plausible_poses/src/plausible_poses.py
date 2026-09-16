#!/usr/bin/env python3
"""Does acoustic verification resolve visual ambiguity? By number of plausible poses.

For every test sample, a hypothesis is *visually plausible* when its log
posterior is within the gate's own threshold of the strongest one:

    plausible(k)  <=>  v_(1) - v_k <= tau_v

with tau_v the visual threshold the validation rooms chose for this benchmark
and backbone (the same number the gate uses), so nothing here is tuned on the
test rooms. N_plausible is the count of plausible hypotheses, grouped as
1, 2, 3, >=4. Per group and per benchmark and backbone:

    vision, +audio, gain      recall at 1 m of the visual top-1 and of the
                              fused pick, with a paired bootstrap over samples
    repair / break            vision wrong -> fused right, and the reverse

and, the decisive subset, samples whose plausible set contains a correct
hypothesis (one within 1 m of the truth): there vision has already proposed
the right place and the question is only which of the plausible ones to pick.
On that subset the table reports the selection accuracy of vision (top-1),
of the fused rule, and of the acoustic evidence alone choosing among the
plausible hypotheses (no visual prior, no gate).

    python src/plausible_poses.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import DS, BB, NAME, C, load_rows, boot_mean, boot_diff, ci, pct, write_md  # noqa: E402

GROUPS = [(1, "1"), (2, "2"), (3, "3"), (4, "≥4")]


def n_plausible(r, tau_v):
    return int(np.sum(r["v"][0] - r["v"] <= tau_v)) if r["v"].size else 0


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    J, L = {}, []
    W = L.append
    W("## Table 1. By number of visually plausible poses (v_(1) - v_k <= tau_v, tau_v from validation)\n")
    W("| benchmark | backbone | tau_v | N_plausible | samples | vision | +audio | gain [95% CI] | repair | break |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    T2 = ["## Table 2. Samples whose plausible set contains a correct hypothesis: which one is picked?\n",
          "| benchmark | backbone | N_plausible | samples | GT in plausible set | vision picks it | audio alone picks it | fused picks it | fused − vision [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds, _ in DS:
        for bb, _ in BB:
            rows, cfg = load_rows(ds, bb)
            if rows is None:
                continue
            tau = float(cfg.tau_v)
            npl = np.array([min(n_plausible(r, tau), 4) for r in rows])
            J[f"{ds}/{bb}"] = dict(tau_v=tau, groups={})
            for g, lab in GROUPS:
                sel = [r for r, n in zip(rows, npl) if n == g]
                if not sel:
                    continue
                v = np.array([r["e_vis"] < 1 for r in sel], float); o = np.array([r["e_ours"] < 1 for r in sel], float)
                gain, gci = boot_diff(v, o)
                rep = int(((v == 0) & (o == 1)).sum()); brk = int(((v == 1) & (o == 0)).sum())
                # the decisive subset: a correct hypothesis is among the plausible ones
                inset = []
                for r in sel:
                    pl = np.nonzero(r["v"][0] - r["v"] <= tau)[0]
                    if (r["d"][pl] < 1).any():
                        inset.append(r)
                if inset:
                    vis_ok = np.array([r["e_vis"] < 1 for r in inset], float)
                    fus_ok = np.array([r["e_ours"] < 1 for r in inset], float)
                    ac_ok = []
                    for r in inset:
                        pl = np.nonzero(r["v"][0] - r["v"] <= tau)[0]
                        ac_ok.append(float(r["d"][pl[int(np.argmax(r["a"][pl]))]] < 1))
                    ac_ok = np.array(ac_ok)
                    dm, dci = boot_diff(vis_ok, fus_ok)
                    T2.append(f"| {NAME[ds]} | {NAME[bb]} | {lab} | {len(sel)} | {len(inset)} ({100*len(inset)/len(sel):.0f}%) | "
                              f"{pct(vis_ok.mean())} | {pct(ac_ok.mean())} | {pct(fus_ok.mean())} | {100*dm:+.1f} [{100*dci[0]:+.1f}, {100*dci[1]:+.1f}] |")
                    sub = dict(n=len(inset), vision=float(vis_ok.mean()), audio=float(ac_ok.mean()), fused=float(fus_ok.mean()), diff=dm, diff_ci=dci)
                else:
                    sub = None
                J[f"{ds}/{bb}"]["groups"][lab] = dict(n=len(sel), vision=float(v.mean()), ours=float(o.mean()), gain=gain, gain_ci=gci,
                                                     repair=rep, brk=brk, gt_in_set=sub)
                W(f"| {NAME[ds]} | {NAME[bb]} | {tau:g} | {lab} | {len(sel)} | {pct(v.mean())} | {pct(o.mean())} | "
                  f"{100*gain:+.1f} [{100*gci[0]:+.1f}, {100*gci[1]:+.1f}] | {rep} | {brk} |")
            print(f"[{ds}/{bb}] tau_v={tau:g} N_plausible counts {np.bincount(npl, minlength=5)[1:]}", flush=True)
    L += [""] + T2
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "plausible_poses.json").write_text(json.dumps(J, indent=1))
    write_md(HERE / "md" / "plausible_poses.md", "Does acoustic verification resolve visual ambiguity?", L)

    # ------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    for col, (ds, name) in enumerate(DS):
        ax = axes[0, col]
        for j, (bb, lab) in enumerate(BB):
            k = f"{ds}/{bb}"
            if k not in J:
                continue
            G = J[k]["groups"]; x = np.arange(len(GROUPS)) + (j - 1) * 0.27
            gains = [100 * G[lb]["gain"] if lb in G else np.nan for _, lb in GROUPS]
            err = [[100 * (G[lb]["gain"] - G[lb]["gain_ci"][0]) if lb in G else 0 for _, lb in GROUPS],
                   [100 * (G[lb]["gain_ci"][1] - G[lb]["gain"]) if lb in G else 0 for _, lb in GROUPS]]
            ax.bar(x, gains, 0.27, yerr=err, capsize=2, label=lab, alpha=0.9)
        ax.axhline(0, color="#999", lw=0.6); ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([lb for _, lb in GROUPS])
        ax.set_title(name); ax.set_xlabel("N plausible poses")
        ax = axes[1, col]
        for j, (bb, lab) in enumerate(BB):
            k = f"{ds}/{bb}"
            if k not in J:
                continue
            G = J[k]["groups"]; x = np.arange(len(GROUPS)) + (j - 1) * 0.27
            sub = [G[lb]["gt_in_set"] if lb in G and G[lb]["gt_in_set"] else None for _, lb in GROUPS]
            ax.plot(x, [100 * s["vision"] if s else np.nan for s in sub], "s", color=C["vision"], alpha=0.45 + 0.27 * j, ms=5, label="vision" if j == 1 else None)
            ax.plot(x, [100 * s["audio"] if s else np.nan for s in sub], "^", color=C["acoustic"], alpha=0.45 + 0.27 * j, ms=5, label="audio alone" if j == 1 else None)
            ax.plot(x, [100 * s["fused"] if s else np.nan for s in sub], "o", color=C["fused"], alpha=0.45 + 0.27 * j, ms=5, label="fused" if j == 1 else None)
        ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([lb for _, lb in GROUPS]); ax.set_ylim(0, 100)
        ax.set_xlabel("N plausible poses (GT among them)")
    axes[0, 0].set_ylabel("gain from sound, points"); axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].set_ylabel("picks the correct plausible pose (%)"); axes[1, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); (HERE / "figs").mkdir(exist_ok=True); fig.savefig(HERE / "figs" / "plausible_poses.png", dpi=170)
    print("wrote md/plausible_poses.md, data/plausible_poses.json, figs/plausible_poses.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
