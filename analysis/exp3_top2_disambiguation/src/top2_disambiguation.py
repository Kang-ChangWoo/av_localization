#!/usr/bin/env python3
"""Can acoustics resolve the strongest visual ambiguity? Top-2 forced choice.

Main protocol. After NMS, take only the two hypotheses vision ranks highest,
h_(1) and h_(2), and keep the samples where exactly one of them is within 1 m
of the truth (the *decidable* samples). On each, two orderings are scored:

    acoustic right   alpha_correct > alpha_incorrect, the acoustic evidence of
                     each hypothesis alone; no visual prior, no gate, no fusion
    visual right     v_correct > v_incorrect, i.e. vision's top-1 is the right
                     one; this is what the backbone would answer on its own

so the question is literally "the two places vision cannot tell apart: can
sound?". An exact tie in either score counts as half right (a coin flip),
which matters on Structured3D, where a fifth of the samples have two
hypotheses with identical posterior. Reported for every benchmark and backbone, on all decidable samples
and on the hardest fifth of them, the samples whose m_v = v_(1) - v_(2) is in
the most ambiguous quintile of the whole test set (equal-count groups by rank;
the boundary is analysis-only). Intervals are bootstraps over samples; one
sample is one pair here, so this is also a pair-level interval.

Supplementary protocol, kept for robustness. All pairs (c, i) with c a
correct and i an incorrect hypothesis among the ten, every sample weighted
equally over its pairs, bootstrap over samples. It has many more trials and
tighter intervals but includes easy negatives; it is reported after the main
table and not in its place.

    python src/top2_disambiguation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import DS, BB, NAME, C, load_rows, quintile_of, boot_mean, boot_diff, ci, write_md  # noqa: E402


def win(x, y):
    """1 if x ranks first, 0 if y does, 0.5 on an exact tie."""
    return 1.0 if x > y else 0.0 if x < y else 0.5


def winm(X, Y):
    return np.where(X > Y, 1.0, np.where(X < Y, 0.0, 0.5))


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    J, L = {}, []
    W = L.append
    W("## Table 1. Main protocol: the two strongest visual hypotheses, exactly one within 1 m\n")
    W("| benchmark | backbone | subset | decidable samples | acoustic right [95% CI] | visual right [95% CI] | acoustic − visual [95% CI] |")
    W("|---|---|---|---|---|---|---|")
    S = ["## Table 2. Supplementary protocol: all correct × incorrect pairs among the ten, samples weighted equally\n",
         "| benchmark | backbone | subset | samples | pairs | acoustic right [95% CI] | visual right [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    curves = {}
    for ds, _ in DS:
        for bb, _ in BB:
            rows, cfg = load_rows(ds, bb)
            if rows is None:
                continue
            q = quintile_of([r["m_v"] for r in rows])
            # ---- main: top-2 decidable
            dec = []
            for r, qq in zip(rows, q):
                if r["d"].size < 2:
                    continue
                t = np.argsort(-r["v"])[:2]; ok = r["d"][t] < 1
                if ok.sum() != 1:
                    continue
                c, i = (t[0], t[1]) if ok[0] else (t[1], t[0])
                dec.append(dict(ac=win(r["a"][c], r["a"][i]), vi=win(r["v"][c], r["v"][i]), q=int(qq), m_v=r["m_v"]))
            J[f"{ds}/{bb}"] = dict(main={}, supp={})
            for name, sel in (("all decidable", [d for d in dec]), ("hardest fifth (Q1 of m_v)", [d for d in dec if d["q"] == 1]),
                              ("Q1–Q2", [d for d in dec if d["q"] <= 2])):
                if not sel:
                    continue
                a = np.array([d["ac"] for d in sel]); v = np.array([d["vi"] for d in sel])
                am, aci = boot_mean(a); vm, vci = boot_mean(v); dm, dci = boot_diff(v, a)
                J[f"{ds}/{bb}"]["main"][name] = dict(n=len(sel), acoustic=am, acoustic_ci=aci, visual=vm, visual_ci=vci, diff=dm, diff_ci=dci)
                W(f"| {NAME[ds]} | {NAME[bb]} | {name} | {len(sel)} | {ci(am, aci)} | {ci(vm, vci)} | {100*dm:+.1f} [{100*dci[0]:+.1f}, {100*dci[1]:+.1f}] |")
            # curve: accuracy on decidable samples with m_v below a threshold
            mv = np.array([d["m_v"] for d in dec]); a = np.array([d["ac"] for d in dec]); v = np.array([d["vi"] for d in dec])
            ths = np.quantile(mv, np.linspace(0.1, 1.0, 10))
            curves[f"{ds}/{bb}"] = [(float(t), float(a[mv <= t].mean()), float(v[mv <= t].mean()), int((mv <= t).sum())) for t in ths]
            # ---- supplementary: all C x I pairs
            per = []
            for r, qq in zip(rows, q):
                good = np.nonzero(r["d"] < 1)[0]; bad = np.nonzero(r["d"] >= 1)[0]
                if good.size and bad.size:
                    A = winm(r["a"][good][:, None], r["a"][bad][None, :]); V = winm(r["v"][good][:, None], r["v"][bad][None, :])
                    per.append((float(A.mean()), float(V.mean()), int(A.size), int(qq)))
            per = np.array(per)
            for name, sel in (("all", np.ones(len(per), bool)), ("Q1 of m_v", per[:, 3] == 1)):
                P = per[sel]
                if len(P) == 0:
                    continue
                am, aci = boot_mean(P[:, 0]); vm, vci = boot_mean(P[:, 1])
                J[f"{ds}/{bb}"]["supp"][name] = dict(samples=int(len(P)), pairs=int(P[:, 2].sum()), acoustic=am, acoustic_ci=aci, visual=vm, visual_ci=vci)
                S.append(f"| {NAME[ds]} | {NAME[bb]} | {name} | {len(P)} | {int(P[:, 2].sum())} | {ci(am, aci)} | {ci(vm, vci)} |")
            print(f"[{ds}/{bb}] decidable {len(dec)}", flush=True)
    L += [""] + S
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "top2_disambiguation.json").write_text(json.dumps(dict(results=J, curves=curves), indent=1))
    write_md(HERE / "md" / "top2_disambiguation.md", "Can acoustics resolve the strongest visual ambiguity?", L)

    # ------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, (ds, name) in zip(axes, DS):
        for j, (bb, lab) in enumerate(BB):
            k = f"{ds}/{bb}"
            if k not in curves:
                continue
            cv = curves[k]; x = [c[3] for c in cv]
            ax.plot(x, [100 * c[1] for c in cv], "o-", color=C["acoustic"], alpha=0.45 + 0.27 * j, ms=3, label=f"acoustic, {lab}" if ds == "replica" else None)
            ax.plot(x, [100 * c[2] for c in cv], "s--", color=C["vision"], alpha=0.45 + 0.27 * j, ms=3, label=f"visual, {lab}" if ds == "replica" else None)
        ax.axhline(50, color="#999", lw=0.6); ax.set_title(name)
        ax.set_xlabel("decidable samples included, hardest first (by m_v)")
    axes[0].set_ylabel("correct one of the top-2 chosen (%)"); axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); (HERE / "figs").mkdir(exist_ok=True); fig.savefig(HERE / "figs" / "top2_disambiguation.png", dpi=170)
    print("wrote md/top2_disambiguation.md, data/, figs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
