#!/usr/bin/env python3
"""Experiment I: does the acoustic score carry relative information at all?

Acoustic-alone recall under a furnished query is 21%, and a reviewer reads that
as "the cue is nearly empty". The end-to-end number cannot answer that, because
it is bounded by the decision rule as much as by the evidence. A forced binary
choice is not: put the correct hypothesis beside one incorrect hypothesis and
ask which the acoustic score prefers. Chance is 50%. If the score is empty, it
stays at 50% in every stratum; if it carries relative information, it rises
above chance even where the end-to-end gain is small.

Four strata, chosen because each is a different reviewer objection:

    visually ambiguous / confident   where the gate would and would not act
    spatially close / far            whether the cue resolves nearby alternatives
                                     or only gross ones

The visual ordering is the competing baseline on the same pairs, so the reader
sees whether acoustics adds to it, agrees with it, or merely repeats it.

    python feasible/run_pairwise.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

BACKBONES = [("f3STFT", "F3Loc mono"), ("unlocSTFT", "UnLoc"), ("discoID", "DisCo-FLoc RRP")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--backbones", nargs="+", default=[b for b, _ in BACKBONES])
    p.add_argument("--ac-col", default="ac_quantile")
    p.add_argument("--vis-col", default="vis_log_lse")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "results" / "I_pairwise.md")
    p.add_argument("--fig", type=Path, default=HERE / "figs" / "I_pairwise.png")
    return p.parse_args()


def read(p: Path) -> dict:
    rows = list(csv.DictReader(open(p, newline="")))
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        if v[0] in ("True", "False"):
            out[k] = np.array([x == "True" for x in v]); continue
        try:
            out[k] = np.array([float(x) if x not in ("", "nan") else np.nan for x in v])
        except ValueError:
            out[k] = np.array(v, dtype=object)
    return out


def main() -> int:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from track1_core.provenance import stamp

    rng = np.random.default_rng(args.seed)

    def boot_mean(x):
        i = rng.integers(0, x.size, size=(args.boot, x.size))
        m = x[i].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    out, js = [], {}
    W = out.append
    W("# I. Pairwise acoustic discrimination\n")
    W("For every query with a correct hypothesis (within 1 m) among its ten, "
      "each incorrect hypothesis is paired with it and the acoustic score is "
      "asked to pick. Chance is 50%. The visual ordering is scored on the same "
      "pairs. Intervals are bootstraps over pairs.\n")
    STRATA = ["all", "visually ambiguous", "visually confident", "spatially close (<2 m)",
              "spatially far (>4 m)"]
    fig, axes = plt.subplots(1, len(args.backbones), figsize=(4.4 * len(args.backbones), 3.8),
                             facecolor="white", squeeze=False)

    for ax, tag in zip(axes[0], args.backbones):
        label = dict(BACKBONES).get(tag, tag)
        mp = args.analysis_dir / f"modes_{args.condition}_{tag}.csv"
        if not mp.exists():
            print(f"[skip] {tag}"); continue
        M = read(mp)
        per: dict = {}
        for i, q in enumerate(M["query_id"]):
            # injected acoustic candidates (mode >= 100, written by the newer extractor) are not part of the visual shortlist
            if int(M["mode"][i]) >= 100:
                continue
            d = per.setdefault(str(q), dict(d=[], a=[], v=[], m=[]))
            d["d"].append(M["dist_gt_m"][i]); d["a"].append(M[args.ac_col][i])
            d["v"].append(M[args.vis_col][i]); d["m"].append(M["mode"][i])
        pairs = []   # (acoustic picks correct, vision picks correct, ambiguity, separation)
        for q, d in per.items():
            o = np.argsort(np.asarray(d["m"], dtype=int))
            dd = np.asarray(d["d"])[o]; aa = np.asarray(d["a"])[o]; vv = np.asarray(d["v"])[o]
            good = np.nonzero(dd < 1.0)[0]
            if good.size == 0:
                continue
            c = int(good[0])                     # the strongest correct hypothesis
            amb = float(vv[0] - vv[1]) if len(vv) > 1 else np.inf
            for j in range(len(dd)):
                if dd[j] < 1.0:
                    continue
                # the two hypotheses' separation in the plane is not stored
                # directly, so it is approximated by the difference of their
                # distances to truth, which is exact when they are collinear
                # with it and a lower bound otherwise
                sep = abs(float(dd[j]) - float(dd[c]))
                pairs.append((aa[c] > aa[j], vv[c] > vv[j], amb, sep))
        P = np.array(pairs, dtype=float)
        if len(P) == 0:
            continue
        ac, vi, amb, sep = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
        thr = np.nanmedian(amb[np.isfinite(amb)])
        masks = {"all": np.ones(len(P), bool),
                 "visually ambiguous": amb < thr,
                 "visually confident": amb >= thr,
                 "spatially close (<2 m)": sep < 2.0,
                 "spatially far (>4 m)": sep > 4.0}
        W(f"\n## {label}\n")
        W(f"{len(per)} queries, {len(P)} pairs. Visual ambiguity is split at its median "
          f"log-odds ({thr:.2f}).\n")
        W("| stratum | pairs | acoustic picks correct | 95% CI | visual ordering | 95% CI |")
        W("|---|---|---|---|---|---|")
        xs, ya, yv, ea, ev = [], [], [], [], []
        for s in STRATA:
            m = masks[s]
            if m.sum() < 10:
                continue
            a_m, a_lo, a_hi = boot_mean(ac[m]); v_m, v_lo, v_hi = boot_mean(vi[m])
            W(f"| {s} | {int(m.sum())} | {100*a_m:.1f}% | [{100*a_lo:.1f}, {100*a_hi:.1f}] | "
              f"{100*v_m:.1f}% | [{100*v_lo:.1f}, {100*v_hi:.1f}] |")
            js.setdefault(label, {})[s] = dict(pairs=int(m.sum()), acoustic=a_m,
                                              acoustic_ci=[a_lo, a_hi], visual=v_m,
                                              visual_ci=[v_lo, v_hi])
            xs.append(s); ya.append(100 * a_m); yv.append(100 * v_m)
            ea.append([100 * (a_m - a_lo), 100 * (a_hi - a_m)])
            ev.append([100 * (v_m - v_lo), 100 * (v_hi - v_m)])
        x = np.arange(len(xs)); wdt = 0.36
        ax.bar(x - wdt / 2, ya, wdt, yerr=np.array(ea).T, color="#1d4ed8", capsize=2,
               label="acoustic score", error_kw=dict(lw=0.8))
        ax.bar(x + wdt / 2, yv, wdt, yerr=np.array(ev).T, color="#9ca3af", capsize=2,
               label="visual ordering", error_kw=dict(lw=0.8))
        ax.axhline(50, color="#111827", ls=":", lw=1.1, label="chance")
        short = {"all": "all", "visually ambiguous": "vis.\nambiguous",
                 "visually confident": "vis.\nconfident",
                 "spatially close (<2 m)": "close\n<2 m", "spatially far (>4 m)": "far\n>4 m"}
        ax.set_xticks(x); ax.set_xticklabels([short.get(s, s) for s in xs], fontsize=7.5)
        ax.set_ylim(30, 100); ax.set_ylabel("correct hypothesis chosen (%)")
        ax.set_title(label, fontsize=10); ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.suptitle("Forced binary choice between the correct hypothesis and one wrong one",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=170, facecolor="white"); plt.close(fig)

    W("\nWhat to read off this. Above 50% everywhere means the cue is not empty. "
      "Where acoustics beats the visual ordering is where the gate should act; "
      "where it does not, the gate should stay closed, and that is what the "
      "visual-ambiguity strata test directly.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    (args.out.parent / "I_pairwise.json").write_text(
        json.dumps(dict(results=js, provenance=stamp()), indent=2, default=str))
    print("\n".join(out)); print(f"\nwrote {args.out} and {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
