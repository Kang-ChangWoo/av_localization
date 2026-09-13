#!/usr/bin/env python3
"""Does the acoustic term do more work as the camera does less?

The paper's mechanism is that acoustics intervenes where the visual posterior
is ambiguous. If that is what is happening, then degrading the query image
should lower vision, leave the acoustic side untouched, and *widen* the gap
between the two, because more queries fall into the regime where the gate lets
acoustics act. If instead the gain shrinks or vanishes as vision degrades, the
acoustic term was riding on a good visual shortlist and the mechanism claimed is
not the one at work.

Each level is a separate extraction with the corrupted image fed to the frozen
backbone; nothing else changes. Scalars are refit by leave-one-room-out at every
level, so a degraded backbone gets thresholds tuned for it rather than
inheriting the clean ones, which would confound "vision is worse" with "the
gate is mis-set".

    python feasible/run_visual_degradation.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

# (tag, family, level, label)
LEVELS = [("f3STFT", "clean", 0, "clean"),
          ("f3loc_mono_deg_blur2", "blur", 2, "blur σ=2"),
          ("f3loc_mono_deg_blur4", "blur", 4, "blur σ=4"),
          ("f3loc_mono_deg_blur8", "blur", 8, "blur σ=8"),
          ("f3loc_mono_deg_dark50", "dark", 0.5, "dark ×0.5"),
          ("f3loc_mono_deg_dark25", "dark", 0.25, "dark ×0.25"),
          ("f3loc_mono_deg_dark10", "dark", 0.1, "dark ×0.1")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "results" / "V_visual_degradation.md")
    p.add_argument("--fig", type=Path, default=HERE / "figs" / "V_visual_degradation.png")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw the figure from the saved JSON without re-evaluating")
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


def group(M: dict) -> dict:
    cols = [c for c in M if c not in ("query_id", "scene", "collection")
            and M[c].dtype not in (object, bool)]
    g: dict = {}
    for i, q in enumerate(M["query_id"]):
        d = g.setdefault(str(q), {c: [] for c in cols})
        for c in cols:
            d[c].append(M[c][i])
    for q, d in g.items():
        o = np.argsort(np.asarray(d["mode"], dtype=int))
        g[q] = {c: np.asarray(v, dtype=float)[o] for c, v in d.items()}
    return g


def main() -> int:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose, evidence_columns
    from track1_core.provenance import stamp

    rng = np.random.default_rng(args.seed)
    scalars = list(itertools.product((0.5, 1.0, 2.0), (0.02, 0.05, 0.1),
                                     (0.005, 0.02, 0.05, 0.1, 0.2), (-2.0, 0.0, 0.2, 0.4)))
    # The structure is held at the paper's shared choice, read from the policy
    # file rather than typed here: a hand-typed "lse" once cost the clean level
    # 16 points against the same rooms scored with the policy's "centre".
    pol = json.loads((REPO_ROOT / "outputs" / "metrics" / "unified_policy.json").read_text())
    VE = pol["policy"]["f3STFT"]["vis_evidence"]; AE = pol["policy"]["f3STFT"]["ac_evidence"]
    print(f"[structure] visual={VE} acoustic={AE} (from unified_policy.json)")

    rows_out, js = [], {}
    if args.plot_only:
        rows_out = json.loads((args.out.parent / "V_visual_degradation.json").read_text())["rows"]
    for tag, fam, lvl, label in ([] if args.plot_only else LEVELS):
        qp = args.analysis_dir / f"queries_{args.condition}_{tag}.csv"
        mp = args.analysis_dir / f"modes_{args.condition}_{tag}.csv"
        if not (qp.exists() and mp.exists()):
            print(f"[skip] {label}: not extracted yet")
            continue
        Q = read(qp); M = group(read(mp))
        qids = [str(x) for x in Q["query_id"]]
        scene = np.array([str(x) for x in Q["scene"]])
        rooms = sorted(set(scene))

        def run(cfg, mask):
            vc, ac = evidence_columns(cfg)
            return np.array([M[qids[i]]["dist_gt_m"][choose(M[qids[i]][vc], M[qids[i]][ac], cfg)[0]]
                             for i in np.nonzero(mask)[0]])

        E = np.zeros(len(qids)); used = np.zeros(len(qids))
        for held in rooms:
            rep = scene == held; fit = ~rep
            best, bs = None, -1.0
            for w, s, tv, ta in scalars:
                c = ModeFusionConfig(vis_evidence=VE, ac_evidence=AE, rule="continuous",
                                     weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta)
                sc = float((run(c, fit) < 1).mean())
                if sc > bs:
                    best, bs = c, sc
            E[rep] = run(best, rep)
            vc, ac = evidence_columns(best)
            used[rep] = [choose(M[qids[i]][vc], M[qids[i]][ac], best)[1]
                         for i in np.nonzero(rep)[0]]
        v = Q["e_vis"]; a = Q["e_ac"]
        d = (E < 1).astype(float) - (v < 1).astype(float)
        i = rng.integers(0, d.size, size=(args.boot, d.size)); m = d[i].mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        # how ambiguous the visual posterior is: the log-odds between the two
        # strongest hypotheses, which is what the gate reads
        amb = np.array([M[q]["vis_log_lse"][0] - M[q]["vis_log_lse"][1] if len(M[q]["vis_log_lse"]) > 1
                        else np.inf for q in qids])
        rows_out.append(dict(tag=tag, family=fam, level=lvl, label=label, n=len(qids),
                             vision=float((v < 1).mean()), acoustic=float((a < 1).mean()),
                             ours=float((E < 1).mean()), gain=float(d.mean()),
                             lo=float(lo), hi=float(hi), audio_used=float(used.mean()),
                             median_ambiguity=float(np.median(amb[np.isfinite(amb)]))))
        print(f"  {label:12s} vision {100*(v<1).mean():5.1f}  ours {100*(E<1).mean():5.1f}  "
              f"gain {100*d.mean():+5.1f} [{100*lo:+5.1f},{100*hi:+5.1f}]  "
              f"audio acts on {100*used.mean():4.0f}%")

    out = []
    W = out.append
    W("# Acoustic gain as the camera degrades\n")
    W("The query image is corrupted before the frozen visual backbone; the "
      "acoustic side is untouched. Scalars are refit by leave-one-room-out at "
      "every level. `audio acts` is the fraction of queries on which the gate "
      "lets the acoustic term change the answer.\n")
    W("| degradation | queries | vision @1m | acoustic alone | ours @1m | gain | 95% CI | audio acts | median visual ambiguity |")
    W("|---|---|---|---|---|---|---|---|---|")
    for r in rows_out:
        W(f"| {r['label']} | {r['n']} | {100*r['vision']:.1f}% | {100*r['acoustic']:.1f}% | "
          f"{100*r['ours']:.1f}% | {100*r['gain']:+.1f} | [{100*r['lo']:+.1f}, {100*r['hi']:+.1f}] | "
          f"{100*r['audio_used']:.0f}% | {r['median_ambiguity']:.2f} |")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), facecolor="white")
    for ax, fam, xl in zip(axes, ("blur", "dark"), ("Gaussian blur σ (px)", "illumination gain")):
        sub = [r for r in rows_out if r["family"] in (fam, "clean")]
        if fam == "dark":
            sub = sorted(sub, key=lambda r: -(r["level"] if r["family"] == "dark" else 1.0))
            xs = [r["level"] if r["family"] == "dark" else 1.0 for r in sub]
        else:
            sub = sorted(sub, key=lambda r: r["level"])
            xs = [r["level"] for r in sub]
        vis = [100 * r["vision"] for r in sub]; ours = [100 * r["ours"] for r in sub]
        lo = [100 * (r["ours"] - r["vision"] - r["lo"]) for r in sub]
        hi = [100 * (r["hi"] - (r["ours"] - r["vision"])) for r in sub]
        ac_alone = 100 * float(np.median([r["acoustic"] for r in sub]))
        ax.axhline(ac_alone, color="#111827", ls=":", lw=1.2,
                   label=f"acoustic alone, whole grid ({ac_alone:.1f})")
        ax.plot(xs, vis, "o-", color="#9ca3af", lw=1.6, label="vision alone")
        ax.plot(xs, ours, "o-", color="#1d4ed8", lw=2.0, label="vision + acoustics")
        ax.fill_between(xs, np.array(ours) - np.array(lo), np.array(ours) + np.array(hi),
                        color="#1d4ed8", alpha=0.12, lw=0)
        ax.set_xlabel(xl); ax.set_ylabel("recall @1 m (%)")
        if fam == "dark":
            ax.set_xscale("log"); ax.invert_xaxis()
        ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"{fam}: acoustic gain "
                     + ", ".join(f"{100*r['gain']:+.1f}" for r in sub), fontsize=9)
    fig.suptitle("As the camera degrades the acoustic gain grows, but a ruined shortlist "
                 "caps the fusion below acoustics alone", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=170, facecolor="white")
    plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    (args.out.parent / "V_visual_degradation.json").write_text(
        json.dumps(dict(rows=rows_out, provenance=stamp()), indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out} and {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
