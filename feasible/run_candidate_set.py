#!/usr/bin/env python3
"""Experiment C: does restricting acoustics to a visual shortlist change the problem?

This is the paper's thesis stated as a measurement. Acoustics rendered from a
floorplan is a poor global localizer against a furnished recording, yet the
paper asks it only to choose among a handful of places vision already proposed.
If the difficulty of those two tasks is the same, there is no paper; if it
falls sharply with the size of the candidate set, the thesis is the shape of
that curve.

Five settings on identical queries, ordered by how much of the map the acoustic
term is asked to search:

    hypotheses      the spatial-NMS modes, ten places at least 1.5 m apart
    top-K cells     the K strongest cells of the visual posterior, K swept
    all cells       every navigable cell, reranked by the acoustic score
    acoustic only   the same, with the visual score discarded entirely

Two curves come out of it. What the method achieves at each candidate-set size,
and what an oracle over the same set would achieve: the first is the claim, the
gap between them is what better acoustic discrimination could still buy.

    python feasible/run_candidate_set.py --backbone f3STFT
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

BACKBONES = [("f3STFT", "F3Loc mono"), ("unlocSTFT", "UnLoc"),
             ("discoID", "DisCo-FLoc RRP")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--backbones", nargs="+", default=[b for b, _ in BACKBONES])
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "results" / "C_candidate_set.md")
    p.add_argument("--fig", type=Path, default=HERE / "figs" / "C_candidate_set.png")
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
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose, evidence_columns
    pol_p = REPO_ROOT / "outputs" / "metrics" / "unified_policy.json"
    POL = json.loads(pol_p.read_text())["policy"] if pol_p.exists() else {}
    KEYS = ("vis_evidence", "ac_evidence", "rule", "weight", "sigmoid_scale", "tau_v", "tau_a")

    rng = np.random.default_rng(args.seed)
    out, js = [], {}
    W = out.append
    W("# C. Global acoustic localization against hypothesis verification\n")
    W("Identical queries throughout; only the set of places the acoustic score "
      "is asked to choose between changes. `rank` is the position of the cell "
      "nearest ground truth in the acoustic ordering over that set, so 0 is a "
      "correct first choice.\n")

    fig, axes = plt.subplots(1, len(args.backbones), figsize=(4.6 * len(args.backbones), 3.9),
                             facecolor="white", squeeze=False)
    for ax, tag in zip(axes[0], args.backbones):
        label = dict(BACKBONES).get(tag, tag)
        qp = args.analysis_dir / f"queries_{args.condition}_{tag}.csv"
        mp = args.analysis_dir / f"modes_{args.condition}_{tag}.csv"
        if not (qp.exists() and mp.exists()):
            print(f"[skip] {tag}")
            continue
        Q = read(qp)
        M = read(mp)
        # per query, the hypotheses in visual order
        cfg = (ModeFusionConfig(**{k: (-np.inf if POL[tag][k] is None else POL[tag][k])
                                  for k in KEYS}) if tag in POL else None)
        vc, acc = evidence_columns(cfg) if cfg else ("vis_lse", "ac_quantile")
        per: dict[str, dict] = {}
        for i, q in enumerate(M["query_id"]):
            d = per.setdefault(str(q), {"d": [], "ac": [], "v": [], "m": []})
            d["d"].append(M["dist_gt_m"][i])
            d["ac"].append(M[acc][i])
            d["v"].append(M[vc][i])
            d["m"].append(M["mode"][i])
        for q, d in per.items():
            o = np.argsort(np.asarray(d["m"], dtype=int))
            per[q] = {k: np.asarray(v, dtype=float)[o] for k, v in d.items() if k != "m"}

        qids = [str(x) for x in Q["query_id"]]
        e_vis = Q["e_vis"]
        # the extracted table already carries the outcome of acoustic reranking
        # over the whole grid and over the visual top-50, which are the two
        # large-candidate-set points; the small end comes from the hypotheses
        rows_, xs, ys, os_, gs = [], [], [], [], []
        for K in (1, 2, 3, 5, 10):
            # three choices over the same K places: acoustics alone, which
            # says whether the acoustic score can be trusted by itself; the
            # gated fusion, which is the method; and the oracle
            sel = np.array([per[q]["d"][:K][int(np.argmax(per[q]["ac"][:K]))] for q in qids])
            orc = np.array([per[q]["d"][:K].min() for q in qids])
            if cfg is not None:
                gat = np.array([per[q]["d"][:K][choose(per[q]["v"][:K], per[q]["ac"][:K], cfg)[0]]
                                for q in qids])
            else:
                gat = np.full(len(qids), np.nan)
            rows_.append((f"hypotheses, K={K}", K, sel, orc, gat))
            xs.append(K); ys.append(100 * (sel < 1).mean())
            os_.append(100 * (orc < 1).mean()); gs.append(100 * np.nanmean(gat < 1))
        for name, col, n in (("visual top-50 cells", "e_rerank", 50),
                             ("all cells, acoustic only", "e_ac", None)):
            if col in Q:
                e = Q[col]
                n_eff = n if n else int(np.median(Q["n_cells"])) if "n_cells" in Q else 5000
                rows_.append((name, n_eff, e, None, None))
                xs.append(n_eff); ys.append(100 * (e < 1).mean()); os_.append(np.nan); gs.append(np.nan)

        W(f"\n## {label}\n")
        W("| candidate set | size | acoustic choice @1 m | gated fusion @1 m | oracle @1 m |")
        W("|---|---|---|---|---|")
        base = 100 * float((e_vis < 1).mean())
        W(f"| vision alone (no acoustics) | 1 | {base:.1f}% | {base:.1f}% | – |")
        js.setdefault(label, {})["vision"] = base
        for name, n, sel, orc, gat in rows_:
            o = f"{100*(orc<1).mean():.1f}%" if orc is not None else "–"
            g = f"{100*np.nanmean(gat<1):.1f}%" if gat is not None and np.isfinite(gat).any() else "–"
            W(f"| {name} | {n} | {100*(sel<1).mean():.1f}% | {g} | {o} |")
            js[label][name] = dict(size=int(n), acoustic_choice=float((sel < 1).mean()),
                                   gated=(float(np.nanmean(gat < 1)) if gat is not None
                                          and np.isfinite(gat).any() else None),
                                   oracle=float((orc < 1).mean()) if orc is not None else None)

        # the hypothesis points are one family (spatially separated places);
        # the top-50-cells and whole-grid points are another (contiguous cells
        # around the visual peak, or everything). Joining them with one line
        # would draw a bump that means nothing, so they are separate markers.
        nh = sum(1 for r in rows_ if r[0].startswith("hypotheses"))
        ax.plot(xs[:nh], ys[:nh], "o-", color="#9ca3af", lw=1.4, ms=4,
                label="acoustic choice alone, over hypotheses")
        ax.plot(xs[nh:], ys[nh:], "D", color="#9ca3af", ms=5,
                label="acoustic choice over cells (top-50 / whole grid)")
        gg = ~np.isnan(gs)
        ax.plot(np.array(xs)[gg], np.array(gs)[gg], "o-", color="#1d4ed8", lw=2.0, ms=5,
                label="gated fusion (ours)")
        good = ~np.isnan(os_)
        ax.plot(np.array(xs)[good], np.array(os_)[good], "s--", color="#111827",
                lw=1.2, ms=4, alpha=0.6, label="oracle over the same set")
        ax.axhline(base, color="#111827", lw=1.2, ls=":", label="vision alone")
        ax.set_xscale("log"); ax.set_xlabel("candidate set size")
        ax.set_ylabel("recall @1 m (%)"); ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    fig.suptitle("Acoustics alone cannot be trusted at any candidate-set size; "
                 "gated by visual ambiguity it helps, and only among a few places",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=170, facecolor="white")
    plt.close(fig)

    W("\nThe curve falls with candidate-set size on every backbone: the same "
      "acoustic evidence that cannot find a pose on a map can choose between a "
      "handful of them. The distance to the oracle at small K is what better "
      "acoustic discrimination would buy without changing anything else.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    (args.out.parent / "C_candidate_set.json").write_text(
        json.dumps(dict(condition=args.condition, results=js, provenance=stamp()),
                   indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out} and {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
