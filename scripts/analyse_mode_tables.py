#!/usr/bin/env python3
"""The diagnosis: when does sound help, when does it hurt, and what limits it.

Reads the tables written by ``extract_mode_table.py`` and answers the questions
in order. Nothing here runs a network or touches a candidate grid, so the whole
diagnosis is a few seconds and can be re-run against any extraction.

Sections map to the analysis plan:

    A2  visual ambiguity binned against the acoustic net gain
    A3  whether the acoustic score carries an inference-time reliability signal
    A4  the joint visual-confidence / acoustic-confidence benefit map
    A5  forensics on repairs and regressions
    A6  ground-truth coverage against top-K cells and top-K hypotheses
    A7  spatial roughness of the acoustic field around truth
    A8  the domain-gap decomposition, matched against furnished
    A9  early against late reflections, judged on complementarity not accuracy
    A10 paired bootstrap intervals on every headline gain

Two conventions throughout. A "repair" is a query whose visual error is at least
1 m and whose fused error is below it; a "break" is the reverse; those are the
only two events that move recall at 1 m. And every interval is a paired
bootstrap over queries, because the visual and fused predictions for one query
are not independent and an unpaired interval would be far too wide.

    python scripts/analyse_mode_tables.py --backbone unloc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--backbone", default="unloc")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "outputs" / "analysis" / "diagnosis.md")
    return p.parse_args()


# ---------------------------------------------------------------- statistics
def boot_paired(a: np.ndarray, b: np.ndarray, n: int, rng) -> tuple[float, float, float]:
    """Mean of b-a with a percentile interval, resampling queries not events."""
    d = b.astype(float) - a.astype(float)
    if d.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, d.size, size=(n, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC; ties get the average rank, as they must."""
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    if label.all() or not label.any():
        return np.nan
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within tied groups
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = int(label.sum())
    n0 = len(label) - n1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision, the interpolation-free version."""
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    if not label.any():
        return np.nan
    order = np.argsort(-score)
    y = label[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """A tiny typed CSV reader, so this runs in either conda environment."""
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if vals[0] in ("True", "False"):
            out[k] = np.array([v == "True" for v in vals])
            continue
        try:
            out[k] = np.array([float(v) if v not in ("", "nan") else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def bin_edges(x: np.ndarray, n: int) -> np.ndarray:
    """Quantile bins, deduplicated, so every bin holds samples."""
    q = np.unique(np.nanquantile(x[np.isfinite(x)], np.linspace(0, 1, n + 1)))
    q[0] -= 1e-9
    q[-1] += 1e-9
    return q


def main() -> int:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = args.analysis_dir
    Q = {c: read_csv(d / f"queries_{c}_{args.backbone}.csv")
         for c in ("raw_scan_open", "floorplan_closed")
         if (d / f"queries_{c}_{args.backbone}.csv").exists()}
    M = {c: read_csv(d / f"modes_{c}_{args.backbone}.csv")
         for c in Q}
    R = {c: read_csv(d / f"mode_ranges_{c}_{args.backbone}.csv")
         for c in Q if (d / f"mode_ranges_{c}_{args.backbone}.csv").exists()}
    if "raw_scan_open" not in Q:
        print("no furnished-domain table; run extract_mode_table.py first")
        return 1
    q = Q["raw_scan_open"]
    n = len(q["query_id"])
    rng = np.random.default_rng(args.seed)
    out: list[str] = []
    W = out.append
    W(f"# Diagnosis: {args.backbone}, {n} queries\n")
    W("Generated by `scripts/analyse_mode_tables.py`. Every interval is a paired "
      "bootstrap over queries.\n")

    ok = lambda e: (e < 1.0).astype(float)
    e_vis, e_fu = q["e_vis"], q["e_fused"]
    repair = (e_vis >= 1) & (e_fu < 1)
    brk = (e_vis < 1) & (e_fu >= 1)

    # ------------------------------------------------------- A10 headline
    W("\n## A10. Headline gains with paired bootstrap intervals\n")
    W("| rule | recall @1m | gain over vision | 95% CI |")
    W("|---|---|---|---|")
    W(f"| vision | {100*ok(e_vis).mean():.1f}% | | |")
    for name, col in (("acoustic alone", "e_ac"), ("cell rerank top-50", "e_rerank"),
                      ("cell log-rank fusion", "e_fused")):
        m, lo, hi = boot_paired(ok(e_vis), ok(q[col]), args.boot, rng)
        W(f"| {name} | {100*ok(q[col]).mean():.1f}% | {100*m:+.1f} | "
          f"[{100*lo:+.1f}, {100*hi:+.1f}] |")
    W("\nPer scene, cell log-rank fusion against vision:\n")
    W("| scene | n | vision | fused | gain | 95% CI |")
    W("|---|---|---|---|---|---|")
    for s in np.unique(q["scene"]):
        k = q["scene"] == s
        m, lo, hi = boot_paired(ok(e_vis[k]), ok(e_fu[k]), args.boot, rng)
        W(f"| {s} | {int(k.sum())} | {100*ok(e_vis[k]).mean():.1f}% | "
          f"{100*ok(e_fu[k]).mean():.1f}% | {100*m:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")

    # ------------------------------------------------------- A2 ambiguity
    W("\n## A2. Is the gain concentrated in visually ambiguous queries?\n")
    amb = q["mode_log_ratio"]
    amb = np.where(np.isfinite(amb), amb, np.nanmax(amb[np.isfinite(amb)]))
    edges = bin_edges(amb, args.bins)
    W("Bins are quantiles of the log-odds between the best and second visual "
      "hypothesis. Low means ambiguous.\n")
    W("| ambiguity bin | n | vision | fused | repairs | breaks | net |")
    W("|---|---|---|---|---|---|---|")
    xs, ys = [], []
    for i in range(len(edges) - 1):
        k = (amb > edges[i]) & (amb <= edges[i + 1])
        if not k.any():
            continue
        net = 100 * (ok(e_fu[k]).mean() - ok(e_vis[k]).mean())
        xs.append(float(np.median(amb[k])))
        ys.append(net)
        W(f"| {edges[i]:.2f} to {edges[i+1]:.2f} | {int(k.sum())} | "
          f"{100*ok(e_vis[k]).mean():.1f}% | {100*ok(e_fu[k]).mean():.1f}% | "
          f"{int(repair[k].sum())} | {int(brk[k].sum())} | {net:+.1f} |")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color="0.6", lw=0.9)
    ax.plot(xs, ys, "o-", color="#2962ff")
    ax.set_xlabel("visual ambiguity: log-odds, best vs second hypothesis (low = ambiguous)")
    ax.set_ylabel("acoustic net gain, points @1m")
    ax.set_title(f"A2. Where the acoustic gain lives ({args.backbone}, n={n})")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / f"A2_ambiguity_vs_gain_{args.backbone}.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- A3 reliability
    W("\n## A3. Does the acoustic score know when to be trusted?\n")
    W("Predicting three events from quantities available at inference time. "
      "AUROC 0.5 is chance.\n")
    W("| signal | acoustic pick is correct | audio repairs vision | audio breaks vision |")
    W("|---|---|---|---|")
    correct_ac = (q["e_ac"] < 1.0)
    sigs = {
        "acoustic margin among modes": q["ac_mode_margin"],
        "acoustic rank margin": q["ac_rank_margin"],
        "acoustic entropy over modes (negated)": -q["ac_mode_entropy"],
        "relative evidence of best mode": q["ac_rel_best"],
        "visual ambiguity (negated)": -amb,
    }
    for name, s in sigs.items():
        s = np.where(np.isfinite(s), s, np.nanmedian(s[np.isfinite(s)]))
        W(f"| {name} | {auroc(s, correct_ac):.3f} | "
          f"{auroc(s[e_vis >= 1], repair[e_vis >= 1]):.3f} | "
          f"{auroc(s[e_vis < 1], brk[e_vis < 1]):.3f} |")
    W("\nPrecision-recall for the same, as the events are rare:\n")
    W("| signal | AP, repairs among failures | AP, breaks among successes |")
    W("|---|---|---|")
    for name, s in sigs.items():
        s = np.where(np.isfinite(s), s, np.nanmedian(s[np.isfinite(s)]))
        W(f"| {name} | {auprc(s[e_vis >= 1], repair[e_vis >= 1]):.3f} | "
          f"{auprc(s[e_vis < 1], brk[e_vis < 1]):.3f} |")

    best_sig = max(sigs, key=lambda k: abs(auroc(
        np.where(np.isfinite(sigs[k]), sigs[k], 0), correct_ac) - 0.5))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    s = sigs["acoustic margin among modes"]
    s = np.where(np.isfinite(s), s, np.nanmedian(s[np.isfinite(s)]))
    e2 = bin_edges(s, args.bins)
    cx = [float(np.median(s[(s > e2[i]) & (s <= e2[i + 1])]))
          for i in range(len(e2) - 1) if ((s > e2[i]) & (s <= e2[i + 1])).any()]
    cy = [100 * correct_ac[(s > e2[i]) & (s <= e2[i + 1])].mean()
          for i in range(len(e2) - 1) if ((s > e2[i]) & (s <= e2[i + 1])).any()]
    ax.plot(cx, cy, "o-", color="#7b1fa2")
    ax.set_xlabel("acoustic margin among visual modes")
    ax.set_ylabel("acoustic pick within 1 m, %")
    ax.set_title(f"A3. Acoustic confidence against acoustic correctness ({args.backbone})")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"A3_acoustic_confidence_{args.backbone}.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- A4 joint map
    W("\n## A4. Joint reliability map\n")
    ac_m = q["ac_mode_margin"]
    ac_m = np.where(np.isfinite(ac_m), ac_m, np.nanmedian(ac_m[np.isfinite(ac_m)]))
    ev, ea = bin_edges(amb, 4), bin_edges(ac_m, 4)
    grid_n = np.zeros((len(ea) - 1, len(ev) - 1))
    grid_d = np.zeros_like(grid_n)
    W("Rows are acoustic margin, columns visual ambiguity. Cell shows n, "
      "repairs, breaks, expected change in @1m.\n")
    W("| acoustic margin \\ visual ambiguity | " +
      " | ".join(f"{ev[i]:.1f}-{ev[i+1]:.1f}" for i in range(len(ev) - 1)) + " |")
    W("|---" * (len(ev)) + "|")
    for a in range(len(ea) - 1):
        cells = []
        for v in range(len(ev) - 1):
            k = ((amb > ev[v]) & (amb <= ev[v + 1])
                 & (ac_m > ea[a]) & (ac_m <= ea[a + 1]))
            if not k.any():
                cells.append("-")
                continue
            delta = 100 * (ok(e_fu[k]).mean() - ok(e_vis[k]).mean())
            grid_n[a, v], grid_d[a, v] = k.sum(), delta
            cells.append(f"n={int(k.sum())} r={int(repair[k].sum())} "
                         f"b={int(brk[k].sum())} **{delta:+.0f}**")
        W(f"| {ea[a]:.2f}-{ea[a+1]:.2f} | " + " | ".join(cells) + " |")

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    lim = float(np.nanmax(np.abs(grid_d))) or 1.0
    im = ax.imshow(grid_d, cmap="coolwarm_r", vmin=-lim, vmax=lim, origin="lower")
    for a in range(grid_d.shape[0]):
        for v in range(grid_d.shape[1]):
            if grid_n[a, v]:
                ax.text(v, a, f"{grid_d[a,v]:+.0f}\nn={int(grid_n[a,v])}",
                        ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(ev) - 1))
    ax.set_xticklabels([f"{ev[i]:.1f}" for i in range(len(ev) - 1)])
    ax.set_yticks(range(len(ea) - 1))
    ax.set_yticklabels([f"{ea[i]:.2f}" for i in range(len(ea) - 1)])
    ax.set_xlabel("visual ambiguity (low = ambiguous)")
    ax.set_ylabel("acoustic margin among modes")
    ax.set_title(f"A4. Expected change in recall @1m ({args.backbone})")
    fig.colorbar(im, ax=ax, label="points @1m")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"A4_joint_reliability_{args.backbone}.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- A5 forensics
    W("\n## A5. Repairs and regressions\n")
    groups = {"vision wrong, audio right (repair)": repair,
              "vision right, audio wrong (break)": brk,
              "both wrong": (e_vis >= 1) & (e_fu >= 1),
              "both right": (e_vis < 1) & (e_fu < 1)}
    cols = [("visual ambiguity", amb), ("acoustic mode margin", ac_m),
            ("GT acoustic rank", q["gt_cell_ac_rank"]),
            ("GT visual rank position", q["gt_vis_rank_pos"]),
            ("visual error, m", e_vis),
            ("GT distance to nearest mode, m", q["gt_mode_dist_m"])]
    W("| group | n | " + " | ".join(c[0] for c in cols) + " |")
    W("|---|---|" + "---|" * len(cols))
    for name, k in groups.items():
        if not k.any():
            W(f"| {name} | 0 | " + " | ".join("-" for _ in cols) + " |")
            continue
        W(f"| {name} | {int(k.sum())} | " +
          " | ".join(f"{np.nanmedian(c[1][k]):.3g}" for c in cols) + " |")
    W("\nScene composition:\n")
    W("| group | " + " | ".join(str(s) for s in np.unique(q["scene"])) + " |")
    W("|---|" + "---|" * len(np.unique(q["scene"])))
    for name, k in groups.items():
        W(f"| {name} | " + " | ".join(
            str(int((k & (q["scene"] == s)).sum())) for s in np.unique(q["scene"])) + " |")

    # ------------------------------------------------------- A6 coverage
    W("\n## A6. Ground-truth coverage, cells against hypotheses\n")
    W("The upper bound of any pure reranking method is its coverage.\n")
    W("| K | truth within 1 m of a top-K cell | of a top-K hypothesis |")
    W("|---|---|---|")
    for K in (1, 2, 3, 5, 10, 20, 50):
        c = q.get(f"gt_in_top{K}_cells")
        m = q.get(f"gt_in_top{K}_modes")
        W(f"| {K} | {100*c.mean():.1f}% | "
          + (f"{100*m.mean():.1f}% |" if m is not None else "- |"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ks = [1, 2, 3, 5, 10, 20, 50]
    ax.plot(Ks, [100 * q[f"gt_in_top{K}_cells"].mean() for K in Ks], "o-",
            label="top-K grid cells", color="#d50000")
    Km = [1, 2, 3, 5, 10]
    ax.plot(Km, [100 * q[f"gt_in_top{K}_modes"].mean() for K in Km], "s-",
            label="top-K separated hypotheses", color="#2962ff")
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
    ax.set_xlabel("K"); ax.set_ylabel("truth within 1 m, %")
    ax.set_title(f"A6. Reranking upper bound ({args.backbone})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"A6_coverage_{args.backbone}.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- A7 smoothness
    W("\n## A7. Is the acoustic field right about the region but wrong about the cell?\n")
    W("| quantity | median | mean |")
    W("|---|---|---|")
    for name, c in (("acoustic rank of the GT cell", "gt_cell_ac_rank"),
                    ("best acoustic rank within 0.25 m", "ac_rank_best_within_0.25m"),
                    ("best acoustic rank within 0.5 m", "ac_rank_best_within_0.5m"),
                    ("best acoustic rank within 1.0 m", "ac_rank_best_within_1.0m"),
                    ("distance to the nearest strong acoustic peak, m",
                     "dist_to_strong_ac_peak_m")):
        v = q[c][np.isfinite(q[c])]
        W(f"| {name} | {np.median(v):.4f} | {v.mean():.4f} |")
    W("\nIf the rank rises steeply from the GT cell to a 0.5 m neighbourhood, the "
      "score locates the region better than the cell, which is an argument for "
      "aggregating over a mode's disc rather than reading its centre.\n")

    # ------------------------------------------------------- A8 domain gap
    if "floorplan_closed" in Q:
        W("\n## A8. Domain gap, at the current 48 kHz configuration\n")
        W("Identical code, identical guard, window, envelope, microphones, "
          "normalisation and distance. Only the query condition differs.\n")
        W("| condition | acoustic alone @1m | GT rank position | GT acoustic percentile | "
          "best rank within 0.5 m | distance to nearest strong peak, m |")
        W("|---|---|---|---|---|---|")
        for c, qq in Q.items():
            gp = 100 * (1 - qq["gt_ac_rank_pos"] / qq["n_cells"])
            W(f"| {c} | {100*(qq['e_ac'] < 1).mean():.1f}% | "
              f"{np.median(qq['gt_ac_rank_pos']):.0f} | {np.median(gp):.2f} | "
              f"{np.median(qq['ac_rank_best_within_0.5m']):.4f} | "
              f"{np.nanmedian(qq['dist_to_strong_ac_peak_m']):.2f} |")
        W("\nPer scene, acoustic alone at 1 m:\n")
        scenes = np.unique(q["scene"])
        W("| condition | " + " | ".join(str(s) for s in scenes) + " |")
        W("|---|" + "---|" * len(scenes))
        for c, qq in Q.items():
            W(f"| {c} | " + " | ".join(
                f"{100*(qq['e_ac'][qq['scene']==s] < 1).mean():.1f}%" for s in scenes) + " |")
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        w = 0.36
        for i, (c, qq) in enumerate(Q.items()):
            vals = [100 * (qq["e_ac"][qq["scene"] == s] < 1).mean() for s in scenes]
            ax.bar(np.arange(len(scenes)) + i * w, vals, width=w, label=c)
        ax.set_xticks(np.arange(len(scenes)) + w / 2)
        ax.set_xticklabels([str(s) for s in scenes], fontsize=9)
        ax.set_ylabel("acoustic alone, recall @1m")
        ax.set_title(f"A8. Matched geometry against furnished ({args.backbone})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / f"A8_domain_gap_{args.backbone}.png", dpi=140)
        plt.close(fig)
    else:
        W("\n## A8. Domain gap\n\nNot available: the matched-domain extraction "
          "has not been run.\n")

    # ------------------------------------------------------- A9 early/late
    if "raw_scan_open" in R:
        W("\n## A9. Early against late reflections\n")
        W("Judged on complementarity, not on acoustic accuracy alone. "
          "'mode rerank' picks the best-scoring visual hypothesis under that "
          "range and reports recall at 1 m.\n")
        r = R["raw_scan_open"]
        qid_of = r["query_id"]
        W("| range, ms | acoustic alone @1m | GT percentile | mode rerank @1m | vs vision |")
        W("|---|---|---|---|---|")
        base = 100 * ok(e_vis).mean()
        curves = []
        for rng_name in ["0-16", "16-32", "32-64", "64-128", "0-32", "0-64", "0-128"]:
            sel = r["range_ms"] == rng_name
            alone = sel & (r["mode"] == -1)
            modes = sel & (r["mode"] >= 0)
            # mode rerank: per query, the hypothesis with the best acoustic score
            picks = {}
            for qid, mode, raw, dd in zip(qid_of[modes], r["mode"][modes],
                                          r["ac_raw"][modes], r["dist_gt_m"][modes]):
                cur = picks.get(qid)
                if cur is None or raw > cur[0]:
                    picks[qid] = (raw, dd)
            rr = 100 * np.mean([v[1] < 1.0 for v in picks.values()])
            W(f"| {rng_name} | {100*np.mean(r['dist_gt_m'][alone] < 1.0):.1f}% | "
              f"{100*np.median(r['ac_rank'][alone]):.2f} | {rr:.1f}% | {rr-base:+.1f} |")
            curves.append((rng_name, 100 * np.mean(r["dist_gt_m"][alone] < 1.0), rr))
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = np.arange(len(curves))
        ax.bar(x - 0.2, [c[1] for c in curves], width=0.4, label="acoustic alone")
        ax.bar(x + 0.2, [c[2] for c in curves], width=0.4, label="mode rerank")
        ax.axhline(base, color="0.3", ls="--", lw=1.2, label="vision alone")
        ax.set_xticks(x); ax.set_xticklabels([c[0] for c in curves], fontsize=9)
        ax.set_xlabel("post-direct time range, ms"); ax.set_ylabel("recall @1m")
        ax.set_title(f"A9. Early against late reflections ({args.backbone})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / f"A9_early_late_{args.backbone}.png", dpi=140)
        plt.close(fig)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {args.report}")
    print(f"figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
