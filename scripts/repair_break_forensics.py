#!/usr/bin/env python3
"""Every repair and every regression, explained by the numbers that caused it.

A heat map cannot say why a switch happened, and a recall table cannot say
whether the switches share a mechanism. This lists them individually: for each
query where the acoustic term changed the outcome at 1 m, it prints the two
hypotheses involved and the quantities that decided between them.

The decision under the recommended rule is a comparison of two numbers, so the
explanation can be complete rather than suggestive. For hypotheses `v` (vision's
pick) and `c` (the one that won):

    visual evidence      log posterior at each hypothesis; their difference is
                         what vision was risking
    acoustic evidence    each hypothesis against the aggregate of all the others,
                         in nats, after per-query standardisation
    the swing            how much acoustic evidence had to overcome the visual
                         preference, and by how much it did

A repair and a regression with the same numbers are the same event with a
different answer, and if that is what the data shows then no threshold on these
quantities can separate them. That is the question this script exists to settle.

    python scripts/repair_break_forensics.py --backbone unloc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--backbone", default="unloc")
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--rule", default="relative", choices=["relative", "mode_rerank", "continuous"])
    p.add_argument("--weight", type=float, default=0.5)
    p.add_argument("--vis-evidence", default="centre")
    p.add_argument("--ac-evidence", default="max")
    p.add_argument("--n-show", type=int, default=10)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "analysis" / "repair_break_forensics.md")
    return p.parse_args()


def read_csv(path: Path) -> dict[str, np.ndarray]:
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if vals[0] in ("True", "False"):
            out[k] = np.array([v == "True" for v in vals]); continue
        try:
            out[k] = np.array([float(v) if v not in ("", "nan") else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def main() -> int:
    args = parse_args()
    from track1_core.likelihood.mode_fusion import (
        ModeFusionConfig, acoustic_evidence, choose, evidence_columns, standardise,
    )

    d = args.analysis_dir
    M = read_csv(d / f"modes_{args.condition}_{args.backbone}.csv")
    cfg = ModeFusionConfig(vis_evidence=args.vis_evidence, ac_evidence=args.ac_evidence,
                           rule=args.rule, weight=args.weight)
    vc, ac = evidence_columns(cfg)

    by: dict[str, dict[str, list]] = {}
    for i, qid in enumerate(M["query_id"]):
        by.setdefault(str(qid), {c: [] for c in M}).setdefault
        g = by[str(qid)]
        for c in M:
            g[c].append(M[c][i])

    events = []
    for qid, g in by.items():
        order = np.argsort(np.asarray(g["mode"], dtype=int))
        vis = np.asarray(g[vc], dtype=float)[order]
        acs = np.asarray(g[ac], dtype=float)[order]
        dist = np.asarray(g["dist_gt_m"], dtype=float)[order]
        k, _ = choose(vis, acs, cfg)
        v = int(vis.argmax())
        if k == v:
            continue
        rel = acoustic_evidence(acs)
        sv = standardise(vis)
        kind = ("repair" if dist[v] >= 1 > dist[k]
                else "break" if dist[k] >= 1 > dist[v] else "neutral")
        events.append(dict(
            qid=qid, scene=str(g["scene"][0]), kind=kind,
            v=v, k=k, e_v=dist[v], e_k=dist[k],
            vis_v=vis[v], vis_k=vis[k],
            sv_gap=float(sv[v] - sv[k]),          # what vision preferred, standardised
            rel_v=float(rel[v]), rel_k=float(rel[k]),
            swing=float(args.weight * (rel[k] - rel[v])),
            n_better=int((dist < 1).sum())))

    rep = [e for e in events if e["kind"] == "repair"]
    brk = [e for e in events if e["kind"] == "break"]
    rep.sort(key=lambda e: -e["e_v"])
    brk.sort(key=lambda e: -e["e_k"])

    out: list[str] = []
    W = out.append
    W(f"# Repairs and regressions, {args.backbone}, {args.condition}\n")
    W(f"Rule `{args.rule}` with visual evidence `{args.vis_evidence}`, acoustic "
      f"evidence `{args.ac_evidence}`, weight {args.weight:g}. "
      f"{len(events)} of {len(by)} queries change hypothesis: "
      f"{len(rep)} repairs, {len(brk)} regressions, "
      f"{len(events)-len(rep)-len(brk)} that cross no threshold.\n")
    W("Columns: `vis gap` is how much vision preferred its own pick, standardised "
      "across the query's hypotheses. `swing` is the weighted acoustic evidence "
      "that overturned it. A switch happens exactly when swing exceeds vis gap.\n")

    for title, evs in (("Repairs: vision was wrong, sound was right", rep),
                       ("Regressions: vision was right, sound was wrong", brk)):
        W(f"\n## {title}\n")
        W("| query | scene | vision mode -> chosen | error | vis gap | rel(vision) | "
          "rel(chosen) | swing | modes within 1 m |")
        W("|---|---|---|---|---|---|---|---|---|")
        for e in evs[: args.n_show]:
            W(f"| {e['qid']} | {e['scene']} | {e['v']} -> {e['k']} | "
              f"{e['e_v']:.1f} m -> {e['e_k']:.1f} m | {e['sv_gap']:.3f} | "
              f"{e['rel_v']:+.3f} | {e['rel_k']:+.3f} | {e['swing']:+.3f} | "
              f"{e['n_better']} |")

    W("\n## Are repairs and regressions distinguishable?\n")
    W("| quantity | repairs, median | regressions, median | overlap |")
    W("|---|---|---|---|")
    for name, key in (("visual gap overturned", "sv_gap"),
                      ("acoustic swing", "swing"),
                      ("relative evidence of the chosen hypothesis", "rel_k"),
                      ("relative evidence of vision's pick", "rel_v")):
        a = np.array([e[key] for e in rep])
        b = np.array([e[key] for e in brk])
        if a.size == 0 or b.size == 0:
            W(f"| {name} | - | - | - |")
            continue
        # fraction of one distribution inside the other's central 90 percent
        lo, hi = np.percentile(b, [5, 95])
        ov = float(np.mean((a >= lo) & (a <= hi)))
        W(f"| {name} | {np.median(a):+.3f} | {np.median(b):+.3f} | {100*ov:.0f}% |")
    W("\nAn overlap near 100% means the quantity cannot separate a repair from a "
      "regression, so no threshold on it can keep the first while rejecting the "
      "second.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
