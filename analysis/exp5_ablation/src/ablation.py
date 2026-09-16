#!/usr/bin/env python3
"""Ablations, assembled across benchmarks and backbones.

Every number here was produced by the selection scripts under
../../feasible (validation rooms choose, test rooms evaluated once) and is
read from their JSON; this script only lays them side by side. Four axes:

  1. projection source: none / trained on the benchmark's own training rooms
     (own) / pooled Replica+Matterport3D (pooled) / pooled plus Structured3D
     self-pairs (three) / searched on validation with everything else
     (VAL_<ds>[_fixed_<src>].json)
  2. rule: three scalars (visual gate only) against four (plus the acoustic
     gate and the relative-evidence transform), structure fixed
  3. beyond the shortlist: A injected acoustic hypotheses, B shortlist
     rejection, C cell-wise product (VAL_<ds>_fixed_indomain.json variants)
  4. the cell-product rule with one scalar, lambda chosen on validation on a
     wide grid (CELL_product.json), and the projection transfer matrix under
     test-room CV (P_matrix_<ds>.json), which is the only test-CV table here
     and is marked as such.

    python src/ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import DS, BB, NAME, RES, C, write_md  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "feasible"))
from run_projection_matrix import IDENT, tag_of  # noqa: E402


def load(p):
    p = RES / p
    return json.loads(p.read_text()) if p.exists() else None


def g(j, bb, variant="selected"):
    r = (j or {}).get("results", {}).get(bb, {}).get(variant)
    return r if r and "gain" in r else None


def fmt(r):
    return "–" if r is None else f"{100*r['gain']:+.1f} [{100*r['ci'][0]:+.1f}, {100*r['ci'][1]:+.1f}]"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    J, L = {}, []
    W = L.append
    W("## Table 1. Projection source (structure and three-scalar rule fixed; scalars chosen on validation), gain at 1 m\n")
    W("| benchmark | backbone | vision | none | own W | pooled W | three W | searched (source, structure, rule on validation) |")
    W("|---|---|---|---|---|---|---|---|")
    for ds, _ in DS:
        own = load(f"VAL_{ds}_fixed_indomain.json"); pooled = load(f"VAL_{ds}_fixed_B.json"); three = load(f"VAL_{ds}_fixed_T.json"); searched = load(f"VAL_{ds}.json")
        for bb, _ in BB:
            o = g(own, bb)
            if o is None:
                continue
            row = dict(vision=o["vision"], none=g(own, bb, "no projection"), own=o, pooled=g(pooled, bb), three=g(three, bb), searched=g(searched, bb))
            J.setdefault("source", {})[f"{ds}/{bb}"] = {k: (v if k == "vision" else (None if v is None else v["gain"])) for k, v in row.items()}
            W(f"| {NAME[ds]} | {NAME[bb]} | {100*o['vision']:.1f} | {fmt(row['none'])} | **{fmt(o)}** | {fmt(row['pooled'])} | {fmt(row['three'])} | {fmt(row['searched'])} |")

    W("\n## Table 2. Rule: three scalars against four, own W, structure fixed\n")
    W("| benchmark | backbone | three-scalar rule | four-scalar rule (acoustic gate + relative evidence) |")
    W("|---|---|---|---|")
    for ds, _ in DS:
        j = load(f"VAL_{ds}.json")            # the searched run records both rules' best
        for bb, _ in BB:
            a, b = g(j, bb, "three-scalar rule"), g(j, bb, "four-scalar rule")
            if a is None:
                continue
            J.setdefault("rule", {})[f"{ds}/{bb}"] = dict(three=a["gain"], four=b["gain"] if b else None)
            W(f"| {NAME[ds]} | {NAME[bb]} | {fmt(a)} | {fmt(b)} |")

    W("\n## Table 3. Beyond the visual shortlist (settings chosen on validation), gain at 1 m\n")
    W("| benchmark | backbone | hypothesis rule | A: injected acoustic hypotheses | B: shortlist rejection | C: cell-wise product (16-point grid) |")
    W("|---|---|---|---|---|---|")
    for ds, _ in DS:
        j = load(f"VAL_{ds}_fixed_indomain.json")
        for bb, _ in BB:
            o = g(j, bb)
            if o is None:
                continue
            A, B, Cc = g(j, bb, "A: injected acoustic hypotheses"), g(j, bb, "B: shortlist rejection"), g(j, bb, "C: cell-wise product")
            J.setdefault("beyond", {})[f"{ds}/{bb}"] = dict(ours=o["gain"], A=A["gain"] if A else None, B=B["gain"] if B else None, C=Cc["gain"] if Cc else None,
                                                        A_setting=A["setting"] if A else None, B_setting=B["setting"] if B else None)
            W(f"| {NAME[ds]} | {NAME[bb]} | {fmt(o)} | {fmt(A)} ({A['setting'] if A else ''}) | {fmt(B)} ({B['setting'] if B else ''}) | {fmt(Cc)} |")

    W("\n## Table 4. The cell-product rule, one scalar λ chosen on validation (wide grid), against the hypothesis rule\n")
    W("| benchmark | backbone | λ | val @1m | hypothesis rule | cell product |")
    W("|---|---|---|---|---|---|")
    cp = load("CELL_product.json") or {}
    for ds, _ in DS:
        j = load(f"VAL_{ds}_fixed_indomain.json")
        for bb, _ in BB:
            o = g(j, bb); c = cp.get(f"{ds}/{bb}")
            if o is None or c is None:
                continue
            J.setdefault("cell", {})[f"{ds}/{bb}"] = dict(lam=c["lam"], val=c["val"], ours=o["gain"], cell=c["gain"], cell_ci=c["ci"])
            W(f"| {NAME[ds]} | {NAME[bb]} | {c['lam']:g} | {100*c['val']:.1f} | {fmt(o)} | {100*c['gain']:+.1f} [{100*c['ci'][0]:+.1f}, {100*c['ci'][1]:+.1f}] |")

    W("\n## Table 5. Projection transfer matrix (test-room CV, three scalars refit per fold; not the headline protocol)\n")
    W("| benchmark | backbone | none | R | M | B | T |")
    W("|---|---|---|---|---|---|---|")
    for ds, _ in DS:
        j = load(f"P_matrix_{ds}.json")
        res = (j or {}).get("results", {})
        for bb, _ in BB:
            cells = []
            for src in (None, "R", "M", "B", "T"):
                t = IDENT[(ds, bb)] if src is None else tag_of(ds, bb, src)
                r = res.get(t); cells.append(f"{100*r['gain']:+.1f}" if r else "–")
            J.setdefault("matrix", {})[f"{ds}/{bb}"] = cells
            W(f"| {NAME[ds]} | {NAME[bb]} | " + " | ".join(cells) + " |")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "ablation.json").write_text(json.dumps(J, indent=1))
    write_md(HERE / "md" / "ablation.md", "Ablations", L)

    # ------------------------------------------------------------ figure: sources and rules
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    keys = [("none", "none"), ("own", "own W"), ("pooled", "pooled W"), ("three", "three W"), ("searched", "searched")]
    for ax, (ds, name) in zip(axes, DS):
        for j, (bb, lab) in enumerate(BB):
            r = J.get("source", {}).get(f"{ds}/{bb}")
            if not r:
                continue
            ax.plot(range(len(keys)), [100 * r[k] if r[k] is not None else np.nan for k, _ in keys], "o-", label=lab)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], rotation=20); ax.set_title(name); ax.axhline(0, color="#999", lw=0.6)
    axes[0].set_ylabel("gain at 1 m (points)"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); (HERE / "figs").mkdir(exist_ok=True); fig.savefig(HERE / "figs" / "projection_source.png", dpi=170)
    print("wrote md/ablation.md, data/, figs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
