#!/usr/bin/env python3
"""Can the system tell when its visual shortlist has collapsed, and switch?

Under heavy corruption the visual posterior stops covering the true cell and
the hypothesis rule has nothing right to choose from; the acoustic score alone
(36% on Replica) then beats it. The earlier attempt to detect this from the
visual log-odds between the two best hypotheses found no signal. This script
tries indicators that look at the whole posterior and at the agreement of the
two modalities, all computable at inference from the dumped fields:

    entropy      normalised entropy of softmax(log pi) over the cells
    pmax         mass of the posterior's largest cell
    za_at_vis    the standardised acoustic score at vision's own argmax cell:
                 when vision is right, sound tends to agree there; when it has
                 collapsed, its argmax is a random cell and the score is ~0
    agree        the visual posterior mass within 1 m of the acoustic top-1

For each indicator it reports how well it separates queries whose shortlist
still covers the truth from those where it does not (AUROC), on the degraded
test levels, and then builds a switch: below/above a threshold the answer is
the acoustic top-1 (or the cell-product rule) instead of the hypothesis rule.
The threshold is chosen on *degraded validation queries*, never on test.

    python feasible/run_collapse_switch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from run_degradation_final import LEVELS, LABEL, tag as deg_tag  # noqa: E402
from run_cell_product import load_fields, errors as cell_errors  # noqa: E402
from scripts.room_cv_eval import read, group  # noqa: E402

AN = ROOT / "outputs" / "analysis"
VAL_LEVELS = [("clean", None), ("blur", 8), ("blur", 16), ("dark", 0.05), ("dark", 0.02), ("noise", 100), ("occlude", 0.5)]


def val_tag(fam, lvl):
    return "unloc_valprojR" if fam == "clean" else f"unloc_valdegP_{fam}_{lvl}"


def indicators(F):
    out = {k: [] for k in ("entropy", "pmax", "za_at_vis", "agree")}
    for lv, za, d in zip(F["logv"], F["za"], F["dist"]):
        lv = lv.astype(np.float64); p = np.exp(lv - lv.max()); p /= p.sum()
        n = p.size
        out["entropy"].append(float(-(p * np.log(p + 1e-300)).sum() / np.log(n)))
        out["pmax"].append(float(p.max()))
        kv = int(np.argmax(lv)); out["za_at_vis"].append(float(za[kv]))
        ka = int(np.argmax(za))
        # mass within 1 m of the acoustic top-1: needs distances between cells,
        # which the dump does not carry; the distance to truth is used as the
        # frame instead, so 'agree' is the mass within 1 m of *truth* when the
        # acoustic pick is itself within 1 m, else 0 (an upper bound on what a
        # geometric version could see)
        out["agree"].append(float(p[d < 1].sum()) if d[ka] < 1 else 0.0)
    return {k: np.array(v) for k, v in out.items()}


def hyp_errors(tag, cfg, vc, ac):
    from track1_core.likelihood.mode_fusion import choose
    Q = read(AN / f"queries_raw_scan_open_{tag}.csv"); M = group(read(AN / f"modes_raw_scan_open_{tag}.csv"))
    e = np.array([M[str(q)]["dist_gt_m"][choose(M[str(q)][vc], M[str(q)][ac], cfg)[0]] for q in Q["query_id"]])
    cov = np.array([(M[str(q)]["dist_gt_m"] < 1).any() for q in Q["query_id"]])
    ac_alone = Q["e_ac"]
    return e, cov, ac_alone, [str(q) for q in Q["query_id"]]


def auroc(score, positive):
    # probability a random positive scores above a random negative
    pos, neg = score[positive], score[~positive]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> int:
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, evidence_columns
    sel = json.loads((HERE / "results" / "VAL_replica_fixed_indomain.json").read_text())["results"]["unloc"]["selected"]
    w, s, tv, ta = sel["scalars"]
    cfg = ModeFusionConfig(vis_evidence=sel["structure"][0], ac_evidence=sel["structure"][1], rule="continuous",
                           weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta, ac_transform="standard")
    vc, ac = evidence_columns(cfg)
    C = json.loads((HERE / "results" / "CELL_product.json").read_text()).get("replica/unloc", {})
    bC, TC = C.get("beta", 1.0), C.get("T", 2.0)

    out = []
    W = out.append
    W("# Detecting visual collapse and switching, UnLoc on Replica\n")

    # ---------------------------------------------- 1. do the indicators see it?
    W("## Separation of 'shortlist covers truth' from 'shortlist broken' (AUROC, higher means the indicator rises with collapse)\n")
    W("| level | broken % | entropy | 1-pmax | -za_at_vis | 1-agree |")
    W("|---|---|---|---|---|---|")
    test = {}
    for fam, lvl in LEVELS:
        t = deg_tag(fam, lvl)
        F = load_fields("raw_scan_open", t)
        if F is None or not (AN / f"queries_raw_scan_open_{t}.csv").exists():
            continue
        e, cov, ac_alone, qids = hyp_errors(t, cfg, vc, ac)
        ind = indicators(F)
        eC = cell_errors(F, bC, TC)
        test[(fam, lvl)] = dict(e=e, cov=cov, ac=ac_alone, eC=eC, ind=ind)
        label = LABEL[fam].format(lvl) if lvl is not None else "clean"
        W(f"| {label} | {100*(~cov).mean():.0f} | {auroc(ind['entropy'], ~cov):.2f} | {auroc(1-ind['pmax'], ~cov):.2f} | "
          f"{auroc(-ind['za_at_vis'], ~cov):.2f} | {auroc(1-ind['agree'], ~cov):.2f} |")

    # ---------------------------------------------- 2. thresholds on degraded validation
    val = {}
    for fam, lvl in VAL_LEVELS:
        t = val_tag(fam, lvl)
        F = load_fields("raw_scan_open", t)
        if F is None or not (AN / f"queries_raw_scan_open_{t}.csv").exists():
            continue
        e, cov, ac_alone, _ = hyp_errors(t, cfg, vc, ac)
        val[(fam, lvl)] = dict(e=e, ac=ac_alone, eC=cell_errors(F, bC, TC), ind=indicators(F))
    W(f"\nValidation levels available: {[LABEL[f].format(l) if l is not None else 'clean' for f, l in val]}\n")

    def pooled(dct, key):
        return np.concatenate([v[key] for v in dct.values()])

    results = {}
    for name, sign in (("entropy", +1), ("pmax", -1), ("za_at_vis", -1), ("agree", -1)):
        sv = sign * np.concatenate([v["ind"][name] for v in val.values()])
        ev, av, cv = pooled(val, "e"), pooled(val, "ac"), pooled(val, "eC")
        base = float((ev < 1).mean())
        for target, ealt in (("acoustic", av), ("cellprod", cv)):
            best = (base, None)
            for th in np.quantile(sv, np.linspace(0.02, 0.98, 49)):
                r = float(np.where(sv > th, ealt < 1, ev < 1).mean())
                if r > best[0] + 0.005:
                    best = (r, float(th))
            results[(name, target)] = best
    W("## Switch thresholds chosen on the pooled degraded validation queries\n")
    W("| indicator | switch to | val recall, no switch | with switch | threshold |")
    W("|---|---|---|---|---|")
    ev = pooled(val, "e"); base = float((ev < 1).mean())
    for (name, target), (r, th) in results.items():
        W(f"| {name} | {target} | {100*base:.1f} | {100*r:.1f} | {'none helps' if th is None else f'{th:.3f}'} |")

    # ---------------------------------------------- 3. apply, frozen, on every test level
    W("\n## Test levels with the chosen switches, frozen\n")
    heads = ["level", "vision", "acoustic", "ours", "C"] + [f"{n}→{t}" for (n, t) in results]
    W("| " + " | ".join(heads) + " |"); W("|" + "---|" * len(heads))
    rows = []
    for (fam, lvl), v in test.items():
        label = LABEL[fam].format(lvl) if lvl is not None else "clean"
        cells = [label, f"{100*np.mean(v['e']<1):.1f}"]
        vis = np.array([d[int(np.argmax(lv))] for lv, d in zip(load_fields('raw_scan_open', deg_tag(fam, lvl))['logv'], load_fields('raw_scan_open', deg_tag(fam, lvl))['dist'])])
        cells = [label, f"{100*np.mean(vis<1):.1f}", f"{100*np.mean(v['ac']<1):.1f}", f"{100*np.mean(v['e']<1):.1f}", f"{100*np.mean(v['eC']<1):.1f}"]
        row = dict(label=label, vision=float(np.mean(vis < 1)), acoustic=float(np.mean(v["ac"] < 1)),
                   ours=float(np.mean(v["e"] < 1)), C=float(np.mean(v["eC"] < 1)))
        for (name, target), (r, th) in results.items():
            sign = +1 if name == "entropy" else -1
            if th is None:
                cells.append("–"); row[f"{name}->{target}"] = None; continue
            sv = sign * v["ind"][name]; ealt = v["ac"] if target == "acoustic" else v["eC"]
            es = np.where(sv > th, ealt, v["e"])
            cells.append(f"{100*np.mean(es<1):.1f} ({100*np.mean(sv>th):.0f}%)")
            row[f"{name}->{target}"] = float(np.mean(es < 1))
        rows.append(row)
        W("| " + " | ".join(cells) + " |")
    (HERE / "results" / "SWITCH_collapse.md").write_text("\n".join(out) + "\n")
    (HERE / "results" / "SWITCH_collapse.json").write_text(json.dumps(dict(rows=rows, thresholds={f"{k[0]}->{k[1]}": v for k, v in results.items()}), indent=2))
    print("\n".join(out)); print("wrote feasible/results/SWITCH_collapse.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
