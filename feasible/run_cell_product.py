#!/usr/bin/env python3
"""The cell-wise product rule, evaluated from the dumped fields.

    k* = argmax_c  log pi(c) + lambda * z_a(c)

over every candidate cell, where pi is the visual posterior (max over heading)
and z_a the projected acoustic score standardised over the cells of the query.
No hypotheses, no gate: one scalar. (An argmax is invariant to a positive
scale, so the two temperatures of the first version, beta on the log-posterior
and T on the score, were one parameter lambda = 1/(beta T) in disguise, which
is why that grid always looked as if the optimum sat on its edge.) lambda is
chosen on the validation rooms over a log-spaced grid and the test rooms are
evaluated once. For UnLoc on Replica the chosen pair is also
applied, frozen, to every degradation level.

Reads outputs/analysis/fields_<cond>_<tag>.npz written by extract_mode_table.py.

    python feasible/run_cell_product.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from run_projection_matrix import COND  # noqa: E402
from run_beyond_shortlist import TEST_TAG, SRC  # noqa: E402
from run_val_pipeline import val_tag  # noqa: E402
from run_degradation_final import LEVELS, LABEL, tag as deg_tag  # noqa: E402

AN = ROOT / "outputs" / "analysis"
LAMBDAS = tuple(float(x) for x in np.round(np.logspace(np.log10(0.005), np.log10(20.0), 25), 4))
LABELS = {"f3loc": "F3Loc mono", "unloc": "UnLoc", "disco": "DisCo-FLoc RRP"}
TH = (0.1, 0.5, 1.0, 2.0, 5.0)


def load_fields(cond, tag):
    p = AN / f"fields_{cond}_{tag}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return dict(scene=[str(s) for s in z["scene"]],
                logv=[np.asarray(a, np.float32) for a in z["logv"]],
                za=[np.asarray(a, np.float32) for a in z["za"]],
                dist=[np.asarray(a, np.float32) for a in z["dist"]])


def errors(F, lam, _T=None):
    if _T is not None:            # old (beta, T) call: the same rule with lambda = 1/(beta T)
        lam = 1.0 / (lam * _T)
    return np.array([d[int(np.argmax(lv + lam * za))] for lv, za, d in zip(F["logv"], F["za"], F["dist"])])


def vision_errors(F):
    return np.array([d[int(np.argmax(lv))] for lv, d in zip(F["logv"], F["dist"])])


def select(F):
    best = (-1.0, None)
    table = {}
    for lam in LAMBDAS:
        r = float((errors(F, lam) < 1).mean()); table[lam] = r
        if r > best[0]:
            best = (r, lam)
    return best[1], best[0], table


def main() -> int:
    rng = np.random.default_rng(0)
    out, js = [], {}
    W = out.append
    W("# Cell-wise product rule, temperatures chosen on validation, tested once\n")
    W("| benchmark | backbone | λ | val @1m | test vision | test C | gain | 95% CI | edge of grid? |")
    W("|---|---|---|---|---|---|---|---|---|")
    for ds in ("replica", "mp3d", "s3d"):
        cond = COND[ds]
        for bb in ("f3loc", "unloc", "disco"):
            Fv = load_fields(cond, val_tag(bb, SRC[ds], ds))
            Ft = load_fields(cond, TEST_TAG[(ds, bb)])
            if Fv is None or Ft is None:
                W(f"| {ds} | {LABELS[bb]} | (fields missing) | | | | | | | |"); continue
            lam, rv, table = select(Fv)
            e = errors(Ft, lam); v = vision_errors(Ft)
            d = (e < 1).astype(float) - (v < 1).astype(float)
            m = d[rng.integers(0, d.size, size=(10000, d.size))].mean(axis=1)
            lo, hi = np.percentile(m, [2.5, 97.5])
            edge = lam in (LAMBDAS[0], LAMBDAS[-1])
            W(f"| {ds} | {LABELS[bb]} | {lam:g} | {100*rv:.1f} | {100*(v<1).mean():.1f} | {100*(e<1).mean():.1f} | "
              f"{100*d.mean():+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {'yes' if edge else 'no'} |")
            js[f"{ds}/{bb}"] = dict(lam=lam, beta=1.0, T=1.0 / lam, val=rv, vision=float((v < 1).mean()), ours=float((e < 1).mean()),
                                    gain=float(d.mean()), ci=[float(lo), float(hi)], n=int(e.size),
                                    recalls={f"{t}m": float((e < t).mean()) for t in TH},
                                    vision_recalls={f"{t}m": float((v < t).mean()) for t in TH},
                                    median=float(np.median(e)), vision_median=float(np.median(v)),
                                    val_table={f"{k:g}": r for k, r in table.items()})
            print(out[-1], flush=True)

    # ------------------------------------------- degradation, UnLoc on Replica
    key = "replica/unloc"
    if key in js:
        lam = js[key]["lam"]
        W(f"\n## UnLoc on Replica under degradation, λ={lam:g} frozen\n")
        W("| degradation | vision | C | gain |")
        W("|---|---|---|---|")
        js["degradation"] = []
        for fam, lvl in LEVELS:
            F = load_fields("raw_scan_open", deg_tag(fam, lvl))
            if F is None:
                continue
            e = errors(F, lam); v = vision_errors(F)
            label = LABEL[fam].format(lvl) if lvl is not None else "clean"
            W(f"| {label} | {100*(v<1).mean():.1f} | {100*(e<1).mean():.1f} | {100*((e<1).mean()-(v<1).mean()):+.1f} |")
            js["degradation"].append(dict(label=label, vision=float((v < 1).mean()), C=float((e < 1).mean())))
    (HERE / "results" / "CELL_product.md").write_text("\n".join(out) + "\n")
    (HERE / "results" / "CELL_product.json").write_text(json.dumps(js, indent=2))
    print("\n".join(out[-6:])); print("wrote feasible/results/CELL_product.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
