#!/usr/bin/env python3
"""Every fusion rule on one axis, plus the two oracles that say what is missing.

Reads the extracted mode tables, so no network runs here and the whole
comparison is a couple of seconds. That matters: thresholds are tuned by an
exhaustive sweep on the validation collection, which would be unaffordable if
each evaluation needed a GPU pass, and which is the only way to keep the
reported numbers honest.

The split is strict. `replica_f` and `replica_g` are independent pose sets over
the same three rooms. Every threshold, weight and evidence choice is selected by
maximising recall at 1 m on the fit collection, and every reported number comes
from the other one. This holds out the protocol, not the room, which is stated
plainly rather than papered over.

Two oracles bound the problem from above:

    mode oracle       picks the hypothesis nearest truth. Not a method; it says
                      how much is reachable if acoustic discrimination were
                      perfect, given the hypotheses vision actually proposes.
    matched-domain    the same rules run on `floorplan_closed` queries, where
                      the recording and the candidate come from the same
                      geometry. If the rules approach the oracle there and not
                      here, the fusion is finished and the domain gap is the
                      whole remaining problem.

    python scripts/mode_fusion_eval.py --backbone unloc --fit-collection replica_f
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--backbone", default="unloc")
    p.add_argument("--fit-collection", default="replica_f")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "analysis" / "fusion_comparison.md")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "mode_fusion_eval.json")
    return p.parse_args()


def read_csv(path: Path) -> dict[str, np.ndarray]:
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
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


def group_modes(M: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Per-query arrays, hypotheses ordered by visual strength as extracted."""
    out: dict[str, dict[str, list]] = {}
    cols = [c for c in M if c not in ("query_id", "scene", "collection")]
    for i, qid in enumerate(M["query_id"]):
        d = out.setdefault(qid, {c: [] for c in cols})
        for c in cols:
            d[c].append(M[c][i])
    for qid, d in out.items():
        order = np.argsort(np.asarray(d["mode"], dtype=int))
        out[qid] = {c: np.asarray(v, dtype=object if isinstance(v[0], str) else float)[order]
                    for c, v in d.items()}
    return out


def boot_paired(a: np.ndarray, b: np.ndarray, n: int, rng):
    d = b.astype(float) - a.astype(float)
    if d.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, d.size, size=(n, d.size))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    args = parse_args()
    from track1_core.likelihood.mode_fusion import (
        ModeFusionConfig, choose, contradiction_is_degenerate, evidence_columns,
    )
    from track1_core.provenance import stamp

    d = args.analysis_dir
    tables = {}
    for cond in ("raw_scan_open", "floorplan_closed"):
        qp = d / f"queries_{cond}_{args.backbone}.csv"
        mp = d / f"modes_{cond}_{args.backbone}.csv"
        if qp.exists() and mp.exists():
            tables[cond] = (read_csv(qp), group_modes(read_csv(mp)))
    if "raw_scan_open" not in tables:
        print("run extract_mode_table.py first")
        return 1

    rng = np.random.default_rng(args.seed)
    out: list[str] = []
    W = out.append
    results: dict = {}

    def evaluate(cond: str, cfg: ModeFusionConfig, qids: list[str]):
        """Recall at 1 m, at 1 m and 30 degrees, and the per-query error vector."""
        _, MM = tables[cond]
        vc, acc = evidence_columns(cfg)
        err, orn, used = [], [], []
        for qid in qids:
            m = MM[qid]
            k, u = choose(m[vc], m[acc], cfg)
            err.append(m["dist_gt_m"][k]); orn.append(m["yaw_err_deg"][k]); used.append(u)
        err, orn = np.asarray(err), np.asarray(orn)
        return err, orn, np.asarray(used)

    def ids(cond: str, coll: str | None):
        Q, _ = tables[cond]
        keep = np.ones(len(Q["query_id"]), bool) if coll is None else (Q["collection"] == coll)
        return [str(x) for x in Q["query_id"][keep]]

    fit_ids = ids("raw_scan_open", args.fit_collection)
    Q, _ = tables["raw_scan_open"]
    rep_coll = [c for c in np.unique(Q["collection"]) if c != args.fit_collection]
    rep_ids = ids("raw_scan_open", rep_coll[0]) if rep_coll else fit_ids
    W(f"# Mode-level fusion, {args.backbone}\n")
    W(f"Thresholds fitted on {args.fit_collection} ({len(fit_ids)} queries), "
      f"reported on {rep_coll[0] if rep_coll else args.fit_collection} "
      f"({len(rep_ids)} queries). Position and yaw always come from the chosen "
      f"hypothesis's own visual centre, so sound never moves the answer off a "
      f"place vision proposed and never touches yaw.\n")

    # ---------------------------------------------------------------- B1
    W("\n## B1/B2. Which evidence summarises a hypothesis best\n")
    W("Selected on the fit collection with the selective rule at its default "
      "thresholds. Reported here on the fit collection only, since this is the "
      "selection step.\n")
    W("| visual evidence | acoustic evidence | fit recall @1m |")
    W("|---|---|---|")
    best_ev, best_score = ("lse", "quantile"), -1.0
    for ve, ae in itertools.product(("centre", "max", "lse"),
                                    ("centre", "max", "quantile", "lse")):
        c = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="selective",
                             tau_v=0.05, tau_a=0.3)
        e, _, _ = evaluate("raw_scan_open", c, fit_ids)
        s = float((e < 1).mean())
        W(f"| {ve} | {ae} | {100*s:.1f}% |")
        if s > best_score:
            best_ev, best_score = (ve, ae), s
    W(f"\nChosen: visual `{best_ev[0]}`, acoustic `{best_ev[1]}`.\n")
    ve, ae = best_ev

    # ---------------------------------------------------------------- B4 sweep
    W("\n## B4. Threshold sweep for the dual-confidence gate\n")
    tv_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf]
    ta_grid = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.2]
    best, best_s = None, -1.0
    for tv, ta in itertools.product(tv_grid, ta_grid):
        c = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="selective",
                             tau_v=tv, tau_a=ta)
        e, _, _ = evaluate("raw_scan_open", c, fit_ids)
        s = float((e < 1).mean())
        if s > best_s:
            best, best_s = (tv, ta), s
    W(f"Best on the fit collection: visual ambiguity below {best[0]:g}, "
      f"acoustic margin above {best[1]:g}, giving {100*best_s:.1f}%.\n")
    W("The acoustic half of the gate is the interesting one. Holding the visual "
      "threshold at its best value, the fit recall as the acoustic threshold "
      "moves says whether acoustic self-confidence carries anything:\n")
    W("| acoustic threshold | fit recall @1m | queries where audio acts |")
    W("|---|---|---|")
    for ta in ta_grid:
        c = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="selective",
                             tau_v=best[0], tau_a=ta)
        e, _, u = evaluate("raw_scan_open", c, fit_ids)
        W(f"| {ta:g} | {100*(e<1).mean():.1f}% | {100*u.mean():.0f}% |")

    # continuous variant, two parameters only
    cbest, cbest_s = None, -1.0
    for w, sc in itertools.product((0.25, 0.5, 1.0, 2.0), (0.02, 0.05, 0.1, 0.3)):
        c = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="continuous",
                             weight=w, tau_v=best[0], tau_a=best[1], sigmoid_scale=sc)
        e, _, _ = evaluate("raw_scan_open", c, fit_ids)
        s = float((e < 1).mean())
        if s > cbest_s:
            cbest, cbest_s = (w, sc), s
    rbest, rbest_s = None, -1.0
    for w in (0.1, 0.25, 0.5, 1.0, 2.0):
        c = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="relative", weight=w)
        e, _, _ = evaluate("raw_scan_open", c, fit_ids)
        s = float((e < 1).mean())
        if s > rbest_s:
            rbest, rbest_s = w, s
    lbest, lbest_s = None, -1.0
    for lam in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        c = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="contradiction", lam=lam)
        e, _, _ = evaluate("raw_scan_open", c, fit_ids)
        s = float((e < 1).mean())
        if s > lbest_s:
            lbest, lbest_s = lam, s

    # ---------------------------------------------------------------- main table
    rules = {
        "vision alone": ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="vision"),
        "mode rerank, unconditional": ModeFusionConfig(vis_evidence=ve, ac_evidence=ae,
                                                       rule="mode_rerank"),
        f"relative evidence (w={rbest:g})": ModeFusionConfig(
            vis_evidence=ve, ac_evidence=ae, rule="relative", weight=rbest),
        f"contradiction only (lam={lbest:g})": ModeFusionConfig(
            vis_evidence=ve, ac_evidence=ae, rule="contradiction", lam=lbest),
        f"selective, visual gate only (tau_v={best[0]:g})": ModeFusionConfig(
            vis_evidence=ve, ac_evidence=ae, rule="selective",
            tau_v=best[0], tau_a=-np.inf),
        f"selective, dual gate (tau_v={best[0]:g}, tau_a={best[1]:g})": ModeFusionConfig(
            vis_evidence=ve, ac_evidence=ae, rule="selective", tau_v=best[0], tau_a=best[1]),
        f"continuous dual gate (w={cbest[0]:g}, s={cbest[1]:g})": ModeFusionConfig(
            vis_evidence=ve, ac_evidence=ae, rule="continuous", weight=cbest[0],
            tau_v=best[0], tau_a=best[1], sigmoid_scale=cbest[1]),
    }

    W("\n## Comparison on the held-out collection\n")
    W("`cell` rows are the existing rules, recomputed from the same tables. "
      "`mode` rows are new.\n")
    W("| level | rule | 0.1 m | 0.5 m | 1 m | 1m/30deg | median | gain @1m | 95% CI |")
    W("|---|---|---|---|---|---|---|---|---|")

    Qr = {str(k): i for i, k in enumerate(Q["query_id"])}
    ridx = np.array([Qr[q] for q in rep_ids])
    base_err = Q["e_vis"][ridx]
    base_ok = (base_err < 1).astype(float)

    def row(level, name, err, orn):
        m, lo, hi = boot_paired(base_ok, (err < 1).astype(float), args.boot, rng)
        # sound supplies no yaw, so a rule with no visual hypothesis behind it
        # has no heading to report and the column is left empty rather than zero
        o30 = ("-" if not np.isfinite(orn).any()
               else f"{100*np.mean((err<1)&(orn<30)):.1f}%")
        W(f"| {level} | {name} | {100*(err<0.1).mean():.1f}% | {100*(err<0.5).mean():.1f}% | "
          f"{100*(err<1).mean():.1f}% | {o30} | {np.median(err):.2f} m | "
          f"{100*m:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
        return dict(r01=float((err < 0.1).mean()), r05=float((err < 0.5).mean()),
                    r10=float((err < 1).mean()),
                    r10_30=(float(np.mean((err < 1) & (orn < 30)))
                            if np.isfinite(orn).any() else None),
                    median=float(np.median(err)), gain=float(m), ci=[float(lo), float(hi)])

    for name, col, ocol in (("vision", "e_vis", "orn_vis"),
                            ("acoustic alone", "e_ac", None),
                            ("cell rerank top-50", "e_rerank", "orn_rerank"),
                            ("cell log-rank fusion", "e_fused", "orn_fused")):
        e = Q[col][ridx]
        o = Q[ocol][ridx] if ocol else np.full(len(e), np.nan)
        results[f"cell/{name}"] = row("cell", name, e, o)
    degen = 0
    for name, cfg in rules.items():
        e, o, u = evaluate("raw_scan_open", cfg, rep_ids)
        results[f"mode/{name}"] = row("mode", name, e, o)
        results[f"mode/{name}"]["audio_used"] = float(u.mean())
    _, MMd = tables["raw_scan_open"]
    vcd, acd = evidence_columns(rules["vision alone"])
    degen = float(np.mean([contradiction_is_degenerate(MMd[q][acd]) for q in rep_ids]))
    W(f"\nThe contradiction variant is algebraically the relative-evidence rule "
      f"on {100*degen:.0f}% of these queries, because with ten hypotheses the "
      f"relative evidence is negative for all of them and the clip does nothing. "
      f"Their rows are identical for that reason, not by coincidence; see "
      f"`contradiction_is_degenerate`.\n")

    # ---------------------------------------------------------------- C1
    W("\n## C1. Oracle over the same hypotheses\n")
    _, MM = tables["raw_scan_open"]
    oe = np.array([MM[q]["dist_gt_m"].min() for q in rep_ids])
    oo = np.array([MM[q]["yaw_err_deg"][int(MM[q]["dist_gt_m"].argmin())] for q in rep_ids])
    results["oracle/mode"] = row("oracle", "pick the hypothesis nearest truth", oe, oo)
    W("\nThis is the ceiling of any rule that chooses among these hypotheses. "
      "The distance between the best real rule and this number is what better "
      "acoustic discrimination could still buy.\n")

    # ---------------------------------------------------------------- C2
    if "floorplan_closed" in tables:
        W("\n## C2. The same rules with matched-geometry acoustics\n")
        W("Query and candidate rendered on the same wall-only mesh, so the "
          "furniture gap is removed and nothing else changes. The visual side is "
          "identical.\n")
        W("| level | rule | 0.1 m | 0.5 m | 1 m | 1m/30deg | median | gain @1m | 95% CI |")
        W("|---|---|---|---|---|---|---|---|---|")
        Qm, _ = tables["floorplan_closed"]
        Qmr = {str(k): i for i, k in enumerate(Qm["query_id"])}
        mrep = [q for q in rep_ids if q in Qmr]
        midx = np.array([Qmr[q] for q in mrep])
        base_ok_m = (Qm["e_vis"][midx] < 1).astype(float)
        saved = base_ok
        base_ok = base_ok_m
        for name, col, ocol in (("vision", "e_vis", "orn_vis"),
                                ("acoustic alone", "e_ac", None),
                                ("cell log-rank fusion", "e_fused", "orn_fused")):
            e = Qm[col][midx]
            o = Qm[ocol][midx] if ocol else np.full(len(e), np.nan)
            results[f"matched/cell/{name}"] = row("cell", name, e, o)
        for name, cfg in rules.items():
            e, o, u = evaluate("floorplan_closed", cfg, mrep)
            results[f"matched/mode/{name}"] = row("mode", name, e, o)
        _, MMm = tables["floorplan_closed"]
        oe2 = np.array([MMm[q]["dist_gt_m"].min() for q in mrep])
        oo2 = np.array([MMm[q]["yaw_err_deg"][int(MMm[q]["dist_gt_m"].argmin())] for q in mrep])
        results["matched/oracle/mode"] = row("oracle", "hypothesis nearest truth", oe2, oo2)
        base_ok = saved
    else:
        W("\n## C2. Matched-geometry acoustics\n\nNot available yet.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dict(
        backbone=args.backbone, fit_collection=args.fit_collection,
        evidence=dict(visual=ve, acoustic=ae),
        thresholds=dict(tau_v=float(best[0]), tau_a=float(best[1]),
                        relative_weight=float(rbest), lam=float(lbest),
                        continuous=[float(cbest[0]), float(cbest[1])]),
        results=results, provenance=stamp()), indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
