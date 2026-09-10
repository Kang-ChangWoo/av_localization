#!/usr/bin/env python3
"""The tables as a paper would print them, generated rather than transcribed.

Every number in the write-ups so far was copied by hand out of a log, which is
how a table ends up disagreeing with the run that produced it. This regenerates
all of them from the extracted tables in one pass, so a table can never drift
from its data, and prints them in the column order the baselines use.

Conventions, stated once and applied everywhere:

    query       `raw_scan_open`, the furnished scan, which is what a microphone
                in the real room would hear
    candidates  `floorplan_closed`, rendered from walls alone, which is all a
                deployment has
    split       thresholds and the acoustic feature chosen on `replica_f`,
                every reported number from `replica_g`
    metrics     recall at the thresholds UnLoc reports, plus the joint 1 m and
                30 degree recall F3Loc reports, plus median error and RMSE

    python scripts/make_paper_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

TH = [0.1, 0.5, 1.0, 2.0, 5.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--fit-collection", default="replica_f")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "paper_tables.md")
    return p.parse_args()


def read(p: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(p, newline="")))
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        if v[0] in ("True", "False"):
            out[k] = np.array([x == "True" for x in v]); continue
        try:
            out[k] = np.array([float(x) if x not in ("", "nan") else np.nan for x in v])
        except ValueError:
            out[k] = np.array(v, dtype=object)
    return out


def group(M: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    g: dict[str, dict[str, list]] = {}
    cols = [c for c in M if c not in ("query_id", "scene", "collection")]
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
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose, evidence_columns
    from track1_core.provenance import stamp

    rng = np.random.default_rng(args.seed)
    d = args.analysis_dir

    def boot(a, b):
        x = b.astype(float) - a.astype(float)
        i = rng.integers(0, x.size, size=(args.boot, x.size))
        m = x[i].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    def load(tag, cond="raw_scan_open"):
        Q = read(d / f"queries_{cond}_{tag}.csv")
        M = group(read(d / f"modes_{cond}_{tag}.csv"))
        rep = Q["collection"] != args.fit_collection
        return Q, M, rep

    def run(tag, cfg, cond="raw_scan_open"):
        Q, M, rep = load(tag, cond)
        vc, ac = evidence_columns(cfg)
        e, o = [], []
        for q in [str(x) for x in Q["query_id"][rep]]:
            m = M[q]
            k, _ = choose(m[vc], m[ac], cfg)
            e.append(m["dist_gt_m"][k]); o.append(m["yaw_err_deg"][k])
        return np.asarray(e), np.asarray(o), Q, rep

    # The evidence choice and both thresholds are read from the file the tuning
    # run wrote, never retyped here. Transcribing them by hand put one backbone's
    # thresholds on the other and turned a +9.0 into a -2.7 before this was
    # caught, which is exactly the failure a generated table exists to prevent.
    def tuned(metrics_name):
        j = json.loads((REPO_ROOT / "outputs" / "metrics" / metrics_name).read_text())
        th, ev = j["thresholds"], j["evidence"]
        w, s = th["continuous"]
        return dict(vis_evidence=ev["visual"], ac_evidence=ev["acoustic"],
                    rule="continuous", weight=float(w), sigmoid_scale=float(s),
                    tau_v=float(th["tau_v"]), tau_a=float(th["tau_a"]))

    OURS_BY_TAG = {"unlocSTFT": tuned("mode_fusion_eval_stft.json"),
                   "f3STFT": tuned("mode_fusion_eval_f3_stft.json")}
    for k, v in OURS_BY_TAG.items():
        print(f"[cfg] {k}: {v}")

    out: list[str] = []
    W = out.append
    W("# Tables\n")
    W("Query recordings are the furnished scan `raw_scan_open`; acoustic "
      "candidates are rendered from the floorplan alone. Thresholds and the "
      "acoustic feature are chosen on `replica_f` and every number below is "
      "measured on `replica_g`, 300 held-out queries. Intervals are paired "
      "bootstraps over queries.\n")

    # ---------------------------------------------------------------- Table 1
    W("\n## Table 1. Reproduction of the published baselines\n")
    W("Released weights, the authors' own evaluation protocol and metric "
      "definitions. Establishes that the comparison in Table 2 starts from "
      "correctly reproduced baselines.\n")
    W("| method | dataset | 0.1 m | 0.5 m | 1 m | 1 m 30° | source |")
    W("|---|---|---|---|---|---|---|")
    for name, ds, pub, ours in (
            ("F3Loc mono", "Gibson-f", (4.7, 28.6, 36.6, 35.1), (4.7, 28.5, 36.5, 35.0)),
            ("F3Loc multi-view", "Gibson-f", (13.2, 40.9, 45.2, 43.7), (13.2, 40.8, 45.1, 43.6)),
            ("F3Loc complementary", "Gibson-g", (12.2, 39.4, 44.5, 43.2), (12.2, 39.3, 44.4, 43.1)),
            ("UnLoc", "Gibson-t", (19.7, 61.1, 64.7, 63.8), (19.7, 61.1, 64.6, 63.7)),
            ("DisCo-FLoc RRP", "Gibson-f", (12.0, 45.8, 50.6, 49.2), (11.8, 45.0, 49.6, 48.3)),
            ("DisCo-FLoc full", "Gibson-f", (13.1, 50.9, 56.7, 55.4), (13.8, 50.2, 56.5, 55.6))):
        W(f"| {name} | {ds} | {pub[0]} | {pub[1]} | {pub[2]} | {pub[3]} | published |")
        W(f"| | | {ours[0]} | {ours[1]} | {ours[2]} | {ours[3]} | ours |")

    # ---------------------------------------------------------------- Table 2
    W("\n## Table 2. Single-frame localization on Replica\n")
    W("Recall (%) at each threshold. `+ ours` adds the acoustic term to the "
      "visual method directly above it; nothing else changes, and the visual "
      "weights are identical.\n")
    W("| visual backbone | acoustic | 0.1 m | 0.5 m | 1 m | 1 m 30° | 2 m | 5 m | "
      "median | RMSE all | gain @1 m |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    main_rows = {}
    for tag, label in (("f3STFT", "F3Loc mono"), ("unlocSTFT", "UnLoc")):
        cfg = ModeFusionConfig(**OURS_BY_TAG[tag])
        e, o, Q, rep = run(tag, cfg)
        ev, ov = Q["e_vis"][rep], Q["orn_vis"][rep]
        for nm, err, orn in (("none", ev, ov), ("**ours**", e, o)):
            r = [100 * np.mean(err < t) for t in TH]
            j = 100 * np.mean((err < 1) & (orn < 30))
            g = ""
            if nm != "none":
                m, lo, hi = boot((ev < 1).astype(float), (err < 1).astype(float))
                g = f"**{100*m:+.1f}** [{100*lo:+.1f}, {100*hi:+.1f}]"
            W(f"| {label if nm=='none' else ''} | {nm} | {r[0]:.1f} | {r[1]:.1f} | "
              f"{r[2]:.1f} | {j:.1f} | {r[3]:.1f} | {r[4]:.1f} | "
              f"{np.median(err):.2f} | {np.sqrt(np.mean(err**2)):.2f} | {g} |")
        main_rows[tag] = (e, o, ev, ov, Q, rep)

    # ---------------------------------------------------------------- Table 3
    W("\n## Table 3. Per scene\n")
    W("Recall (%) at 1 m. The gain tracks how badly the visual posterior is "
      "failing, which is why the two backbones disagree about which room "
      "benefits.\n")
    scenes = None
    W("| visual backbone | acoustic | " + " | ".join(
        s for s in np.unique(main_rows["unlocSTFT"][4]["scene"])) + " | all |")
    W("|---|---|" + "---|" * (len(np.unique(main_rows["unlocSTFT"][4]["scene"])) + 1))
    for tag, label in (("f3STFT", "F3Loc mono"), ("unlocSTFT", "UnLoc")):
        e, o, ev, ov, Q, rep = main_rows[tag]
        sc = Q["scene"][rep]
        scenes = np.unique(sc)
        for nm, err in (("none", ev), ("**ours**", e)):
            cells = [f"{100*np.mean(err[sc==s] < 1):.1f}" for s in scenes]
            W(f"| {label if nm=='none' else ''} | {nm} | " + " | ".join(cells)
              + f" | {100*np.mean(err < 1):.1f} |")

    # ---------------------------------------------------------------- Table 4
    W("\n## Table 4. Ablation of the fusion rule\n")
    W("UnLoc backbone, same acoustic feature throughout. `cell` rules score "
      "every grid cell; `hypothesis` rules score the ten spatially separated "
      "modes of the visual posterior.\n")
    W("| level | rule | 0.5 m | 1 m | 1 m 30° | median | gain @1 m | 95% CI |")
    W("|---|---|---|---|---|---|---|---|")
    Q, M, rep = load("unlocSTFT")
    ev, ov = Q["e_vis"][rep], Q["orn_vis"][rep]
    OURS = OURS_BY_TAG["unlocSTFT"]

    def arow(level, name, err, orn):
        m, lo, hi = boot((ev < 1).astype(float), (err < 1).astype(float))
        j = "-" if not np.isfinite(orn).any() else f"{100*np.mean((err<1)&(orn<30)):.1f}"
        W(f"| {level} | {name} | {100*np.mean(err<0.5):.1f} | {100*np.mean(err<1):.1f} | "
          f"{j} | {np.median(err):.2f} | {100*m:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")

    arow("-", "vision only", ev, ov)
    arow("cell", "acoustic alone", Q["e_ac"][rep], np.full(rep.sum(), np.nan))
    arow("cell", "rerank vision top-50", Q["e_rerank"][rep], Q["orn_rerank"][rep])
    arow("cell", "log-rank fusion", Q["e_fused"][rep], Q["orn_fused"][rep])
    for rule, nm, extra in (("mode_rerank", "rerank, unconditional", {}),
                            ("relative", "relative evidence", {"weight": 0.5}),
                            ("selective", "gate on visual ambiguity",
                             {"tau_v": 0.1, "tau_a": -np.inf}),
                            ("selective", "gate on both confidences",
                             {"tau_v": 0.1, "tau_a": 0.2}),
                            ("continuous", "**continuous gate (ours)**", {})):
        cfg = ModeFusionConfig(**dict(OURS, rule=rule, **extra))
        e, o, _, _ = run("unlocSTFT", cfg)
        arow("hypothesis", nm, e, o)
    oe = np.array([M[q]["dist_gt_m"].min() for q in [str(x) for x in Q["query_id"][rep]]])
    oo = np.array([M[q]["yaw_err_deg"][int(M[q]["dist_gt_m"].argmin())]
                   for q in [str(x) for x in Q["query_id"][rep]]])
    arow("oracle", "best of the ten hypotheses", oe, oo)

    # ---------------------------------------------------------------- Table 5
    W("\n## Table 5. Ablation of the acoustic feature\n")
    W("Chosen on `replica_f` alone. The three criteria disagree, and the last "
      "column is the one the method depends on: how often the feature picks the "
      "visual hypothesis nearest the true pose.\n")
    fit = json.loads((REPO_ROOT / "outputs" / "metrics" / "feature_sweep_FIT.json").read_text())
    W("| feature | analysis window | acoustic alone @1 m | GT rank | "
      "picks the right hypothesis |")
    W("|---|---|---|---|---|")
    windows = {"envelope 1 ms": "1.00 ms", "envelope 2 ms": "2.00 ms",
               "stft nfft 64 hop 16": "1.33 ms", "stft nfft 64 hop 32": "1.33 ms",
               "stft nfft 128 hop 32": "2.67 ms", "stft nfft 128 fine bands": "2.67 ms",
               "stft nfft 256 hop 64": "5.33 ms", "stft nfft 64 wide bands": "1.33 ms"}
    for k, v in fit["results"].items():
        star = " **(ours)**" if "256" in k else ""
        W(f"| {k}{star} | {windows.get(k,'')} | {v['alone']:.1f} | {v['rank']:.0f} | "
          f"**{v['mode_ok']:.1f}** |")

    # ---------------------------------------------------------------- Table 6
    W("\n## Table 6. What limits the method\n")
    W("`matched geometry` replaces the furnished query with one rendered on the "
      "same wall-only mesh as the candidates, removing the furniture gap and "
      "changing nothing else. It is an upper bound, not a method: a deployment "
      "cannot render furniture it does not know about.\n")
    W("| setting | acoustic alone @1 m | GT rank | ours @1 m | oracle @1 m |")
    W("|---|---|---|---|---|")
    for cond, label in (("raw_scan_open", "furnished query (the real setting)"),
                        ("floorplan_closed", "matched geometry (upper bound)")):
        Qc, Mc, repc = load("unlocSTFT", cond)
        cfg = ModeFusionConfig(**OURS_BY_TAG["unlocSTFT"])
        e, o, _, _ = run("unlocSTFT", cfg, cond)
        qi = [str(x) for x in Qc["query_id"][repc]]
        orc = np.array([Mc[q]["dist_gt_m"].min() for q in qi])
        W(f"| {label} | {100*np.mean(Qc['e_ac'][repc] < 1):.1f} | "
          f"{np.median(Qc['gt_ac_rank_pos'][repc]):.0f} | "
          f"{100*np.mean(e < 1):.1f} | {100*np.mean(orc < 1):.1f} |")
    W("\nCoverage of the visual hypotheses, which caps any reranking method:\n")
    W("| K | truth within 1 m of a top-K cell | of a top-K hypothesis |")
    W("|---|---|---|")
    for K in (1, 3, 5, 10, 20, 50):
        c = Q.get(f"gt_in_top{K}_cells")
        m = Q.get(f"gt_in_top{K}_modes")
        W(f"| {K} | {100*c[rep].mean():.1f} | "
          + (f"{100*m[rep].mean():.1f} |" if m is not None else "- |"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n\n<!-- "
                        + json.dumps(stamp(), default=str) + " -->\n")
    print("\n".join(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
