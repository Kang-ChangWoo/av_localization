#!/usr/bin/env python3
"""Pool the motion collections, hold out rooms, and report cross-validated recall.

Why this replaces the collection split. The `_f` and `_g` collections are two
motion regimes over the same rooms, matching Gibson's forward and general sets.
They are different queries, but they are not spatially separated: on the Replica
test rooms, 73% of the general poses have a forward pose within 0.1 m, one grid
cell, and 100% have one within 1 m, which is the threshold we report. The
acoustic score is a function of position alone, flat across yaw, so a scalar
fitted on the forward set has already seen the acoustic evidence of the general
set at nearly every reported position. The split holds out the visual side and
leaks the acoustic one, which is the side being tuned.

So the collections are pooled into one test set, and the four scalars are chosen
by leave-one-room-out cross-validation: fit on the other rooms, predict the held
out room, concatenate. Every reported query comes from a room no scalar ever
saw, the test set doubles in size, and nothing new has to be rendered.

The structure, meaning which summary represents a hypothesis under each
modality, is selected inside each fold as well rather than once globally, and
the script reports how often each structure wins so the reader can see whether
it is stable.

    python scripts/room_cv_eval.py --backbones f3STFT unlocSTFT discoID
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

LABELS = {"f3STFT": "F3Loc mono", "unlocSTFT": "UnLoc", "discoID": "DisCo-FLoc RRP",
          "f3loc_mono_s3d": "F3Loc mono", "disco_rrp_s3d": "DisCo-FLoc RRP",
          "unloc_s3d": "UnLoc", "f3loc_mono_mp3dpre": "F3Loc mono",
          "f3loc_mono_mp3d9": "F3Loc mono"}
TH = [0.1, 0.5, 1.0, 2.0, 5.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--backbones", nargs="+", default=["f3STFT", "unlocSTFT", "discoID"])
    p.add_argument("--condition", default="raw_scan_open",
                   choices=["raw_scan_open", "floorplan_closed"])
    p.add_argument("--folds", type=int, default=0,
                   help="0 means leave one room out. A positive value groups the "
                        "rooms into that many folds, for datasets with too many "
                        "rooms to refit once each.")
    p.add_argument("--only-collection", default=None,
                   help="restrict to one motion collection, e.g. mp3d_f, so a "
                        "backbone that collapses on in-place rotation can be "
                        "judged on the motion it was built for")
    p.add_argument("--fix-structure", nargs=2, metavar=("VIS", "AC"), default=None,
                   help="hold the structure at one (visual, acoustic) summary and refit "
                        "only the four scalars per fold. With three rooms a per-fold "
                        "structure search overfits two rooms and can pick a summary that "
                        "fails on the third; the paper's shared structure is centre quantile")
    p.add_argument("--grid", choices=["absolute", "quantile"], default="absolute",
                   help="how the visual gate's threshold and softness are gridded. "
                        "'absolute' uses one fixed list of log-odds values for every "
                        "backbone. 'quantile' places them at quantiles of the fitting "
                        "rooms' own visual ambiguity, so a backbone whose log-odds live "
                        "an order of magnitude lower (DisCo-FLoc's do) gets a gate that "
                        "can actually close on its confident queries. Nothing from the "
                        "held-out room enters either grid")
    p.add_argument("--simple", action="store_true",
                   help="the stripped rule: no acoustic gate (tau_a off) and the "
                        "standardised acoustic summary in place of the relative "
                        "evidence, so two scalars fewer and one transform fewer. "
                        "If this matches the full rule, the full rule is decoration")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "room_cv.md")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "room_cv.json")
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
    cols = [c for c in M if c not in ("query_id", "scene", "collection")
            and M[c].dtype not in (object, bool)]
    g: dict[str, dict[str, list]] = {}
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
    from track1_core.likelihood.mode_fusion import (ModeFusionConfig, choose, evidence_columns,
                                                    visual_ambiguity)
    from track1_core.provenance import stamp

    rng = np.random.default_rng(args.seed)
    WEIGHTS = (0.5, 1.0, 2.0)
    TAU_A = (-2.0,) if args.simple else (-2.0, 0.0, 0.2, 0.4)
    AC_TRANSFORM = "standard" if args.simple else "relative"
    absolute = list(itertools.product(WEIGHTS, (0.02, 0.05, 0.1),
                                      (0.005, 0.02, 0.05, 0.1, 0.2), TAU_A))

    def quantile_grid(amb_fit: np.ndarray) -> list[tuple]:
        """The same shape of grid, placed on the fitting rooms' ambiguity scale.

        Five thresholds at the 20th to 90th percentiles of the fitting queries'
        log-odds and three softnesses as fractions of their interquartile range,
        so the gate has the same number of settings on every backbone and each
        of them lands somewhere the backbone's queries actually are.
        """
        q = np.quantile(amb_fit, (0.2, 0.4, 0.6, 0.75, 0.9))
        iqr = float(np.quantile(amb_fit, 0.75) - np.quantile(amb_fit, 0.25)) or 1e-3
        return list(itertools.product(WEIGHTS, tuple(iqr * f for f in (0.1, 0.25, 0.5)),
                                      tuple(float(x) for x in q), TAU_A))
    structures = ([tuple(args.fix_structure)] if args.fix_structure else
                  list(itertools.product(("centre", "max", "lse"),
                                         ("centre", "max", "quantile", "lse"))))

    out: list[str] = []
    W = out.append
    js: dict = {}
    W("# Rooms held out, motion collections pooled\n")
    W("The `_f` and `_g` collections are two motion regimes over the same rooms. "
      "They are pooled here into one test set, because they are not spatially "
      "separated: on Replica, every general-motion pose has a forward-motion "
      "pose within the $1$ m threshold being reported, and the acoustic score "
      "depends on position alone. The four scalars are instead chosen by "
      "leave-one-room-out cross-validation, so every reported query comes from a "
      "room whose scalars were fitted without it.\n")
    W("\n| dataset | backbone | rooms | queries | vision @1m | ours @1m | gain | 95% CI |")
    W("|---|---|---|---|---|---|---|---|")

    for tag in args.backbones:
        qp = args.analysis_dir / f"queries_{args.condition}_{tag}.csv"
        if not qp.exists():
            print(f"[skip] {qp.name} not on disk")
            continue
        Q = read(qp)
        if args.only_collection:
            keep = Q["collection"] == args.only_collection
            Q = {k: v[keep] for k, v in Q.items()}
        M = group(read(args.analysis_dir / f"modes_{args.condition}_{tag}.csv"))
        qids = [str(x) for x in Q["query_id"]]
        rooms = sorted({str(x) for x in Q["scene"]})
        scene = np.array([str(x) for x in Q["scene"]])

        if args.folds and args.folds < len(rooms):
            groups = [set(rooms[i::args.folds]) for i in range(args.folds)]
        else:
            groups = [{r} for r in rooms]

        def run(cfg, mask):
            vc, ac = evidence_columns(cfg)
            e, o = [], []
            for i in np.nonzero(mask)[0]:
                m = M[qids[i]]
                k, _ = choose(m[vc], m[ac], cfg)
                e.append(m["dist_gt_m"][k]); o.append(m["yaw_err_deg"][k])
            return np.asarray(e), np.asarray(o)

        errs, orns, vis, structure_votes = [], [], [], Counter()
        fold_policies: list[dict] = []
        for held in groups:
            rep = np.array([s in held for s in scene])
            fit = ~rep
            best, bs = None, -1.0
            for ve, ae in structures:
                if args.grid == "quantile":
                    vc = evidence_columns(ModeFusionConfig(vis_evidence=ve, ac_evidence=ae))[0]
                    scalars = quantile_grid(np.array([visual_ambiguity(M[qids[i]][vc])
                                                      for i in np.nonzero(fit)[0]]))
                else:
                    scalars = absolute
                for w, sg, tv, ta in scalars:
                    c = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae,
                                         rule="continuous", weight=w,
                                         sigmoid_scale=sg, tau_v=tv, tau_a=ta,
                                         ac_transform=AC_TRANSFORM)
                    e, _ = run(c, fit)
                    sc = float((e < 1).mean())
                    if sc > bs:
                        best, bs = c, sc
            structure_votes[(best.vis_evidence, best.ac_evidence)] += 1
            # the scalars a held-out room was scored under, so a qualitative
            # figure can draw a room with the same rule the table scored it with
            fold_policies.append(dict(held_out=sorted(held), fit_recall=bs,
                                      policy=dict(vars(best))))
            e, o = run(best, rep)
            errs.append(e); orns.append(o); vis.append(Q["e_vis"][rep])

        e = np.concatenate(errs); o = np.concatenate(orns); v = np.concatenate(vis)
        d = (e < 1).astype(float) - (v < 1).astype(float)
        bi = rng.integers(0, d.size, size=(args.boot, d.size))
        m = d[bi].mean(axis=1)
        lo, hi = np.percentile(m, 2.5), np.percentile(m, 97.5)
        ds = "Structured3D" if tag.endswith("_s3d") else (
             "Matterport3D" if "mp3d" in tag else "Replica")
        W(f"| {ds} | {LABELS.get(tag, tag)} | {len(rooms)} | {len(e)} | "
          f"{100*(v<1).mean():.1f}% | {100*(e<1).mean():.1f}% | "
          f"{100*d.mean():+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
        js[tag] = dict(dataset=ds, rooms=len(rooms), n=len(e),
                       vision=float((v < 1).mean()), ours=float((e < 1).mean()),
                       gain=float(d.mean()), ci=[float(lo), float(hi)],
                       recalls={f"{t}m": float((e < t).mean()) for t in TH},
                       vision_recalls={f"{t}m": float((v < t).mean()) for t in TH},
                       median=float(np.median(e)), vision_median=float(np.median(v)),
                       structures={f"{a}/{b}": n for (a, b), n in structure_votes.items()},
                       folds=fold_policies)
        print(f"[{tag}] structures chosen per fold: {dict(structure_votes)}")

    W("\nThe structure chosen inside each fold is listed in the JSON. A structure "
      "that wins every fold is a property of the method; one that changes fold to "
      "fold would mean the summaries are not doing what we claim.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dict(condition=args.condition, grid=args.grid,
                                             fix_structure=args.fix_structure,
                                             simple=args.simple, results=js,
                                             provenance=stamp()), indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
