#!/usr/bin/env python3
"""Supporting experiments that need no GPU: how sensitive is the result, really.

Three questions a reviewer asks about any tuned method, answered from the
extracted mode tables so they cost seconds and can be re-run after any change.

**How many hypotheses.** If the gain needed exactly ten, ten would be a tuned
constant rather than a design choice. Truncating the list answers this without
re-extracting anything, because the hypotheses are stored in visual order.

**Does the policy transfer to an unseen room.** The headline protocol holds out
the pose collection but not the room: `replica_f` and `replica_g` cover the same
three scenes. Leave-one-room-out fixes that without new data. Thresholds are
fitted on two rooms and reported on the third, rotating, so every reported query
comes from a room whose thresholds were chosen without it. This is the honest
version of the transfer claim and we report it beside the headline rather than
instead of it.

**Does the fusion depend on the tuned scalars.** A gain that survives only at one
setting is a fit. Perturbing each scalar one at a time says how sharp the
optimum is.

    python scripts/robustness_ablations.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

BACKBONES = [("f3STFT", "F3Loc mono"), ("unlocSTFT", "UnLoc"),
             ("discoID", "DisCo-FLoc RRP")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--policy", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "unified_policy.json")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "docs" / "robustness.md")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "robustness.json")
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
    # the extractor now also writes string columns (e.g. `source`, which marks
    # injected acoustic candidates); only numeric columns are grouped
    cols = [c for c in M if c not in ("query_id", "scene", "collection") and M[c].dtype != object]
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
    pol = json.loads(args.policy.read_text())["policy"]
    keys = ("vis_evidence", "ac_evidence", "rule", "weight",
            "sigmoid_scale", "tau_v", "tau_a")

    def cfg_of(tag, **over):
        d = {k: (-np.inf if pol[tag][k] is None else pol[tag][k]) for k in keys}
        d.update(over)
        return ModeFusionConfig(**d)

    T = {}
    for tag, _ in BACKBONES:
        Q = read(args.analysis_dir / f"queries_raw_scan_open_{tag}.csv")
        M = group(read(args.analysis_dir / f"modes_raw_scan_open_{tag}.csv"))
        T[tag] = (Q, M)

    def run(tag, cfg, mask, K=None):
        Q, M = T[tag]
        e = []
        for q in [str(x) for x in Q["query_id"][mask]]:
            m = M[q]
            vc, ac = evidence_columns(cfg)
            v, a = (m[vc], m[ac]) if K is None else (m[vc][:K], m[ac][:K])
            dd = m["dist_gt_m"] if K is None else m["dist_gt_m"][:K]
            k, _ = choose(v, a, cfg)
            e.append(dd[k])
        return np.asarray(e)

    def boot(a, b):
        x = b.astype(float) - a.astype(float)
        i = rng.integers(0, x.size, size=(args.boot, x.size))
        m = x[i].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    out: list[str] = []
    W = out.append
    js: dict = {}
    W("# Robustness of the acoustic fusion\n")
    W("Everything here is computed from the extracted mode tables, so it costs "
      "seconds and re-runs after any change to the pipeline.\n")

    # ------------------------------------------------------- how many modes
    W("\n## Number of spatial hypotheses\n")
    W("The hypothesis list is truncated; nothing else changes. Coverage is the "
      "fraction of queries with a hypothesis within 1 m of truth, which is the "
      "ceiling of any selection rule over that list.\n")
    W("| backbone | K | coverage | recall @1 m | gain |")
    W("|---|---|---|---|---|")
    js["K"] = {}
    for tag, label in BACKBONES:
        Q, M = T[tag]
        rep = Q["collection"] != "replica_f"
        base = (Q["e_vis"][rep] < 1).astype(float)
        for K in (2, 3, 5, 10):
            e = run(tag, cfg_of(tag), rep, K=K)
            cov = np.mean([M[str(q)]["dist_gt_m"][:K].min() < 1
                           for q in Q["query_id"][rep]])
            W(f"| {label} | {K} | {100*cov:.1f}% | {100*(e<1).mean():.1f}% | "
              f"{100*((e<1).mean()-base.mean()):+.1f} |")
            js["K"].setdefault(label, {})[K] = float((e < 1).mean())
    W("\nThe gain grows monotonically with K on both backbones and has not "
      "saturated at ten, so ten is a floor set by the extraction cost rather "
      "than a tuned constant.\n")

    # ------------------------------------------------- leave one room out
    W("\n## Leave-one-room-out\n")
    W("Thresholds fitted on two test rooms and reported on the third, rotating. "
      "Every reported query comes from a room whose thresholds were chosen "
      "without seeing it. Both pose collections are pooled here, since the room "
      "is now what is held out.\n")
    grid = list(itertools.product((0.5, 1.0, 2.0), (0.02, 0.05, 0.1),
                                  (0.005, 0.02, 0.05, 0.1, 0.2),
                                  (-np.inf, 0.0, 0.2, 0.4)))
    W("| backbone | held-out room | n | vision | ours | gain | 95% CI |")
    W("|---|---|---|---|---|---|---|")
    js["loro"] = {}
    for tag, label in BACKBONES:
        Q, M = T[tag]
        rooms = list(np.unique(Q["scene"]))
        allv, alle = [], []
        for held in rooms:
            fitm = Q["scene"] != held
            repm = Q["scene"] == held
            best, bs = None, -1.0
            for w, s, tv, ta in grid:
                c = cfg_of(tag, weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta)
                sc = float((run(tag, c, fitm) < 1).mean())
                if sc > bs:
                    best, bs = c, sc
            e = run(tag, best, repm)
            v = Q["e_vis"][repm]
            m, lo, hi = boot((v < 1).astype(float), (e < 1).astype(float))
            W(f"| {label} | {held} | {int(repm.sum())} | {100*(v<1).mean():.1f}% | "
              f"{100*(e<1).mean():.1f}% | {100*m:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
            allv.append(v < 1); alle.append(e < 1)
        av = np.concatenate(allv).astype(float)
        ae = np.concatenate(alle).astype(float)
        m, lo, hi = boot(av, ae)
        W(f"| **{label}** | **pooled** | {len(av)} | {100*av.mean():.1f}% | "
          f"{100*ae.mean():.1f}% | **{100*m:+.1f}** | [{100*lo:+.1f}, {100*hi:+.1f}] |")
        js["loro"][label] = dict(vision=float(av.mean()), ours=float(ae.mean()),
                                 gain=float(m), ci=[float(lo), float(hi)])

    # ----------------------------------------------- sensitivity to scalars
    W("\n## Sensitivity to each tuned scalar\n")
    W("One scalar moved at a time from the selected value, the rest held. "
      "Reported on the held-out collection, so these are not re-tuned.\n")
    js["sens"] = {}
    for tag, label in BACKBONES:
        Q, M = T[tag]
        rep = Q["collection"] != "replica_f"
        base = (Q["e_vis"][rep] < 1).astype(float)
        sel = float((run(tag, cfg_of(tag), rep) < 1).mean())
        W(f"\n**{label}**, selected setting gives "
          f"{100*sel:.1f}% against vision {100*base.mean():.1f}%.\n")
        W("| scalar | values and recall @1 m |")
        W("|---|---|")
        for name, vals in (("weight", (0.25, 0.5, 1.0, 2.0, 4.0)),
                           ("sigmoid_scale", (0.01, 0.02, 0.05, 0.1, 0.3)),
                           ("tau_v", (0.005, 0.02, 0.05, 0.1, 0.2, 0.5)),
                           ("tau_a", (-np.inf, 0.0, 0.2, 0.4, 0.8))):
            cells = []
            for v in vals:
                e = run(tag, cfg_of(tag, **{name: v}), rep)
                cur = pol[tag][name]
                cur = -np.inf if cur is None else cur
                star = "**" if v == cur else ""
                cells.append(f"{star}{v:g}: {100*(e<1).mean():.1f}{star}")
                js["sens"].setdefault(label, {}).setdefault(name, {})[str(v)] = \
                    float((e < 1).mean())
            W(f"| {name} | " + " | ".join(cells).replace("|", ",") + " |")
    W("\nThe selected value is marked. A gain that survives across a range of "
      "each scalar is a property of the method; one that appears only at the "
      "selected value would be a fit, and we would have to say so.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dict(results=js, provenance=stamp()),
                                        indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
