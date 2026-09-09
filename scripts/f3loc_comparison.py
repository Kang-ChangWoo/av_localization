#!/usr/bin/env python3
"""Put every number this repo produced next to F3Loc's own published table.

Three comparisons, only one of which is apples to apples, so they are kept
separate rather than stacked into one table:

  1. reproduction   our Gibson checkpoints against the published Gibson numbers.
                    Same dataset, same protocol, same metric definitions, so a
                    difference here is a difference in training, nothing else.
  2. baseline       the same architectures retrained on this Replica-derived
                    dataset. Not comparable to the paper -- different rooms,
                    different renderer -- but it is the only valid reference for
                    anything measured on Replica.
  3. acoustic       vision-only against vision plus the gated acoustic re-rank,
                    on Replica. The paper's mono-to-multiview gap is used as the
                    yardstick: it says what a whole extra camera view is worth
                    under F3Loc's own metric, which is the fairest way to size a
                    gain that costs no extra view.

    python scripts/f3loc_comparison.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# F3Loc, CVPR 2024, observation-module table. (0.1 m, 0.5 m, 1 m, 1 m/30 deg)
PAPER = {
    ("mono", "gibson_f"): (0.047, 0.286, 0.366, 0.351),
    ("mv", "gibson_f"): (0.132, 0.409, 0.452, 0.437),
    ("comp", "gibson_g"): (0.122, 0.394, 0.445, 0.432),
}
PAPER_NAME = {"mono": "Ours_s", "mv": "Ours_m", "comp": "Ours_f"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gated", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "margin_gated_band_all.json")
    p.add_argument("--gate-row", default="hard gate q=0.6")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def row(label: str, m, note: str = "") -> str:
    return (f"{label:34s} {100*m[0]:6.1f}% {100*m[1]:6.1f}% {100*m[2]:6.1f}% "
            f"{100*m[3]:7.1f}%  {note}")


def main() -> int:
    metrics = REPO_ROOT / "outputs" / "metrics"
    hdr = f"{'':34s} {'0.1m':>7s} {'0.5m':>7s} {'1m':>7s} {'1m/30d':>8s}"
    summary = {}

    # --- 1. did the reproduction land on the paper? -------------------------
    print("1. REPRODUCTION -- our Gibson checkpoints vs the published table")
    print("   same dataset, same protocol, so these are directly comparable\n")
    print(hdr)
    print("-" * 72)
    repro = {}
    for net in ("mono", "mv", "comp"):
        for ds in ("gibson_f", "gibson_g"):
            if (net, ds) not in PAPER:
                continue
            cands = sorted(glob.glob(str(metrics / f"{net}_*_{ds}_upstream.json")))
            if not cands:
                continue
            best, bm = None, -1.0
            for c in cands:
                d = json.load(open(c))
                if d["recall_1m"] > bm:
                    best, bm = d, d["recall_1m"]
            ours = (best["recall_0.1m"], best["recall_0.5m"],
                    best["recall_1m"], best["recall_1m_30deg"])
            print(row(f"F3Loc paper  {PAPER_NAME[net]:7s} ({ds})", PAPER[(net, ds)]))
            print(row(f"  ours       {net:7s} ({ds})", ours,
                      f"{100*(ours[2]-PAPER[(net,ds)][2]):+.1f} at 1m"))
            repro[f"{net}/{ds}"] = dict(paper=list(PAPER[(net, ds)]), ours=list(ours),
                                        delta_1m=ours[2] - PAPER[(net, ds)][2])
    summary["reproduction_gibson"] = repro

    # --- 2 and 3. Replica: vision alone, then with the acoustic re-rank -----
    args = parse_args()
    if not args.gated.exists():
        print(f"\n(no gated results at {args.gated} yet)")
        return 0
    g = json.load(open(args.gated))
    print("\n\n2+3. REPLICA -- vision alone, then with the gated acoustic re-rank")
    print("     the paper rows are Gibson and are shown only for scale;")
    print("     the valid comparison is each 'vision only' row against the")
    print("     'gated' row directly beneath it\n")
    print(hdr + f" {'vs vision':>10s}")
    print("-" * 84)
    ours = {}
    for spec, blob in g["models"].items():
        net = spec.split(":")[1]
        rows = blob["rows"]
        v, a = rows["vision only"], rows.get(args.gate_row)
        hv = rows.get("[holdout replica_g] vision")
        ha = next((m for k, m in rows.items() if k.startswith("[holdout") and "gated" in k), None)
        if a is None:
            continue
        key = lambda m: (m["r01"], m["r05"], m["r1"], m["r1_30"])
        print(row(f"{net} vision only  (replica)", key(v)))
        print(row(f"{net} + acoustic   (replica)", key(a),
                  f"{100*(a['r1']-v['r1']):+.1f} at 1m"))
        if hv and ha:
            print(row(f"  {net} holdout replica_g vision", key(hv)))
            print(row(f"  {net} holdout replica_g gated", key(ha),
                      f"{100*(ha['r1']-hv['r1']):+.1f} at 1m"))
        print()
        ours[net] = dict(vision=key(v), gated=key(a),
                         holdout_vision=key(hv) if hv else None,
                         holdout_gated=key(ha) if ha else None,
                         gain_1m=a["r1"] - v["r1"],
                         holdout_gain_1m=(ha["r1"] - hv["r1"]) if hv and ha else None)
    summary["replica"] = ours

    # --- the yardstick: what is an extra camera view worth in the paper? ----
    print("\nHOW BIG IS THE ACOUSTIC GAIN, IN F3LOC'S OWN TERMS")
    print("-" * 72)
    view_gap = PAPER[("mv", "gibson_f")][2] - PAPER[("mono", "gibson_f")][2]
    print(f"paper mono -> multiview at 1m: {100*PAPER[('mono','gibson_f')][2]:.1f}% -> "
          f"{100*PAPER[('mv','gibson_f')][2]:.1f}%  = {100*view_gap:+.1f} pp")
    print("that is what a second camera view buys under this metric.\n")
    for net, d in ours.items():
        gain = d["holdout_gain_1m"] if d["holdout_gain_1m"] is not None else d["gain_1m"]
        tag = "held out" if d["holdout_gain_1m"] is not None else "in sample"
        print(f"  {net:5s} acoustic re-rank ({tag}): {100*gain:+5.1f} pp "
              f"= {100*gain/view_gap:4.0f}% of an extra camera view")
    summary["view_gap_1m"] = view_gap
    print("\nCaveat: the paper's gap is measured on Gibson and ours on Replica, so")
    print("this is a sense of scale, not an equality. What it does establish is")
    print("that the gain is the size of an architectural change, not noise.")

    out = args.out or metrics / "f3loc_comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
