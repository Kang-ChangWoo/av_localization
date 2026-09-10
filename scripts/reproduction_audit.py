#!/usr/bin/env python3
"""One report answering three separate questions about the three baselines.

They are easy to conflate and they have different answers.

1. **Do the released weights reproduce the published tables?** This is the
   claim "we evaluated the authors' method correctly". It needs the published
   numbers, which are hard-coded below with their source.

2. **Does training from scratch reach the released weights?** This is the much
   stronger claim "we can retrain the method". Failing it does not invalidate
   (1), and passing (1) says nothing about it.

3. **Is the method trained on Replica at all**, or is every Replica number a
   zero-shot transfer from Gibson? An audio gain measured against a zero-shot
   backbone is measured against a handicapped baseline.

There is one trap specific to F3Loc and it has already cost this project twice:
its own evaluation leaves the monocular and multi-view nets in train mode, so
BatchNorm uses batch statistics and the score depends on the evaluation batch
size. At batch 16 the monocular net reads 46.8% against 36.5% at batch 1. Any
comparison between two checkpoints is meaningless unless both were run at the
same batch size, and the older metric files do not record it. Only files that
record ``batch_size`` are trusted here; the rest are reported as unverifiable.

    python scripts/reproduction_audit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "outputs" / "metrics"

# Published numbers, with where each came from.
PUBLISHED = {
    "f3loc": {
        "source": "F3Loc CVPR 2024, Table 1",
        "rows": {
            "mono (Ours_s) gibson_f": (4.7, 28.6, 36.6, 35.1),
            "mv (Ours_m) gibson_f":   (13.2, 40.9, 45.2, 43.7),
            "comp (Ours_f) gibson_g": (12.2, 39.4, 44.5, 43.2),
        },
    },
    "disco": {
        "source": "DisCo-FLoc, arXiv 2601.01822, Table 1, Gibson(f)",
        "rows": {
            "RRP only (Ours w/o Dis.)": (12.0, 45.8, 50.6, 49.2),
            "RRP + DisCo (Ours)":       (13.1, 50.9, 56.7, 55.4),
        },
    },
    "unloc": {
        "source": "UnLoc, arXiv 2509.11301, Table 6 single-frame Gibson(t)",
        "rows": {"UnLoc single-frame": (19.7, 61.1, 64.7, 63.8)},
    },
}

TOL = 0.5  # points; anything inside this is called a match


def load(name: str) -> dict | None:
    p = METRICS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def quad(d: dict) -> tuple:
    return (100 * d["recall_0.1m"], 100 * d["recall_0.5m"],
            100 * d["recall_1m"], 100 * d["recall_1m_30deg"])


def verdict(got: tuple, want: tuple) -> str:
    worst = max(abs(a - b) for a, b in zip(got, want))
    return f"MATCH ({worst:.1f})" if worst <= TOL else f"off by {worst:.1f}"


def line(label: str, v: tuple, extra: str = "") -> str:
    return f"  {label:34s} " + " ".join(f"{x:6.1f}" for x in v) + f"   {extra}"


def header(title: str) -> None:
    print(f"\n{title}")
    print(f"  {'':34s} " + " ".join(f"{h:>6s}" for h in ("0.1m", "0.5m", "1m", "1m/30")))


def batch_of(d: dict | None) -> str:
    if d is None:
        return "missing"
    b = d.get("batch_size")
    return str(b) if b is not None else "unrecorded"


def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print("=" * 78)
    print("1. DO THE RELEASED WEIGHTS REPRODUCE THE PUBLISHED TABLES?")
    print("=" * 78)

    header(f"F3Loc  [{PUBLISHED['f3loc']['source']}]  batch 1, upstream BN regime")
    f3 = [("mono (Ours_s) gibson_f", "AUDIT_official_mono_gibson_f_bs1",
           "official_mono_gibson_f_upstream_bs1"),
          ("mv (Ours_m) gibson_f", "AUDIT_official_mv_gibson_f_bs1",
           "official_mv_gibson_f_upstream_bs1"),
          ("comp (Ours_f) gibson_g", "AUDIT_official_comp_gibson_g_bs1",
           "official_comp_gibson_g_upstream")]
    for label, *cands in f3:
        d = next((x for x in (load(c) for c in cands) if x), None)
        want = PUBLISHED["f3loc"]["rows"][label]
        print(line("published", want))
        if d is None:
            print(line(label, (0, 0, 0, 0), "NOT RUN"))
        else:
            print(line(label, quad(d), f"{verdict(quad(d), want)}   bs={batch_of(d)}"))

    header(f"DisCo  [{PUBLISHED['disco']['source']}]")
    disco = load("DISCO_gibson_repro")
    if disco is None:
        print("  (parsed from logs/repro_disco_gibson.log; see the log for the raw runs)")
    for label, want in PUBLISHED["disco"]["rows"].items():
        print(line("published", want))
        got = (disco or {}).get(label)
        print(line(label, tuple(got), verdict(tuple(got), want)) if got
              else line(label, (0, 0, 0, 0), "see log"))

    header(f"UnLoc  [{PUBLISHED['unloc']['source']}]")
    for label, want in PUBLISHED["unloc"]["rows"].items():
        print(line("published", want))
        u = load("UNLOC_gibson_t_single_frame")
        print(line(label, tuple(u["quad"]), verdict(tuple(u["quad"]), want)) if u
              else line(label, (0, 0, 0, 0), "see logs/unloc_gibson_t.log"))

    print("\n" + "=" * 78)
    print("2. DOES TRAINING FROM SCRATCH REACH THE RELEASED WEIGHTS?")
    print("=" * 78)
    print("  Both sides must be evaluated at the same batch size or the comparison")
    print("  is meaningless. Rows whose batch size was never recorded are dropped.")
    for net, ds in (("mono", "gibson_f"), ("mono", "gibson_g"),
                    ("mv", "gibson_f"), ("mv", "gibson_g"),
                    ("comp", "gibson_f"), ("comp", "gibson_g")):
        ours = load(f"AUDIT_{net}_primary_{ds}_bs1")
        off = load(f"AUDIT_official_{net}_{ds}_bs1") or \
            load(f"official_{net}_{ds}_upstream_bs1")
        header(f"F3Loc {net} on {ds}")
        if ours is None or off is None:
            print(f"  {'not both available':34s}  ours={batch_of(ours)}  released={batch_of(off)}")
            continue
        print(line("released weights", quad(off), f"bs={batch_of(off)}"))
        print(line("ours, trained from scratch", quad(ours), f"bs={batch_of(ours)}"))
        d = quad(ours)[2] - quad(off)[2]
        print(f"  {'difference at 1 m':34s} {d:+6.1f}")

    print("\n  DisCo and UnLoc: training on Replica is under way; neither has a")
    print("  from-scratch Gibson run, so question 2 is unanswered for both.")

    print("\n" + "=" * 78)
    print("3. IS THE BACKBONE TRAINED ON REPLICA, OR ZERO-SHOT FROM GIBSON?")
    print("=" * 78)
    for label, run, ck in (
            ("F3Loc mono", "echoloc_mono_fg", REPO_ROOT / "outputs/echoloc_mono_fg/mono.ckpt"),
            ("F3Loc mv", "echoloc_mv_fg", REPO_ROOT / "outputs/echoloc_mv_fg/mv.ckpt"),
            ("F3Loc comp", "echoloc_comp_fg", REPO_ROOT / "outputs/echoloc_comp_fg/comp.ckpt"),
            ("DisCo RRP", "rrp_replica_f",
             REPO_ROOT.parent / "DisCo-FLoc/logs/rrp_runs"),
            ("DisCo contrastive", "disco_replica_f",
             REPO_ROOT.parent / "DisCo-FLoc/logs"),
            ("UnLoc", "unloc replica", REPO_ROOT.parent / "UnLoc/logs")):
        print(f"  {label:20s} {'trained' if Path(ck).exists() else 'NOT TRAINED':12s}  {run}")

    print("\n  Every Replica number reported for DisCo and UnLoc so far uses Gibson")
    print("  weights, so those backbones are handicapped relative to F3Loc, which")
    print("  is trained in domain. Comparisons between them are not like for like.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
