#!/usr/bin/env python3
"""Robustness to a degrading camera, with everything frozen.

Reads the degradation study of ../../feasible (UnLoc on Replica: projection
trained on Replica's training rooms and the three scalars chosen on clean
validation rooms, frozen; the query image corrupted at test time at 28
levels across five families) and the collapse-switch study, and lays them
out: vision alone, acoustic alone, the hypothesis rule, and the beyond-
shortlist variants A, B, C at every level, plus whether any indicator of
visual collapse could switch to the acoustic answer without hurting clean
queries.

    python src/robustness.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import RES, C, write_md  # noqa: E402
FIGS_SRC = HERE.parents[1] / "feasible" / "figs"


def main() -> int:
    V = json.loads((RES / "V_final_unloc.json").read_text())
    L = []
    W = L.append
    W("## Table 1. UnLoc on Replica, recall at 1 m, nothing refit on the degraded queries\n")
    W(f"Policy: `{V.get('policy', '')}`. Columns A/B/C use the settings the clean validation rooms chose.\n")
    W("| degradation | vision | acoustic alone | hypothesis rule | gain [95% CI] | gate open | A inject | B reject | C product |")
    W("|---|---|---|---|---|---|---|---|---|")
    for r in V["rows"]:
        W(f"| {r['label']} | {100*r['vision']:.1f} | {100*r['acoustic']:.1f} | {100*r['ours']:.1f} | {100*r['gain']:+.1f} [{100*r['lo']:+.1f}, {100*r['hi']:+.1f}] | "
          f"{100*r['audio_used']:.0f}% | {100*r.get('A', np.nan):.1f} | {100*r.get('B', np.nan):.1f} | {100*r.get('C', np.nan):.1f} |")
    cp = RES / "CELL_product.json"
    if cp.exists():
        cj = json.loads(cp.read_text())
        if "degradation" in cj:
            W(f"\n## Table 2. The cell-product rule with λ chosen on clean validation (λ={cj['replica/unloc']['lam']:g}), frozen\n")
            W("| degradation | vision | cell product | gain |")
            W("|---|---|---|---|")
            for r in cj["degradation"]:
                W(f"| {r['label']} | {100*r['vision']:.1f} | {100*r['C']:.1f} | {100*(r['C']-r['vision']):+.1f} |")
    sw = RES / "SWITCH_collapse.md"
    if sw.exists():
        txt = sw.read_text()
        W("\n## Table 3. Can visual collapse be detected and switched on? (from the collapse-switch study)\n")
        W("Indicator AUROC for 'shortlist covers the truth' against 'shortlist broken', per level; then the switch thresholds chosen on *degraded validation* queries and their effect on test.\n")
        W(txt[txt.index("| level | broken %"):].replace("## ", "### "))
    (HERE / "figs").mkdir(exist_ok=True)
    for f in ("V_final_unloc.png",):
        if (FIGS_SRC / f).exists():
            shutil.copy(FIGS_SRC / f, HERE / "figs" / f)
    (HERE / "data").mkdir(exist_ok=True)
    shutil.copy(RES / "V_final_unloc.json", HERE / "data" / "degradation.json")
    if (RES / "SWITCH_collapse.json").exists():
        shutil.copy(RES / "SWITCH_collapse.json", HERE / "data" / "collapse_switch.json")
    write_md(HERE / "md" / "robustness.md", "Robustness to a degrading camera", L)
    print("wrote md/robustness.md, data/, figs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
