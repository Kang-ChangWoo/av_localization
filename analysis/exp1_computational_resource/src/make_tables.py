#!/usr/bin/env python3
"""The compute tables, regenerated inside md/computational_resource.md.

The file holds the four tables between ``<!-- tables:start -->`` and
``<!-- tables:end -->`` and, after them, prose written by hand. This script
rewrites only the block between the markers, so a regenerated table never
silently rewrites a sentence. Reads data/cost.json (measure_cost.py),
data/render_cost.json (measure_render_cost.py) and data/backbone_<tag>.json
(measure_backbones.py).

    python src/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DATA, MD = HERE / "data", HERE / "md"
STAGES = [  # key, label, branch, device
    ("image_io", "image read and normalisation", "f", "CPU"),
    ("vis_encoder", "UnLoc encoder, 640×480", "f", "GPU"),
    ("vis_rays", "ray extraction from predicted depth", "b", "CPU"),
    ("vis_localize", "match against the pose grid, 36 headings", "b", "GPU"),
    ("ac_io", "impulse response read, 6 ch at 48 kHz", "b", "CPU"),
    ("ac_feature", "banded STFT of the recording", "b", "CPU"),
    ("ac_project", "projection of the recording, 1746 → 128", "b", "CPU"),
    ("ac_score", "L1 score of every candidate cell (128-d)", "b", "CPU"),
    ("modes", "hypothesis rule: ten hypotheses and disc evidence", "b", "CPU"),
    ("fusion", "hypothesis rule: gated selection", "b", "CPU"),
    ("cell_product", "cell-product rule: argmax of log π + λ z_a", "b", "CPU"),
]


def mb(x):
    return f"{x/2**20:.1f} MB"


def main() -> int:
    cost = json.loads((DATA / "cost.json").read_text())
    render = json.loads((DATA / "render_cost.json").read_text())["results"]
    hw = cost["hardware"]
    scenes = [s["scene"] for s in cost["online"]["per_scene_ms"]]
    L = []
    W = L.append
    W("# Computational resource, av_localization\n")
    W(f"Measured on one {hw['gpu']} and a {hw['cpu']}, PyTorch {hw['torch']}. Per-query times are "
      f"medians over {cost['online']['n_queries_per_scene']} held-out Replica queries per scene after "
      f"warm-up. (f) front branch: the image encoder, which the acoustic branch does not touch. "
      f"(b) back branch: everything after it.\n")

    # ---------------------------------------------------------- parameters
    W("## Table 1. Learned parameters\n")
    W("| component | branch | parameters | note |")
    W("|---|---|---|---|")
    for tag, label in (("unloc", "UnLoc"), ("f3loc_mono", "F3Loc mono"), ("disco_rrp", "DisCo-FLoc RRP")):
        f = DATA / f"backbone_{tag}.json"
        if not f.exists():
            continue
        b = json.loads(f.read_text())["params"]
        enc = {"unloc": "ViT-L, frozen", "f3loc_mono": "ResNet-50, layer 1", "disco_rrp": "ViT-S, frozen"}[tag]
        W(f"| {label} encoder | (f) | {b['encoder']/1e6:.1f} M | {enc} |")
        W(f"| {label} head | (b) | {b['head']/1e6:.2f} M | features → rays |")
        if b.get("unused_at_inference"):
            W(f"| {label} depth head | – | {b['unused_at_inference']/1e6:.1f} M | in the checkpoint, never called |")
    W("| floorplan match | (b) | 0 | L1 against the DESDF |")
    Wp = cost["params"].get("acoustic projection W", {})
    W(f"| acoustic projection W | (b) | {Wp.get('total', 0)/1e6:.2f} M | linear, {'×'.join(map(str, Wp.get('shape', [1746, 128])))}, "
      f"the branch's only learned part |")
    W("| acoustic feature and score | (b) | 0 | banded STFT, L1 |")
    W("| hypothesis rule | (b) | 3 | weight, sigmoid scale, visual threshold |")
    W("| cell-product rule | (b) | 1 | λ |")

    # ------------------------------------------------------------ offline
    W("\n## Table 2. Cost per building, paid once\n")
    W("| scene | cells | grid on disk | render wall | core-hours | s / cell | featurise | project | feature kept (raw → projected) | DESDF |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    tot_cells = tot_ch = 0
    for s in scenes:
        o = cost["offline"][s]; r = render["render"].get(s, {})
        ch = r.get("core_hours", float("nan")); tot_cells += o["cells"]; tot_ch += ch
        W(f"| {s} | {o['cells']} | {o['grid_bytes']/1e9:.2f} GB | {r.get('wall_min', float('nan')):.0f} min ({r.get('shards', '?')} workers) | "
          f"{ch:.1f} | {3600*ch/o['cells']:.1f} | {o['featurise_s']:.0f} s | {o.get('project_s', float('nan')):.2f} s | "
          f"{mb(o['feature_bytes'])} → {mb(o.get('feature_bytes_projected', float('nan')))} | {o['desdf_bytes']/2**20:.1f} MB |")
    W(f"\nAll scenes: {tot_cells} cells, {tot_ch:.1f} core-hours, {3600*tot_ch/tot_cells:.1f} s of single-core simulation per cell.\n")

    # ------------------------------------------------------------- online
    W("## Table 3. Time per query, ms (UnLoc)\n")
    W("| stage | branch | device | " + " | ".join(scenes) + " |")
    W("|---|---|---|" + "---|" * len(scenes))
    per = {s["scene"]: s for s in cost["online"]["per_scene_ms"]}
    for key, label, br, dev in STAGES:
        if not all(key in per[s] for s in scenes):
            continue
        W(f"| {label} | ({br}) | {dev} | " + " | ".join(f"{per[s][key]:.1f}" if per[s][key] < 10 else f"{per[s][key]:.0f}" for s in scenes) + " |")
    front = {s: per[s]["image_io"] + per[s]["vis_encoder"] for s in scenes}
    vis_b = {s: per[s]["vis_rays"] + per[s]["vis_localize"] for s in scenes}
    ac_common = {s: per[s]["ac_io"] + per[s]["ac_feature"] + per[s].get("ac_project", 0) + per[s]["ac_score"] for s in scenes}
    hyp = {s: ac_common[s] + per[s]["modes"] + per[s]["fusion"] for s in scenes}
    cp = {s: ac_common[s] + per[s].get("cell_product", 0) for s in scenes}
    W("| **front branch** | (f) | | " + " | ".join(f"**{front[s]:.0f}**" for s in scenes) + " |")
    W("| **back branch, visual** | (b) | | " + " | ".join(f"**{vis_b[s]:.0f}**" for s in scenes) + " |")
    W("| **added by sound, hypothesis rule** | (b) | | " + " | ".join(f"**{hyp[s]:.0f}**" for s in scenes) + " |")
    W("| **added by sound, cell-product rule** | (b) | | " + " | ".join(f"**{cp[s]:.0f}**" for s in scenes) + " |")
    W("| **total (hypothesis rule)** | | | " + " | ".join(f"**{per[s]['total']:.0f}**" for s in scenes) + " |")
    W(f"\nPeak GPU memory {max(per[s]['gpu_peak_mb'] for s in scenes)/1024:.1f} GB; model load "
      f"{cost['online']['model_load_s']:.1f} s, once. Stage rows are medians and do not sum to the total row, "
      f"which is the median of the whole. The two rules share every row above them; only their last one or "
      f"two rows differ.\n")

    # ----------------------------------------------------------- replace
    W("## Table 4. What the acoustic branch could replace\n")
    W("| backbone | encoder (f) params | head (b) params | encoder (f) ms | head (b) ms | match (b) ms | visual total ms | acoustic branch ms (no I/O) |")
    W("|---|---|---|---|---|---|---|---|")
    ac_ms = float(np.median([per[s]["ac_feature"] + per[s].get("ac_project", 0) + per[s]["ac_score"] + per[s].get("cell_product", 0) for s in scenes]))
    for tag, label in (("unloc", "UnLoc"), ("f3loc_mono", "F3Loc mono"), ("disco_rrp", "DisCo-FLoc RRP")):
        f = DATA / f"backbone_{tag}.json"
        if not f.exists():
            continue
        b = json.loads(f.read_text())
        m = {k: float(np.median([x[k] for x in b["per_scene_ms"]])) for k in ("encoder", "head", "localize", "visual_total")}
        W(f"| {label} | {b['params']['encoder']/1e6:.1f} M | {b['params']['head']/1e6:.2f} M | {m['encoder']:.0f} | "
          f"{m['head']:.1f} | {m['localize']:.1f} | {m['visual_total']:.0f} | {ac_ms:.1f} |")
    MD.mkdir(exist_ok=True)
    target = MD / "computational_resource.md"
    block = "\n".join(L[1:]).rstrip() + "\n\n"      # L[0] is the title, kept in the file
    start, end = "<!-- tables:start -->", "<!-- tables:end -->"
    if target.exists() and start in target.read_text() and end in target.read_text():
        text = target.read_text()
        text = text[: text.index(start) + len(start)] + "\n" + block + text[text.index(end):]
    else:
        text = L[0] + "\n\n" + start + "\n" + block + end + "\n"
    target.write_text(text)
    print(f"wrote {target} (tables block)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
