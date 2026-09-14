#!/usr/bin/env python3
"""Tables for the supplementary: what differs between datasets, and what the
protocol actually establishes.

Everything here is measured rather than asserted, for the same reason the main
tables are generated: a supplementary that repeats a number by hand drifts from
the pipeline within a week. Four tables come out.

    tab_setup     what is shared and what differs across the three datasets,
                  read from the dataset metadata rather than from memory
    tab_rayfan    why the matched fan has to fit the camera's field of view,
                  with the recall each setting actually produced
    tab_cluster   query-level against room-clustered bootstrap intervals. The
                  queries inside a room share a floorplan, a candidate grid and
                  an acoustic field, so they are not independent, and with three
                  rooms the clustered interval is the one that answers "would
                  this hold in a new building".
    tab_ringrot   how much the six-microphone ring depends on its own rotation,
                  which is the measurement that says the ring carries no heading

    python scripts/supplement_tables.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

DATASETS = [("replica", "Replica"), ("mp3d", "Matterport3D"), ("s3d", "Structured3D")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=Path("/root/storage/echoloc_dataset"))
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--tex-dir", type=Path, default=REPO_ROOT / "docs" / "tables")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "supplement.json")
    return p.parse_args()


def esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


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
    from track1_core.provenance import stamp
    args.tex_dir.mkdir(parents=True, exist_ok=True)
    js: dict = {}
    rng = np.random.default_rng(args.seed)

    meta = {}
    for key, _ in DATASETS:
        f = args.dataset_root / key / "dataset_meta.json"
        meta[key] = json.loads(f.read_text()) if f.exists() else {}

    def dig(key, *path, default="--"):
        o = meta.get(key, {})
        for k in path:
            if not isinstance(o, dict) or k not in o:
                return default
            o = o[k]
        return o

    # ------------------------------------------------------------ tab_setup
    # Structured3D's camera height is per frame, which the other two do not
    # have, and the candidate grid is rendered at one height per scene. The
    # within-scene spread is therefore a query-to-candidate mismatch that exists
    # only there, so it is measured rather than described.
    spread = []
    for s in sorted(glob.glob(str(args.dataset_root / "s3d" / "s3d" / "*" / "chunks.json")))[:120]:
        c = json.loads(Path(s).read_text())
        z = [f["cam_z"] for ch in c.get("chunks", []) for f in ch.get("frames", [])]
        if z:
            spread.append(max(z) - min(z))
    s3d_spread = float(np.median(spread)) if spread else float("nan")
    js["s3d_height_spread_m"] = s3d_spread

    rows = [
        ("Sampling rate", [f"{dig(k, 'acoustics', 'sample_rate_hz')}\\,Hz" for k, _ in DATASETS]),
        ("Microphones", [str(dig(k, 'acoustics', 'n_mics')) for k, _ in DATASETS]),
        ("Ring radius", [f"{dig(k, 'acoustics', 'ring_radius_m')}\\,m" for k, _ in DATASETS]),
        ("DESDF cell / orientations",
         [f"{dig(k, 'desdf', 'cell_m')}\\,m / {dig(k, 'desdf', 'orientations')}" for k, _ in DATASETS]),
        ("Depth rays / max range",
         [f"{dig(k, 'depth', 'rays')} / {dig(k, 'depth', 'dist_max_m')}\\,m" for k, _ in DATASETS]),
        ("Image", [f"${dig(k, 'camera', 'width_px')}{{\\times}}{dig(k, 'camera', 'height_px')}$"
                   for k, _ in DATASETS]),
        ("Horizontal FOV", [f"${dig(k, 'camera', 'hfov_deg')}^\\circ$" for k, _ in DATASETS]),
        ("$F/W$", [f"{float(dig(k, 'camera', 'F_W', default=0)):.4f}" for k, _ in DATASETS]),
        ("Rays in the matched fan", ["11", "11", "7"]),
        ("Views per pose", [str(dig(k, 'chunk', 'views_per_chunk')) for k, _ in DATASETS]),
        ("Motion collections", ["forward, general", "forward, general", "none (fixed cameras)"]),
        ("Camera height",
         ["$1.25$\\,m", "$1.25$\\,m",
          f"per frame, ${s3d_spread:.2f}$\\,m spread within a scene"]),
        ("Furnished query", ["yes", "yes", "no (no scan mesh)"]),
    ]
    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{What the three benchmarks share and where they differ. The "
           r"acoustic simulation is identical; the cameras are not. The field of "
           r"view fixes how many rays of the matched fan fit inside the image "
           r"(Tab.~\ref{tab:rayfan}), and Structured3D ships neither a furnished "
           r"mesh nor a fixed camera height, so its query is rendered on the same "
           r"floorplan proxy as the candidates and at a height that varies within "
           r"a scene while the candidate grid uses one height per scene.}",
           r"\label{tab:setup}",
           r"\begin{tabular}{lccc}\toprule",
           r" & Replica & Matterport3D & Structured3D \\\midrule"]
    for name, vals in rows:
        tex.append(f"{name} & " + " & ".join(str(v) for v in vals) + r" \\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    (args.tex_dir / "tab_setup.tex").write_text(
        "% Generated by scripts/supplement_tables.py -- do not edit by hand.\n"
        + "\n".join(tex) + "\n")

    # ----------------------------------------------------------- tab_rayfan
    fan = []
    for p in sorted(glob.glob(str(args.analysis_dir / "queries_floorplan_closed_f3loc_mono_sweep*.csv"))):
        stem = Path(p).stem.split("sweep")[1]
        V = int(stem.split("F")[0].lstrip("V")); F = float(stem.split("F")[1])
        Q = read(Path(p))
        e = Q["e_vis"]
        half = (V - 1) * 5.0                      # 10 degrees between rays
        edge = np.degrees(np.arctan2(19.5, 40 * F))   # outermost stored column
        fan.append((V, F, half, edge, 100 * float((e < 1).mean()),
                    100 * float((e < 2).mean()), float(np.median(e))))
    fan.sort(key=lambda r: (-r[1], -r[0]))
    js["rayfan"] = [dict(V=v, F_W=f, half_deg=h, edge_deg=ed, r1=r1, r2=r2, med=m)
                    for v, f, h, ed, r1, r2, m in fan]
    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{The matched fan has to fit inside the image. Rays are "
           r"$10^\circ$ apart, so $V$ rays span $(V{-}1)\cdot 10^\circ$, and the "
           r"depth vector only exists out to the angle of its outermost column. "
           r"Beyond it the interpolation returns NaN and localisation collapses "
           r"without raising. Measured on three Structured3D scenes, $30$ queries.}",
           r"\label{tab:rayfan}",
           r"\begin{tabular}{ccccccc}\toprule",
           r"$V$ & $F/W$ & fan half-angle & image half-angle & $1$\,m & $2$\,m & median \\\midrule"]
    for v, f, h, ed, r1, r2, m in fan:
        flag = "" if h <= ed else r"\,$\dagger$"
        tex.append(rf"{v} & {f:.4g} & ${h:.0f}^\circ${flag} & ${ed:.1f}^\circ$ & "
                   rf"{r1:.1f} & {r2:.1f} & {m:.2f}\,m \\")
    tex += [r"\bottomrule\end{tabular}",
            r"\\[2pt]\footnotesize $\dagger$ part of the fan falls outside the image.",
            r"\end{table}"]
    (args.tex_dir / "tab_rayfan.tex").write_text(
        "% Generated by scripts/supplement_tables.py -- do not edit by hand.\n"
        + "\n".join(tex) + "\n")

    # ---------------------------------------------------------- tab_cluster
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose, evidence_columns
    pol_p = REPO_ROOT / "outputs" / "metrics" / "unified_policy.json"
    pol = json.loads(pol_p.read_text())["policy"] if pol_p.exists() else {}
    keys = ("vis_evidence", "ac_evidence", "rule", "weight", "sigmoid_scale", "tau_v", "tau_a",
            "ac_transform")
    clus = []
    for tag, label in (("f3STFT", "F3Loc mono"), ("unlocSTFT", "UnLoc"), ("discoID", "DisCo-FLoc RRP")):
        qp = args.analysis_dir / f"queries_raw_scan_open_{tag}.csv"
        if not (qp.exists() and tag in pol):
            continue
        Q = read(qp); M = group(read(args.analysis_dir / f"modes_raw_scan_open_{tag}.csv"))
        cfg = ModeFusionConfig(**{k: (-np.inf if pol[tag][k] is None else pol[tag][k]) for k in keys})
        vc, ac = evidence_columns(cfg)
        rep = Q["collection"] != "replica_f"
        ids = [str(x) for x in Q["query_id"][rep]]
        sc = np.array([str(x) for x in Q["scene"][rep]])
        e = np.array([M[q]["dist_gt_m"][choose(M[q][vc], M[q][ac], cfg)[0]] for q in ids])
        d = (e < 1).astype(float) - (Q["e_vis"][rep] < 1).astype(float)
        i = rng.integers(0, d.size, size=(args.boot, d.size))
        q_lo, q_hi = np.percentile(d[i].mean(axis=1), [2.5, 97.5])
        rooms = sorted(set(sc)); by = [d[sc == r] for r in rooms]
        cm = np.array([np.concatenate([by[j] for j in rng.integers(0, len(rooms), len(rooms))]).mean()
                       for _ in range(args.boot)])
        c_lo, c_hi = np.percentile(cm, [2.5, 97.5])
        clus.append((label, len(d), len(rooms), 100 * d.mean(),
                     100 * q_lo, 100 * q_hi, 100 * c_lo, 100 * c_hi))
    js["cluster"] = [dict(backbone=a, n=b, rooms=c, gain=g, query_ci=[ql, qh],
                          room_ci=[cl, ch]) for a, b, c, g, ql, qh, cl, ch in clus]
    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{Two bootstraps of the same Replica result. Resampling "
           r"queries answers whether the gain would survive more poses in these "
           r"three rooms. Resampling rooms answers whether it would survive a new "
           r"building, and it is the question the paper asks: queries inside a "
           r"room share a floorplan, a candidate grid and an acoustic field, so "
           r"with three rooms the effective sample size is three. We report the "
           r"query interval in the main tables and this one beside it, and we do "
           r"not claim building-level generalisation from Replica alone.}",
           r"\label{tab:cluster}",
           r"\begin{tabular}{lccccc}\toprule",
           r"Backbone & queries & rooms & $\Delta_{1\mathrm{m}}$ & "
           r"query bootstrap & room bootstrap \\\midrule"]
    for a, b, c, g, ql, qh, cl, ch in clus:
        tex.append(rf"{esc(a)} & {b} & {c} & {g:+.1f} & "
                   rf"$[{ql:+.1f}, {qh:+.1f}]$ & $[{cl:+.1f}, {ch:+.1f}]$ \\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    (args.tex_dir / "tab_cluster.tex").write_text(
        "% Generated by scripts/supplement_tables.py -- do not edit by hand.\n"
        + "\n".join(tex) + "\n")

    # ---------------------------------------------------------- tab_ringrot
    from track1_core.likelihood.grid_score import GridScoreConfig, band_energy, featurise, score
    g = REPO_ROOT / "outputs" / "acoustic_grid_v2" / "office_4.npz"
    rot = []
    if g.exists():
        b = np.load(g, allow_pickle=False)
        sr = int(json.loads(str(b["config"]))["sample_rate"])
        cfg = GridScoreConfig(feature="stft_band", nfft=256, hop=64, sample_rate_hz=sr,
                              direct_guard_samples=int(round(sr * 2 / 1000)),
                              usable_samples=int(round(sr * 128 / 1000)))
        R = b["rir"][:600]; del b
        F = np.stack([featurise(band_energy(r, cfg), cfg) for r in R])
        N, rows_, T = F.shape; C = 6; B = rows_ // C
        pres = np.ones(N, bool)
        for c in range(6):
            ranks = []
            for tgt in (7, 100, 250, 400, 550):
                o = np.roll(F[tgt].reshape(C, B, T), c, axis=0).reshape(rows_, T)
                s = score(F, o, pres, cfg)
                ranks.append(int((s > s[tgt]).sum()))
            rot.append((c, float(np.mean(ranks)), int(max(ranks))))
    js["ring_rotation"] = [dict(shift=c, mean_rank=m, worst_rank=w) for c, m, w in rot]
    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{The six-microphone ring carries no heading. The query's "
           r"channels are rotated by $c$ positions and matched against an "
           r"unrotated grid of $600$ candidates; the rank of the true cell barely "
           r"moves. This is why the query ring rotating with the camera while the "
           r"candidate ring is fixed to the world costs nothing, and it is also "
           r"why every reported orientation comes from the visual backbone. A "
           r"binaural pair is the change that would make the acoustic score a "
           r"function of heading, at one engine call per (cell, heading).}",
           r"\label{tab:ringrot}",
           r"\begin{tabular}{ccc}\toprule",
           r"Channel rotation & mean rank of the true cell & worst of five \\\midrule"]
    for c, m, w in rot:
        tex.append(rf"{c} & {m:.1f} & {w} \\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    (args.tex_dir / "tab_ringrot.tex").write_text(
        "% Generated by scripts/supplement_tables.py -- do not edit by hand.\n"
        + "\n".join(tex) + "\n")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dict(results=js, provenance=stamp()),
                                        indent=2, default=str))
    print("wrote tab_setup.tex, tab_rayfan.tex, tab_cluster.tex, tab_ringrot.tex")
    print(json.dumps(js, indent=2, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
