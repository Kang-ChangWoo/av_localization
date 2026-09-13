#!/usr/bin/env python3
"""Vision plus binaural, against vision plus the ring, on the same queries.

The heading test said a binaural pair recovers no orientation here: at the true
cell, choosing the best of 36 headings lands within 30 degrees 17.5% of the time
against a 16.7% chance rate, and matching the geometry does not change that. The
likely reason is the setup rather than the receiver: the source is co-located
with the listener, so there is no direction of arrival for a head-related
transfer function to resolve.

That does not settle whether binaural helps with *position*, which is what the
method actually uses the acoustic term for, and it is the question this script
answers. A binaural grid holds 36 entries per cell; collapsing the per-heading
scores down to one number per cell gives a drop-in replacement for the ring's
candidate feature, and everything downstream, the visual posterior, the mode
extraction and the fusion rule, is left identical. Two collapses are reported,
the best heading and a soft maximum over headings, because a deployment that
does not know its heading has to do one of the two.

    python scripts/binaural_fusion_eval.py --scenes office_4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=Path("/root/storage/echoloc_dataset/replica"))
    p.add_argument("--bin-grid", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_binaural")
    p.add_argument("--ring-grid", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_v2")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=["office_4"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/visual/replica/mono_lr3e4/mono.ckpt"))
    p.add_argument("--n-poses", type=int, default=40, help="per scene and collection")
    p.add_argument("--n-modes", type=int, default=10)
    p.add_argument("--nms-radius-m", type=float, default=1.5)
    p.add_argument("--local-radius-m", type=float, default=0.5)
    p.add_argument("--gpu", default="3")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "binaural_fusion.md")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "binaural_fusion.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import cv2
    import torch
    from track1_core._vendor import ensure_on_path
    ensure_on_path()
    from utils.localization_utils import get_ray_from_depth, localize
    from track1_core.models import MonoDepthModule
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, featurise, observation_rate, score,
    )
    from track1_core.modes import ModeConfig, aggregate, extract_modes, local_discs
    from track1_core.provenance import stamp

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MonoDepthModule.load_from_checkpoint(args.checkpoint).to(dev).eval()
    F_W = 1 / (2 * np.tan(np.deg2rad(106.2602) / 2))
    mcfg = ModeConfig(nms_radius_m=args.nms_radius_m, n_modes=args.n_modes,
                      local_radius_m=args.local_radius_m)

    def posterior(img, dt):
        x = img[:, :, ::-1].astype(np.float64) / 255.0
        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        x = torch.tensor(np.transpose(x, (2, 0, 1))[None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred = net.encoder(x, None)[0].squeeze(0).float().cpu().numpy()
        rays = torch.tensor(get_ray_from_depth(pred, V=11, F_W=F_W), device=dev,
                            dtype=torch.float32)
        _, pd_, orn, _ = localize(dt, rays, return_np=False)
        return np.asarray(pd_.cpu(), np.float64), np.asarray(orn.cpu())

    VARIANTS = ["ring", "bin_best", "bin_soft"]
    rec: list[dict] = []

    for scene in args.scenes:
        bg, rg = args.bin_grid / f"{scene}.npz", args.ring_grid / f"{scene}.npz"
        if not (bg.exists() and rg.exists()):
            print(f"[skip] {scene}: need both grids")
            continue
        B = np.load(bg, allow_pickle=False)
        cj = json.loads(str(B["config"])); sr = int(cj["sample_rate"])
        n_yaw = int(cj.get("yaw_bins", 36))
        cfg = GridScoreConfig(feature="stft_band", nfft=256, hop=64, sample_rate_hz=sr,
                              direct_guard_samples=int(round(sr * 2 / 1000)),
                              usable_samples=int(round(sr * 128 / 1000)))
        bidx = B["index"]
        print(f"[grid] {scene}: binaural {len(bidx)} entries, featurising")
        FB = np.stack([featurise(band_energy(r, cfg), cfg) for r in B["rir"]])
        del B
        R = np.load(rg, allow_pickle=False)
        ridx = R["index"]
        FR = np.stack([featurise(band_energy(r, cfg), cfg) for r in R["rir"]])
        del R

        desdf = np.load(args.dataset_root / "desdf" / scene / "desdf.npy",
                        allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(args.dataset_root / args.collections[0] / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        res = pg.grid_resolution_m
        rows, cols = np.nonzero(valid_pose_mask(occ, pg))
        dt = torch.tensor(desdf["desdf"], device=dev)

        # ring: one entry per cell, exactly as the main pipeline
        lut = -np.ones(occ.shape[:2], np.int64)
        lut[ridx[:, 0], ridx[:, 1]] = np.arange(len(ridx))
        r_pick = lut[rows, cols]; r_present = r_pick >= 0
        CR = np.zeros((len(rows),) + FR.shape[1:], np.float32)
        CR[r_present] = FR[r_pick[r_present]]
        del FR

        # binaural: 36 entries per cell, kept separate so the collapse is a choice
        slot = -np.ones((occ.shape[0], occ.shape[1], n_yaw), np.int64)
        slot[bidx[:, 0], bidx[:, 1], bidx[:, 2]] = np.arange(len(bidx))
        b_pick = slot[rows, cols]                       # (cells, n_yaw)
        b_present = (b_pick >= 0).any(axis=1)
        CB = np.zeros((len(rows), n_yaw) + FB.shape[1:], np.float32)
        for y in range(n_yaw):
            ok = b_pick[:, y] >= 0
            CB[ok, y] = FB[b_pick[ok, y]]
        b_any = b_pick >= 0
        del FB

        for coll in args.collections:
            poses = np.array([[float(v) for v in l.split()]
                              for l in open(args.dataset_root / coll / scene / "poses.txt")
                              if l.strip()])
            rd = args.dataset_root / "rir" / coll / args.condition / scene
            if not rd.is_dir():
                continue
            dirs = sorted(d for d in os.listdir(rd) if d.startswith("pose_"))
            picks = np.linspace(0, len(dirs) - 1, min(args.n_poses, len(dirs))).astype(int)
            for k in picks:
                dn = dirs[int(k)]; i = int(dn.split("_")[1])
                fr = rd / dn / "rir.npy"; fb = rd / dn / "rir_binaural.npy"
                img_p = args.dataset_root / coll / scene / "rgb" / f"{i//4:05d}-{i%4}.png"
                if not (fr.exists() and fb.exists() and img_p.exists() and i < len(poses)):
                    continue
                pd_, orns = posterior(cv2.imread(str(img_p), cv2.IMREAD_COLOR), dt)
                vis = pd_[rows, cols]
                yaw = pg.bin_to_yaw(orns[rows, cols])
                gx, gy, _ = pg.pose_metric_to_grid(poses[i, :3])
                dist = np.hypot(cols - gx, rows - gy) * res

                obs_r = featurise(band_energy(np.load(fr), cfg, observation_rate(fr)), cfg)
                obs_b = featurise(band_energy(np.load(fb), cfg, observation_rate(fb)), cfg)
                ac = {"ring": score(CR, obs_r, r_present, cfg)}
                # score every heading, then collapse: a deployment that knows no
                # heading must either take the best or average over them
                per = np.full((len(rows), n_yaw), -np.inf, np.float32)
                for y in range(n_yaw):
                    ok = b_any[:, y]
                    if ok.any():
                        per[ok, y] = score(CB[:, y][ok], obs_b, np.ones(int(ok.sum()), bool), cfg)
                ac["bin_best"] = np.where(b_present, per.max(axis=1), -np.inf)
                mx = np.where(np.isfinite(per), per, -np.inf).max(axis=1, keepdims=True)
                soft = mx.squeeze(1) + np.log(np.exp(np.where(np.isfinite(per), per - mx, -np.inf)).sum(axis=1))
                ac["bin_soft"] = np.where(b_present, soft, -np.inf)

                modes = extract_modes(vis, rows, cols, res, mcfg)
                discs = local_discs(modes, rows, cols, res, mcfg.local_radius_m)
                lv = aggregate(np.log(np.clip(vis, 1e-300, None)), modes, discs, mcfg)
                row = dict(scene=scene, collection=coll, pose=i,
                           e_vis=float(dist[modes[0]]),
                           o_vis=float(abs(((yaw[modes[0]] - poses[i, 2] + np.pi)
                                            % (2 * np.pi)) - np.pi) * 180 / np.pi),
                           d_modes=[float(dist[m]) for m in modes],
                           o_modes=[float(abs(((yaw[m] - poses[i, 2] + np.pi)
                                               % (2 * np.pi)) - np.pi) * 180 / np.pi)
                                    for m in modes],
                           vis_lse=[float(v) for v in lv["lse"]])
                for v in VARIANTS:
                    row[f"ac_{v}"] = [float(x) for x in
                      aggregate(ac[v], modes, discs, mcfg)["quantile"]]
                    row[f"acell_{v}"] = float(dist[int(np.nanargmax(
                        np.where(np.isfinite(ac[v]), ac[v], np.nan)))])
                rec.append(row)
                if len(rec) % 20 == 0:
                    print(f"  {len(rec)} queries")
        del CR, CB

    if not rec:
        print("nothing measured")
        return 1

    # ------------------------------------------------------- fuse and report
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose
    import itertools
    scene_of = np.array([r["scene"] for r in rec])
    rooms = sorted(set(scene_of))
    grid = list(itertools.product((0.5, 1.0, 2.0), (0.02, 0.05, 0.1),
                                  (0.005, 0.02, 0.05, 0.1, 0.2), (-2.0, 0.0, 0.2, 0.4)))

    def run(v, cfg, idx):
        e, o = [], []
        for j in idx:
            r = rec[j]
            k, _ = choose(np.asarray(r["vis_lse"]), np.asarray(r[f"ac_{v}"]), cfg)
            e.append(r["d_modes"][k]); o.append(r["o_modes"][k])
        return np.asarray(e), np.asarray(o)

    rng = np.random.default_rng(0)
    out, js = [], {}
    W = out.append
    W("# Vision plus binaural, against vision plus the ring\n")
    W(f"Same queries, same visual posterior, same fusion rule; only the acoustic "
      f"candidate changes. {len(rec)} queries over {len(rooms)} room(s), "
      f"condition `{args.condition}`. Scalars are fitted by leave-one-room-out "
      f"where more than one room is present, and on the queries themselves "
      f"otherwise, which is stated because it flatters every variant equally.\n")
    # The cost columns belong in the same row as the recall. A receiver that
    # wins on recall and loses by a factor of thirty-six on the offline render
    # is not obviously the better choice, and a reader cannot see that if the
    # two numbers are in different tables.
    COST = {"ring": (1, "ring6"), "bin_best": (n_yaw, "binaural"),
            "bin_soft": (n_yaw, "binaural")}
    gb = {}
    for name, d in (("ring", args.ring_grid), ("binaural", args.bin_grid)):
        gb[name] = sum((d / f"{s}.npz").stat().st_size for s in args.scenes
                       if (d / f"{s}.npz").exists()) / 1e9
    TH = [0.1, 0.5, 1.0, 2.0, 5.0]
    W("\n| acoustic candidate | calls/cell | grid | alone @1m | 0.1 m | 0.5 m | "
      "1 m | 1m/30deg | 2 m | 5 m | median | RMSE | gain @1m | 95% CI |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ev = np.array([r["e_vis"] for r in rec]); ov = np.array([r["o_vis"] for r in rec])
    rv = [100 * float((ev < t).mean()) for t in TH]
    W(f"| vision only | – | – | – | {rv[0]:.1f} | {rv[1]:.1f} | {rv[2]:.1f} | "
      f"{100*np.mean((ev<1)&(ov<30)):.1f} | {rv[3]:.1f} | {rv[4]:.1f} | "
      f"{np.median(ev):.2f} | {np.sqrt(np.mean(ev**2)):.2f} | – | – |")
    js["vision"] = dict(recalls={f"{t}m": float((ev < t).mean()) for t in TH},
                        joint=float(np.mean((ev < 1) & (ov < 30))),
                        median=float(np.median(ev)),
                        rmse=float(np.sqrt(np.mean(ev ** 2))))
    js["grid_gb"] = gb
    base = (ev < 1).astype(float)
    for v in VARIANTS:
        if len(rooms) > 1:
            E, O = np.zeros(len(rec)), np.zeros(len(rec))
            for held in rooms:
                fit = np.nonzero(scene_of != held)[0]; rep = np.nonzero(scene_of == held)[0]
                best, bs = None, -1.0
                for w, s, tv, ta in grid:
                    c = ModeFusionConfig(vis_evidence="lse", ac_evidence="quantile",
                                         rule="continuous", weight=w, sigmoid_scale=s,
                                         tau_v=tv, tau_a=ta)
                    sc = float((run(v, c, fit)[0] < 1).mean())
                    if sc > bs:
                        best, bs = c, sc
                e, o = run(v, best, rep); E[rep] = e; O[rep] = o
        else:
            idx = np.arange(len(rec)); best, bs = None, -1.0
            for w, s, tv, ta in grid:
                c = ModeFusionConfig(vis_evidence="lse", ac_evidence="quantile",
                                     rule="continuous", weight=w, sigmoid_scale=s,
                                     tau_v=tv, tau_a=ta)
                sc = float((run(v, c, idx)[0] < 1).mean())
                if sc > bs:
                    best, bs = c, sc
            E, O = run(v, best, idx)
        alone = np.array([r[f"acell_{v}"] for r in rec])
        d = (E < 1).astype(float) - base
        i = rng.integers(0, d.size, size=(10000, d.size)); m = d[i].mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        calls, gname = COST[v]
        r = [100 * float((E < t).mean()) for t in TH]
        W(f"| {v} | {calls} | {gb[gname]:.1f} GB | {100*(alone<1).mean():.1f} | "
          f"{r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | {100*np.mean((E<1)&(O<30)):.1f} | "
          f"{r[3]:.1f} | {r[4]:.1f} | {np.median(E):.2f} | "
          f"{np.sqrt(np.mean(E**2)):.2f} | {100*d.mean():+.1f} | "
          f"[{100*lo:+.1f}, {100*hi:+.1f}] |")
        js[v] = dict(alone=float((alone < 1).mean()), calls_per_cell=calls,
                     grid_gb=gb[gname],
                     recalls={f"{t}m": float((E < t).mean()) for t in TH},
                     joint=float(np.mean((E < 1) & (O < 30))),
                     median=float(np.median(E)),
                     rmse=float(np.sqrt(np.mean(E ** 2))),
                     gain=float(d.mean()), ci=[float(lo), float(hi)])
    W("\n`bin_best` takes the best of the 36 headings at each cell, which needs no "
      "heading but is optimistic; `bin_soft` sums over them, which is what a "
      "deployment with no heading prior would do. `ring` is the shipped six-mic "
      "candidate and is the number in the main tables.\n")

    # ------------------------------------------------------------------ tex
    LBL = {"ring": r"$6$-mic ring", "bin_best": r"binaural, best heading",
           "bin_soft": r"binaural, summed over headings"}
    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{Receiver comparison on Replica, everything else held "
           r"fixed: same queries, same visual posterior, same fusion rule, only "
           r"the acoustic candidate changes. The cost columns are in the same "
           r"rows deliberately. A binaural candidate grid needs one engine call "
           r"per (cell, heading) rather than one per cell, and a receiver that "
           r"wins on recall while costing thirty-six times the offline render is "
           r"not obviously the better choice. `Alone' is the acoustic score "
           r"without the visual term. Scalars are fitted by leave-one-room-out.}",
           r"\label{tab:receiver}",
           r"\begin{tabular}{lcccccccccc}\toprule",
           r"Receiver & calls/cell & grid & alone & $0.5$\,m & $1$\,m & "
           r"$1$\,m\,$30^\circ$ & $2$\,m & Median & RMSE & $\Delta_{1\mathrm{m}}$ \\\midrule"]
    vj = js["vision"]
    tex.append(rf"\emph{{vision only}} & \textendash & \textendash & \textendash & "
               rf"{100*vj['recalls']['0.5m']:.1f} & {100*vj['recalls']['1.0m']:.1f} & "
               rf"{100*vj['joint']:.1f} & {100*vj['recalls']['2.0m']:.1f} & "
               rf"{vj['median']:.2f} & {vj['rmse']:.2f} & \textendash \\")
    for v in VARIANTS:
        d_ = js[v]
        lo_, hi_ = 100 * d_["ci"][0], 100 * d_["ci"][1]
        # an interval that spans zero must not be bold: the table would be
        # claiming a result the statistics do not support
        bold = lo_ > 0
        r05 = 100 * d_["recalls"]["0.5m"]
        r10 = 100 * d_["recalls"]["1.0m"]
        r20 = 100 * d_["recalls"]["2.0m"]
        cells = [f"{r05:.1f}", f"{r10:.1f}", f"{100*d_['joint']:.1f}", f"{r20:.1f}",
                 f"{d_['median']:.2f}", f"{d_['rmse']:.2f}",
                 f"{100*d_['gain']:+.1f}" + r"\," +
                 rf"{{\scriptsize[{lo_:+.1f},{hi_:+.1f}]}}"]
        if bold:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        tex.append(rf"{LBL[v]} & {d_['calls_per_cell']} & {d_['grid_gb']:.1f}\,GB & "
                   rf"{100*d_['alone']:.1f} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    tex_dir = REPO_ROOT / "docs" / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "tab_receiver.tex").write_text(
        "% Generated by scripts/binaural_fusion_eval.py -- do not edit by hand.\n"
        + "\n".join(tex) + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dict(condition=args.condition, n=len(rec),
                                             results=js, provenance=stamp()),
                                        indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
