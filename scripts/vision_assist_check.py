#!/usr/bin/env python3
"""Feasibility check: can acoustics assist vision, and which model is good enough?

Vision is the reference, not a competitor. It runs alone at every pose and
defines the baseline; each acoustic candidate model is then allowed only to
re-order vision's own shortlist. The question is not whether sound localizes
well by itself but whether it moves vision's number, and by how much.

Re-ranking rather than multiplying is deliberate: the answer can never leave
vision's top-K, so a confident-but-wrong acoustic model costs nothing. The
shortlist size K is the knob, and the curve against K is the result.

    python scripts/vision_assist_check.py --scene office_4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STYLE = {"A": ("#8d99ae", "2D walls"),
         "A+": ("#457b9d", "+ floor/ceiling"),
         "B": ("#f4a261", "image sources"),
         "C": ("#e63946", "SoundSpaces")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", default="office_4")
    p.add_argument("--run-name", default="echoloc_mono")
    p.add_argument("--net", default="mono", choices=["mono", "mv"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--feature", default="energy", choices=["energy", "chan_shape"],
                   help="chan_shape keeps the ring's relative channel levels as an "
                        "explicit part of the vector; see scripts/acoustic_feature_sweep.py")
    p.add_argument("--topk", type=int, nargs="+",
                   default=[1, 3, 5, 10, 25, 50, 100, 250, 500, 1000])
    p.add_argument("--n-poses", type=int, default=120)
    p.add_argument("--chunk", type=int, default=48)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import tqdm

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, free_space, valid_pose_mask
    from track1_core.likelihood.acoustic import (
        AcousticProxyConfig, synthesize_proxy, trace_echoes,
    )
    from track1_core.models import MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    scene, L = args.scene, 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
    Ks = np.array(args.topk)

    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    desdf["desdf"][desdf["desdf"] > 10] = 10
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, pg)
    rows, cols = np.nonzero(mask)
    desdf_t = torch.tensor(desdf["desdf"], device=device)
    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / scene / "poses.txt") if l.strip()])

    fp = json.loads((root / "floorplan_proxy" / scene / "floorplan.json").read_text())
    cdir = root / "rir" / args.collection / "floorplan_closed" / scene
    md = json.loads((cdir / sorted(d for d in os.listdir(cdir) if d.startswith("pose_"))[0]
                     / "rir_metadata.json").read_text())
    room_h, src_h = float(fp["room_height_m"]), \
        float(md["array_center_habitat"][1]) - float(fp["z_floor"])

    # ---- candidate models, all on the same cells ---------------------------
    models = {}

    def analytic(v_order):
        cfg = AcousticProxyConfig(n_rays=72, max_order=2, energy_window_ms=args.window_ms,
                                  vertical_order=v_order, room_height_m=room_h,
                                  source_height_m=src_h, usable_samples=args.usable_samples,
                                  direct_guard_samples=args.guard_samples)
        free = torch.tensor(free_space(occ), device=device)
        origins = torch.tensor(pg.grid_to_map(np.stack([cols, rows], axis=-1)),
                               dtype=torch.float32, device=device)
        yaws = torch.zeros(1, device=device)
        out = []
        for s in range(0, len(origins), args.chunk):
            o = origins[s: s + args.chunk]
            ech = trace_echoes(free, o, cfg, pg.map_resolution_m)
            out.append(synthesize_proxy(ech, o, yaws, pg, cfg).squeeze(1).cpu().numpy())
            del ech
            torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)

    models["A"] = analytic(0)
    models["A+"] = analytic(1)
    lut = -np.ones(mask.shape, dtype=int)
    for tag, fname in (("B", f"{scene}_pra.npz"), ("C", f"{scene}.npz")):
        f = REPO_ROOT / "outputs" / "acoustic_grid" / fname
        if not f.exists():
            continue
        blob = np.load(f)
        cand, index = blob["rir"], blob["index"]
        env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
        lut[:] = -1
        lut[index[:, 0], index[:, 1]] = np.arange(len(index))
        pick = lut[rows, cols]
        aligned = np.zeros((len(rows), env.shape[1], env.shape[2]))
        aligned[pick >= 0] = env[pick[pick >= 0]]
        models[tag] = aligned
    for k in models:
        models[k] = models[k] / models[k].sum(axis=(1, 2), keepdims=True).clip(1e-20)

    def featurize(env):
        if args.feature == "energy":
            return env
        per = env / env.sum(axis=-1, keepdims=True).clip(1e-12)
        total = env.sum(axis=(-1, -2))
        split = env.sum(axis=-1) / np.clip(total[..., None], 1e-12, None)
        return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)

    models = {k: featurize(v) for k, v in models.items()}
    print(f"[assist] models: {list(models)}   candidates: {len(rows)}   feature: {args.feature}")

    # ---- vision, then each model re-ranking vision's shortlist --------------
    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    cls = MonoDepthModule if args.net == "mono" else MVDepthModule
    model = cls.load_from_checkpoint(str(ckpt)).to(device).eval()
    dataset_dir = str(root / args.collection)
    dataset = GridSeqDataset(dataset_dir, [scene], L=L, depth_dir=dataset_dir,
                             depth_suffix="depth40" if args.net == "mono" else "depth160")
    rir_dir = root / args.collection and root / "rir" / args.collection / args.condition / scene

    picks = np.linspace(0, len(dataset) - 1, min(args.n_poses, len(dataset))).astype(int)
    vis_err, cover, oracle = [], [], []
    rerank = {k: [] for k in models}
    for chunk in tqdm.tqdm(picks, desc=scene):
        pose_idx = int(chunk) * (L + 1) + L
        f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
        if not f.exists():
            continue
        data = dataset[int(chunk)]
        with torch.no_grad():
            if args.net == "mono":
                pred, _, _ = model.encoder(
                    torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
            else:
                b = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                     for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                b["ref_mask"] = b["src_mask"] = None
                pred = model.net(b)["d"]
        rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                            device=device, dtype=torch.float32)
        _, prob_dist, _, _ = localize(desdf_t, rays)
        v = np.array(prob_dist, dtype=np.float64)[mask]
        order = np.argsort(-v)

        gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
        dist = np.hypot(cols - gx, rows - gy) * pg.grid_resolution_m
        vis_err.append(float(dist[order[0]]))
        cover.append([bool((dist[order[:K]] <= 1.0).any()) for K in Ks])
        oracle.append([float(dist[order[:K]].min()) for K in Ks])

        obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
        obs = featurize(obs / max(obs.sum(), 1e-20))
        for k, env in models.items():
            nw = min(env.shape[2], obs.shape[1])
            sc = np.exp(-np.abs(env[:, :, :nw] - obs[None, :, :nw]).sum(axis=(1, 2))
                        / args.temperature)
            rerank[k].append([float(dist[order[:K]][int(sc[order[:K]].argmax())]) for K in Ks])

    vis_err = np.array(vis_err); cover = np.array(cover); oracle = np.array(oracle)
    base = float((vis_err < 1).mean())
    print(f"\n[assist] {scene} / {args.condition} / {len(vis_err)} poses")
    print(f"    vision alone: recall@1m {100*base:.0f}%, median {np.median(vis_err):.2f} m")
    print(f"    {'K':>5s} {'cover':>7s} {'oracle':>7s} " +
          " ".join(f"{k:>7s}" for k in models))
    summary = {"scene": scene, "condition": args.condition, "poses": int(len(vis_err)),
               "vision_recall_1m": base, "vision_median_m": float(np.median(vis_err)),
               "K": Ks.tolist(), "coverage": cover.mean(axis=0).tolist(),
               "oracle_recall_1m": (oracle < 1).mean(axis=0).tolist(), "models": {}}
    for k in models:
        summary["models"][k] = (np.array(rerank[k]) < 1).mean(axis=0).tolist()
    for i, K in enumerate(Ks):
        print(f"    {K:5d} {100*cover[:, i].mean():6.0f}% {100*(oracle[:, i]<1).mean():6.0f}% " +
              " ".join(f"{100*summary['models'][k][i]:6.0f}%" for k in models))

    out_json = REPO_ROOT / "outputs" / "metrics" / f"vision_assist_{scene}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # ---- figure: vision is the reference line ------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].axhline(100 * base, color="black", lw=2.5, label=f"vision alone ({100*base:.0f}%)")
    ax[0].plot(Ks, 100 * (oracle < 1).mean(axis=0), "k--", lw=1.2, alpha=0.6,
               label="oracle (best in shortlist)")
    for k in models:
        c, lab = STYLE[k]
        ax[0].plot(Ks, 100 * np.array(summary["models"][k]), "-o", ms=4, color=c,
                   label=f"+ {k}  {lab}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("shortlist size K (vision's top-K)")
    ax[0].set_ylabel("recall @ 1 m [%]"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_title("Acoustics re-ranks vision's shortlist", fontsize=11)

    gains = {k: max(np.array(summary["models"][k]) - base) * 100 for k in models}
    ax[1].bar(range(len(gains)), list(gains.values()),
              color=[STYLE[k][0] for k in gains])
    for i, (k, g) in enumerate(gains.items()):
        best_K = Ks[int(np.argmax(summary["models"][k]))]
        ax[1].text(i, g, f"{g:+.0f}pp\n@K={best_K}", ha="center",
                   va="bottom" if g >= 0 else "top", fontsize=9)
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(range(len(gains)))
    ax[1].set_xticklabels([f"{k}\n{STYLE[k][1]}" for k in gains], fontsize=9)
    ax[1].set_ylabel("best gain over vision alone [pp]")
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title("How much each candidate model adds", fontsize=11)

    fig.suptitle(f"Can sound assist vision?  {scene}, recording made in the real room\n"
                 f"vision alone {100*base:.0f}% @1m over {len(vis_err)} poses, "
                 f"{len(rows)} candidate positions", fontsize=12)
    fig.tight_layout()
    out = args.out or REPO_ROOT / "outputs" / "viz" / f"vision_assist_{scene}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n[assist] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
