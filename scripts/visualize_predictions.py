#!/usr/bin/env python3
"""Render qualitative panels for a trained observation model.

Recall alone does not say whether a model failed because its depth prediction is
wrong or because the floorplan is ambiguous. Each panel therefore shows the
three things together for one test sample:

  1. the input image
  2. predicted vs ground-truth structural depth, in metres
  3. the floorplan with the localization posterior, the ground-truth pose, the
     predicted pose, and the 11 rays the localizer actually matched

Samples are picked across the error distribution (best, median, worst) so the
failure modes are visible, not just the successes.

    python scripts/visualize_predictions.py --run-name echoloc_mono --net mono \
        --dataset replica_f --dataset-root /root/storage/echoloc_dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--net", default="mono", choices=["mono", "mv"])
    parser.add_argument("--dataset", default="gibson_f")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data" / "f3loc"))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-panels", type=int, default=9, help="samples to render, spread over the error range")
    parser.add_argument("--max-samples", type=int, default=400, help="samples to score before picking panels")
    parser.add_argument("--bn-mode", default="eval", choices=["eval", "upstream"])
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import tqdm

    from track1_core.datasets.gibson_f3loc import load_split
    from track1_core.models import MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, V, DV, F_W = 3, 11, 10, 3 / 8
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "viz" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.ckpt or REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    dataset_dir = os.path.join(args.dataset_root, args.dataset)
    depth_suffix = "depth40" if args.net == "mono" else "depth160"
    scenes = load_split(dataset_dir)[args.split]
    dataset = GridSeqDataset(dataset_dir, scenes, L=L, depth_dir=dataset_dir, depth_suffix=depth_suffix)

    cls = MonoDepthModule if args.net == "mono" else MVDepthModule
    model = cls.load_from_checkpoint(str(ckpt)).to(device)
    model.train() if args.bn_mode == "upstream" else model.eval()

    # --- scene geometry -----------------------------------------------------
    desdfs, gt_poses, maps = {}, {}, {}
    for scene in scenes:
        d = np.load(os.path.join(args.dataset_root, "desdf", scene, "desdf.npy"), allow_pickle=True).item()
        d["desdf"][d["desdf"] > 10] = 10
        d["t_desdf"] = torch.tensor(d["desdf"], device=device)
        desdfs[scene] = d
        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        maps[scene] = occ
        h, w = occ.shape
        rows = [l.strip() for l in open(os.path.join(dataset_dir, scene, "poses.txt")) if l.strip()]
        p = np.zeros((len(rows), 3), np.float32)
        for i, r in enumerate(rows):
            x, y, th = (float(v) for v in r.split(" "))
            p[i] = (x / 0.01 + w / 2, y / 0.01 + h / 2, th)
        gt_poses[scene] = p

    starts = np.array(dataset.scene_start_idx)
    n_score = min(args.max_samples, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, n_score).astype(int)

    # --- score, keeping what the panels need -------------------------------
    records = []
    for data_idx in tqdm.tqdm(idxs, desc="scoring"):
        data = dataset[int(data_idx)]
        scene_idx = int(np.sum(data_idx >= starts) - 1)
        scene = dataset.scene_names[scene_idx]
        within = int(data_idx - starts[scene_idx])
        desdf = desdfs[scene]

        batch = {
            "ref_img": torch.tensor(data["ref_img"], device=device).unsqueeze(0),
            "src_img": torch.tensor(data["src_img"], device=device).unsqueeze(0),
            "ref_pose": torch.tensor(data["ref_pose"], device=device).unsqueeze(0),
            "src_pose": torch.tensor(data["src_pose"], device=device).unsqueeze(0),
            "ref_mask": None, "src_mask": None,
        }
        with torch.no_grad():
            if args.net == "mono":
                pred, _, _ = model.encoder(batch["ref_img"], None)
            else:
                pred = model.net(batch)["d"]
        pred = pred.squeeze(0).float().cpu().numpy()
        gt = data["ref_depth"]

        rays = torch.tensor(get_ray_from_depth(pred, V=V, dv=DV, F_W=F_W), device=device, dtype=torch.float32)
        _, prob_dist, _, pose_pred = localize(desdf["t_desdf"], rays)

        gt_desdf = gt_poses[scene][within * (L + 1) + L].copy()
        gt_desdf[0] = (gt_desdf[0] - desdf["l"]) / 10
        gt_desdf[1] = (gt_desdf[1] - desdf["t"]) / 10
        err = float(np.linalg.norm(pose_pred[:2] - gt_desdf[:2]) * 0.1)

        records.append(dict(data_idx=int(data_idx), scene=scene, within=within, pred=pred, gt=gt,
                            prob=prob_dist, pose_pred=pose_pred, gt_desdf=gt_desdf, err=err,
                            l1=float(np.abs(pred - gt).mean())))

    errs = np.array([r["err"] for r in records])
    order = np.argsort(errs)
    picks = np.unique(np.linspace(0, len(order) - 1, args.n_panels).astype(int))
    chosen = [records[order[i]] for i in picks]

    # --- panels -------------------------------------------------------------
    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * n))
    if n == 1:
        axes = axes[None, :]
    for row, rec in enumerate(chosen):
        scene, desdf = rec["scene"], desdfs[rec["scene"]]

        img_path = os.path.join(dataset_dir, scene, "rgb", f"{rec['within']:05d}-{L}.png")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{scene}  chunk {rec['within']}", fontsize=9)
        axes[row, 0].axis("off")

        ax = axes[row, 1]
        u = np.arange(len(rec["gt"]))
        ax.plot(u, rec["gt"], lw=2, label="ground truth")
        ax.plot(u, rec["pred"], lw=2, ls="--", label="predicted")
        ax.set_title(f"structural depth   L1 = {rec['l1']:.3f} m", fontsize=9)
        ax.set_xlabel("image column sample (left to right)", fontsize=8)
        ax.set_ylabel("forward depth [m]", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[row, 2]
        occ = maps[scene]
        H, W = rec["prob"].shape
        crop = occ[desdf["t"]:desdf["t"] + H * 10, desdf["l"]:desdf["l"] + W * 10]
        crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST)
        ax.imshow(crop, cmap="gray", origin="lower", alpha=0.55)
        ax.imshow(rec["prob"], cmap="viridis", origin="lower", alpha=0.55)
        gx, gy, gth = rec["gt_desdf"]
        px, py, pth = rec["pose_pred"]
        # the 11 rays the localizer matched, drawn from the ground-truth pose
        rays_m = get_ray_from_depth(rec["pred"], V=V, dv=DV, F_W=F_W)
        angs = (np.arange(V) - np.arange(V).mean()) * DV / 180 * np.pi
        for r_m, a in zip(rays_m, angs):
            ax.plot([gx, gx + r_m * 10 * np.cos(gth + a)], [gy, gy + r_m * 10 * np.sin(gth + a)],
                    color="cyan", lw=0.7, alpha=0.8)
        ax.plot(gx, gy, "o", ms=9, mfc="none", mec="lime", mew=2, label="ground truth")
        ax.plot(px, py, "x", ms=9, color="red", mew=2, label="prediction")
        ax.set_title(f"posterior over floorplan   error = {rec['err']:.2f} m", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{args.run_name} / {args.net} / {args.dataset} {args.split}  (bn={args.bn_mode})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    panel_path = out_dir / f"panels_{args.dataset}_{args.bn_mode}.png"
    fig.savefig(panel_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- summary ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(errs, bins=40, color="steelblue")
    for thr, c in ((0.1, "green"), (0.5, "orange"), (1.0, "red")):
        ax[0].axvline(thr, color=c, ls="--", lw=1, label=f"{thr} m")
    ax[0].set_xlabel("localization error [m]"); ax[0].set_ylabel("samples")
    ax[0].set_title("error distribution"); ax[0].legend(fontsize=8)

    xs = np.sort(errs)
    ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs) * 100, lw=2)
    for thr, c in ((0.1, "green"), (0.5, "orange"), (1.0, "red")):
        r = float((errs < thr).mean() * 100)
        ax[1].axvline(thr, color=c, ls="--", lw=1)
        ax[1].annotate(f"{thr}m: {r:.1f}%", (thr, r), fontsize=8, color=c,
                       xytext=(4, -10), textcoords="offset points")
    ax[1].set_xlim(0, min(10, xs.max())); ax[1].set_ylim(0, 100)
    ax[1].set_xlabel("error threshold [m]"); ax[1].set_ylabel("recall [%]")
    ax[1].set_title("recall curve"); ax[1].grid(alpha=0.3)
    fig.suptitle(f"{args.run_name} on {args.dataset} {args.split}  ({len(errs)} samples, bn={args.bn_mode})")
    fig.tight_layout()
    summary_path = out_dir / f"summary_{args.dataset}_{args.bn_mode}.png"
    fig.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"scored {len(errs)} samples | mean depth L1 = {np.mean([r['l1'] for r in records]):.3f} m")
    print(f"  recall  0.1m={100*(errs<0.1).mean():.1f}%  0.5m={100*(errs<0.5).mean():.1f}%  1m={100*(errs<1).mean():.1f}%")
    print(f"  median error = {np.median(errs):.2f} m")
    print(f"wrote {panel_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
