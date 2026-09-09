#!/usr/bin/env python3
"""Show what the acoustic likelihood does to the visual posterior.

One row per test sample, four panels:

  1. the input image
  2. the visual posterior over the floorplan (F3Loc rays matched to the DESDF)
  3. the acoustic likelihood (SoundSpaces candidate grid vs the recording)
  4. the product of the two, on the shared valid-pose support

The visual posterior is what the monocular network alone believes; the point of
the figure is where it is multi-modal and whether acoustics removes the wrong
modes. Both are normalised over the same ``valid_pose_mask``, so the comparison
is between distributions on one support, not between arbitrary scores.

    python scripts/visualize_fusion.py --run-name echoloc_mono --scene office_4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="echoloc_mono")
    p.add_argument("--net", default="mono", choices=["mono", "mv"])
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scene", default="office_4")
    p.add_argument("--condition", default="raw_scan_open",
                   choices=["raw_scan_open", "floorplan_closed"])
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--acoustic-temp", type=float, default=0.5)
    p.add_argument("--n-panels", type=int, default=6)
    p.add_argument("--max-samples", type=int, default=120)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out-dir", type=Path, default=None)
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
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.models import MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    scene = args.scene
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    out_dir = args.out_dir or REPO_ROOT / "outputs" / "viz" / f"{args.run_name}_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- shared pose grid and support -------------------------------------
    desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
    desdf["desdf"][desdf["desdf"] > 10] = 10
    occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
    pg = PoseGrid.from_desdf(desdf, occ.shape)
    mask = valid_pose_mask(occ, pg)
    desdf_t = torch.tensor(desdf["desdf"], device=device)

    # ---- acoustic candidate grid ------------------------------------------
    blob = np.load(REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz")
    cand, index = blob["rir"], blob["index"]
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
    cand_env = np.stack([envelope(c, args.guard_samples, args.usable_samples, win) for c in cand])
    cand_n = cand_env / cand_env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
    cell_rows, cell_cols = index[:, 0], index[:, 1]

    # ---- visual model ------------------------------------------------------
    ckpt = REPO_ROOT / "outputs" / args.run_name / f"{args.net}.ckpt"
    cls = MonoDepthModule if args.net == "mono" else MVDepthModule
    model = cls.load_from_checkpoint(str(ckpt)).to(device).eval()

    dataset_dir = str(root / args.collection)
    dataset = GridSeqDataset(dataset_dir, [scene], L=L, depth_dir=dataset_dir,
                             depth_suffix="depth40" if args.net == "mono" else "depth160")
    poses = np.array([[float(v) for v in l.split()]
                      for l in open(root / args.collection / scene / "poses.txt") if l.strip()])
    rir_dir = root / "rir" / args.collection / args.condition / scene

    n = min(args.max_samples, len(dataset))
    picks = np.linspace(0, len(dataset) - 1, n).astype(int)

    records = []
    for chunk in tqdm.tqdm(picks, desc="scoring"):
        pose_idx = int(chunk) * (L + 1) + L          # RIRs exist for reference frames
        f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
        if not f.exists():
            continue
        data = dataset[int(chunk)]

        with torch.no_grad():
            if args.net == "mono":
                pred, _, _ = model.encoder(
                    torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
            else:
                batch = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                         for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                batch["ref_mask"] = batch["src_mask"] = None
                pred = model.net(batch)["d"]
        pred = pred.squeeze(0).float().cpu().numpy()

        rays = torch.tensor(get_ray_from_depth(pred), device=device, dtype=torch.float32)
        _, prob_dist, _, _ = localize(desdf_t, rays)
        visual = np.array(prob_dist, dtype=np.float64)
        visual[~mask] = 0.0
        visual /= max(visual.sum(), 1e-20)

        obs = envelope(np.load(f), args.guard_samples, args.usable_samples, win)
        obs_n = obs / max(obs.sum(), 1e-20)
        l1 = np.abs(cand_n - obs_n[None]).sum(axis=(1, 2))
        acoustic = np.zeros_like(visual)
        acoustic[cell_rows, cell_cols] = np.exp(-l1 / args.acoustic_temp)
        acoustic[~mask] = 0.0
        acoustic /= max(acoustic.sum(), 1e-20)

        fused = visual * acoustic
        if fused.sum() <= 0:
            continue
        fused /= fused.sum()

        gx, gy, _ = pg.pose_metric_to_grid(poses[pose_idx, :3])
        def err(p):
            r, c = np.unravel_index(int(p.argmax()), p.shape)
            return float(np.hypot(c - gx, r - gy) * pg.grid_resolution_m)
        records.append(dict(chunk=int(chunk), pose_idx=pose_idx, gx=gx, gy=gy,
                            visual=visual, acoustic=acoustic, fused=fused,
                            e_v=err(visual), e_a=err(acoustic), e_f=err(fused)))

    if not records:
        print("no samples with both an image and a recording")
        return 1

    ev = np.array([r["e_v"] for r in records])
    ea = np.array([r["e_a"] for r in records])
    ef = np.array([r["e_f"] for r in records])
    summary = {
        "scene": scene, "condition": args.condition, "run": args.run_name,
        "samples": len(records),
        "visual":   {"median_m": float(np.median(ev)), "recall_1m": float((ev < 1).mean())},
        "acoustic": {"median_m": float(np.median(ea)), "recall_1m": float((ea < 1).mean())},
        "fused":    {"median_m": float(np.median(ef)), "recall_1m": float((ef < 1).mean())},
        "fused_better_than_visual": float((ef < ev).mean()),
    }
    print(json.dumps(summary, indent=2))
    (REPO_ROOT / "outputs" / "metrics" / f"fusion_{scene}_{args.condition}.json").write_text(
        json.dumps(summary, indent=2))

    # ---- panels: the cases where fusion changes the answer -----------------
    gain = ev - ef
    order = np.argsort(-gain)                     # biggest improvement first
    chosen = [records[i] for i in order[: args.n_panels]]

    extent_crop = occ[pg.top:pg.top + pg.height * 10, pg.left:pg.left + pg.width * 10]
    crop = cv2.resize(extent_crop, (pg.width, pg.height), interpolation=cv2.INTER_NEAREST)

    rowsn = len(chosen)
    fig, axes = plt.subplots(rowsn, 4, figsize=(17, 3.5 * rowsn))
    if rowsn == 1:
        axes = axes[None, :]
    for i, rec in enumerate(chosen):
        img = cv2.cvtColor(cv2.imread(str(root / args.collection / scene / "rgb" /
                                          f"{rec['chunk']:05d}-{L}.png")), cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"chunk {rec['chunk']}", fontsize=9)
        for j, (key, title, e) in enumerate(
                [("visual", "visual posterior", rec["e_v"]),
                 ("acoustic", "acoustic likelihood", rec["e_a"]),
                 ("fused", "fused", rec["e_f"])], start=1):
            ax = axes[i, j]
            ax.imshow(crop, cmap="gray", origin="lower", alpha=0.5)
            ax.imshow(np.ma.masked_where(~mask, rec[key]), cmap="viridis", origin="lower", alpha=0.75)
            ax.plot(rec["gx"], rec["gy"], "o", ms=10, mfc="none", mec="lime", mew=2)
            r_, c_ = np.unravel_index(int(rec[key].argmax()), rec[key].shape)
            ax.plot(c_, r_, "x", ms=9, color="red", mew=2)
            ax.set_title(f"{title}   err {e:.2f} m", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.run_name} on {scene} ({args.condition})   "
                 f"median error  visual {np.median(ev):.2f} m -> fused {np.median(ef):.2f} m", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    panels = out_dir / f"fusion_{scene}_{args.condition}.png"
    fig.savefig(panels, dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- aggregate ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for e, lab in ((ev, "visual"), (ea, "acoustic"), (ef, "fused")):
        xs = np.sort(e)
        ax[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs) * 100, lw=2, label=lab)
    ax[0].set_xlim(0, 8); ax[0].set_ylim(0, 100); ax[0].grid(alpha=0.3)
    ax[0].set_xlabel("error [m]"); ax[0].set_ylabel("recall [%]"); ax[0].legend()
    ax[0].set_title("localization recall")
    ax[1].scatter(ev, ef, s=14, alpha=0.6)
    lim = max(ev.max(), ef.max()) * 1.05
    ax[1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
    ax[1].set_xlabel("visual error [m]"); ax[1].set_ylabel("fused error [m]")
    ax[1].set_title(f"per sample  ({100*summary['fused_better_than_visual']:.0f}% improved)")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{scene} / {args.condition} / {len(records)} samples")
    fig.tight_layout()
    summary_png = out_dir / f"fusion_summary_{scene}_{args.condition}.png"
    fig.savefig(summary_png, dpi=120, bbox_inches="tight"); plt.close(fig)

    print(f"wrote {panels}")
    print(f"wrote {summary_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
