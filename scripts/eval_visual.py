#!/usr/bin/env python3
"""Evaluate trained visual observation models with the F3Loc protocol.

Same localization pipeline as upstream ``eval_observation.py`` (predict the 1D
structural depth, convert to 11 rays, match against the scene DESDF), with two
changes: checkpoints are the ones produced by ``train_visual.py``, and network
inference is batched on the GPU instead of running one sample at a time.

Example:
    python scripts/eval_visual.py --net mono --dataset gibson_f \
        --ckpt outputs/mono/mono.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", required=True, choices=["mono", "mv", "comp", "comp_s"])
    parser.add_argument(
        "--dataset",
        default="gibson_f",
        help="collection directory under --dataset-root (gibson_f, replica_f, ...)",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "data" / "f3loc"),
        help="staged dataset copy; see scripts/stage_f3loc_dataset.py",
    )
    parser.add_argument("--ckpt", type=Path, default=None, help="checkpoint for mono/mv/comp")
    parser.add_argument("--mono-ckpt", type=Path, default=None, help="required for comp_s")
    parser.add_argument("--mv-ckpt", type=Path, default=None, help="required for comp_s")
    parser.add_argument("--gpu", default="1", help="single CUDA device id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--split", default="test")
    parser.add_argument("--trans-thresh", type=float, default=0.005, help="comp_s translation variance threshold")
    parser.add_argument(
        "--bn-mode",
        default="eval",
        choices=["eval", "upstream"],
        help="'eval' puts BatchNorm in inference mode (running statistics). "
        "'upstream' reproduces eval_observation.py, which leaves the monocular "
        "and multi-view nets in train mode, so BatchNorm normalises by each "
        "single sample; use it to compare against the published numbers.",
    )
    parser.add_argument("--out", type=Path, default=None, help="write metrics as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))

    import cv2
    import numpy as np
    import torch
    import tqdm
    from torch.utils.data import DataLoader

    from track1_core.datasets.gibson_f3loc import collate_with_optional_masks, load_split
    from track1_core.models import CompDepthModule, MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    dataset_dir = os.path.join(args.dataset_root, args.dataset)
    desdf_path = os.path.join(args.dataset_root, "desdf")
    depth_suffix = "depth40" if args.net == "mono" else "depth160"

    scenes = load_split(dataset_dir)[args.split]
    dataset = GridSeqDataset(
        dataset_dir, scenes, L=L, depth_dir=dataset_dir, depth_suffix=depth_suffix
    )

    # ---- models ----------------------------------------------------------
    def set_mode(module):
        # upstream's eval_observation.py never calls .eval() on the mono/mv nets,
        # so their BatchNorm layers normalise by the single sample in flight.
        return module.train() if args.bn_mode == "upstream" else module.eval()

    mono_net = mv_net = comp_net = None
    if args.net == "mono":
        mono_net = set_mode(MonoDepthModule.load_from_checkpoint(str(args.ckpt)).to(device))
    elif args.net == "mv":
        mv_net = set_mode(MVDepthModule.load_from_checkpoint(str(args.ckpt)).to(device))
    elif args.net == "comp":
        comp_net = CompDepthModule.load_from_checkpoint(
            str(args.ckpt), mono_ckpt=None, mv_ckpt=None
        ).to(device)
        comp_net.eval()  # required: disables BatchNorm updates in the selector
    else:  # comp_s: hard threshold on translation variance
        mono_net = set_mode(MonoDepthModule.load_from_checkpoint(str(args.mono_ckpt)).to(device))
        mv_net = set_mode(MVDepthModule.load_from_checkpoint(str(args.mv_ckpt)).to(device))

    # ---- ground truth in map frame ---------------------------------------
    desdfs, gt_poses = {}, {}
    for scene in tqdm.tqdm(scenes, desc="load desdf/poses"):
        desdf = np.load(os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10  # truncate, as upstream does
        desdf["desdf_t"] = torch.tensor(desdf["desdf"], device=device)
        desdfs[scene] = desdf

        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        h, w = occ.shape[0], occ.shape[1]
        with open(os.path.join(dataset_dir, scene, "poses.txt"), "r") as handle:
            rows = [line.strip() for line in handle if line.strip()]
        poses = np.zeros([len(rows), 3], dtype=np.float32)
        for i, row in enumerate(rows):
            x, y, th = (float(v) for v in row.split(" "))
            poses[i, :] = (x / 0.01 + w / 2, y / 0.01 + h / 2, th)
        gt_poses[scene] = poses

    scene_start_idx = np.array(dataset.scene_start_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_optional_masks,
    )

    acc_record, acc_orn_record = [], []
    global_idx = 0
    for batch in tqdm.tqdm(loader, desc=f"eval {args.net} on {args.dataset}"):
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        n = batch["ref_img"].shape[0]

        with torch.no_grad():
            if args.net == "mono":
                pred, _, _ = mono_net.encoder(batch["ref_img"], batch["ref_mask"])
            elif args.net == "mv":
                pred = mv_net.net(batch)["d"]
            elif args.net == "comp":
                pred = comp_net.comp_d_net(batch)["d_comp"]
            else:
                # per-sample choice between mono and mv on translation variance
                pose_var = (
                    torch.cat((batch["ref_pose"].unsqueeze(1), batch["src_pose"]), dim=1)
                    .var(dim=1)[:, :2]
                    .sum(dim=-1)
                )
                pred_mv = mv_net.net(batch)["d"]
                pred_mono, _, _ = mono_net.encoder(batch["ref_img"], batch["ref_mask"])
                pred = [
                    (pred_mono[i] if pose_var[i] < args.trans_thresh else pred_mv[i])
                    for i in range(n)
                ]
        pred_list = (
            [p.float().cpu().numpy() for p in pred]
            if isinstance(pred, list)
            else list(pred.float().cpu().numpy())
        )

        for i in range(n):
            data_idx = global_idx + i
            scene_idx = int(np.sum(data_idx >= scene_start_idx) - 1)
            scene = dataset.scene_names[scene_idx]
            idx_within_scene = data_idx - scene_start_idx[scene_idx]
            desdf = desdfs[scene]

            gt_pose_desdf = gt_poses[scene][idx_within_scene * (L + 1) + L, :].copy()
            gt_pose_desdf[0] = (gt_pose_desdf[0] - desdf["l"]) / 10
            gt_pose_desdf[1] = (gt_pose_desdf[1] - desdf["t"]) / 10

            rays = torch.tensor(get_ray_from_depth(pred_list[i]), device=device, dtype=torch.float32)
            _, _, _, pose_pred = localize(desdf["desdf_t"], rays)

            acc = float(np.linalg.norm(pose_pred[:2] - gt_pose_desdf[:2], 2.0) * 0.1)
            acc_record.append(acc)
            acc_orn = (pose_pred[2] - gt_pose_desdf[2]) % (2 * np.pi)
            acc_orn_record.append(min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180)
        global_idx += n

    acc = np.array(acc_record)
    orn = np.array(acc_orn_record)
    metrics = {
        "net": args.net,
        "bn_mode": args.bn_mode,
        "dataset": args.dataset,
        "split": args.split,
        "samples": int(acc.shape[0]),
        "recall_0.1m": float(np.mean(acc < 0.1)),
        "recall_0.5m": float(np.mean(acc < 0.5)),
        "recall_1m": float(np.mean(acc < 1)),
        "recall_1m_30deg": float(np.mean((acc < 1) & (orn < 30))),
        "median_error_m": float(np.median(acc)),
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
