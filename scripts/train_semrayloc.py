#!/usr/bin/env python3
"""Train SemRayLoc's ray networks (semantic or depth) on echoloc_dataset.

The SemRayLoc clone (../SemRayLoc, upstream, untouched) expects one folder per
scene with rgb/<i>.png and depth/semantic/pitch/roll/room files; this script
keeps that code as it is and feeds it our collections through a small dataset
adapter instead: rgb in filename order (= poses.txt order), depth<N>.txt and
semantic<N>.txt from scripts/make_semantic_rays.py, roll = pitch = 0, and the
room head trained on the single label "undefined" (no room annotation here).

    python scripts/train_semrayloc.py --net semantic --dataset-root /root/storage/echoloc_dataset/replica \
        --collections replica_f replica_g --run-name srl_sem_replica --gpus 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRL_ROOT = REPO_ROOT.parent / "SemRayLoc"
sys.path.insert(0, str(SRL_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", choices=["semantic", "depth"], required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--collections", nargs="+", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--gpus", default="0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ray-n", type=int, default=40)
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="resize the query image (SemRayLoc trained at 360x640)")
    p.add_argument("--class-weights", type=float, nargs=4, default=None,
                   help="cross-entropy weights for wall/window/door/unknown (semantic net)")
    p.add_argument("--precision", default="bf16-mixed",
                   help="fp16 mixed precision drove the depth net's attention to NaN within the first "
                        "epoch on every benchmark (the semantic net was fine); bf16 has the fp32 range")
    p.add_argument("--monitor", default="loss-valid",
                   help="validation metric that picks the checkpoint: loss-valid (SemRayLoc's own) or, "
                        "for the semantic net, acc_rays_val (the ray class is what the localiser uses; on "
                        "Replica the validation loss rises from epoch 1 through over-confidence while "
                        "the ray accuracy keeps improving)")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit-train-batches", type=float, default=1.0)
    p.add_argument("--limit-val-batches", type=float, default=1.0)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class EchoLocRays:
    """One sample per frame of the given collections and scenes."""

    def __init__(self, root: Path, collections, scenes, ray_n, resize=None, with_semantic=True):
        import torch  # noqa: F401
        self.items = []
        self.resize = resize
        for col in collections:
            for sc in scenes:
                d = root / col / sc
                if not (d / "poses.txt").exists():
                    continue
                names = sorted(f for f in os.listdir(d / "rgb") if f.endswith(".png"))
                depth = np.loadtxt(d / f"depth{ray_n}.txt", ndmin=2, dtype=np.float32)
                sem = (np.loadtxt(d / f"semantic{ray_n}.txt", ndmin=2, dtype=np.int64)
                       if with_semantic else np.zeros_like(depth, dtype=np.int64))
                assert len(names) == len(depth) == len(sem), (d, len(names), len(depth), len(sem))
                for i, n in enumerate(names):
                    self.items.append((str(d / "rgb" / n), depth[i], sem[i]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import torch
        path, depth, sem = self.items[i]
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            img = cv2.resize(img, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
        img = (img / 255.0).astype(np.float32).transpose(2, 0, 1)
        return {
            "ref_img": img,
            "ref_mask": np.ones(img.shape[1:], dtype=np.uint8),
            "ref_depth": depth.astype(np.float32),
            "ref_semantics": sem.astype(np.float32),
            "room_label": torch.tensor(15, dtype=torch.long),   # "undefined" in SemRayLoc's table
        }


def main() -> int:
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    import torch
    import torch.nn.functional as F
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from modules.depth.depth_net_pl import depth_net_pl
    from modules.semantic.semantic_net_pl import semantic_net_pl

    seed_everything(a.seed)
    root = Path(a.dataset_root)
    meta = json.loads((root / "dataset_meta.json").read_text())
    split = meta["split"]
    tr, va = split["train"], split["val"]
    if a.max_scenes:
        tr, va = tr[: a.max_scenes], va[: a.max_scenes]
    ds_tr = EchoLocRays(root, a.collections, tr, a.ray_n, a.resize, a.net == "semantic")
    ds_va = EchoLocRays(root, a.collections, va, a.ray_n, a.resize, a.net == "semantic")
    print(f"[data] train {len(ds_tr)} frames  val {len(ds_va)} frames", flush=True)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, drop_last=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)

    if a.net == "semantic":
        model = semantic_net_pl(num_ray_classes=4, num_room_types=16, lr=a.lr)
        if a.class_weights is not None:
            w = torch.tensor(a.class_weights, dtype=torch.float32)

            def training_step(batch, batch_idx, _m=model):
                ray_logits, room_logits, _ = _m.model(batch["ref_img"], mask=batch.get("ref_mask"))
                loss_rays = F.cross_entropy(ray_logits.permute(0, 2, 1), batch["ref_semantics"].long(), weight=w.to(ray_logits.device))
                loss_room = F.cross_entropy(room_logits, batch["room_label"].long())
                loss = loss_rays + loss_room
                _m.log("loss_rays_train", loss_rays, prog_bar=True)
                _m.log("loss_train", loss)
                return loss
            model.training_step = training_step
    else:
        f_w = float(meta["camera"].get("F_W") or 1 / (2 * np.tan(np.deg2rad(meta["camera"]["hfov_deg"]) / 2)))
        model = depth_net_pl(shape_loss_weight=20, lr=a.lr, d_max=20.0, F_W=f_w)

    out = a.output_dir / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    mode = "max" if a.monitor.startswith("acc") else "min"
    ck = ModelCheckpoint(monitor=a.monitor, dirpath=out, filename=f"{a.net}_net-{{epoch:02d}}-{{{a.monitor}:.3f}}",
                         save_top_k=2, mode=mode, save_last=True)
    trainer = Trainer(max_epochs=a.epochs, max_steps=a.max_steps, callbacks=[ck], accelerator="gpu", devices=1,
                      precision=a.precision, log_every_n_steps=50, default_root_dir=out,
                      limit_train_batches=a.limit_train_batches, limit_val_batches=a.limit_val_batches,
                      gradient_clip_val=1.0)
    trainer.fit(model, dl_tr, dl_va)
    trainer.save_checkpoint(out / f"final_{a.net}.ckpt")
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    (out / "final_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("[done]", json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
