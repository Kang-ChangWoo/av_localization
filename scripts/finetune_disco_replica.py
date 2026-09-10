#!/usr/bin/env python3
"""Fine-tune DisCo's contrastive reranker on Replica instead of training it fresh.

Training the reranker from scratch on Replica overfits hard: 98% training
accuracy against 29% validation, with validation loss flat from the third epoch.
Replica's training split is 11 scenes and about 13k samples, where the released
Gibson model saw roughly a hundred scenes, so twenty epochs of a Gibson-sized
recipe is enough to memorise this set.

Starting from the released Gibson weights is the standard answer when the target
set is small, and here it is also the right answer for the question being asked.
The reranker is the stage that *lost* 5.4 points when transferred to Replica
zero-shot, so what we want to know is whether a modest amount of in-domain data
repairs that transfer, not whether Replica alone can train the stage.

Everything except initialisation, learning rate and epoch count matches the
authors' Replica-shaped config, and the from-scratch run is kept, so the two are
comparable.

    python scripts/finetune_disco_replica.py --epochs 8 --lr 1e-5
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DISCO_ROOT = REPO_ROOT.parent / "DisCo-FLoc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/replica_avfploc/disco_replica_f.yaml")
    p.add_argument("--init-from", default="checkpoints/DisCo_gibson_f_best.ckpt",
                   help="released Gibson reranker; empty string trains from scratch")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="an order of magnitude below the from-scratch rate, "
                        "because the weights already encode the task")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-name", default="disco_replica_finetune")
    p.add_argument("--gpu", default="1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    os.chdir(DISCO_ROOT)
    sys.path.insert(0, str(DISCO_ROOT))

    import pytorch_lightning as pl
    import torch
    import yaml
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from training.DisCo_lightning_module import DisCoLocModel
    from DisCo_model.disco_dataset import DisCo_Dataset

    config = yaml.safe_load(open(args.config))
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["run_name"] = args.run_name
    config.setdefault("dptv2_ckpt_path", "checkpoints/depth_anything_v2_vits.pth")
    dcfg = config["datasets"]
    fp = tuple(dcfg.get("floorplan_img_size", [256, 256]))

    mk = lambda split, aug: DisCo_Dataset(
        data_folder=dcfg["data_folder"], data_splits_path=dcfg["data_splits"],
        split=split, floorplan_img_size=fp, pose_aug_params=aug, dataset_cfg=dcfg)
    train = mk("train", dcfg.get("pose_aug", {"enable": True, "trans_range": 25,
                                              "rot_range": 0.26}))
    val = mk(dcfg.get("val_split", "test"), None)
    print(f"[data] train {len(train)}, val {len(val)}")

    model = DisCoLocModel(config)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init] {args.init_from}: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        if len(missing) > 50:
            print("[init] WARNING: that is a lot of missing keys; the checkpoint "
                  "may not match this model definition")
    else:
        print("[init] from scratch")

    nw = config.get("num_workers", 4)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                    num_workers=nw, pin_memory=True, drop_last=True)
    vl = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                    num_workers=nw, drop_last=True)

    folder = os.path.join("logs", "disco_runs", args.run_name)
    os.makedirs(folder, exist_ok=True)
    ck = ModelCheckpoint(dirpath=os.path.join(folder, "checkpoints"),
                         filename="{epoch:02d}-{val_acc:.3f}_"
                                  + datetime.now().strftime("%Y%m%d_%H%M%S"),
                         save_top_k=3, monitor="val_acc", mode="max")
    pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
               devices=1, max_epochs=args.epochs, callbacks=[ck],
               default_root_dir=folder, log_every_n_steps=10).fit(model, tl, vl)
    print(f"[done] best {ck.best_model_path} at val_acc {ck.best_model_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
