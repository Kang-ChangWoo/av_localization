#!/usr/bin/env python3
"""Train the F3Loc visual observation networks (mono -> mv -> comp).

Upstream F3Loc never released its training script; this is the AV-FPLoc
reproduction entry point. It reads a YAML config from ``configs/``, allows
targeted command-line overrides, and runs multi-GPU DDP.

Example:
    python scripts/train_visual.py --net mono --config configs/visual_mono.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", required=True, choices=["mono", "mv", "comp"])
    parser.add_argument("--config", type=Path, default=None, help="defaults to configs/visual_<net>.yaml")
    parser.add_argument("--gpus", default=None, help="CUDA device ids, e.g. '1,2,3,4,5,6,7' (config default otherwise)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="dataloader workers per GPU")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="Gibson collections to train on")
    parser.add_argument("--max-scenes", type=int, default=None, help="limit scenes per split (smoke tests)")
    parser.add_argument("--limit-train-batches", type=float, default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="hard stop after N optimiser steps")
    parser.add_argument("--mono-ckpt", type=Path, default=None, help="comp only")
    parser.add_argument("--mv-ckpt", type=Path, default=None, help="comp only")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override any config entry, e.g. --set model.shape_loss_weight=null "
        "(value is parsed as YAML, so null/true/2.0 all work)",
    )
    return parser.parse_args()


def apply_overrides(config: dict, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        path, raw = item.split("=", 1)
        keys = path.split(".")
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = yaml.safe_load(raw)


def load_config(net: str, path: Path | None) -> dict:
    path = path or REPO_ROOT / "configs" / f"visual_{net}.yaml"
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    args = parse_args()
    config = load_config(args.net, args.config)
    apply_overrides(config, args.overrides)

    train_cfg = config.setdefault("train", {})
    data_cfg = config.setdefault("data", {})
    model_cfg = config.setdefault("model", {})

    # CUDA_VISIBLE_DEVICES has to be set before torch initialises the driver.
    gpus = args.gpus or train_cfg.get("gpus")
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
    n_devices = len(str(gpus).split(",")) if gpus else 1

    # Heavy imports come after the device mask is in place.
    sys.path.insert(0, str(REPO_ROOT))
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
    from lightning.pytorch.loggers import CSVLogger

    from track1_core.datasets.gibson_f3loc import GibsonF3LocDataModule
    from track1_core.models import build_module

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(train_cfg.get("matmul_precision", "high"))

    # ---- overrides -------------------------------------------------------
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.precision is not None:
        train_cfg["precision"] = args.precision
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.max_scenes is not None:
        data_cfg["max_scenes"] = args.max_scenes
    if args.datasets is not None:
        data_cfg["datasets"] = args.datasets
    if args.lr is not None:
        model_cfg["lr"] = args.lr
    if args.mono_ckpt is not None:
        model_cfg["mono_ckpt"] = str(args.mono_ckpt)
    if args.mv_ckpt is not None:
        model_cfg["mv_ckpt"] = str(args.mv_ckpt)

    dataset_root = os.path.expandvars(
        data_cfg.get("root")
        or os.path.join(os.environ.get("AVFPLOC_DATA_ROOT", ""), "datasets", "f3loc")
    )
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"dataset root not found: {dataset_root}")

    output_dir = args.output_dir or Path(train_cfg.get("output_dir", REPO_ROOT / "outputs"))
    run_name = args.run_name or train_cfg.get("run_name", args.net)
    run_dir = Path(output_dir) / run_name

    # ---- data ------------------------------------------------------------
    datamodule = GibsonF3LocDataModule(
        net_type=args.net,
        dataset_root=dataset_root,
        dataset_names=data_cfg.get("datasets", ["gibson_f", "gibson_g"]),
        L=data_cfg.get("L", 3),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 8),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        views=data_cfg.get("views", "all"),
        add_rp=data_cfg.get("add_rp", False),
        roll=data_cfg.get("roll", 0.0),
        pitch=data_cfg.get("pitch", 0.0),
        max_scenes=data_cfg.get("max_scenes"),
    )
    datamodule.setup()

    epochs = train_cfg.get("epochs", 20)
    world = max(n_devices, 1)
    steps_per_epoch = math.ceil(len(datamodule.train_set) / (datamodule.batch_size * world))
    total_steps = steps_per_epoch * epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    model_cfg.setdefault("max_steps", total_steps)

    # ---- model -----------------------------------------------------------
    model = build_module(args.net, **model_cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        # metric names contain '-', which str.format cannot interpolate,
        # so the filename carries only epoch/step and ranking uses `monitor`
        filename="epoch{epoch:03d}-step{step:06d}",
        monitor="loss-valid",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_cb, LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=20)]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=world,
        strategy=(
            "ddp_find_unused_parameters_true" if args.net == "comp" and world > 1
            else ("ddp" if world > 1 else "auto")
        ),
        precision=train_cfg.get("precision", "bf16-mixed"),
        max_epochs=epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        limit_train_batches=args.limit_train_batches or 1.0,
        limit_val_batches=args.limit_val_batches or 1.0,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(output_dir), name=run_name),
        default_root_dir=str(run_dir),
        num_sanity_val_steps=2,
    )

    if trainer.is_global_zero:
        print(f"[train_visual] net={args.net} devices={world} gpus={gpus}")
        print(f"[train_visual] dataset_root={dataset_root} datasets={data_cfg.get('datasets')}")
        print(f"[train_visual] train={len(datamodule.train_set)} val={len(datamodule.val_set)} samples")
        print(f"[train_visual] steps/epoch={steps_per_epoch} epochs={epochs} total_steps={total_steps}")
        print(f"[train_visual] run_dir={run_dir}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=str(args.resume) if args.resume else None)

    if trainer.is_global_zero:
        best = checkpoint_cb.best_model_path
        print(f"[train_visual] best checkpoint: {best} ({checkpoint_cb.best_model_score})")
        if best:
            # Stable name the evaluation scripts and the comp stage look for.
            alias = run_dir / f"{args.net}.ckpt"
            alias.write_bytes(Path(best).read_bytes())
            print(f"[train_visual] wrote {alias}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
