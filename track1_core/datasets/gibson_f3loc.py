"""Gibson Floorplan Localization dataset adapters for visual training.

Upstream F3Loc ships ``GridSeqDataset`` for evaluation but never released the
training script, so the batch keys its Lightning wrappers expect do not line up
with what the dataset emits. This module owns that adapter layer:

* ``MonoFrameDataset``   -> ``{"img", "gt_rays", "mask"}``   (monocular net)
* ``GridSeqDataset``     -> ``{"ref_img", "src_img", ...}``  (multi-view / comp)

Ground-truth conventions follow the dataset README: ``poses.txt`` and the depth
files are ordered by ascending rgb filename, so for the four-view collections
line ``chunk * (L + 1) + view`` describes ``{chunk:05d}-{view}.png``.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, default_collate

import lightning.pytorch as pl

from track1_core import _vendor  # noqa: F401  (puts vendored f3loc on sys.path)
from utils.data_utils import GridSeqDataset
from utils.utils import gravity_align

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MASK_KEYS = ("mask", "ref_mask", "src_mask")


def load_split(dataset_dir: str) -> dict:
    with open(os.path.join(dataset_dir, "split.yaml"), "r") as handle:
        return yaml.safe_load(handle)


def worker_init(_worker_id: int) -> None:
    """Stop each dataloader worker from oversubscribing the machine.

    OpenCV defaults to one thread per core (96 here), and every worker inherits
    that. With seven DDP ranks the pool would try to run thousands of threads
    over 96 cores, and the resulting contention starves the GPUs.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def collate_with_optional_masks(batch):
    """Collate, then materialise the mask keys the vendored nets index blindly.

    ``mv_depth_net.forward`` reads ``x["ref_mask"]`` / ``x["src_mask"]``
    unconditionally and treats ``None`` as "no gravity-alignment mask".
    ``default_collate`` cannot carry ``None`` through, so the keys are added
    after collation instead of inside ``__getitem__``.
    """
    collated = default_collate(batch)
    for key in MASK_KEYS:
        collated.setdefault(key, None)
    return collated


def _normalize_bgr(img: np.ndarray) -> np.ndarray:
    """BGR uint8/float image -> ImageNet-normalised RGB, matching upstream."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img -= IMAGENET_MEAN
    img /= IMAGENET_STD
    return img


class MonoFrameDataset(Dataset):
    """Single-frame samples for the monocular structural depth network.

    ``GridSeqDataset`` loads all ``L + 1`` views per item, but the monocular
    network only ever consumes the reference frame. Indexing frames directly
    cuts image I/O by ``L + 1`` and, with ``views="all"``, exposes every frame
    as its own training sample instead of discarding the ``L`` source views.

    views:
        ``"all"`` - every frame is a sample (default; more data, same I/O/sample)
        ``"ref"`` - only the frame the evaluation protocol uses as reference
    """

    def __init__(
        self,
        dataset_dir: str,
        scene_names: Sequence[str],
        L: int = 3,
        depth_dir: str | None = None,
        depth_suffix: str = "depth40",
        views: str = "all",
        add_rp: bool = False,
        roll: float = 0.0,
        pitch: float = 0.0,
    ) -> None:
        super().__init__()
        if views not in ("all", "ref"):
            raise ValueError(f"views must be 'all' or 'ref', got {views!r}")
        self.dataset_dir = dataset_dir
        self.scene_names = list(scene_names)
        self.L = L
        self.depth_dir = depth_dir or dataset_dir
        self.depth_suffix = depth_suffix
        self.views = views
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch

        # index: (scene_name, chunk, view) plus the matching depth row
        self.samples: list[tuple[str, int, int]] = []
        self.gt_depth: dict[str, np.ndarray] = {}
        self._build_index()

    def _build_index(self) -> None:
        stride = self.L + 1
        for scene in self.scene_names:
            depth_file = os.path.join(self.depth_dir, scene, self.depth_suffix + ".txt")
            with open(depth_file, "r") as handle:
                rows = [line.strip() for line in handle if line.strip()]
            depths = np.array(
                [[float(v) for v in row.split(" ")] for row in rows], dtype=np.float32
            )
            self.gt_depth[scene] = depths

            n_chunks = depths.shape[0] // stride
            view_range = range(stride) if self.views == "all" else (self.L,)
            for chunk in range(n_chunks):
                for view in view_range:
                    self.samples.append((scene, chunk, view))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        scene, chunk, view = self.samples[idx]
        image_path = os.path.join(
            self.dataset_dir, scene, "rgb", f"{chunk:05d}-{view}.png"
        )
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            # A collection built with L=0 has one frame per chunk and no view to
            # disambiguate, so it names files `00000.png` rather than
            # `00000-0.png`. Structured3D is such a collection. Falling back is
            # safe because the suffixed name is tried first and only a dataset
            # that has no such file reaches here.
            flat = os.path.join(self.dataset_dir, scene, "rgb", f"{chunk:05d}.png")
            img = cv2.imread(flat, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"{image_path} (also tried {flat})")
            image_path = flat
        img = img.astype(np.float32)

        data = {}
        if self.add_rp:
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            mask = np.ones(list(img.shape[:2]))
            mask = gravity_align(mask, r, p, mode=1)
            mask[mask < 1] = 0
            mask = mask.astype(np.uint8)
            img = _normalize_bgr(img)
            img[mask == 0, :] = 0
            data["mask"] = mask
        else:
            img = _normalize_bgr(img)

        data["img"] = np.transpose(img, (2, 0, 1)).astype(np.float32)
        data["gt_rays"] = self.gt_depth[scene][chunk * (self.L + 1) + view]
        return data


def build_dataset(
    net_type: str,
    dataset_root: str,
    dataset_names: Iterable[str],
    split: str,
    L: int = 3,
    views: str = "all",
    add_rp: bool = False,
    roll: float = 0.0,
    pitch: float = 0.0,
    max_scenes: int | None = None,
) -> Dataset:
    """Concatenate one dataset per Gibson collection for the requested split."""
    depth_suffix = "depth40" if net_type == "mono" else "depth160"
    parts = []
    for name in dataset_names:
        dataset_dir = os.path.join(dataset_root, name)
        scenes = load_split(dataset_dir)[split]
        if max_scenes is not None:
            scenes = scenes[:max_scenes]
        if net_type == "mono":
            parts.append(
                MonoFrameDataset(
                    dataset_dir,
                    scenes,
                    L=L,
                    depth_dir=dataset_dir,
                    depth_suffix=depth_suffix,
                    views=views,
                    add_rp=add_rp,
                    roll=roll,
                    pitch=pitch,
                )
            )
        else:
            parts.append(
                GridSeqDataset(
                    dataset_dir,
                    scenes,
                    L=L,
                    depth_dir=dataset_dir,
                    depth_suffix=depth_suffix,
                    add_rp=add_rp,
                    roll=roll,
                    pitch=pitch,
                )
            )
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


class GibsonF3LocDataModule(pl.LightningDataModule):
    """Train/val loaders over the Gibson Floorplan Localization Dataset."""

    def __init__(
        self,
        net_type: str,
        dataset_root: str,
        dataset_names: Sequence[str] = ("gibson_f", "gibson_g"),
        L: int = 3,
        batch_size: int = 16,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        views: str = "all",
        add_rp: bool = False,
        roll: float = 0.0,
        pitch: float = 0.0,
        max_scenes: int | None = None,
    ) -> None:
        super().__init__()
        if net_type not in ("mono", "mv", "comp"):
            raise ValueError(f"unknown net_type {net_type!r}")
        self.net_type = net_type
        self.dataset_root = dataset_root
        self.dataset_names = list(dataset_names)
        self.L = L
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.views = views
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.max_scenes = max_scenes
        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None

    def _build(self, split: str) -> Dataset:
        return build_dataset(
            self.net_type,
            self.dataset_root,
            self.dataset_names,
            split,
            L=self.L,
            views=self.views,
            add_rp=self.add_rp and split == "train",
            roll=self.roll,
            pitch=self.pitch,
            max_scenes=self.max_scenes,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is None:
            self.train_set = self._build("train")
        if self.val_set is None:
            self.val_set = self._build("val")

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_with_optional_masks,
            worker_init_fn=worker_init,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
            # A multi-view sample is ~15 MB of float32, so in-flight batches add
            # up fast across ranks; deep prefetch evicts the page cache holding
            # the dataset and ends up slower than no prefetch at all.
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)
