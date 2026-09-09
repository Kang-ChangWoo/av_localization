"""Contract tests for the visual training adapters.

These cover the three upstream/adapter mismatches that silently break training:
batch key names, the mask keys the multi-view network indexes blindly, and the
frame-to-depth-row alignment the monocular sampler relies on.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path(
    os.environ.get("AVFPLOC_F3LOC_DATA", "/root/jeongeon/AV-FPLoc/datasets/f3loc")
)

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "gibson_f" / "split.yaml").exists(),
    reason=f"Gibson dataset not available at {DATA_ROOT}",
)


def _scene(dataset: str = "gibson_f") -> tuple[str, list[str]]:
    from track1_core.datasets.gibson_f3loc import load_split

    dataset_dir = str(DATA_ROOT / dataset)
    return dataset_dir, load_split(dataset_dir)["train"][:1]


def test_vendored_baseline_is_importable_and_pinned():
    from track1_core import _vendor

    assert (_vendor.VENDOR_ROOT / "modules" / "mv" / "mv_depth_net.py").exists()
    assert len(_vendor.pinned_revision()) == 40


def test_mono_dataset_emits_the_keys_depth_net_pl_expects():
    from track1_core.datasets.gibson_f3loc import MonoFrameDataset

    dataset_dir, scenes = _scene()
    dataset = MonoFrameDataset(dataset_dir, scenes, L=3, depth_suffix="depth40")
    sample = dataset[0]

    assert set(sample) >= {"img", "gt_rays"}
    assert sample["img"].shape == (3, 480, 640)
    assert sample["gt_rays"].shape == (40,)
    assert sample["img"].dtype == np.float32


def test_mono_views_all_yields_every_frame():
    from track1_core.datasets.gibson_f3loc import MonoFrameDataset

    dataset_dir, scenes = _scene()
    ref_only = MonoFrameDataset(dataset_dir, scenes, L=3, views="ref")
    every = MonoFrameDataset(dataset_dir, scenes, L=3, views="all")
    assert len(every) == 4 * len(ref_only)


def test_mono_frame_matches_grid_seq_reference_frame():
    """Frame (chunk, view=L) must carry the same target as GridSeqDataset's ref."""
    from utils.data_utils import GridSeqDataset

    from track1_core.datasets.gibson_f3loc import MonoFrameDataset

    dataset_dir, scenes = _scene()
    grid = GridSeqDataset(
        dataset_dir, scenes, L=3, depth_dir=dataset_dir, depth_suffix="depth40"
    )
    mono = MonoFrameDataset(dataset_dir, scenes, L=3, views="ref", depth_suffix="depth40")

    for idx in (0, 5, len(mono) - 1):
        np.testing.assert_allclose(mono[idx]["gt_rays"], grid[idx]["ref_depth"])
        np.testing.assert_allclose(mono[idx]["img"], grid[idx]["ref_img"])


def test_collate_supplies_the_mask_keys_mv_indexes():
    from torch.utils.data import DataLoader

    from track1_core.datasets.gibson_f3loc import collate_with_optional_masks
    from utils.data_utils import GridSeqDataset

    dataset_dir, scenes = _scene()
    dataset = GridSeqDataset(
        dataset_dir, scenes, L=3, depth_dir=dataset_dir, depth_suffix="depth160"
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_with_optional_masks)))

    # mv_depth_net.forward reads these unconditionally
    assert batch["ref_mask"] is None
    assert batch["src_mask"] is None
    assert batch["ref_img"].shape == (2, 3, 480, 640)
    assert batch["src_img"].shape == (2, 3, 3, 480, 640)
    assert batch["ref_depth"].shape == (2, 160)


def test_comp_module_trains_only_the_selector():
    from track1_core.models import CompDepthModule

    module = CompDepthModule()
    trainable = {n for n, p in module.named_parameters() if p.requires_grad}
    assert trainable
    assert all("selector" in name for name in trainable)

    # frozen observation nets stay in eval mode even after .train()
    module.train()
    assert not module.comp_d_net.mv_net.training
    assert not module.comp_d_net.mono_net.training
    assert module.comp_d_net.selector.training


def test_checkpoints_stay_loadable_by_the_upstream_wrappers():
    """Submodule attribute names must match upstream's Lightning wrappers."""
    from modules.mono.depth_net_pl import depth_net_pl
    from modules.mv.mv_depth_net_pl import mv_depth_net_pl

    from track1_core.models import MVDepthModule, MonoDepthModule

    mono_keys = set(MonoDepthModule().state_dict())
    assert mono_keys == set(depth_net_pl().state_dict())

    mv_keys = set(MVDepthModule().state_dict())
    assert mv_keys == set(mv_depth_net_pl(d_hyp=-0.2).state_dict())
