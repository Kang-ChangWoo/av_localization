#!/usr/bin/env python3
"""Which parts of the acoustic feature survive furniture, measured not assumed.

The candidate grid can only ever be rendered from a floorplan, because a
floorplan is all a deployment has. Rendering candidates on a furnished mesh
would close the domain gap by using information the method is not allowed to
have, so it is not a method. The legitimate move is the opposite one: keep only
the components of the feature that a floorplan can predict, and stop charging
the score for the components it cannot.

Which components those are is measurable, because the dataset contains the same
pose rendered twice, once on the furnished scan and once on the walls alone.
Every component of the feature therefore has two variances that can be estimated
separately:

    signal   how much the component varies from pose to pose in the wall-only
             domain. A component that is the same everywhere in the room carries
             no position information however clean it is.
    noise    how much furniture moves the component, at a fixed pose. This is the
             error the score pays when it compares a furnished recording against
             a wall-only candidate.

The weight is the ratio the two imply,

    w = signal / (signal + noise)

which is one for a component furniture leaves alone and zero for one furniture
destroys, with no threshold to choose. The weighted L1 that uses it is the same
distance as before with a per-component reliability, and nothing else changes.

Two constraints make this an honest procedure rather than a fit.

**The weights come from the training scenes only** and are applied unchanged to
the test scenes. The test rooms are never used to decide what to keep.

**One weighting for every scene.** A per-scene weighting would be eleven small
fits reported as one method. This script therefore also measures whether the
weights agree across scenes, and says so plainly if they do not, because a
weighting that differs per room is evidence against the whole idea rather than
a licence to fit one per room.

    python scripts/furniture_invariance.py --n-poses 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TRAIN_SCENES = ["frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
                "frl_apartment_3", "hotel_0", "office_0", "office_1",
                "office_2", "room_0", "room_1", "room_2"]
VAL_SCENES = ["apartment_1", "frl_apartment_4", "office_3"]
TEST_SCENES = ["apartment_2", "frl_apartment_5", "office_4"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--scenes", nargs="+", default=TRAIN_SCENES)
    p.add_argument("--check-scenes", nargs="+", default=VAL_SCENES,
                   help="held-out scenes used only to check that the weights transfer")
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--n-poses", type=int, default=150, help="per scene")
    p.add_argument("--floor", type=float, default=0.0,
                   help="weights below this are set to zero, dropping the component")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "analysis" / "furniture_weights.npz")
    p.add_argument("--report", type=Path,
                   default=REPO_ROOT / "outputs" / "analysis" / "furniture_invariance.md")
    p.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tqdm

    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, featurise, observation_rate,
    )
    from track1_core.provenance import stamp

    root = Path(args.dataset_root)
    probe = root / "rir" / args.collection / "raw_scan_open" / args.scenes[0] / "pose_00003"
    sr = observation_rate(probe / "rir.npy")
    cfg = GridScoreConfig(feature="envelope", window_ms=args.window_ms,
                          sample_rate_hz=sr,
                          direct_guard_samples=int(round(sr * 2 / 1000)),
                          usable_samples=int(round(sr * 128 / 1000)))
    print(f"[cfg] {sr} Hz, guard 2 ms, window 128 ms, envelope {args.window_ms:g} ms")

    def pairs_for(scene: str):
        """Featurised (wall-only, furnished) pairs at the same poses."""
        pd_ = root / "rir" / args.collection / "floorplan_closed" / scene
        fd = root / "rir" / args.collection / "raw_scan_open" / scene
        names = sorted(set(os.listdir(pd_)) & set(os.listdir(fd)))
        names = [n for n in names if n.startswith("pose_")]
        picks = np.linspace(0, len(names) - 1, min(args.n_poses, len(names))).astype(int)
        P, F = [], []
        for i in picks:
            a, b = pd_ / names[int(i)] / "rir.npy", fd / names[int(i)] / "rir.npy"
            if not a.exists() or not b.exists():
                continue
            P.append(featurise(band_energy(np.load(a), cfg, observation_rate(a)), cfg))
            F.append(featurise(band_energy(np.load(b), cfg, observation_rate(b)), cfg))
        return np.asarray(P), np.asarray(F)

    def stats(P: np.ndarray, F: np.ndarray):
        """Per-component signal and furniture noise variances, and their ratio."""
        signal = P.var(axis=0)                 # spread across poses, wall-only domain
        noise = (F - P).var(axis=0)            # what furniture does at a fixed pose
        w = signal / (signal + noise + 1e-30)
        return signal, noise, w

    per_scene = {}
    allP, allF = [], []
    for s in tqdm.tqdm(args.scenes, desc="train scenes"):
        P, F = pairs_for(s)
        if len(P) < 10:
            continue
        per_scene[s] = stats(P, F)[2]
        allP.append(P); allF.append(F)
    if not allP:
        print("no pairs found")
        return 1
    P = np.concatenate(allP); F = np.concatenate(allF)
    signal, noise, w = stats(P, F)
    if args.floor > 0:
        w = np.where(w >= args.floor, w, 0.0)
    print(f"[fit] {len(P)} pose pairs over {len(per_scene)} training scenes, "
          f"feature {w.shape}")

    # ---- does one weighting serve every scene? --------------------------
    keys = list(per_scene)
    cross = np.ones((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            x, y = per_scene[a].ravel(), per_scene[b].ravel()
            cross[i, j] = float(np.corrcoef(x, y)[0, 1])
    off = cross[~np.eye(len(keys), dtype=bool)]

    # ---- do they transfer to scenes never used to fit them? -------------
    # Two granularities, because they can transfer very differently. The full
    # map has one weight per microphone row and frame, and its row structure is
    # a directional property of the room. The time profile averages the rows
    # away and asks only "how far into the response can a floorplan still be
    # trusted", which is a physical question with a room-independent answer if
    # it has any answer at all.
    nf = cfg.usable_samples // cfg.window_samples
    prof = w[:, :nf].mean(axis=0)
    checks, checks_prof = {}, {}
    for s in tqdm.tqdm(args.check_scenes + TEST_SCENES, desc="held-out scenes"):
        P2, F2 = pairs_for(s)
        if len(P2) < 10:
            continue
        w2 = stats(P2, F2)[2]
        checks[s] = float(np.corrcoef(w2.ravel(), w.ravel())[0, 1])
        checks_prof[s] = float(np.corrcoef(w2[:, :nf].mean(axis=0), prof)[0, 1])
    cross_prof = np.ones((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            cross_prof[i, j] = float(np.corrcoef(
                per_scene[a][:, :nf].mean(axis=0), per_scene[b][:, :nf].mean(axis=0))[0, 1])
    off_prof = cross_prof[~np.eye(len(keys), dtype=bool)]

    n_frames = cfg.usable_samples // cfg.window_samples
    per_frame = w[:, :n_frames].mean(axis=0)
    ms = (np.arange(n_frames) + 0.5) * args.window_ms

    out: list[str] = []
    W = out.append
    W("# Which acoustic components survive furniture\n")
    W(f"Fitted on {len(per_scene)} training scenes, {len(P)} pose pairs, "
      f"{args.collection}. The test rooms are not used anywhere in this file "
      f"except as a transfer check.\n")
    W(f"\nFeature shape {tuple(w.shape)}: {w.shape[0]} microphone rows by "
      f"{w.shape[1]} columns, of which the first {n_frames} are 2 ms envelope "
      f"frames and the last {w.shape[1]-n_frames} are the per-row energy split.\n")

    W("\n## Reliability against arrival time\n")
    W("| post-direct time, ms | mean weight | signal variance | furniture variance |")
    W("|---|---|---|---|")
    edges = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
    for lo, hi in edges:
        k = (ms >= lo) & (ms < hi)
        if not k.any():
            continue
        W(f"| {lo}-{hi} | {per_frame[k].mean():.3f} | "
          f"{signal[:, :n_frames][:, k].mean():.3e} | "
          f"{noise[:, :n_frames][:, k].mean():.3e} |")
    W(f"\nEnergy-split columns: mean weight {w[:, n_frames:].mean():.3f}.\n")

    W("\n## Is one weighting enough for every scene?\n")
    W("Two granularities. The full map is one weight per microphone row and "
      "frame; the time profile averages the rows away.\n")
    W(f"\nAgreement between training scenes: full map median "
      f"{np.median(off):.3f} (range {off.min():.3f} to {off.max():.3f}), "
      f"time profile median {np.median(off_prof):.3f} "
      f"(range {off_prof.min():.3f} to {off_prof.max():.3f}).\n")
    W("\nTransfer to scenes never used to fit the weights:\n")
    W("| scene | split | full map | time profile |")
    W("|---|---|---|---|")
    for s in checks:
        W(f"| {s} | {'test' if s in TEST_SCENES else 'val'} | "
          f"{checks[s]:.3f} | {checks_prof[s]:.3f} |")
    W(f"\nMean over held-out scenes: full map "
      f"{np.mean(list(checks.values())):.3f}, time profile "
      f"{np.mean(list(checks_prof.values())):.3f}.\n")
    W("\nA high correlation means the same components survive furniture in rooms "
      "the weighting has never seen, which is what licenses one weighting for "
      "every scene. A low one would mean the effect is per-room and the whole "
      "approach is a fit.\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for s, ws in per_scene.items():
        axes[0].plot(ms, ws[:, :n_frames].mean(axis=0), lw=0.9, alpha=0.45, color="0.5")
    axes[0].plot(ms, per_frame, lw=2.6, color="#2962ff", label="fitted, all training scenes")
    axes[0].set_xlabel("post-direct arrival time, ms")
    axes[0].set_ylabel("reliability weight  signal / (signal + furniture)")
    axes[0].set_title("Furniture robustness against arrival time\n"
                      "grey: individual training scenes")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1)
    im = axes[1].imshow(w[:, :n_frames], aspect="auto", cmap="viridis",
                        vmin=0, vmax=1, extent=[0, n_frames * args.window_ms, w.shape[0], 0])
    axes[1].set_xlabel("post-direct arrival time, ms")
    axes[1].set_ylabel("microphone row")
    axes[1].set_title("Per-component weight")
    fig.colorbar(im, ax=axes[1], label="weight")
    fig.tight_layout()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_dir / "furniture_invariance.png", dpi=140)
    plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, weights=w.astype(np.float32),
             profile=prof.astype(np.float32),
             signal=signal.astype(np.float32), noise=noise.astype(np.float32),
             per_frame=per_frame.astype(np.float32),
             config=json.dumps(dict(window_ms=args.window_ms, sample_rate=sr,
                                    scenes=list(per_scene), n_pairs=int(len(P)),
                                    collection=args.collection, floor=args.floor)))
    args.report.write_text("\n".join(out) + "\n"
                           + "\n<!-- " + json.dumps(stamp()) + " -->\n")
    print("\n".join(out))
    print(f"\nwrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
