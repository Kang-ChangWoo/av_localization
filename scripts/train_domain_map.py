#!/usr/bin/env python3
"""Learn the furnished-to-floorplan mapping, and test it on rooms it never saw.

A scalar reliability per feature component does not transfer across rooms: fitted
on eleven training scenes it correlates 0.04 and -0.05 with the same quantity
measured in two of the three test rooms. That failure is specific, though. It
says a *diagonal, linear* correction is room-specific. It does not say the
furnished and wall-only domains are unrelated, and a nonlinear map could still
be room-independent where a per-component scale is not.

This is a better-posed learning problem than the one that already failed here. A
contrastive acoustic encoder trained on this data reached 13.3% validation top-1
and then 2.7% on the test rooms, worse than chance, because it had to learn a
metric space from scratch. Here the supervision is exact and paired: the same
pose is rendered on both meshes, so the target is known for every input, the
output lives in the same 6 by 68 feature space as the input, and the model only
has to learn a correction rather than a representation.

The protocol is the only part that matters for whether the answer is believable.

    train      eleven Replica training rooms
    validate   three validation rooms, used for early stopping only
    test       the three evaluation rooms, touched once at the end

and the test is not the regression loss, which would flatter any smoothing. It
is whether mapping the query before scoring moves the true pose up the candidate
ranking, measured with the same code as everything else.

The identity map is the control. A model that cannot beat "do nothing" has
learned to smooth, and the correct conclusion is that the domain gap is not a
learnable function of the recording alone.

    python scripts/train_domain_map.py --epochs 60
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
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--grid-dir", default="outputs/acoustic_grid_v2")
    p.add_argument("--window-ms", type=float, default=2.0)
    p.add_argument("--n-poses", type=int, default=200, help="per scene and collection")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--residual", action="store_true", default=True,
                   help="predict a correction to the input rather than the output")
    p.add_argument("--eval-poses", type=int, default=100, help="test poses per scene")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "domain_map")
    p.add_argument("--report", type=Path,
                   default=REPO_ROOT / "outputs" / "analysis" / "domain_map.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import torch.nn as nn
    import tqdm

    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, candidate_features, featurise,
        observation_rate, score,
    )
    from track1_core.provenance import stamp

    root = Path(args.dataset_root)
    probe = root / "rir" / args.collections[0] / "raw_scan_open" / TRAIN_SCENES[0] / "pose_00003"
    sr = observation_rate(probe / "rir.npy")
    cfg = GridScoreConfig(feature="envelope", window_ms=args.window_ms,
                          sample_rate_hz=sr,
                          direct_guard_samples=int(round(sr * 2 / 1000)),
                          usable_samples=int(round(sr * 128 / 1000)))

    def load_pairs(scenes, n_poses):
        """(furnished, wall-only) featurised pairs at identical poses."""
        X, Y, meta = [], [], []
        for coll in args.collections:
            for s in scenes:
                pd_ = root / "rir" / coll / "floorplan_closed" / s
                fd = root / "rir" / coll / "raw_scan_open" / s
                if not pd_.is_dir() or not fd.is_dir():
                    continue
                names = sorted(n for n in (set(os.listdir(pd_)) & set(os.listdir(fd)))
                               if n.startswith("pose_"))
                picks = np.linspace(0, len(names) - 1,
                                    min(n_poses, len(names))).astype(int)
                for i in picks:
                    a = pd_ / names[int(i)] / "rir.npy"
                    b = fd / names[int(i)] / "rir.npy"
                    if not a.exists() or not b.exists():
                        continue
                    Y.append(featurise(band_energy(np.load(a), cfg, observation_rate(a)), cfg))
                    X.append(featurise(band_energy(np.load(b), cfg, observation_rate(b)), cfg))
                    meta.append((coll, s, int(names[int(i)].split("_")[1])))
        return np.asarray(X, np.float32), np.asarray(Y, np.float32), meta

    print("[data] loading pairs")
    Xtr, Ytr, _ = load_pairs(TRAIN_SCENES, args.n_poses)
    Xva, Yva, _ = load_pairs(VAL_SCENES, args.n_poses // 2)
    print(f"[data] train {Xtr.shape}, val {Xva.shape}")
    rows, width = Xtr.shape[1], Xtr.shape[2]

    # Features are per-row shape vectors summing to one and spanning orders of
    # magnitude, so the network sees them in log space and the loss is taken
    # there too. An L1 in log space is scale-free per component, which matches
    # the scoring distance better than an L1 on the raw values would.
    def to_log(a):
        return np.log(np.clip(a, 1e-12, None))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.tensor(to_log(Xtr)); Ytr_t = torch.tensor(to_log(Ytr))
    Xva_t = torch.tensor(to_log(Xva)).to(dev); Yva_t = torch.tensor(to_log(Yva)).to(dev)
    mu, sd = Xtr_t.mean(), Xtr_t.std()

    class Map(nn.Module):
        """A small per-row temporal convolution with a channel-mixing stage.

        Deliberately small. The brief rules out a large learned audio model as a
        primary direction, and the question here is whether *any* room-independent
        mapping exists, which a bigger model would not answer more honestly.
        """

        def __init__(self, rows, width, w, depth):
            super().__init__()
            layers = [nn.Conv1d(rows, w, 5, padding=2), nn.GELU()]
            for _ in range(depth - 1):
                layers += [nn.Conv1d(w, w, 5, padding=2), nn.GELU()]
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(w, rows, 1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        def forward(self, x):           # x: (B, rows, width), standardised
            return self.head(self.body(x))

    net = Map(rows, width, args.width, args.depth).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"[model] {n_par} parameters, residual={args.residual}")

    def predict(xlog_t):
        z = (xlog_t - mu.to(xlog_t.device)) / sd.to(xlog_t.device)
        d = net(z)
        return xlog_t + d if args.residual else d * sd.to(xlog_t.device) + mu.to(xlog_t.device)

    # the identity map is the control every epoch is measured against
    id_val = float((Xva_t - Yva_t).abs().mean())
    best, best_state = id_val, None
    hist = []
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i: i + args.batch_size]
            xb, yb = Xtr_t[idx].to(dev), Ytr_t[idx].to(dev)
            loss = (predict(xb) - yb).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float((predict(Xva_t) - Yva_t).abs().mean())
        hist.append((ep, tot / len(perm), v))
        if v < best:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train {tot/len(perm):.4f}  val {v:.4f}  "
                  f"(identity {id_val:.4f})")
    improved = best < id_val
    print(f"[fit] best val L1 {best:.4f} against identity {id_val:.4f} "
          f"({'better' if improved else 'NOT better'})")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # ---- the test that matters: does it move truth up the ranking? -------
    out: list[str] = []
    W = out.append
    W("# A learned furnished-to-floorplan mapping\n")
    W(f"{n_par} parameters, trained on {len(TRAIN_SCENES)} rooms "
      f"({len(Xtr)} pose pairs), early stopped on {len(VAL_SCENES)} rooms, "
      f"evaluated on {len(TEST_SCENES)} rooms never seen.\n")
    W(f"\nValidation L1 in log space: identity {id_val:.4f}, learned {best:.4f}. "
      f"The regression {'does' if improved else 'does not'} beat doing nothing.\n")
    W("\n## The ranking test on the held-out rooms\n")
    W("`raw` scores the furnished recording against the wall-only candidate grid, "
      "which is the current method. `mapped` passes the recording through the "
      "network first. Everything else is identical.\n")
    W("| scene | n | GT rank, raw | GT rank, mapped | recall @1m, raw | "
      "recall @1m, mapped |")
    W("|---|---|---|---|---|---|")

    rows_out = {}
    for scene in TEST_SCENES:
        g = REPO_ROOT / args.grid_dir / f"{scene}.npz"
        if not g.exists():
            continue
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        occ = cv2.imread(str(root / args.collections[0] / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        rr, cc = np.nonzero(mask)
        cand, present = candidate_features(g, rr, cc, mask.shape, cfg)

        raw_rank, map_rank, raw_ok, map_ok = [], [], [], []
        for coll in args.collections:
            poses = np.array([[float(v) for v in l.split()]
                              for l in open(root / coll / scene / "poses.txt") if l.strip()])
            fd = root / "rir" / coll / "raw_scan_open" / scene
            names = sorted(n for n in os.listdir(fd) if n.startswith("pose_"))
            picks = np.linspace(0, len(names) - 1,
                                min(args.eval_poses, len(names))).astype(int)
            for i in tqdm.tqdm(picks, desc=f"{coll}/{scene}", leave=False):
                f = fd / names[int(i)] / "rir.npy"
                pi = int(names[int(i)].split("_")[1])
                if not f.exists() or pi >= len(poses):
                    continue
                obs = featurise(band_energy(np.load(f), cfg, observation_rate(f)), cfg)
                with torch.no_grad():
                    m = predict(torch.tensor(to_log(obs))[None].to(dev))
                obs_m = np.exp(m[0].cpu().numpy())
                gx, gy, _ = pg.pose_metric_to_grid(poses[pi, :3])
                dist = np.hypot(cc - gx, rr - gy) * pg.grid_resolution_m
                gt = int(dist.argmin())
                for feat, rk, okl in ((obs, raw_rank, raw_ok), (obs_m, map_rank, map_ok)):
                    s = score(cand, feat.astype(np.float32), present, cfg)
                    rk.append(int((s > s[gt]).sum()) + 1)
                    okl.append(dist[int(s.argmax())] < 1.0)
        rows_out[scene] = dict(n=len(raw_rank),
                               raw_rank=float(np.median(raw_rank)),
                               map_rank=float(np.median(map_rank)),
                               raw_ok=100 * float(np.mean(raw_ok)),
                               map_ok=100 * float(np.mean(map_ok)))
        r = rows_out[scene]
        W(f"| {scene} | {r['n']} | {r['raw_rank']:.0f} | {r['map_rank']:.0f} | "
          f"{r['raw_ok']:.1f}% | {r['map_ok']:.1f}% |")
    if rows_out:
        tot_n = sum(r["n"] for r in rows_out.values())
        W(f"| **all** | {tot_n} | "
          f"{np.mean([r['raw_rank'] for r in rows_out.values()]):.0f} | "
          f"{np.mean([r['map_rank'] for r in rows_out.values()]):.0f} | "
          f"{np.mean([r['raw_ok'] for r in rows_out.values()]):.1f}% | "
          f"{np.mean([r['map_ok'] for r in rows_out.values()]):.1f}% |")
    W("\nA lower ground-truth rank is better. If the mapped column is not clearly "
      "below the raw one on rooms the network never saw, the mapping has not "
      "learned anything room-independent, and no amount of capacity will change "
      "that without more rooms.\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), rows=rows, width=width,
                    args=dict(vars(args), out_dir=str(args.out_dir),
                              report=str(args.report)),
                    mu=float(mu), sd=float(sd)), args.out_dir / "domain_map.pt")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(out) + "\n\n<!-- "
                           + json.dumps(stamp(), default=str) + " -->\n")
    (args.out_dir / "history.json").write_text(json.dumps(
        dict(history=hist, identity_val=id_val, best_val=best,
             test=rows_out), indent=2))
    print("\n".join(out))
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
