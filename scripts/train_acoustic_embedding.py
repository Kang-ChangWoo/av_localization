#!/usr/bin/env python3
"""Learn an embedding where a floorplan response and a real recording of the same
pose agree, so the geometry-domain gap stops dominating the acoustic score.

Every training-free score in this project is bounded by one measurement. With the
recording rendered on the floorplan's own extrusion, the low band reaches 98-100%
recall at 1 m. With the recording taken on the real Replica scan, the same
feature and the same candidates reach 4-26%. Nothing about the comparison rule
closes a gap that large; the two signals genuinely differ, because furniture is
in one and not the other.

That gap is learnable here because the supervision already exists. The dataset
renders both conditions at the same poses, so for 6,600 training poses there is
an exact correspondence between the response a floorplan can produce and the
response the real room produces. Training a shared encoder to map the two onto
each other is ordinary metric learning, not a new modelling assumption.

    f(scan @ p)  close to  f(floorplan @ p),  far from  f(floorplan @ q != p)

At inference the candidate grid is encoded once per scene and the observation
once per query, and the score is a dot product, which drops into exactly the
slot the L1 currently occupies.

Only the 11 training scenes are used. The three evaluation scenes are never
seen, so every number reported elsewhere in this project stays comparable.

    python scripts/train_acoustic_embedding.py --epochs 60
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--bands", type=float, nargs="+", default=[0, 500, 1500, 4000],
                   help="band edges in Hz; the sweep decides whether to narrow this")
    p.add_argument("--first-frame", type=int, default=0,
                   help="0 keeps the whole window; 15 keeps only arrivals past "
                        "30 ms, which is where non-line-of-sight paths live")
    p.add_argument("--nfft", type=int, default=64)
    p.add_argument("--hop", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def build_encoder(in_rows: int, n_frames: int, width: int, embed_dim: int):
    """A deliberately small 1D convolutional encoder over the time axis.

    Eleven scenes and 6,600 poses is not much, and the failure mode to avoid is
    memorising rooms rather than learning what furniture does to a response. The
    rows (channel x band) are the input channels and the network is convolutional
    in time only, so a delayed arrival looks the same wherever it lands.
    """
    import torch.nn as nn

    def block(cin, cout, stride=2):
        return nn.Sequential(nn.Conv1d(cin, cout, 5, stride=stride, padding=2),
                             nn.BatchNorm1d(cout), nn.GELU())

    return nn.Sequential(
        block(in_rows, width), block(width, width * 2), block(width * 2, width * 2),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(width * 2, embed_dim))


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)

    import torch
    import torch.nn.functional as F
    import tqdm
    import yaml

    from scripts.margin_gated_fusion import band_envelope
    from track1_core.likelihood.events import EventMatchConfig

    ecfg = EventMatchConfig()
    root = Path(args.dataset_root)
    edges = args.bands
    bands = list(zip(edges[:-1], edges[1:]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def describe(path: Path) -> np.ndarray:
        e = band_envelope(np.load(path), ecfg.direct_guard_samples, 1024,
                          args.nfft, args.hop, ecfg.sample_rate_hz, bands)
        e = e[:, args.first_frame:]
        # per-row shape plus overall level, the same normalisation the
        # training-free score uses, so the two are directly comparable
        return (e / np.clip(e.sum(axis=-1, keepdims=True), 1e-20, None)).astype(np.float32)

    split = yaml.safe_load(open(root / args.collections[0] / "split.yaml"))
    pairs = {k: [] for k in ("train", "val")}
    for k in ("train", "val"):
        for coll in args.collections:
            for scene in split[k]:
                fp = root / "rir" / coll / "floorplan_closed" / scene
                sc = root / "rir" / coll / "raw_scan_open" / scene
                if not fp.is_dir() or not sc.is_dir():
                    continue
                for d in sorted(os.listdir(fp)):
                    a, b = fp / d / "rir.npy", sc / d / "rir.npy"
                    if a.exists() and b.exists():
                        pairs[k].append((a, b, scene))
    print(f"[data] train {len(pairs['train'])} pairs over "
          f"{len({s for _,_,s in pairs['train']})} scenes, "
          f"val {len(pairs['val'])} pairs")
    if not pairs["train"]:
        print("no paired data")
        return 1

    def load_split(key):
        X = np.stack([describe(a) for a, _, _ in tqdm.tqdm(pairs[key], desc=f"load {key} model")])
        Y = np.stack([describe(b) for _, b, _ in tqdm.tqdm(pairs[key], desc=f"load {key} obs")])
        return torch.tensor(X), torch.tensor(Y)

    Xtr, Ytr = load_split("train")
    Xva, Yva = load_split("val")
    rows, frames = Xtr.shape[1], Xtr.shape[2]
    print(f"[data] each sample is {rows} rows x {frames} frames "
          f"({len(bands)} bands, first frame {args.first_frame})")

    net = build_encoder(rows, frames, args.width, args.embed_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = Xtr.shape[0]
    print(f"[model] {sum(p.numel() for p in net.parameters())/1e3:.1f}k parameters")

    def evaluate(X, Y, chunk=512):
        """Retrieval accuracy: for each real recording, is the floorplan response
        of its own pose the nearest of all candidates in the split?"""
        net.eval()
        with torch.no_grad():
            ex = torch.cat([F.normalize(net(X[i:i+chunk].to(device)), dim=1).cpu()
                            for i in range(0, X.shape[0], chunk)])
            ey = torch.cat([F.normalize(net(Y[i:i+chunk].to(device)), dim=1).cpu()
                            for i in range(0, Y.shape[0], chunk)])
        sim = ey @ ex.T
        rank = (sim > sim.diag()[:, None]).sum(1)
        return dict(top1=float((rank == 0).float().mean()),
                    top5=float((rank < 5).float().mean()),
                    median_rank=float(rank.float().median()) + 1, n=int(sim.shape[0]))

    best, hist = -1.0, []
    out = args.out or REPO_ROOT / "outputs" / "acoustic_embed"
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n - 1, args.batch_size):
            b = perm[i:i + args.batch_size]
            if b.numel() < 8:
                continue
            zx = F.normalize(net(Xtr[b].to(device)), dim=1)
            zy = F.normalize(net(Ytr[b].to(device)), dim=1)
            logits = zy @ zx.T / args.temperature
            tgt = torch.arange(b.numel(), device=device)
            # symmetric InfoNCE: the recording must find its own floorplan
            # response and the floorplan response must find its own recording
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * b.numel()
        sched.step()
        va = evaluate(Xva, Yva)
        hist.append(dict(epoch=ep, loss=tot / n, **va))
        if va["top1"] > best:
            best = va["top1"]
            torch.save(dict(state=net.state_dict(), rows=rows, frames=frames,
                            bands=bands, first_frame=args.first_frame,
                            width=args.width, embed_dim=args.embed_dim,
                            nfft=args.nfft, hop=args.hop), out / "encoder.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot/n:.4f}   val top1 {100*va['top1']:5.1f}%  "
                  f"top5 {100*va['top5']:5.1f}%  median rank {va['median_rank']:.0f}"
                  f"{'   *' if va['top1'] >= best else ''}")

    print(f"\nbest val top-1 {100*best:.1f}% over {evaluate(Xva, Yva)['n']} val pairs")
    print(f"chance would be {100/evaluate(Xva, Yva)['n']:.2f}%")
    (out / "history.json").write_text(json.dumps(dict(
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        best_val_top1=best, history=hist), indent=2))
    print(f"wrote {out}/encoder.pt and history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
