#!/usr/bin/env python3
"""A domain-invariant ranking projection for the acoustic feature.

The furniture gap is the thing that limits the method: the same feature that
finds the true cell 94% of the time on matched geometry finds it 21% of the
time when the query is furnished. A regression from the furnished to the
wall-only domain was tried and learned the average effect of furniture, which
is common to every candidate and cancels in a ranking. This is the other
objective. Every pose in every room has both a furnished and a wall-only
recording, so a projection can be trained to make the furnished recording
closer to its own wall-only twin than to the wall-only recordings of other
poses in the same room, which is a ranking loss and learns discriminative
directions rather than a mean shift. No candidate grid is needed to train it.

Selection is on the validation rooms with the same in-room retrieval; the
final test uses the real candidate grid on the test rooms, so the number that
matters is measured exactly where the pipeline measures it.

    python feasible/run_invariant_projection.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=Path("/root/storage/echoloc_dataset/replica"))
    p.add_argument("--grid-dir", type=Path, default=REPO_ROOT / "outputs" / "acoustic_grid_v2")
    p.add_argument("--collection", default="replica_f")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--per-room", type=int, default=400, help="poses per room used for training")
    p.add_argument("--gpu", default="0")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "outputs" / "metrics" / "proj_feature_cache.npz")
    p.add_argument("--out", type=Path, default=HERE / "results" / "P_invariant_projection.md")
    p.add_argument("--weights", type=Path, default=REPO_ROOT / "outputs" / "metrics" / "acoustic_projection.npz")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import cv2, torch, yaml
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.likelihood.grid_score import (
        GridScoreConfig, band_energy, featurise, observation_rate,
    )
    from track1_core.provenance import stamp

    R = args.dataset_root
    sp = yaml.safe_load((R / args.collection / "split.yaml").read_text())
    sr = 48000
    cfg = GridScoreConfig(feature="stft_band", nfft=256, hop=64, sample_rate_hz=sr,
                          direct_guard_samples=int(round(sr * 2 / 1000)),
                          usable_samples=int(round(sr * 128 / 1000)))

    # ------------------------------------------------------------ features
    def load_room(scene, n):
        P = np.array([[float(v) for v in l.split()]
                      for l in open(R / args.collection / scene / "poses.txt") if l.strip()])
        rd_f = R / "rir" / args.collection / "raw_scan_open" / scene
        rd_c = R / "rir" / args.collection / "floorplan_closed" / scene
        dirs = sorted(d for d in os.listdir(rd_f) if d.startswith("pose_"))
        picks = np.linspace(0, len(dirs) - 1, min(n, len(dirs))).astype(int)
        XF, XC, XY = [], [], []
        for k in picks:
            dn = dirs[int(k)]; i = int(dn.split("_")[1])
            ff, fc = rd_f / dn / "rir.npy", rd_c / dn / "rir.npy"
            if not (ff.exists() and fc.exists()) or i >= len(P):
                continue
            XF.append(featurise(band_energy(np.load(ff), cfg, observation_rate(ff)), cfg))
            XC.append(featurise(band_energy(np.load(fc), cfg, observation_rate(fc)), cfg))
            XY.append(P[i, :2])
        return np.stack(XF).astype(np.float32), np.stack(XC).astype(np.float32), np.array(XY)

    if args.cache.exists():
        z = np.load(args.cache, allow_pickle=True)
        data = {k: z[k].item() for k in ("train", "val", "test")}
        print(f"[cache] {args.cache}")
    else:
        data = {}
        for split, n in (("train", args.per_room), ("val", args.per_room), ("test", 1200)):
            data[split] = {}
            for s in sp[split]:
                xf, xc, xy = load_room(s, n)
                data[split][s] = dict(f=xf, c=xc, xy=xy)
                print(f"[feat] {split}/{s}: {len(xf)} poses, feature {xf.shape[1:]}")
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **{k: np.array(v, dtype=object) for k, v in data.items()})

    D = int(np.prod(next(iter(data["train"].values()))["f"].shape[1:]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def flat(x):
        return torch.tensor(x.reshape(len(x), -1), device=dev)

    # ------------------------------------------------------- retrieval eval
    def in_room_recall(W, split):
        """Furnished query against the wall-only twins of every pose in its own room."""
        hits, tot = 0, 0
        for s, d in data[split].items():
            f, c = flat(d["f"]), flat(d["c"])
            if W is not None:
                f, c = f @ W, c @ W
            dist = torch.cdist(f, c, p=1)                   # (n, n)
            j = dist.argmin(dim=1).cpu().numpy()
            err = np.linalg.norm(d["xy"][j] - d["xy"], axis=1)
            hits += int((err < 1).sum()); tot += len(err)
        return hits / max(tot, 1)

    # --------------------------------------------------------------- train
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.eye(D, args.dim, device=dev) * 0.1
                           + 0.01 * torch.randn(D, args.dim, device=dev))
    opt = torch.optim.Adam([W], lr=args.lr)
    rooms = list(data["train"].items())
    best_val, best_W, hist = -1.0, None, []
    base_val = in_room_recall(None, "val")
    print(f"[val] identity projection: in-room recall@1m {100*base_val:.1f}%")
    for ep in range(args.epochs):
        np.random.shuffle(rooms); tot = 0.0
        for s, d in rooms:
            f, c = flat(d["f"]) @ W, flat(d["c"]) @ W
            # in-room InfoNCE on negative L1 distances: the furnished recording
            # must be nearer its own wall-only twin than any other pose's
            logits = -torch.cdist(f, c, p=1) / args.temperature
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(f), device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        v = in_room_recall(W.detach(), "val")
        hist.append((ep, tot / len(rooms), v))
        print(f"[ep {ep:2d}] loss {tot/len(rooms):.3f}  val in-room recall@1m {100*v:.1f}%")
        if v > best_val:
            best_val, best_W = v, W.detach().clone()

    # --------------------------------------------------- test on the grid
    lines = ["# P. Domain-invariant ranking projection\n",
             f"Linear projection {D} -> {args.dim}, trained on the {len(rooms)} training "
             f"rooms with an in-room InfoNCE over (furnished query, wall-only twin) pairs, "
             f"selected on the validation rooms. In-room retrieval recall@1m on validation: "
             f"identity {100*base_val:.1f}%, projected {100*best_val:.1f}%.\n",
             "| test room | queries | acoustic alone @1m, identity | projected | median GT rank, identity | projected |",
             "|---|---|---|---|---|---|"]
    js = dict(dim=args.dim, val_identity=base_val, val_projected=best_val, test={})
    Wn = best_W.cpu().numpy()
    tot_e = {"id": [], "pr": []}; tot_r = {"id": [], "pr": []}
    for s, d in data["test"].items():
        b = np.load(args.grid_dir / f"{s}.npz", allow_pickle=False)
        idx = b["index"]
        Fc = np.stack([featurise(band_energy(r, cfg), cfg) for r in b["rir"]]).astype(np.float32)
        del b
        desdf = np.load(R / "desdf" / s / "desdf.npy", allow_pickle=True).item()
        occ = cv2.imread(str(R / args.collection / s / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape); res = pg.grid_resolution_m
        rows, cols = np.nonzero(valid_pose_mask(occ, pg))
        lut = -np.ones(occ.shape[:2], np.int64); lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
        pick = lut[rows, cols]; pres = pick >= 0
        C = np.zeros((len(rows), Fc.shape[1] * Fc.shape[2]), np.float32)
        C[pres] = Fc[pick[pres]].reshape(int(pres.sum()), -1)
        Ct = torch.tensor(C, device=dev); Cp = Ct @ best_W
        Q = flat(d["f"]); Qp = Q @ best_W
        xy = d["xy"]
        e = {"id": [], "pr": []}; rk = {"id": [], "pr": []}
        for i in range(len(Q)):
            gx, gy, _ = pg.pose_metric_to_grid(np.array([xy[i, 0], xy[i, 1], 0.0]))
            dist = np.hypot(cols - gx, rows - gy) * res
            gt = int(dist.argmin())
            for name, q, c in (("id", Q[i:i+1], Ct), ("pr", Qp[i:i+1], Cp)):
                sc = -torch.cdist(q, c, p=1)[0].cpu().numpy()
                sc = np.where(pres, sc, sc.min() - 1)
                e[name].append(float(dist[int(sc.argmax())])); rk[name].append(int((sc > sc[gt]).sum()))
        for k in e:
            tot_e[k] += e[k]; tot_r[k] += rk[k]
        lines.append(f"| {s} | {len(Q)} | {100*np.mean(np.array(e['id'])<1):.1f}% | "
                     f"{100*np.mean(np.array(e['pr'])<1):.1f}% | {np.median(rk['id']):.0f} | {np.median(rk['pr']):.0f} |")
        js["test"][s] = dict(n=len(Q), r1_identity=float(np.mean(np.array(e['id']) < 1)),
                             r1_projected=float(np.mean(np.array(e['pr']) < 1)),
                             rank_identity=float(np.median(rk['id'])),
                             rank_projected=float(np.median(rk['pr'])))
        del Fc, C, Ct, Cp
    ei, ep_ = np.array(tot_e["id"]), np.array(tot_e["pr"])
    lines.append(f"| **all** | {len(ei)} | {100*(ei<1).mean():.1f}% | {100*(ep_<1).mean():.1f}% | "
                 f"{np.median(tot_r['id']):.0f} | {np.median(tot_r['pr']):.0f} |")
    js["test"]["all"] = dict(n=len(ei), r1_identity=float((ei < 1).mean()),
                             r1_projected=float((ep_ < 1).mean()))
    lines.append("\nAcoustic score alone over the whole candidate grid, furnished query, "
                 "test rooms never seen in training or selection. If the projected column "
                 "is not above the identity column here, the feature cannot close the gap "
                 "by a linear change of basis, whatever it does in-room.\n")
    np.savez(args.weights, W=Wn, dim=args.dim, feature_shape=np.array(next(iter(data["train"].values()))["f"].shape[1:]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (args.out.parent / "P_invariant_projection.json").write_text(
        json.dumps(dict(results=js, history=hist, provenance=stamp()), indent=2, default=str))
    print("\n".join(lines)); print(f"\nwrote {args.out} and {args.weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
