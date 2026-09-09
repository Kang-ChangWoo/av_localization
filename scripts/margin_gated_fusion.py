#!/usr/bin/env python3
"""Re-rank only where vision is uncertain, and report it in F3Loc's metrics.

The stratified evaluation showed the acoustic term is not a uniform improvement.
Split by the gap between the best visual pose and its strongest competitor:

    low margin (ambiguous)   vision 20.2%  fused 26.2%   +6.0
    high margin (confident)  vision 70.7%  fused 65.8%   -4.8

Averaged together those nearly cancel, which is exactly what the +2.0 overall
gain was. So the fix is not a better acoustic feature -- it is not applying the
acoustic feature when vision has already decided. The margin is computable at
inference time, so the gate is a legitimate part of the method rather than an
oracle.

Two gate shapes are compared against always-on and never-on:

  hard    re-rank only when margin < tau
  soft    weight the acoustic term by sigmoid(-(margin - tau)/beta), so the
          shortlist is reordered gently near the threshold instead of switching

Sweeping tau over quantiles of the margin distribution shows whether a single
operating point exists that beats vision everywhere, or whether the gain is an
artefact of choosing the threshold after seeing the answer.

    python scripts/margin_gated_fusion.py --models echoloc_mono_fg:mono
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--collections", nargs="+", default=["replica_f", "replica_g"])
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--models", nargs="+",
                   default=["echoloc_mono_fg:mono", "echoloc_mv_fg:mv", "echoloc_comp_fg:comp"])
    p.add_argument("--condition", default="raw_scan_open")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--mode-sep-m", type=float, default=1.5)
    p.add_argument("--window-ms", type=float, default=0.125)
    p.add_argument("--guard-samples", type=int, default=16)
    p.add_argument("--usable-samples", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--feature", default="chan_shape", choices=["chan_shape", "band"],
                   help="broadband energy envelope, or the same split into "
                        "frequency bands")
    p.add_argument("--nfft", type=int, default=64)
    p.add_argument("--hop", type=int, default=16)
    p.add_argument("--bands", nargs="+", default=None,
                   help="band edges in Hz, e.g. '0 500 1500 4000'. One band "
                        "('0 4000') is the control that isolates the coarser "
                        "time resolution from the frequency split.")
    p.add_argument("--n-poses", type=int, default=300)
    p.add_argument("--holdout", default="replica_g",
                   help="collection kept out when the threshold is chosen")
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# Bands, in Hz. 8 kHz sampling puts Nyquist at 4 kHz. The split is by
# wavelength against furniture size: 343/500 = 69 cm diffracts around a chair
# and carries wall geometry, 343/3000 = 11 cm scatters off it and carries the
# clutter the floorplan does not have.
STFT_BANDS = [(0, 500), (500, 1500), (1500, 4000)]


def featurize(env: np.ndarray) -> np.ndarray:
    """``chan_shape``: per-channel temporal shape, plus the energy split across
    channels, which is what carries direction."""
    per = env / env.sum(axis=-1, keepdims=True).clip(1e-12)
    total = env.sum(axis=(-1, -2))
    split = env.sum(axis=-1) / np.clip(total[..., None], 1e-12, None)
    return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)


def band_envelope(rir: np.ndarray, guard: int, usable: int, nfft: int, hop: int,
                  sample_rate: int, bands: list) -> np.ndarray:
    """Energy per (channel, band, frame): ``chan_shape`` with a frequency axis.

    The broadband envelope sums over frequency, so a wall return and the
    furniture scatter arriving in the same 0.125 ms window are indistinguishable.
    Splitting by band keeps them apart wherever they differ in spectrum, which
    is the only axis along which the model's error is known to be structured.
    """
    peak = int(np.abs(rir).max(axis=0).argmax())
    seg = rir[:, peak + guard: peak + guard + usable]
    if seg.shape[1] < usable:
        seg = np.pad(seg, ((0, 0), (0, usable - seg.shape[1])))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    sel = [(freqs >= lo) & (freqs < hi) for lo, hi in bands]
    w = np.hanning(nfft)
    frames = 1 + (usable - nfft) // hop
    out = np.empty((seg.shape[0], len(bands), frames), dtype=np.float32)
    for t in range(frames):
        p = np.abs(np.fft.rfft(seg[:, t * hop: t * hop + nfft] * w, axis=-1)) ** 2
        for b, s in enumerate(sel):
            out[:, b, t] = p[:, s].sum(-1)
    return out.reshape(seg.shape[0] * len(bands), frames)


# The band description is laid out as (channel x band, frame), i.e. the same
# shape contract as the broadband envelope with the rows split by frequency, so
# the identical normalisation applies: temporal shape within each row, plus the
# energy split across rows, with the overall render gain cancelling either way.
featurize_band = featurize


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT))
    sys.stdout.reconfigure(line_buffering=True)

    import cv2
    import torch
    import tqdm

    from scripts.acoustic_grid_probe import envelope
    from track1_core.floorplan import PoseGrid, valid_pose_mask
    from track1_core.models import CompDepthModule, MVDepthModule, MonoDepthModule
    from utils.data_utils import GridSeqDataset
    from utils.localization_utils import get_ray_from_depth, localize

    root = Path(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 3
    scenes = args.scenes or sorted(os.listdir(root / "desdf"))
    win = max(1, int(round(args.sample_rate * args.window_ms / 1000.0)))
    CLS = {"mono": MonoDepthModule, "mv": MVDepthModule, "comp": CompDepthModule}

    if args.feature == "chan_shape":
        def describe(rir):
            return envelope(rir, args.guard_samples, args.usable_samples, win)
        featurize_fn = featurize
    else:
        if args.bands:
            e = [float(x) for x in args.bands]
            bands = list(zip(e[:-1], e[1:]))
        else:
            bands = STFT_BANDS

        def describe(rir):
            return band_envelope(rir, args.guard_samples, args.usable_samples,
                                 args.nfft, args.hop, args.sample_rate, bands)
        featurize_fn = featurize_band
        print(f"[gate] bands={bands}  nfft={args.nfft} hop={args.hop} "
              f"({1000*args.hop/args.sample_rate:.2f} ms per frame)")
    print(f"[gate] feature={args.feature}")

    # geometry and acoustic candidates are per (collection, scene) because the
    # floorplan raster differs between the two collections
    geo = {}
    for coll in args.collections:
        for scene in scenes:
            g = REPO_ROOT / "outputs" / "acoustic_grid" / f"{scene}.npz"
            if not g.exists() or not (root / coll / scene / "map.png").exists():
                continue
            desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            desdf["desdf"][desdf["desdf"] > 10] = 10
            occ = cv2.imread(str(root / coll / scene / "map.png"))[:, :, 0]
            pg = PoseGrid.from_desdf(desdf, occ.shape)
            blob = np.load(g)
            cand, index = blob["rir"], blob["index"]
            env = np.stack([describe(c) for c in cand])
            env /= env.sum(axis=(1, 2), keepdims=True).clip(1e-20)
            acoustic = np.zeros((pg.height, pg.width, env.shape[1], env.shape[2]),
                                dtype=np.float32)
            acoustic[index[:, 0], index[:, 1]] = env
            geo[(coll, scene)] = dict(
                pg=pg, mask=valid_pose_mask(occ, pg),
                desdf_t=torch.tensor(desdf["desdf"], device=device),
                acoustic=featurize_fn(acoustic),
                poses=np.array([[float(v) for v in l.split()]
                                for l in open(root / coll / scene / "poses.txt") if l.strip()]))
    print(f"[gate] {len(geo)} (collection, scene) pairs, K={args.topk}")

    results = {}
    for spec in args.models:
        run, net = spec.split(":")
        ckpt = REPO_ROOT / "outputs" / run / f"{net}.ckpt"
        if not ckpt.exists():
            print(f"[gate] {spec}: no checkpoint, skipping")
            continue
        kw = dict(mono_ckpt=None, mv_ckpt=None) if net == "comp" else {}
        model = CLS[net].load_from_checkpoint(str(ckpt), **kw).to(device).eval()

        recs = []
        for (coll, scene), G in geo.items():
            pg, mask, poses = G["pg"], G["mask"], G["poses"]
            rir_dir = root / "rir" / coll / args.condition / scene
            if not rir_dir.exists():
                continue
            dsdir = str(root / coll)
            dataset = GridSeqDataset(dsdir, [scene], L=L, depth_dir=dsdir,
                                     depth_suffix="depth40" if net == "mono" else "depth160")
            picks = np.linspace(0, len(dataset) - 1,
                                min(args.n_poses, len(dataset))).astype(int)
            for chunk in tqdm.tqdm(picks, desc=f"{run} {coll}/{scene}", leave=False):
                pose_idx = int(chunk) * (L + 1) + L
                f = rir_dir / f"pose_{pose_idx:05d}" / "rir.npy"
                if not f.exists():
                    continue
                data = dataset[int(chunk)]
                with torch.no_grad():
                    if net == "mono":
                        pred, _, _ = model.encoder(
                            torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)
                    else:
                        b = {k: torch.tensor(data[k], device=device).unsqueeze(0)
                             for k in ("ref_img", "src_img", "ref_pose", "src_pose")}
                        b["ref_mask"] = b["src_mask"] = None
                        pred = (model.net(b)["d"] if net == "mv"
                                else model.comp_d_net(b)["d_comp"])
                rays = torch.tensor(get_ray_from_depth(pred.squeeze(0).float().cpu().numpy()),
                                    device=device, dtype=torch.float32)
                prob_vol, _, _, _ = localize(G["desdf_t"], rays)
                vol = np.array(prob_vol, dtype=np.float64)
                vol[~mask] = 0.0

                gx, gy, gyaw = pg.pose_metric_to_grid(poses[pose_idx, :3])
                flat = vol.reshape(-1)
                order = np.argsort(-flat)[: args.topk]
                r_, c_, o_ = np.unravel_index(order, vol.shape)

                # margin against the strongest spatially separated competitor
                far = (np.hypot(c_ - c_[0], r_ - r_[0]) * pg.grid_resolution_m
                       > args.mode_sep_m)
                second = flat[order][far].max() if far.any() else 0.0
                margin = float(np.log((flat[order[0]] + 1e-300) / (second + 1e-300)))

                obs = describe(np.load(f))
                obs_f = featurize_fn(obs / max(obs.sum(), 1e-20))
                cand = G["acoustic"][r_, c_]
                nw = min(cand.shape[2], obs_f.shape[1])
                ac = -np.abs(cand[:, :, :nw] - obs_f[None, :, :nw]).sum(axis=(1, 2))

                err = np.hypot(c_ - gx, r_ - gy) * pg.grid_resolution_m
                yaw = np.array([PoseGrid.orientation_error_deg(pg.bin_to_yaw(o), gyaw)
                                for o in o_])
                # visual score on the shortlist, log domain, for the soft gate
                vlog = np.log(np.clip(flat[order], 1e-300, None))
                recs.append(dict(coll=coll, scene=scene, margin=margin, err=err, yaw=yaw,
                                 ac=ac, vlog=vlog))

        if not recs:
            continue
        margins = np.array([r["margin"] for r in recs])

        def evaluate(pick_fn, subset=None):
            g = recs if subset is None else subset
            e = np.array([r["err"][pick_fn(r)] for r in g])
            y = np.array([r["yaw"][pick_fn(r)] for r in g])
            return dict(n=int(len(e)), r01=float((e < 0.1).mean()),
                        r05=float((e < 0.5).mean()), r1=float((e < 1.0).mean()),
                        r1_30=float(((e < 1.0) & (y < 30)).mean()),
                        median_m=float(np.median(e)))

        rows = {"vision only": evaluate(lambda r: 0),
                "acoustic always on": evaluate(lambda r: int(r["ac"].argmax()))}

        # --- hard gate, swept over the margin distribution -------------------
        sweep = []
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            tau = float(np.quantile(margins, q))
            m = evaluate(lambda r, t=tau: int(r["ac"].argmax()) if r["margin"] < t else 0)
            sweep.append(dict(quantile=q, tau=tau, **m))
            rows[f"hard gate q={q:.1f}"] = m

        # --- soft gate: acoustic weight fades in as the margin shrinks -------
        for beta in (0.5, 1.0, 2.0):
            tau = float(np.quantile(margins, 0.5))

            def pick(r, t=tau, b=beta):
                w = 1.0 / (1.0 + np.exp((r["margin"] - t) / b))
                z = (r["vlog"] - r["vlog"].max())
                a = (r["ac"] - r["ac"].max()) / max(np.abs(r["ac"]).max(), 1e-12)
                return int((z + w * a * np.abs(z).max()).argmax())

            rows[f"soft gate beta={beta}"] = evaluate(pick)

        # --- per scene and per collection, so an average cannot hide a gain
        # that only one room produces -------------------------------------
        tau_mid = float(np.quantile(margins, 0.6))
        gate = lambda r, t=tau_mid: int(r["ac"].argmax()) if r["margin"] < t else 0
        breakdown = []
        for coll in sorted({r["coll"] for r in recs}):
            for scene in sorted({r["scene"] for r in recs if r["coll"] == coll}):
                sub = [r for r in recs if r["coll"] == coll and r["scene"] == scene]
                v = evaluate(lambda r: 0, sub)
                a = evaluate(gate, sub)
                breakdown.append(dict(collection=coll, scene=scene, n=v["n"],
                                      vision_r1=v["r1"], gated_r1=a["r1"],
                                      gain=a["r1"] - v["r1"],
                                      vision_r05=v["r05"], gated_r05=a["r05"]))

        # --- honest check: choose tau on one collection, report on the other -
        if args.holdout in {r["coll"] for r in recs}:
            fit = [r for r in recs if r["coll"] != args.holdout]
            hold = [r for r in recs if r["coll"] == args.holdout]
            if fit and hold:
                best_tau, best_r1 = None, -1.0
                for q in np.linspace(0.05, 1.0, 20):
                    t = float(np.quantile([r["margin"] for r in fit], q))
                    e = np.array([r["err"][int(r["ac"].argmax()) if r["margin"] < t else 0]
                                  for r in fit])
                    if (e < 1.0).mean() > best_r1:
                        best_r1, best_tau = float((e < 1.0).mean()), t
                he = np.array([r["err"][int(r["ac"].argmax()) if r["margin"] < best_tau else 0]
                               for r in hold])
                hy = np.array([r["yaw"][int(r["ac"].argmax()) if r["margin"] < best_tau else 0]
                               for r in hold])
                ve = np.array([r["err"][0] for r in hold])
                vy = np.array([r["yaw"][0] for r in hold])
                rows[f"[holdout {args.holdout}] vision"] = dict(
                    n=int(len(ve)), r01=float((ve < .1).mean()), r05=float((ve < .5).mean()),
                    r1=float((ve < 1).mean()), r1_30=float(((ve < 1) & (vy < 30)).mean()),
                    median_m=float(np.median(ve)))
                rows[f"[holdout {args.holdout}] gated tau={best_tau:.2f}"] = dict(
                    n=int(len(he)), r01=float((he < .1).mean()), r05=float((he < .5).mean()),
                    r1=float((he < 1).mean()), r1_30=float(((he < 1) & (hy < 30)).mean()),
                    median_m=float(np.median(he)))

        results[spec] = dict(rows=rows, sweep=sweep, breakdown=breakdown,
                             margin_quantiles={str(q): float(np.quantile(margins, q))
                                               for q in (0.1, 0.25, 0.5, 0.75, 0.9)})

        base = rows["vision only"]["r1"]
        print(f"\n=== {net}   ({rows['vision only']['n']} poses)")
        print(f"{'variant':36s} {'0.1m':>7s} {'0.5m':>7s} {'1m':>7s} "
              f"{'1m/30d':>8s} {'vs vision':>10s}")
        for k, m in rows.items():
            d = "" if k == "vision only" else f"{100*(m['r1']-base):+9.1f}"
            print(f"{k:36s} {100*m['r01']:6.1f}% {100*m['r05']:6.1f}% "
                  f"{100*m['r1']:6.1f}% {100*m['r1_30']:7.1f}% {d:>10s}")

        print(f"\n--- {net}: per scene, gate at q=0.6 "
              f"(does every room gain, or only one?)")
        print(f"{'collection':12s} {'scene':18s} {'n':>4s} {'vision 1m':>10s} "
              f"{'gated 1m':>9s} {'gain':>7s}")
        for b in breakdown:
            print(f"{b['collection']:12s} {b['scene']:18s} {b['n']:4d} "
                  f"{100*b['vision_r1']:9.1f}% {100*b['gated_r1']:8.1f}% "
                  f"{100*b['gain']:+6.1f}")

    out = args.out or REPO_ROOT / "outputs" / "metrics" / "margin_gated_fusion.json"
    out.write_text(json.dumps({"topk": args.topk, "condition": args.condition,
                               "models": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
