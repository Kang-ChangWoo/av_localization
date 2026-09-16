#!/usr/bin/env python3
"""Stage 1c: split each visual backbone into the part that would stay and the
part the acoustic branch could replace, and time both.

The question is whether the network after the image encoder can be dropped
once sound is available. That needs three numbers per backbone: what the
frozen encoder costs, what the trained head after it costs (the conv and
attention that turn features into rays), and what the non-learned floorplan
match costs. Timed with warm-up and medians, parameters counted from the
checkpoint by module, written to ``data/metrics/backbone_<tag>.json``.

One backbone per process, because F3Loc and DisCo each vendor a module called
``utils.localization_utils`` and they cannot share a Python path.

    /opt/conda/envs/unloc/bin/python src/measure_backbones.py --backbone unloc --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]          # analysis/exp1_computational_resource
REPO_ROOT = HERE.parents[1]                          # av_localization
UNLOC_ROOT = REPO_ROOT.parent / "UnLoc"
DISCO_ROOT = REPO_ROOT.parent / "DisCo-FLoc"
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["unloc", "f3loc_mono", "disco_rrp"])
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset/replica")
    p.add_argument("--collection", default="replica_g")
    p.add_argument("--scenes", nargs="+", default=["office_4", "apartment_2", "frl_apartment_5"])
    p.add_argument("--n-queries", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.stdout.reconfigure(line_buffering=True)
    import cv2
    import torch
    from track1_core.floorplan import PoseGrid
    from track1_core.floorplan import valid_pose_mask

    device = "cuda"
    sync = torch.cuda.synchronize
    F_W = 1 / (2 * np.tan(np.deg2rad(106.2602) / 2))

    # ---- each backbone exposes: prep(img) -> x, encode(x) -> feats,
    #      head(x, feats) -> prediction, rays(pred), localize(dt, rays) ------
    if args.backbone == "unloc":
        sys.path.insert(0, str(UNLOC_ROOT))
        from modules.depth_net_pl import UnLocDepthModule
        from utils.localization_utils import get_ray_from_depth_uncertainty, localize_uncertainty
        ck = UNLOC_ROOT / "tb_logs/my_model/version_1/checkpoints/epoch=19-step=1040.ckpt"
        cwd = os.getcwd(); os.chdir(UNLOC_ROOT)
        try:
            net = UnLocDepthModule.load_from_checkpoint(checkpoint_path=str(ck), strict=False).to(device).eval()
        finally:
            os.chdir(cwd)
        label = "UnLoc"
        encoder_prefixes = ("encoder.depth_feature.da.pretrained",)
        unused_prefixes = ("encoder.depth_feature.da.depth_head",)

        def prep(img):
            x = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            return torch.tensor(np.transpose(x, (2, 0, 1))[None], dtype=torch.float32, device=device)

        def encode(x):
            with torch.no_grad():
                return net.encoder.depth_feature._encode_with_da(x)

        def full(x):
            with torch.no_grad():
                loc, scl = net.encoder(x, None)[:2]
            return loc.squeeze(0).cpu().numpy(), scl.squeeze(0).cpu().numpy()

        def rays(pred):
            return get_ray_from_depth_uncertainty(*pred)

        def localize(dt, r):
            pr, ps = r
            _, pd_, orn, _ = localize_uncertainty(dt, torch.tensor(pr, device=device),
                                                  torch.tensor(ps, device=device),
                                                  return_np=False, orn_slice=36)
            return pd_

    elif args.backbone == "disco_rrp":
        sys.path.insert(0, str(DISCO_ROOT)); sys.path.insert(0, str(DISCO_ROOT / "eval"))
        import torchvision.transforms as T
        cwd = os.getcwd(); os.chdir(DISCO_ROOT)
        from utils.localization_utils import get_ray_from_depth, localize as _localize
        from training.RRP_lightning_module import RRPLightningModule
        ck = DISCO_ROOT / "checkpoints" / "RRP_gibson_f_best.ckpt"
        rrp = RRPLightningModule.load_from_checkpoint(str(ck), map_location=device).to(device).eval()
        os.chdir(cwd)
        label = "DisCo-FLoc RRP"
        encoder_prefixes = ("model.dptv2_encoder.pretrained",)
        unused_prefixes = ()
        tf = T.Compose([T.ToTensor(), T.Resize((256, 256), antialias=True),
                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        enc = rrp.model.dptv2_encoder

        def prep(img):
            return tf(img[:, :, ::-1].copy()).unsqueeze(0).to(device)

        def encode(x):
            import math
            import torch.nn.functional as F
            B, C, H, W = x.shape
            th, tw = int(math.ceil(H / 14) * 14), int(math.ceil(W / 14) * 14)
            xp = F.pad(x, (0, tw - W, 0, th - H))
            with torch.no_grad():
                return enc.pretrained.get_intermediate_layers(xp, [enc.intermediate_layer_idx],
                                                              return_class_token=True)

        def full(x):
            with torch.no_grad():
                ft = rrp("encode", obs_img=x)
                return rrp("decoder_inference", depth_cond=ft).squeeze(0).cpu().numpy()

        def rays(pred):
            return torch.tensor(get_ray_from_depth(pred, V=11, F_W=F_W), device=device, dtype=torch.float32)

        def localize(dt, r):
            _, pd_, orn, _ = _localize(dt, r, return_np=False)
            return pd_

    else:
        sys.path.insert(0, str(REPO_ROOT))
        from track1_core._vendor import ensure_on_path
        ensure_on_path()
        from utils.localization_utils import get_ray_from_depth, localize as _localize
        from track1_core.models import MonoDepthModule
        ck = REPO_ROOT / "outputs/echoloc_mono_fg/mono.ckpt"
        net = MonoDepthModule.load_from_checkpoint(str(ck)).to(device).eval()
        label = "F3Loc mono"
        encoder_prefixes = ("encoder.depth_feature.resnet",)
        unused_prefixes = ()

        def prep(img):
            x = img[:, :, ::-1].astype(np.float64) / 255.0
            x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            return torch.tensor(np.transpose(x, (2, 0, 1))[None], dtype=torch.float32, device=device)

        def encode(x):
            with torch.no_grad():
                return net.encoder.depth_feature.resnet(x)["feat"]

        def full(x):
            with torch.no_grad():
                return net.encoder(x, None)[0].squeeze(0).float().cpu().numpy()

        def rays(pred):
            return torch.tensor(get_ray_from_depth(pred, V=11, F_W=F_W), device=device, dtype=torch.float32)

        def localize(dt, r):
            _, pd_, orn, _ = _localize(dt, r, return_np=False)
            return pd_

    # ---- parameters, split by what runs at inference ----------------------
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    enc_n = sum(v.numel() for k, v in sd.items() if k.startswith(encoder_prefixes))
    unused_n = sum(v.numel() for k, v in sd.items() if k.startswith(unused_prefixes))
    total_n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    params = {"total": int(total_n), "encoder": int(enc_n), "unused_at_inference": int(unused_n),
              "head": int(total_n - enc_n - unused_n)}
    print(f"[params] {label}: encoder {enc_n/1e6:.1f} M, head {params['head']/1e6:.2f} M, "
          f"unused {unused_n/1e6:.1f} M")

    root = Path(args.dataset_root)
    per_scene = []
    for scene in args.scenes:
        desdf = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
        desdf["desdf"][desdf["desdf"] > 10] = 10
        occ = cv2.imread(str(root / args.collection / scene / "map.png"))[:, :, 0]
        pg = PoseGrid.from_desdf(desdf, occ.shape)
        mask = valid_pose_mask(occ, pg)
        dt = torch.tensor(desdf["desdf"], device=device)
        n_img = len(os.listdir(root / args.collection / scene / "rgb"))
        picks = np.linspace(0, n_img - 1, args.n_queries + args.warmup).astype(int)
        T_ = {k: [] for k in ("encoder", "head", "rays", "localize", "visual_total")}
        for n, i in enumerate(picks):
            img = cv2.imread(str(root / args.collection / scene / "rgb" / f"{i // 4:05d}-{i % 4}.png"))
            if img is None:
                continue
            x = prep(img); sync()
            t0 = time.perf_counter(); encode(x); sync(); t_enc = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = full(x); sync(); t_full = time.perf_counter() - t0
            t0 = time.perf_counter(); r = rays(pred); sync(); t_rays = time.perf_counter() - t0
            t0 = time.perf_counter(); localize(dt, r); sync(); t_loc = time.perf_counter() - t0
            if n >= args.warmup:
                T_["encoder"].append(t_enc); T_["head"].append(max(t_full - t_enc, 0.0))
                T_["rays"].append(t_rays); T_["localize"].append(t_loc)
                T_["visual_total"].append(t_full + t_rays + t_loc)
        med = {k: float(np.median(v)) * 1000 for k, v in T_.items()}
        med["scene"] = scene; med["n"] = len(T_["encoder"]); med["cells"] = int(mask.sum())
        per_scene.append(med)
        print(f"[time] {label} {scene}: encoder {med['encoder']:.0f} ms, head {med['head']:.1f}, "
              f"rays {med['rays']:.1f}, localize {med['localize']:.1f}, total {med['visual_total']:.0f}")

    out = {"backbone": args.backbone, "label": label, "params": params,
           "gpu": torch.cuda.get_device_name(0), "per_scene_ms": per_scene,
           "n_queries_per_scene": args.n_queries,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = HERE / "data" / f"backbone_{args.backbone}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
