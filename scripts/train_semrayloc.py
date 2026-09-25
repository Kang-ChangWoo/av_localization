#!/usr/bin/env python3
"""Train SemRayLoc's ray networks (semantic or depth) on echoloc_dataset.

The SemRayLoc clone (../SemRayLoc, upstream, untouched) expects one folder per
scene with rgb/<i>.png and depth/semantic/pitch/roll/room files; this script
keeps that code as it is and feeds it our collections through a small dataset
adapter instead: rgb in filename order (= poses.txt order), depth<N>.txt and
semantic<N>.txt from scripts/make_semantic_rays.py, roll = pitch = 0.

Room labels (--room-labels, changed 2026-09-25):
  ZInD ships a free-text room label on every panorama and our builder already
  stores it in chunks.json as frames[].label (coverage 45261/45261 on the train
  split, audited in /root/local1/changwoo/zind_sem_r2/results/label_audit.json).
  --room-labels real reads it and maps it with upstream's rule
  (SemRayLoc/data_utils/zind/map_zind_rooms.py:4-44) onto
  modules.semantic.semantic_mapper.zind_room_type_to_id, where undefined is 13
  and the table has 14 entries -- NOT the 16-entry S3D room_type_to_id whose
  undefined is 15.  --room-labels const reproduces the old behaviour (one
  constant id, room loss collapses to 0 and acc_room_val to 1.0), kept as the
  control arm.

    python scripts/train_semrayloc.py --net semantic --dataset-root /root/storage/echoloc_dataset/zind \
        --collections zind --run-name zind_r2/real --room-labels real --num-room-types 14 --gpus 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRL_ROOT = REPO_ROOT.parent / "SemRayLoc"
sys.path.insert(0, str(SRL_ROOT))

RAY_NAMES = ["wall", "window", "door", "unknown"]

# upstream mapping rule, transcribed from SemRayLoc/data_utils/zind/map_zind_rooms.py:4-44.
# That file cannot be imported: it runs an os.walk over a hardcoded Windows path at
# module level.
_ALLOWED_ROOM_KEYS = ["bedroom", "closet", "hallway", "living room", "kitchen",
                      "bathroom", "basement", "dining room", "loft", "garage",
                      "laundry", "office", "stairs"]


def map_zind_label(label: str) -> str:
    n = label.lower()
    if "breakfast nook" in n:
        return "dining room"
    if "family room" in n:
        return "living room"
    if any(x in n for x in ["attic", "pantry", "utility"]):
        return "closet"
    if "fireplace" in n:
        return "living room"
    if "stair" in n:
        return "stairs"
    for k in _ALLOWED_ROOM_KEYS:
        if k in n:
            return k
    return "undefined"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", choices=["semantic", "depth"], required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--collections", nargs="+", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--gpus", default="0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ray-n", type=int, default=40)
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="resize the query image (SemRayLoc trained at 360x640)")
    p.add_argument("--class-weights", type=float, nargs=4, default=None,
                   help="cross-entropy weights for wall/window/door/unknown (semantic net). "
                        "Applied to the training loss AND to the reported weighted validation "
                        "loss; loss_rays_val stays unweighted so runs remain comparable.")
    p.add_argument("--room-labels", choices=["const", "real"], default="const",
                   help="const: one constant room id for every frame (the pre-2026-09-25 "
                        "behaviour, room head carries no signal). real: per-frame ZInD label "
                        "from chunks.json frames[].label, mapped with upstream's rule.")
    p.add_argument("--const-room-id", type=int, default=15,
                   help="room id used by --room-labels const (15 = undefined in the 16-entry "
                        "S3D table, which is what this script used before 2026-09-25)")
    p.add_argument("--num-room-types", type=int, default=16,
                   help="room head width. Use 14 with --room-labels real on ZInD "
                        "(zind_room_type_to_id has 14 entries, undefined = 13).")
    p.add_argument("--room-loss-weight", type=float, default=1.0)
    p.add_argument("--precision", default="bf16-mixed",
                   help="fp16 mixed precision drove the depth net's attention to NaN within the first "
                        "epoch on every benchmark (the semantic net was fine); bf16 has the fp32 range")
    p.add_argument("--monitor", default="loss-valid",
                   help="validation metric that picks the checkpoint: loss-valid (SemRayLoc's own), "
                        "acc_rays_val (global, saturates early under the 83/9/8 ray imbalance), or "
                        "macro_f1_rays3_val (macro F1 over wall/window/door -- the metric the "
                        "2026-09-25 re-run pre-registers)")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="0 disables early stopping; otherwise EarlyStopping on --monitor")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit-train-batches", type=float, default=1.0)
    p.add_argument("--limit-val-batches", type=float, default=1.0)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _scene_room_ids(scene_dir: Path, n_frames: int, id_table):
    """Per-frame room ids for one scene, in rgb-filename order.

    chunks.json stores chunks[].frames[] in the same order as poses.txt; that was
    verified on all 2443 ZInD floors (max |poses.txt[:, :3] - frames[x, y, yaw]|
    == 0 on every one, see results/label_audit.json pose_mismatch).
    """
    c = json.loads((scene_dir / "chunks.json").read_text())
    labels = []
    for ch in c["chunks"]:
        for fr in ch["frames"]:
            labels.append(str(fr.get("label") or ""))
    if len(labels) != n_frames:
        raise RuntimeError("chunks.json frame count %d != rgb count %d in %s"
                           % (len(labels), n_frames, scene_dir))
    ids, raw = [], []
    for lb in labels:
        if not lb.strip():
            raise RuntimeError("empty frames[].label in %s -- audit said coverage was 100%%" % scene_dir)
        m = map_zind_label(lb)
        ids.append(id_table[m])
        raw.append(m)
    return ids, raw


class EchoLocRays:
    """One sample per frame of the given collections and scenes."""

    def __init__(self, root: Path, collections, scenes, ray_n, resize=None, with_semantic=True,
                 room_labels="const", const_room_id=15, id_table=None):
        import torch  # noqa: F401
        self.items = []
        self.resize = resize
        self.room_hist = {}
        for col in collections:
            for sc in scenes:
                d = root / col / sc
                if not (d / "poses.txt").exists():
                    continue
                names = sorted(f for f in os.listdir(d / "rgb") if f.endswith(".png"))
                depth = np.loadtxt(d / f"depth{ray_n}.txt", ndmin=2, dtype=np.float32)
                sem = (np.loadtxt(d / f"semantic{ray_n}.txt", ndmin=2, dtype=np.int64)
                       if with_semantic else np.zeros_like(depth, dtype=np.int64))
                assert len(names) == len(depth) == len(sem), (d, len(names), len(depth), len(sem))
                if room_labels == "real":
                    rids, rnames = _scene_room_ids(d, len(names), id_table)
                    for nm in rnames:
                        self.room_hist[nm] = self.room_hist.get(nm, 0) + 1
                else:
                    rids = [const_room_id] * len(names)
                    self.room_hist["const:%d" % const_room_id] = \
                        self.room_hist.get("const:%d" % const_room_id, 0) + len(names)
                for i, n in enumerate(names):
                    self.items.append((str(d / "rgb" / n), depth[i], sem[i], rids[i]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import torch
        path, depth, sem, room = self.items[i]
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            img = cv2.resize(img, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
        img = (img / 255.0).astype(np.float32).transpose(2, 0, 1)
        return {
            "ref_img": img,
            "ref_mask": np.ones(img.shape[1:], dtype=np.uint8),
            "ref_depth": depth.astype(np.float32),
            "ref_semantics": sem.astype(np.float32),
            "room_label": torch.tensor(room, dtype=torch.long),
        }


def _prf(conf):
    """Per-class precision / recall / F1 / support from an (C, C) confusion matrix
    indexed [true, pred].

    A class the model never predicts scores precision 0, not NaN (sklearn's
    zero_division=0).  Letting it be NaN made macro F1 NaN, which silently turned
    the ModelCheckpoint monitor into -inf -- the smoke run collapsed to all-wall
    and still "improved" forever.  NaN is reserved for classes with no support at
    all (ZInD's "unknown" ray class), which are dropped from every macro.
    """
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    sup = conf.sum(axis=1)
    pred = conf.sum(axis=0)
    prec = np.zeros_like(tp)
    rec = np.zeros_like(tp)
    f1 = np.zeros_like(tp)
    for i in range(conf.shape[0]):
        if sup[i] == 0 and pred[i] == 0:
            prec[i] = rec[i] = f1[i] = np.nan
            continue
        rec[i] = tp[i] / sup[i] if sup[i] > 0 else np.nan
        prec[i] = tp[i] / pred[i] if pred[i] > 0 else 0.0
        if sup[i] == 0:
            f1[i] = np.nan
        else:
            den = prec[i] + rec[i]
            f1[i] = (2 * prec[i] * rec[i] / den) if den > 0 else 0.0
    return prec, rec, f1, sup


def build_semantic_model(a, id_table):
    """semantic_net_pl plus whole-epoch per-class ray/room metrics.

    Upstream's validation_step only logs the two global accuracies
    (SemRayLoc/modules/semantic/semantic_net_pl.py:106-107), which cannot tell a
    real model from the "always wall" predictor at 0.828 on ZInD.  This subclass
    accumulates confusion matrices over the whole validation epoch (not a mean of
    per-batch means) and writes them out.
    """
    import torch
    import torch.nn.functional as F
    from modules.semantic.semantic_net_pl import semantic_net_pl

    inv_room = {v: k for k, v in (id_table or {}).items()}
    n_ray = 4
    n_room = a.num_room_types
    weights = (torch.tensor(a.class_weights, dtype=torch.float32)
               if a.class_weights is not None else None)
    jsonl = Path(a.output_dir) / a.run_name / "metrics_by_epoch.jsonl"

    class SemanticPerClass(semantic_net_pl):
        def __init__(self):
            super().__init__(num_ray_classes=n_ray, num_room_types=n_room, lr=a.lr)
            self.register_buffer("_w", weights if weights is not None else torch.ones(n_ray),
                                 persistent=False)
            self._use_w = weights is not None
            self._conf_ray = None
            self._conf_room = None

        def _losses(self, batch):
            ray_logits, room_logits, _ = self.model(batch["ref_img"], mask=batch.get("ref_mask"))
            ray_t = batch["ref_semantics"].long()
            room_t = batch["room_label"].long()
            lp = ray_logits.permute(0, 2, 1)
            unw = F.cross_entropy(lp, ray_t)
            w = F.cross_entropy(lp, ray_t, weight=self._w.to(lp.device)) if self._use_w else unw
            lroom = F.cross_entropy(room_logits, room_t)
            return lp, room_logits, ray_t, room_t, unw, w, lroom

        def training_step(self, batch, batch_idx):
            _, _, _, _, unw, w, lroom = self._losses(batch)
            loss = w + a.room_loss_weight * lroom
            self.log("loss_rays_train", w, prog_bar=True)
            self.log("loss_rays_train_unw", unw)
            self.log("loss_room_train", lroom, prog_bar=True)
            self.log("loss_train", loss, prog_bar=True)
            return loss

        def on_validation_epoch_start(self):
            self._conf_ray = torch.zeros(n_ray, n_ray, dtype=torch.long, device=self.device)
            self._conf_room = torch.zeros(n_room, n_room, dtype=torch.long, device=self.device)

        def validation_step(self, batch, batch_idx):
            lp, room_logits, ray_t, room_t, unw, w, lroom = self._losses(batch)
            loss = unw + a.room_loss_weight * lroom
            pred_ray = lp.argmax(dim=1)
            pred_room = room_logits.argmax(dim=1)

            self._conf_ray.index_put_(
                (ray_t.reshape(-1), pred_ray.reshape(-1)),
                torch.ones_like(ray_t.reshape(-1)), accumulate=True)
            self._conf_room.index_put_(
                (room_t.reshape(-1), pred_room.reshape(-1)),
                torch.ones_like(room_t.reshape(-1)), accumulate=True)

            self.log("loss-valid", loss)
            self.log("cross_entropy_loss-valid", loss)
            self.log("loss_rays_val", unw)
            self.log("loss_rays_val_w", w)
            self.log("loss_room_val", lroom)
            return {"loss": loss}

        def on_validation_epoch_end(self):
            cr = self._conf_ray.detach().cpu().numpy()
            cm = self._conf_room.detach().cpu().numpy()
            rec = {"epoch": int(self.current_epoch), "global_step": int(self.global_step)}

            tot = cr.sum()
            acc_rays = float(np.diag(cr).sum()) / max(int(tot), 1)
            prec, recall, f1, sup = _prf(cr)
            present = [i for i in range(n_ray) if sup[i] > 0]
            for i in range(n_ray):
                nm = RAY_NAMES[i]
                rec["ray_support_" + nm] = int(sup[i])
                for tag, arr in (("prec", prec), ("rec", recall), ("f1", f1)):
                    v = arr[i]
                    rec["ray_%s_%s" % (tag, nm)] = None if np.isnan(v) else float(v)
            # wall/window/door only: ZInD's "unknown" ray class has ~18 of 1.81M
            # training rays, so a 4-class macro is dominated by an empty class.
            core = [i for i in (0, 1, 2) if sup[i] > 0]
            macro3 = float(np.mean([f1[i] for i in core])) if core else 0.0
            macro_all = float(np.mean([f1[i] for i in present])) if present else 0.0
            dummy = float(sup.max()) / max(float(sup.sum()), 1.0)
            rec.update(acc_rays_val=acc_rays, macro_f1_rays3_val=macro3,
                       macro_f1_rays_present_val=macro_all,
                       dummy_majority_acc_rays=dummy, ray_conf=cr.tolist())

            mtot = cm.sum()
            acc_room = float(np.diag(cm).sum()) / max(int(mtot), 1)
            mp, mr, mf, ms = _prf(cm)
            rpresent = [i for i in range(n_room) if ms[i] > 0]
            for i in rpresent:
                nm = inv_room.get(i, "id%d" % i)
                rec["room_support_" + nm] = int(ms[i])
                rec["room_rec_" + nm] = None if np.isnan(mr[i]) else float(mr[i])
                rec["room_f1_" + nm] = None if np.isnan(mf[i]) else float(mf[i])
            macro_room = float(np.mean([mf[i] for i in rpresent])) if rpresent else 0.0
            bal_room = float(np.nanmean([mr[i] for i in rpresent])) if rpresent else 0.0
            rec.update(acc_room_val=acc_room, macro_f1_room_val=macro_room,
                       bal_acc_room_val=bal_room, n_room_present=len(rpresent),
                       dummy_majority_acc_room=float(ms.max()) / max(float(ms.sum()), 1.0))

            # never log NaN: a NaN monitor makes ModelCheckpoint's best score -inf
            # and EarlyStopping blind.
            macro3 = 0.0 if np.isnan(macro3) else macro3
            macro_room = 0.0 if np.isnan(macro_room) else macro_room
            self.log("acc_rays_val", acc_rays, prog_bar=True)
            self.log("macro_f1_rays3_val", macro3, prog_bar=True)
            self.log("acc_room_val", acc_room)
            self.log("macro_f1_room_val", macro_room, prog_bar=True)
            self.log("bal_acc_room_val", bal_room)
            for i in (1, 2):
                v = f1[i]
                self.log("f1_" + RAY_NAMES[i], 0.0 if np.isnan(v) else float(v), prog_bar=True)

            if self.trainer is not None and not self.trainer.sanity_checking:
                jsonl.parent.mkdir(parents=True, exist_ok=True)
                with open(str(jsonl), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")

    return SemanticPerClass()


def main() -> int:
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader
    from modules.depth.depth_net_pl import depth_net_pl
    from modules.semantic.semantic_mapper import zind_room_type_to_id

    id_table = zind_room_type_to_id
    if a.room_labels == "real":
        assert len(id_table) == 14 and id_table["undefined"] == 13, "upstream ZInD table changed"
        if a.num_room_types < len(id_table):
            raise SystemExit("--num-room-types %d is smaller than the ZInD table (%d)"
                             % (a.num_room_types, len(id_table)))
    else:
        if a.const_room_id >= a.num_room_types:
            raise SystemExit("--const-room-id %d out of range for --num-room-types %d"
                             % (a.const_room_id, a.num_room_types))

    seed_everything(a.seed)
    root = Path(a.dataset_root)
    meta = json.loads((root / "dataset_meta.json").read_text())
    split = meta["split"]
    tr, va = split["train"], split["val"]
    if a.max_scenes:
        tr, va = tr[: a.max_scenes], va[: a.max_scenes]
    kw = dict(room_labels=a.room_labels, const_room_id=a.const_room_id, id_table=id_table)
    ds_tr = EchoLocRays(root, a.collections, tr, a.ray_n, a.resize, a.net == "semantic", **kw)
    ds_va = EchoLocRays(root, a.collections, va, a.ray_n, a.resize, a.net == "semantic", **kw)
    print("[data] train %d frames  val %d frames" % (len(ds_tr), len(ds_va)), flush=True)
    print("[rooms] mode=%s num_room_types=%d" % (a.room_labels, a.num_room_types), flush=True)
    for tag, ds in (("train", ds_tr), ("val", ds_va)):
        h = dict(sorted(ds.room_hist.items(), key=lambda kv: -kv[1]))
        n = max(sum(h.values()), 1)
        print("[rooms/%s] %s" % (tag, json.dumps({k: [v, round(v / n, 5)] for k, v in h.items()})),
              flush=True)

    out = a.output_dir / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_stats.json").write_text(json.dumps({
        "train_frames": len(ds_tr), "val_frames": len(ds_va),
        "room_mode": a.room_labels, "num_room_types": a.num_room_types,
        "room_hist_train": ds_tr.room_hist, "room_hist_val": ds_va.room_hist,
        "class_weights": a.class_weights, "monitor": a.monitor,
        "epochs": a.epochs, "early_stop_patience": a.early_stop_patience,
        "seed": a.seed, "lr": a.lr, "batch_size": a.batch_size,
        "resize": a.resize, "precision": a.precision, "argv": sys.argv,
    }, indent=1))

    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, drop_last=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)

    if a.net == "semantic":
        model = build_semantic_model(a, id_table)
    else:
        f_w = float(meta["camera"].get("F_W") or 1 / (2 * np.tan(np.deg2rad(meta["camera"]["hfov_deg"]) / 2)))
        model = depth_net_pl(shape_loss_weight=20, lr=a.lr, d_max=20.0, F_W=f_w)

    mode = "max" if (a.monitor.startswith("acc") or "f1" in a.monitor) else "min"
    cbs = [ModelCheckpoint(monitor=a.monitor, dirpath=out,
                           filename="%s_net-{epoch:02d}-{%s:.4f}" % (a.net, a.monitor),
                           save_top_k=2, mode=mode, save_last=True)]
    if a.early_stop_patience > 0:
        cbs.append(EarlyStopping(monitor=a.monitor, mode=mode, patience=a.early_stop_patience,
                                 min_delta=a.early_stop_min_delta, verbose=True))
    trainer = Trainer(max_epochs=a.epochs, max_steps=a.max_steps, callbacks=cbs, accelerator="gpu", devices=1,
                      precision=a.precision, log_every_n_steps=50, default_root_dir=out,
                      limit_train_batches=a.limit_train_batches, limit_val_batches=a.limit_val_batches,
                      num_sanity_val_steps=0, gradient_clip_val=1.0)
    trainer.fit(model, dl_tr, dl_va)
    trainer.save_checkpoint(out / ("final_%s.ckpt" % a.net))
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            pass
    metrics["stopped_epoch"] = int(trainer.current_epoch)
    metrics["best_model_path"] = str(cbs[0].best_model_path)
    try:
        metrics["best_monitor_value"] = float(cbs[0].best_model_score)
    except (TypeError, ValueError):
        pass
    (out / "final_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("[done]", json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
