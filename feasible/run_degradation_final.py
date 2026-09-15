#!/usr/bin/env python3
"""Visual degradation under the final protocol, UnLoc on Replica.

The earlier degradation study (run_visual_degradation.py) used no projection
and refit the scalars by leave-one-room-out at every level. This one is the
deployment question asked properly: the projection trained on Replica's
training rooms and the three scalars chosen on the clean validation rooms are
frozen, and the query image is corrupted at test time. Nothing is refit on the
degraded queries, so the curve says what a deployed system does when its camera
degrades, not what a re-tuned one could do.

Queues the eighteen extractions (UnLoc, projected acoustic score, one
corruption each) on GPUs with enough free memory, then evaluates every level
with the frozen configuration and writes feasible/results/V_final_unloc.{md,json}
and feasible/figs/V_final_unloc.png.

    python -u feasible/run_degradation_final.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run_projection_matrix import PY, W, log  # noqa: E402
from run_val_pipeline import gpu_free_mib  # noqa: E402

AN = ROOT / "outputs" / "analysis"
UNLOC_CK = (ROOT.parent / "UnLoc" / "tb_logs" / "my_model" / "version_1" / "checkpoints"
            / "epoch=19-step=1040.ckpt")
BASE = ("--dataset-root /root/storage/echoloc_dataset/replica --collections replica_f replica_g "
        "--scenes office_4 apartment_2 frl_apartment_5 --condition raw_scan_open "
        f"--grid-dir outputs/acoustic_grid_v2 --backbone unloc --checkpoint {UNLOC_CK} "
        "--feature stft_band --nfft 256 --hop 64 --n-poses 100")
# the last two of each family are past anything a working camera produces; they
# are there to show where the shortlist itself gives out
LEVELS = [("clean", None), ("blur", 1), ("blur", 2), ("blur", 4), ("blur", 8), ("blur", 16), ("blur", 32),
          ("dark", 0.75), ("dark", 0.5), ("dark", 0.25), ("dark", 0.1), ("dark", 0.05), ("dark", 0.02),
          ("noise", 10), ("noise", 25), ("noise", 50), ("noise", 100), ("noise", 150),
          ("occlude", 0.1), ("occlude", 0.3), ("occlude", 0.5), ("occlude", 0.7), ("occlude", 0.9),
          ("downscale", 2), ("downscale", 4), ("downscale", 8), ("downscale", 16), ("downscale", 32)]
LABEL = {"clean": "clean", "blur": "blur σ={}", "dark": "dark ×{}", "noise": "noise σ={}",
         "occlude": "occlude {:.0%}", "downscale": "downscale {}×"}
# UnLoc at batch 1 takes about 3.5 GB, so a card that other tenants leave 6 GB
# on can still take a job, and a card with more room can take two
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
SLOTS = [(g, i) for g in GPUS for i in (0, 1)]
NEED = 5000


def tag(fam, lvl):
    return "unloc_uproj" if fam == "clean" else f"unloc_udegP_{fam}_{lvl}"


def table(t):
    return AN / f"queries_raw_scan_open_{t}.csv"


def main() -> int:
    jobs = [(f, l) for f, l in LEVELS if not table(tag(f, l)).exists()]
    log(f"{len(jobs)} degraded extractions to run")
    running = {}
    while jobs or running:
        for slot, (p, key) in list(running.items()):
            if p.poll() is not None:
                log(f"gpu {slot[0]}: {key} {'done' if table(tag(*key)).exists() else 'FAILED'}")
                del running[slot]
        for slot in SLOTS:
            g = slot[0]
            if not jobs or slot in running or gpu_free_mib(g) < NEED:
                continue
            fam, lvl = jobs.pop(0)
            t = tag(fam, lvl)
            if subprocess.run(["pgrep", "-f", f"--tag {t[len('unloc'):]} "], capture_output=True).returncode == 0:
                continue      # already running from an earlier driver
            cmd = (f"{PY} scripts/extract_mode_table.py --gpu {g} {BASE} --degrade {fam}:{lvl} "
                   f"--tag {t[len('unloc'):]} --acoustic-projection {W['R']} > logs/extract_{t}.log 2>&1")
            running[slot] = (subprocess.Popen(cmd, shell=True, cwd=ROOT), (fam, lvl))
            log(f"gpu {g}: started {t}")
            time.sleep(20)      # let the new process claim its memory before the next check
        time.sleep(60)

    # ---------------------------------------------------------- evaluate
    from scripts.room_cv_eval import read, group
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose, evidence_columns
    from track1_core.provenance import stamp
    sel = json.loads((HERE / "results" / "VAL_replica_fixed_indomain.json").read_text())["results"]["unloc"]["selected"]
    w, s, tv, ta = sel["scalars"]
    cfg = ModeFusionConfig(vis_evidence=sel["structure"][0], ac_evidence=sel["structure"][1], rule="continuous",
                           weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta,
                           ac_transform="standard" if sel["rule"] == "simple" else "relative")
    vc, ac = evidence_columns(cfg)
    rng = np.random.default_rng(0)

    # settings the validation rooms chose for the three beyond-shortlist variants
    VJ = json.loads((HERE / "results" / "VAL_replica_fixed_indomain.json").read_text())["results"]["unloc"]
    A = VJ.get("A: injected acoustic hypotheses"); B = VJ.get("B: shortlist rejection"); Cv = VJ.get("C: cell-wise product")
    mA = int(A["setting"].split("=")[1].split()[0]) if A and A["setting"].startswith("m=") and not A["setting"].startswith("m=0") else 0
    scA = A["scalars"] if A else None
    if A and mA > 0:
        parts = dict(x.split("=") for x in A["setting"].split()[1:])
        scA = (float(parts["w"]), float(parts["s"]), float(parts["τv"]), -2.0)
    tgB = float(B["setting"].split(",")[0].split("=")[1]) if B else np.inf
    colC = "e_cellprod_" + Cv["setting"] if Cv else None
    print(f"[variants] A m={mA} scalars={scA}  B tau_g={tgB}  C {colC}", flush=True)

    def variant_errors(Q, M_all, e0):
        outv = {}
        if mA > 0:
            Mm = group(read(AN / f"modes_raw_scan_open_{t}.csv"), inject=mA)
            V = np.full((len(Q["query_id"]), 10 + mA), -np.inf); Aa = np.full_like(V, np.nan); D = np.full_like(V, np.inf)
            for i, q in enumerate(Q["query_id"]):
                mm = Mm[str(q)]; k = len(mm["dist_gt_m"]); V[i, :k] = mm[vc]; Aa[i, :k] = mm[ac]; D[i, :k] = mm["dist_gt_m"]
            zv = (V - np.nanmean(np.where(np.isfinite(V), V, np.nan), 1, keepdims=True)) / np.maximum(np.nanstd(np.where(np.isfinite(V), V, np.nan), 1, keepdims=True), 1e-12)
            za = (Aa - np.nanmean(Aa, 1, keepdims=True)) / np.maximum(np.nanstd(Aa, 1, keepdims=True), 1e-12)
            top = -np.sort(-np.where(np.isfinite(V), V, -np.inf), 1)
            amb = np.where(np.isfinite(top[:, 1]), top[:, 0] - top[:, 1], np.inf)
            gate = scA[0] / (1 + np.exp((amb - scA[2]) / scA[1]))
            k = np.argmax(np.where(np.isfinite(V), zv, -np.inf) + gate[:, None] * np.where(np.isfinite(V), za, -np.inf), 1)
            outv["A"] = D[np.arange(len(k)), k]
        else:
            outv["A"] = e0
        if np.isfinite(tgB):
            M1 = group(read(AN / f"modes_raw_scan_open_{t}.csv"), inject=1)
            g, einj = [], []
            for q in Q["query_id"]:
                mm = M1[str(q)]; a_vis, a_inj = mm[ac][:-1], mm[ac][-1]
                sd = float(np.std(a_vis)); g.append((a_inj - a_vis.max()) / (sd if sd > 1e-12 else 1.0)); einj.append(mm["dist_gt_m"][-1])
            outv["B"] = np.where(np.array(g) > tgB, np.array(einj), e0)
        else:
            outv["B"] = e0
        outv["C"] = Q[colC] if colC and colC in Q else np.full_like(e0, np.nan)
        return outv

    rows, out = [], []
    out.append("# Visual degradation under the final protocol: UnLoc, Replica\n")
    out.append(f"Projection R and the scalars chosen on the clean validation rooms are frozen "
               f"({cfg}); the query image is corrupted at test time and nothing is refit. "
               f"`acoustic alone` is the projected score's own top-1 over the whole grid; "
               f"`audio acts` is the fraction of queries where the gate weight exceeds 0.05.\n")
    out.append(f"The three beyond-shortlist variants use the settings the clean validation rooms "
               f"chose (A m={mA}, B tau_g={tgB:g}, C {colC}), frozen like everything else.\n")
    out.append("| degradation | queries | vision @1m | acoustic alone | ours @1m | gain | 95% CI | audio acts | A inject | B reject | C product |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for fam, lvl in LEVELS:
        t = tag(fam, lvl)
        if not table(t).exists():
            out.append(f"| {LABEL[fam].format(lvl)} | (missing) | | | | | | |"); continue
        Q = read(table(t)); M = group(read(AN / f"modes_raw_scan_open_{t}.csv"))
        e, used = [], []
        for q in Q["query_id"]:
            m = M[str(q)]; k, acted = choose(m[vc], m[ac], cfg); e.append(m["dist_gt_m"][k]); used.append(acted)
        e = np.asarray(e); v = Q["e_vis"]; a = Q["e_ac"]
        var = variant_errors(Q, M, e)
        d = (e < 1).astype(float) - (v < 1).astype(float)
        mb = d[rng.integers(0, d.size, size=(10000, d.size))].mean(axis=1)
        lo, hi = np.percentile(mb, [2.5, 97.5])
        label = LABEL[fam].format(lvl) if lvl is not None else "clean"
        rows.append(dict(family=fam, level=lvl, label=label, n=int(e.size), vision=float((v < 1).mean()),
                         acoustic=float((a < 1).mean()), ours=float((e < 1).mean()), gain=float(d.mean()),
                         lo=float(lo), hi=float(hi), audio_used=float(np.mean(used)),
                         A=float(np.nanmean(var["A"] < 1)), B=float(np.nanmean(var["B"] < 1)),
                         C=float(np.nanmean(var["C"] < 1))))
        r = rows[-1]
        out.append(f"| {label} | {r['n']} | {100*r['vision']:.1f}% | {100*r['acoustic']:.1f}% | {100*r['ours']:.1f}% | "
                   f"{100*r['gain']:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {100*r['audio_used']:.0f}% | "
                   f"{100*r['A']:.1f}% | {100*r['B']:.1f}% | {100*r['C']:.1f}% |")
        print(out[-1], flush=True)

    # ------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = ["blur", "dark", "noise", "occlude", "downscale"]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.4 * len(fams), 3.2), sharey=True)
    clean = next((r for r in rows if r["family"] == "clean"), None)
    for ax, fam in zip(axes, fams):
        rs = [r for r in rows if r["family"] == fam]
        if clean:
            rs = [clean] + rs
        x = np.arange(len(rs)); lab = [r["label"].replace(fam + " ", "") for r in rs]
        ax.plot(x, [100 * r["vision"] for r in rs], "o-", color="#0072B2", label="vision")
        ax.plot(x, [100 * r["ours"] for r in rs], "s-", color="#009E73", label="ours")
        ax.plot(x, [100 * r["acoustic"] for r in rs], "^--", color="#E69F00", label="acoustic alone")
        ax.plot(x, [100 * r["A"] for r in rs], "d:", color="#CC79A7", label="A: injected hypotheses")
        ax.plot(x, [100 * r["B"] for r in rs], "v:", color="#56B4E9", label="B: shortlist rejection")
        ax.plot(x, [100 * r["C"] for r in rs], "x:", color="#999999", label="C: cell product")
        ax.fill_between(x, [100 * r["vision"] for r in rs], [100 * r["ours"] for r in rs], color="#009E73", alpha=0.15)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7, rotation=30); ax.set_title(fam)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recall @ 1 m (%)"); axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("UnLoc on Replica, frozen projection and scalars, query image corrupted at test time", fontsize=9)
    fig.tight_layout()
    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs" / "V_final_unloc.png", dpi=180)
    (HERE / "results" / "V_final_unloc.md").write_text("\n".join(out) + "\n")
    (HERE / "results" / "V_final_unloc.json").write_text(json.dumps(dict(rows=rows, policy=str(cfg), provenance=stamp()), indent=2))
    log("DEGRADATION_FINAL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
