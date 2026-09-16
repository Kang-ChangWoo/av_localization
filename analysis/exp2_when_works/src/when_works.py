#!/usr/bin/env python3
"""When does acoustic verification help? Measured under the paper's protocol.

Reads the test tables the headline table was built from (projection trained on
the benchmark's own rooms, fixed structure, three scalars chosen on validation
rooms) and asks, for every benchmark and backbone:

  A. where the gain is: recall at 1 m with and without sound per scene, so a
     reader sees which rooms carry the improvement and how it relates to how
     well vision does there;
  B. what the outcome of each query is: repaired (vision wrong, fused right),
     regressed (vision right, fused wrong), both wrong, both right;
  C. how the gain depends on visual ambiguity: queries binned by the log-odds
     between vision's two best hypotheses, the quantity the gate reads;
  D. whether the acoustic evidence is real on the decision the method makes:
     a forced choice between the correct hypothesis and one incorrect one,
     acoustic against visual ordering, by ambiguity stratum;
  E. what the acoustic score can do alone as the candidate set grows: choosing
     among the top-K hypotheses by itself, the gated fusion over the same K,
     and the oracle.

Writes data/when_works.json, figs/*.png and the table block of
md/when_works.md (between the markers; the prose after them is by hand).

    python src/when_works.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]           # analysis/exp2_when_works
ROOT = HERE.parents[1]                                # av_localization
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "feasible"))
from scripts.room_cv_eval import read, group          # noqa: E402
from run_projection_matrix import COND                # noqa: E402

AN = ROOT / "outputs" / "analysis"
RES = ROOT / "feasible" / "results"
TEST_TAG = {("replica", "unloc"): "unloc_uproj", ("mp3d", "unloc"): "unloc_mp3d12proj", ("s3d", "unloc"): "unloc_s3dprojB",
            ("replica", "f3loc"): "f3loc_mono_proj", ("mp3d", "f3loc"): "f3loc_mono_mp3d12proj", ("s3d", "f3loc"): "f3loc_mono_s3dprojB",
            ("replica", "disco"): "disco_rrpprojR", ("mp3d", "disco"): "disco_rrp_mp3d12proj", ("s3d", "disco"): "disco_rrp_s3dprojB"}
DS = [("replica", "Replica"), ("mp3d", "Matterport3D"), ("s3d", "Structured3D")]
BB = [("f3loc", "F3Loc mono"), ("unloc", "UnLoc"), ("disco", "DisCo-FLoc RRP")]
C = {"vision": "#0072B2", "acoustic": "#E69F00", "fused": "#009E73", "bad": "#D55E00", "grey": "#8C8C8C"}


def config(ds, bb):
    from track1_core.likelihood.mode_fusion import ModeFusionConfig
    sel = json.loads((RES / f"VAL_{ds}_fixed_indomain.json").read_text())["results"][bb]["selected"]
    w, s, tv, ta = sel["scalars"]
    return ModeFusionConfig(vis_evidence=sel["structure"][0], ac_evidence=sel["structure"][1], rule="continuous",
                            weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta,
                            ac_transform="standard" if sel["rule"] == "simple" else "relative")


def evaluate(ds, bb):
    """Per-query arrays under the paper's rule for one benchmark and backbone."""
    from track1_core.likelihood.mode_fusion import choose, evidence_columns
    cfg = config(ds, bb); vc, ac = evidence_columns(cfg)
    tag = TEST_TAG[(ds, bb)]
    Q = read(AN / f"queries_{COND[ds]}_{tag}.csv"); M = group(read(AN / f"modes_{COND[ds]}_{tag}.csv"))
    rows = []
    for i, q in enumerate(Q["query_id"]):
        m = M[str(q)]; k, acted = choose(m[vc], m[ac], cfg); kv = int(np.argmax(m[vc]))
        top = np.sort(m[vc])[::-1]
        rows.append(dict(scene=str(Q["scene"][i]), e_vis=float(m["dist_gt_m"][kv]), e_ours=float(m["dist_gt_m"][k]),
                         e_ac=float(Q["e_ac"][i]), amb=float(top[0] - top[1]) if top.size > 1 else np.inf,
                         acted=bool(acted), covered=bool((m["dist_gt_m"] < 1).any()),
                         d=m["dist_gt_m"], v=m[vc], a=m[ac], n_cells=int(Q["n_cells"][i])))
    return rows, cfg


def per_scene(rows):
    out = []
    for s in sorted({r["scene"] for r in rows}):
        rs = [r for r in rows if r["scene"] == s]
        v = np.mean([r["e_vis"] < 1 for r in rs]); o = np.mean([r["e_ours"] < 1 for r in rs])
        out.append(dict(scene=s, n=len(rs), cells=int(np.median([r["n_cells"] for r in rs])), vision=float(v), ours=float(o),
                        gain=float(o - v), acoustic=float(np.mean([r["e_ac"] < 1 for r in rs]))))
    return out


def outcomes(rows):
    o = dict(repaired=0, regressed=0, both_wrong=0, both_right=0)
    for r in rows:
        vr, fr = r["e_vis"] < 1, r["e_ours"] < 1
        o["repaired" if (not vr and fr) else "regressed" if (vr and not fr) else "both_right" if vr else "both_wrong"] += 1
    return o


def by_ambiguity(rows, bins=5):
    amb = np.array([r["amb"] for r in rows]); q = np.quantile(amb, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        sel = (amb >= q[i]) & (amb <= q[i + 1]) if i == bins - 1 else (amb >= q[i]) & (amb < q[i + 1])
        rs = [r for r, ok in zip(rows, sel) if ok]
        if not rs:
            continue
        out.append(dict(quantile=i + 1, lo=float(q[i]), hi=float(q[i + 1]), n=len(rs),
                        vision=float(np.mean([r["e_vis"] < 1 for r in rs])), ours=float(np.mean([r["e_ours"] < 1 for r in rs])),
                        acted=float(np.mean([r["acted"] for r in rs]))))
    return out


def pairwise(rows):
    """Correct hypothesis against each incorrect one; who orders them right."""
    amb = np.array([r["amb"] for r in rows]); terc = np.quantile(amb[np.isfinite(amb)], (1 / 3, 2 / 3))
    out = {}
    for name, sel in (("ambiguous", amb <= terc[0]), ("middle", (amb > terc[0]) & (amb <= terc[1])), ("confident", amb > terc[1]), ("all", np.ones_like(amb, bool))):
        na = nv = n = 0
        for r, ok in zip(rows, sel):
            if not ok:
                continue
            good = np.nonzero(r["d"] < 1)[0]; bad = np.nonzero(r["d"] >= 1)[0]
            if good.size == 0 or bad.size == 0:
                continue
            g = int(good[np.argmax(r["v"][good])])       # the best-ranked correct hypothesis
            for b in bad:
                n += 1; na += int(r["a"][g] > r["a"][b]); nv += int(r["v"][g] > r["v"][b])
        out[name] = dict(pairs=n, acoustic=float(na / max(n, 1)), visual=float(nv / max(n, 1)))
    return out


def candidate_set(rows, cfg):
    from track1_core.likelihood.mode_fusion import choose
    out = []
    for K in (1, 2, 3, 5, 10):
        alone = np.mean([r["d"][:K][int(np.argmax(r["a"][:K]))] < 1 for r in rows])
        gated = np.mean([r["d"][:K][choose(r["v"][:K], r["a"][:K], cfg)[0]] < 1 for r in rows])
        oracle = np.mean([(r["d"][:K] < 1).any() for r in rows])
        out.append(dict(K=K, acoustic_alone=float(alone), gated=float(gated), oracle=float(oracle)))
    out.append(dict(K="all cells", acoustic_alone=float(np.mean([r["e_ac"] < 1 for r in rows])), gated=None, oracle=None))
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figs").mkdir(exist_ok=True); (HERE / "data").mkdir(exist_ok=True); (HERE / "md").mkdir(exist_ok=True)
    J = {}
    for ds, _ in DS:
        for bb, _ in BB:
            if not (AN / f"queries_{COND[ds]}_{TEST_TAG[(ds, bb)]}.csv").exists():
                continue
            rows, cfg = evaluate(ds, bb)
            J[f"{ds}/{bb}"] = dict(n=len(rows), vision=float(np.mean([r["e_vis"] < 1 for r in rows])),
                                   ours=float(np.mean([r["e_ours"] < 1 for r in rows])),
                                   scenes=per_scene(rows), outcomes=outcomes(rows), ambiguity=by_ambiguity(rows),
                                   pairwise=pairwise(rows), candidate_set=candidate_set(rows, cfg))
            print(f"[{ds}/{bb}] {len(rows)} queries, vision {100*J[f'{ds}/{bb}']['vision']:.1f} -> ours {100*J[f'{ds}/{bb}']['ours']:.1f}", flush=True)
    (HERE / "data" / "when_works.json").write_text(json.dumps(J, indent=1))

    # ------------------------------------------------------------ figures
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    # A. per-scene gain against visual recall, every benchmark and backbone
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (ds, name) in zip(axes, DS):
        for bb, lab in BB:
            k = f"{ds}/{bb}"
            if k not in J:
                continue
            sc = J[k]["scenes"]
            ax.scatter([100 * s["vision"] for s in sc], [100 * s["gain"] for s in sc], s=18, alpha=0.8, label=lab)
        ax.axhline(0, color="#999", lw=0.6); ax.set_title(name); ax.set_xlabel("vision alone, recall @1 m (%)")
    axes[0].set_ylabel("gain from sound (points)"); axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(HERE / "figs" / "A_gain_per_scene.png", dpi=170); plt.close(fig)
    # C. gain by visual-ambiguity quintile (UnLoc)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, (ds, name) in zip(axes, DS):
        k = f"{ds}/unloc"
        if k not in J:
            continue
        qs = J[k]["ambiguity"]; x = np.arange(len(qs))
        ax.bar(x - 0.2, [100 * q["vision"] for q in qs], 0.4, color=C["vision"], label="vision")
        ax.bar(x + 0.2, [100 * q["ours"] for q in qs], 0.4, color=C["fused"], label="with sound")
        ax.set_xticks(x); ax.set_xticklabels([f"Q{q['quantile']}" for q in qs]); ax.set_title(f"{name}, UnLoc")
        ax.set_xlabel("visual ambiguity quintile (Q1 most ambiguous)")
    axes[0].set_ylabel("recall @1 m (%)"); axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(HERE / "figs" / "C_by_ambiguity.png", dpi=170); plt.close(fig)
    # D. pairwise forced choice
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, (ds, name) in zip(axes, DS):
        for j, (bb, lab) in enumerate(BB):
            k = f"{ds}/{bb}"
            if k not in J:
                continue
            pw = J[k]["pairwise"]; strata = ["ambiguous", "middle", "confident"]; x = np.arange(3) + (j - 1) * 0.27
            ax.plot(x, [100 * pw[s]["acoustic"] for s in strata], "o-", color=C["acoustic"], alpha=0.5 + 0.25 * j, label=f"acoustic, {lab}" if ds == "replica" else None)
            ax.plot(x, [100 * pw[s]["visual"] for s in strata], "s--", color=C["vision"], alpha=0.5 + 0.25 * j, label=f"visual, {lab}" if ds == "replica" else None)
        ax.axhline(50, color="#999", lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(strata); ax.set_title(name)
    axes[0].set_ylabel("correct hypothesis ranked first (%)"); axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(HERE / "figs" / "D_pairwise.png", dpi=170); plt.close(fig)
    # E. candidate set
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, (ds, name) in zip(axes, DS):
        k = f"{ds}/unloc"
        if k not in J:
            continue
        cs = [c for c in J[k]["candidate_set"] if c["K"] != "all cells"]; x = [c["K"] for c in cs]
        ax.plot(x, [100 * c["acoustic_alone"] for c in cs], "^-", color=C["acoustic"], label="acoustic alone among K")
        ax.plot(x, [100 * c["gated"] for c in cs], "s-", color=C["fused"], label="gated fusion over K")
        ax.plot(x, [100 * c["oracle"] for c in cs], ":", color=C["grey"], label="oracle over K")
        ax.axhline(100 * J[k]["candidate_set"][-1]["acoustic_alone"], color=C["acoustic"], lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.set_xlabel("K hypotheses"); ax.set_title(f"{name}, UnLoc")
    axes[0].set_ylabel("recall @1 m (%)"); axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(HERE / "figs" / "E_candidate_set.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- tables
    L = []
    W = L.append
    W("## Table 1. Outcome of every test query, recall at 1 m\n")
    W("| benchmark | backbone | queries | vision | with sound | repaired | regressed | both wrong | both right |")
    W("|---|---|---|---|---|---|---|---|---|")
    for ds, name in DS:
        for bb, lab in BB:
            k = f"{ds}/{bb}"
            if k not in J:
                continue
            o = J[k]["outcomes"]
            W(f"| {name} | {lab} | {J[k]['n']} | {100*J[k]['vision']:.1f} | {100*J[k]['ours']:.1f} | {o['repaired']} | {o['regressed']} | {o['both_wrong']} | {o['both_right']} |")
    W("\n## Table 2. Per scene, UnLoc: where the gain is (sorted by gain)\n")
    W("| benchmark | scene | queries | cells | vision | acoustic alone | with sound | gain |")
    W("|---|---|---|---|---|---|---|---|")
    for ds, name in DS:
        k = f"{ds}/unloc"
        if k not in J:
            continue
        for s in sorted(J[k]["scenes"], key=lambda s: -s["gain"]):
            W(f"| {name} | {s['scene']} | {s['n']} | {s['cells']} | {100*s['vision']:.1f} | {100*s['acoustic']:.1f} | {100*s['ours']:.1f} | {100*s['gain']:+.1f} |")
    W("\n## Table 3. Recall by visual-ambiguity quintile, UnLoc (Q1 most ambiguous)\n")
    W("| benchmark | quintile | log-odds range | n | vision | with sound | gain | gate open |")
    W("|---|---|---|---|---|---|---|---|")
    for ds, name in DS:
        k = f"{ds}/unloc"
        if k not in J:
            continue
        for q in J[k]["ambiguity"]:
            W(f"| {name} | Q{q['quantile']} | {q['lo']:.3f}–{q['hi']:.3f} | {q['n']} | {100*q['vision']:.1f} | {100*q['ours']:.1f} | {100*(q['ours']-q['vision']):+.1f} | {100*q['acted']:.0f}% |")
    W("\n## Table 4. Forced choice between the correct hypothesis and an incorrect one\n")
    W("| benchmark | backbone | stratum | pairs | acoustic right | visual right |")
    W("|---|---|---|---|---|---|")
    for ds, name in DS:
        for bb, lab in BB:
            k = f"{ds}/{bb}"
            if k not in J:
                continue
            for st in ("ambiguous", "middle", "confident", "all"):
                p = J[k]["pairwise"][st]
                W(f"| {name} | {lab} | {st} | {p['pairs']} | {100*p['acoustic']:.1f} | {100*p['visual']:.1f} |")
    W("\n## Table 5. The acoustic score as the candidate set grows, UnLoc\n")
    W("| benchmark | K | acoustic alone among K | gated fusion over K | oracle over K |")
    W("|---|---|---|---|---|")
    for ds, name in DS:
        k = f"{ds}/unloc"
        if k not in J:
            continue
        for c in J[k]["candidate_set"]:
            f = lambda x: "–" if x is None else f"{100*x:.1f}"
            W(f"| {name} | {c['K']} | {f(c['acoustic_alone'])} | {f(c['gated'])} | {f(c['oracle'])} |")
    target = HERE / "md" / "when_works.md"
    start, end = "<!-- tables:start -->", "<!-- tables:end -->"
    block = "\n".join(L).rstrip() + "\n\n"
    if target.exists() and start in target.read_text() and end in target.read_text():
        t = target.read_text(); t = t[: t.index(start) + len(start)] + "\n" + block + t[t.index(end):]
    else:
        t = "# When does acoustic verification help?\n\n" + start + "\n" + block + end + "\n"
    target.write_text(t)
    print(f"wrote {target} (tables block), data/when_works.json, figs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
