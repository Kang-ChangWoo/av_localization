#!/usr/bin/env python3
"""Additional Analysis: when acoustic verification helps, why, and what stops it.

This is not an ablation sweep. Each block answers one question that the paper's
central claim depends on, and each is designed so that a negative outcome would
be reportable rather than merely absent.

    A  When does it help?            visual ambiguity against acoustic gain
    B  Does it really discriminate?  forced choice between two visually
                                     plausible hypotheses, against chance
    D  What is the ceiling?          shortlist oracle against achieved, per K
    E  What limits it?               furnished against matched geometry
    F  Is the gain real?             the same pipeline with the audio detached
                                     from its pose
    G  Is it complementary?          scene-level visual difficulty against gain
    I  Does sound know it is wrong?  acoustic self-confidence as a predictor

Block B is the one that tests the paper's actual task. Acoustic-only recall over
a whole floorplan measures a job the method never does; a forced choice between
two places vision already considers plausible measures the job it does. Block F
is the control that makes any of it believable: permuting the correspondence
between a recording and its pose, while leaving every score distribution
untouched, must destroy the gain, and if it does not then the gain was never
about acoustics.

Everything reads the extracted mode tables, so this runs in seconds and can be
re-run after any pipeline change.

    python analysis/run_analysis.py --backbone unlocSTFT
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--policy", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "unified_policy.json")
    p.add_argument("--backbone", default="unlocSTFT",
                   help="primary backbone for the narrative")
    p.add_argument("--also", nargs="*", default=["f3STFT"],
                   help="backbones repeated only where the claim needs a second one")
    p.add_argument("--fit-collection", default="replica_f")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=HERE / "results")
    return p.parse_args()


def read(p: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(p, newline="")))
    out: dict[str, np.ndarray] = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        if v[0] in ("True", "False"):
            out[k] = np.array([x == "True" for x in v]); continue
        try:
            out[k] = np.array([float(x) if x not in ("", "nan") else np.nan for x in v])
        except ValueError:
            out[k] = np.array(v, dtype=object)
    return out


def group(M: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    g: dict[str, dict[str, list]] = {}
    cols = [c for c in M if c not in ("query_id", "scene", "collection")]
    for i, q in enumerate(M["query_id"]):
        d = g.setdefault(str(q), {c: [] for c in cols})
        for c in cols:
            d[c].append(M[c][i])
    for q, d in g.items():
        o = np.argsort(np.asarray(d["mode"], dtype=int))
        g[q] = {c: np.asarray(v, dtype=float)[o] for c, v in d.items()}
    return g


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson interval, which behaves near 0 and 1 where the normal one does not."""
    if n == 0:
        return (np.nan, np.nan)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main() -> int:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from track1_core.likelihood.mode_fusion import (
        ModeFusionConfig, acoustic_evidence, choose, evidence_columns,
    )
    from track1_core.provenance import stamp

    rng = np.random.default_rng(args.seed)
    pol = json.loads(args.policy.read_text())["policy"]
    KEYS = ("vis_evidence", "ac_evidence", "rule", "weight",
            "sigmoid_scale", "tau_v", "tau_a")

    def cfg_of(tag, **over):
        d = {k: (-np.inf if pol[tag][k] is None else pol[tag][k]) for k in KEYS}
        d.update(over)
        return ModeFusionConfig(**d)

    def load(tag, cond="raw_scan_open"):
        Q = read(args.analysis_dir / f"queries_{cond}_{tag}.csv")
        M = group(read(args.analysis_dir / f"modes_{cond}_{tag}.csv"))
        rep = Q["collection"] != args.fit_collection
        return Q, M, rep

    def predict(tag, cfg, cond="raw_scan_open", perm=None):
        """Errors under `cfg`; `perm` optionally detaches audio from its pose."""
        Q, M, rep = load(tag, cond)
        vc, ac = evidence_columns(cfg)
        qids = [str(x) for x in Q["query_id"][rep]]
        # a query can yield fewer than K separated modes, so the across-query
        # swap has to draw from queries with the same count or the vectors
        # would not line up with the hypotheses they are being attached to
        pool: dict[int, list] = {}
        for q in qids:
            pool.setdefault(len(M[q][ac]), []).append(M[q][ac])
        e = []
        for i, q in enumerate(qids):
            m = M[q]
            a = m[ac]
            if perm == "within":
                a = rng.permutation(a)
            elif perm == "across":
                cand = pool[len(a)]
                a = cand[int(rng.integers(len(cand)))]
            k, _ = choose(m[vc], a, cfg)
            e.append(m["dist_gt_m"][k])
        return np.asarray(e), Q, rep

    def boot(a, b):
        x = b.astype(float) - a.astype(float)
        i = rng.integers(0, x.size, size=(args.boot, x.size))
        m = x[i].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    W = out.append
    js: dict = {}
    tag = args.backbone
    Q, M, rep = load(tag)
    ev, ov = Q["e_vis"][rep], Q["orn_vis"][rep]
    qids = [str(x) for x in Q["query_id"][rep]]
    cfg = cfg_of(tag)
    vc, ac = evidence_columns(cfg)
    e_ours, _, _ = predict(tag, cfg)
    n = len(ev)

    W(f"# Additional analysis\n")
    W(f"Primary backbone `{tag}`, {n} held-out queries on `replica_g`. Query "
      f"acoustics are the furnished scan; candidates are rendered from the "
      f"floorplan alone. Every parameter was chosen on `replica_f`.\n")
    W(f"\nVision alone {100*(ev<1).mean():.1f}% at 1 m, with acoustic "
      f"verification {100*(e_ours<1).mean():.1f}%.\n")

    # ------------------------------------------------------------------ A
    W("\n## A. When does acoustic verification help?\n")
    W("Queries are binned by visual ambiguity, the log-odds between the two "
      "strongest hypotheses. Low means the backbone cannot separate its own top "
      "two candidates.\n")
    amb = Q["mode_log_ratio"][rep]
    amb = np.where(np.isfinite(amb), amb, np.nanmax(amb[np.isfinite(amb)]))
    edges = np.unique(np.quantile(amb, np.linspace(0, 1, args.bins + 1)))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    W("| ambiguity | n | vision | + acoustic | repairs | breaks | net |")
    W("|---|---|---|---|---|---|---|")
    xs, ys, cis = [], [], []
    for i in range(len(edges) - 1):
        k = (amb > edges[i]) & (amb <= edges[i + 1])
        if not k.any():
            continue
        m, lo, hi = boot((ev[k] < 1).astype(float), (e_ours[k] < 1).astype(float))
        W(f"| {edges[i]:.3f}-{edges[i+1]:.3f} | {int(k.sum())} | "
          f"{100*(ev[k]<1).mean():.1f}% | {100*(e_ours[k]<1).mean():.1f}% | "
          f"{int(((ev[k]>=1)&(e_ours[k]<1)).sum())} | "
          f"{int(((ev[k]<1)&(e_ours[k]>=1)).sum())} | **{100*m:+.1f}** |")
        xs.append(float(np.median(amb[k]))); ys.append(100 * m)
        cis.append((100 * lo, 100 * hi))
    js["A"] = dict(x=xs, gain=ys, ci=cis)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.axhline(0, color="0.6", lw=0.9)
    ax.errorbar(xs, ys, yerr=[[y - c[0] for y, c in zip(ys, cis)],
                              [c[1] - y for y, c in zip(ys, cis)]],
                fmt="o-", color="#2962ff", capsize=3)
    ax.set_xlabel("visual ambiguity: log-odds between the top two hypotheses")
    ax.set_ylabel(r"$\Delta$ Recall@1m (points)")
    ax.set_title("A. Acoustic gain against visual ambiguity")
    fig.tight_layout(); fig.savefig(args.out_dir / "A_ambiguity.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ B
    W("\n## B. Can sound choose between two visually plausible hypotheses?\n")
    W("A forced binary choice. For each query we take vision's two strongest "
      "hypotheses and keep the *decidable* pairs, those where exactly one lies "
      "within 1 m of the true pose, so a coin flip scores 50%. This measures the "
      "task the method performs, unlike acoustic-only recall over a whole "
      "floorplan, which measures a task it never performs.\n")
    W("| subset | n | acoustic picks correctly | 95% CI | vision picks correctly |")
    W("|---|---|---|---|---|")
    rows_b = []
    for name, sel in (("all decidable pairs", lambda d, v, s: True),
                      ("visually close (margin below median)", None),
                      ("spatially separated by >2 m", None),
                      ("close and separated", None)):
        rows_b.append(name)
    pairs = []
    for q in qids:
        m = M[q]
        d = m["dist_gt_m"][:2]
        if (d < 1).sum() != 1:
            continue
        right = int(np.argmin(d))
        a = m[ac][:2]
        v = m[vc][:2]
        pairs.append(dict(right=right, ac_ok=int(np.argmax(a)) == right,
                          vis_ok=int(np.argmax(v)) == right,
                          margin=float(v[0] - v[1]),
                          sep=float(m["dist_to_best_mode_m"][1])))
    if pairs:
        marg = np.array([p["margin"] for p in pairs])
        med = float(np.median(marg))
        subsets = {
            "all decidable pairs": np.ones(len(pairs), bool),
            f"visually close (margin < {med:.3f})": marg < med,
            "spatially separated by > 2 m": np.array([p["sep"] > 2 for p in pairs]),
            "close and separated": (marg < med) & np.array([p["sep"] > 2 for p in pairs]),
        }
        js["B"] = {}
        for name, k in subsets.items():
            if k.sum() == 0:
                continue
            ok = int(sum(p["ac_ok"] for p, kk in zip(pairs, k) if kk))
            vo = int(sum(p["vis_ok"] for p, kk in zip(pairs, k) if kk))
            lo, hi = wilson(ok, int(k.sum()))
            W(f"| {name} | {int(k.sum())} | **{100*ok/k.sum():.1f}%** | "
              f"[{lo:.1f}, {hi:.1f}] | {100*vo/k.sum():.1f}% |")
            js["B"][name] = dict(n=int(k.sum()), acc=ok / int(k.sum()), ci=[lo, hi],
                                 vis=vo / int(k.sum()))
        W("\nChance is 50%. An interval whose lower bound clears 50 is the "
          "cleanest statement of the paper's core ability that this data can "
          "produce; one that straddles 50 would say the method works only "
          "through the visual prior it is attached to.\n")

    # ------------------------------------------------------------------ D
    W("\n## D. Is the shortlist or the verifier the bottleneck?\n")
    W("The oracle picks the hypothesis nearest truth and is not a method. The "
      "gap between it and the achieved number is what better acoustic "
      "discrimination could still buy at that K.\n")
    W("| K | oracle @1 m | achieved @1 m | unused headroom |")
    W("|---|---|---|---|")
    js["D"] = {}
    for K in (1, 2, 3, 5, 10):
        orc = np.array([M[q]["dist_gt_m"][:K].min() for q in qids])
        e = []
        for q in qids:
            m = M[q]
            k, _ = choose(m[vc][:K], m[ac][:K], cfg)
            e.append(m["dist_gt_m"][:K][k])
        e = np.asarray(e)
        W(f"| {K} | {100*(orc<1).mean():.1f}% | {100*(e<1).mean():.1f}% | "
          f"{100*((orc<1).mean()-(e<1).mean()):.1f} |")
        js["D"][K] = dict(oracle=float((orc < 1).mean()), got=float((e < 1).mean()))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    Ks = sorted(js["D"])
    ax.plot(Ks, [100 * js["D"][k]["oracle"] for k in Ks], "s--", color="0.35",
            label="oracle over the K hypotheses")
    ax.plot(Ks, [100 * js["D"][k]["got"] for k in Ks], "o-", color="#2962ff",
            label="acoustic verification")
    ax.axhline(100 * (ev < 1).mean(), color="#d50000", lw=1.2, ls=":",
               label="vision alone")
    ax.set_xlabel("number of spatial hypotheses K"); ax.set_ylabel("Recall@1m (%)")
    ax.set_title("D. Shortlist ceiling against achieved")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "D_oracle.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ E
    W("\n## E. What limits the acoustic evidence?\n")
    W("The only change between the two rows is the mesh the *query* was rendered "
      "on. Candidates are floorplan-only in both. This is a control, not a "
      "method: a deployment cannot render furniture it does not know about.\n")
    W("| query geometry | acoustic alone @1 m | median GT rank | with verification @1 m |")
    W("|---|---|---|---|")
    js["E"] = {}
    for cond, label in (("raw_scan_open", "furnished (the real setting)"),
                        ("floorplan_closed", "matched to the candidates")):
        try:
            Qc, Mc, repc = load(tag, cond)
        except FileNotFoundError:
            continue
        ec, _, _ = predict(tag, cfg, cond)
        W(f"| {label} | {100*(Qc['e_ac'][repc]<1).mean():.1f}% | "
          f"{np.median(Qc['gt_ac_rank_pos'][repc]):.0f} | {100*(ec<1).mean():.1f}% |")
        js["E"][cond] = dict(alone=float((Qc["e_ac"][repc] < 1).mean()),
                             gt_rank=float(np.median(Qc["gt_ac_rank_pos"][repc])),
                             ours=float((ec < 1).mean()))

    # ------------------------------------------------------------------ F
    W("\n## F. Is the gain actually acoustic?\n")
    W("Two permutation controls. `within query` keeps every acoustic score but "
      "reassigns which hypothesis it belongs to; `across queries` gives each "
      "query another query's acoustic evidence. Both leave the score "
      "distribution exactly as it was and destroy only the correspondence "
      "between a recording and the pose it was recorded at.\n")
    W("| audio | recall @1 m | gain over vision | 95% CI |")
    W("|---|---|---|---|")
    base = (ev < 1).astype(float)
    js["F"] = {}
    for name, perm in (("as recorded", None), ("permuted within query", "within"),
                       ("swapped across queries", "across")):
        acc = []
        for _ in range(5 if perm else 1):
            e, _, _ = predict(tag, cfg, perm=perm)
            acc.append(e)
        e = np.concatenate(acc)
        b = np.tile(base, len(acc))
        m, lo, hi = boot(b, (e < 1).astype(float))
        W(f"| {name} | {100*(e<1).mean():.1f}% | {100*m:+.1f} | "
          f"[{100*lo:+.1f}, {100*hi:+.1f}] |")
        js["F"][name] = dict(recall=float((e < 1).mean()), gain=float(m),
                             ci=[float(lo), float(hi)])
    W("\nIf a permuted control retained the gain, the fusion would be exploiting "
      "something about the score distribution rather than the recording, and "
      "nothing else in this document would mean anything.\n")

    # ------------------------------------------------------------------ G
    W("\n## G. Does sound complement visual failure?\n")
    W("Scene-level, both backbones, so the axis is not a property of one model.\n")
    W("| backbone | scene | vision @1 m | + acoustic | gain |")
    W("|---|---|---|---|---|")
    js["G"] = []
    for t in [tag] + list(args.also):
        try:
            Qg, Mg, repg = load(t)
        except FileNotFoundError:
            continue
        eg, _, _ = predict(t, cfg_of(t))
        vg = Qg["e_vis"][repg]
        sc = Qg["scene"][repg]
        for s in np.unique(sc):
            k = sc == s
            g = 100 * ((eg[k] < 1).mean() - (vg[k] < 1).mean())
            W(f"| {t} | {s} | {100*(vg[k]<1).mean():.1f}% | "
              f"{100*(eg[k]<1).mean():.1f}% | {g:+.1f} |")
            js["G"].append(dict(backbone=t, scene=str(s),
                                vision=float((vg[k] < 1).mean()), gain=float(g)))
    if len(js["G"]) > 2:
        x = np.array([r["vision"] for r in js["G"]]) * 100
        y = np.array([r["gain"] for r in js["G"]])
        r = float(np.corrcoef(x, y)[0, 1])
        W(f"\nCorrelation between a scene's visual recall and the acoustic gain "
          f"there: **{r:+.2f}** over {len(x)} scene-backbone pairs. A strong "
          f"negative value says sound is useful exactly where vision fails, "
          f"which is a different and stronger statement than sound being better "
          f"in acoustically easy rooms.\n")
        js["G_corr"] = r
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        for t, mk in zip([tag] + list(args.also), "os^"):
            k = [i for i, rr in enumerate(js["G"]) if rr["backbone"] == t]
            ax.scatter(x[k], y[k], marker=mk, s=70, label=t)
        ax.axhline(0, color="0.6", lw=0.9)
        ax.set_xlabel("scene visual Recall@1m (%)")
        ax.set_ylabel(r"$\Delta$ Recall@1m from acoustics")
        ax.set_title(f"G. Complementarity, r = {r:+.2f}")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out_dir / "G_complementarity.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ I
    W("\n## I. Does the acoustic score know when it is wrong?\n")
    W("Reported because it is negative and because it explains the design: the "
      "gate leans on visual ambiguity precisely because no acoustic "
      "self-confidence measure predicts its own correctness.\n")

    def auroc(s, y):
        s, y = np.asarray(s, float), np.asarray(y).astype(bool)
        if y.all() or not y.any():
            return np.nan
        o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
        n1 = int(y.sum()); n0 = len(y) - n1
        return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    # queries differ in how many separated hypotheses they yield, so the
    # per-query evidence vectors are ragged and only their summaries stack
    rels = [acoustic_evidence(M[q][ac]) for q in qids]
    top = np.array([float(r.max()) for r in rels])
    mrg = np.array([float(r.max() - np.sort(r)[-2]) if r.size > 1 else np.inf
                    for r in rels])
    mrg = np.where(np.isfinite(mrg), mrg, np.nanmax(mrg[np.isfinite(mrg)]))
    sigs = {"acoustic margin among hypotheses": mrg,
            "relative evidence of the top hypothesis": top,
            "visual ambiguity": -amb}
    ac_ok = np.array([M[q]["dist_gt_m"][int(np.argmax(M[q][ac]))] < 1 for q in qids])
    W("| signal | acoustic pick correct | repairs vision | breaks vision |")
    W("|---|---|---|---|")
    js["I"] = {}
    for name, s in sigs.items():
        a1 = auroc(s, ac_ok)
        a2 = auroc(s[ev >= 1], (e_ours[ev >= 1] < 1))
        a3 = auroc(s[ev < 1], (e_ours[ev < 1] >= 1))
        W(f"| {name} | {a1:.3f} | {a2:.3f} | {a3:.3f} |")
        js["I"][name] = [a1, a2, a3]
    W("\nAUROC 0.5 is chance.\n")

    rep_md = args.out_dir / "additional_analysis.md"
    rep_md.write_text("\n".join(out) + "\n")
    (args.out_dir / "additional_analysis.json").write_text(
        json.dumps(dict(backbone=tag, n=n, results=js, provenance=stamp()),
                   indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {rep_md} and three figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
