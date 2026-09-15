#!/usr/bin/env python3
"""Select everything on validation rooms, then touch the test rooms once.

The paper's fusion has a handful of discrete choices (which projection source,
which summary represents a hypothesis under each modality, whether the rule
carries an acoustic gate) and three or four scalars. Choosing any of them on
the test rooms, even by cross-validation, lets the test rooms shape the method.
This script makes every one of those choices on the validation rooms of a
benchmark, pooled, by recall at 1 m, and then evaluates the chosen
configuration on the test rooms exactly once with no refitting.

It also records, under the same validation-only selection, the best
configuration without a projection and the best with the four-scalar rule, so
the contribution of each part is measured without looking at the test rooms.

    python scripts/val_select.py --dataset replica
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feasible"))

from scripts.room_cv_eval import read, group          # noqa: E402
from run_projection_matrix import COND, IDENT, PREFIX, SOURCES, tag_of  # noqa: E402

AN = REPO_ROOT / "outputs" / "analysis"
LABEL = {"f3loc": "F3Loc mono", "unloc": "UnLoc", "disco": "DisCo-FLoc RRP"}
TH = [0.1, 0.5, 1.0, 2.0, 5.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["replica", "mp3d", "s3d"])
    p.add_argument("--backbones", nargs="+", default=["f3loc", "unloc", "disco"])
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--simplicity-margin", type=float, default=0.01,
                   help="a part (the projection, the acoustic gate with its transform) is kept "
                        "only if it raises validation recall by more than this. Stated in advance: "
                        "with a few hundred validation queries a half-point difference is noise, "
                        "and the smaller rule is preferred at a tie")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--source", choices=["val", "indomain", "B"], default="val",
                   help="how the projection source is chosen. 'val' searches it with the rest; "
                        "'indomain' is the source trained on the same benchmark (R on Replica, M "
                        "on Matterport3D; Structured3D cannot train one and uses B); 'B' is the "
                        "pooled projection everywhere. Both fixed choices are made in advance")
    p.add_argument("--fixed", action="store_true",
                   help="the paper's protocol: one method everywhere. The projection source (B), "
                        "the hypothesis summaries (centre/quantile) and the rule (three scalars) "
                        "are fixed across benchmarks and backbones; only the three scalars are "
                        "chosen on the validation rooms. Writes VAL_<dataset>_fixed.*")
    return p.parse_args()


def val_tag(backbone: str, source, ds: str = "replica") -> str:
    return f"{PREFIX[backbone]}_val" + ("" if ds == "replica" else ds) + (f"proj{source}" if source else "")


def test_tag(dataset: str, backbone: str, source) -> str:
    return IDENT[(dataset, backbone)] if source is None else tag_of(dataset, backbone, source)


def load(cond: str, tag: str):
    qp = AN / f"queries_{cond}_{tag}.csv"
    if not qp.exists():
        return None
    Q = read(qp)
    M = group(read(AN / f"modes_{cond}_{tag}.csv"))
    return Q, M


def main() -> int:
    args = parse_args()
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose, evidence_columns
    from track1_core.provenance import stamp
    cond = COND[args.dataset]
    rng = np.random.default_rng(0)

    structures = list(itertools.product(("centre", "max", "lse"), ("centre", "max", "quantile", "lse")))
    if args.fixed:
        structures = [("centre", "quantile")]
    base = list(itertools.product((0.5, 1.0, 2.0), (0.02, 0.05, 0.1), (0.005, 0.02, 0.05, 0.1, 0.2)))
    rules = {"simple": [(w, s, tv, -2.0, "standard") for w, s, tv in base],
             "full": [(w, s, tv, ta, "relative") for (w, s, tv), ta in
                      itertools.product(base, (-2.0, 0.0, 0.2, 0.4))]}
    if args.fixed:
        rules = {"simple": rules["simple"]}
    INDOMAIN = {"replica": "R", "mp3d": "M", "s3d": "B"}
    allowed_sources = {"val": (None,) + tuple(SOURCES),
                       "indomain": (INDOMAIN[args.dataset], None),
                       "B": ("B", None)}[args.source]

    # The selection grid is ~13,500 configurations per source, so the rule is
    # evaluated on padded (queries x hypotheses) arrays rather than query by
    # query. ``choose`` is the reference; ``check_vectorised`` below verifies the
    # two agree on every query before any selection is trusted.
    def pack(QM, vc, ac):
        Q, M = QM
        n = len(Q["query_id"]); K = max(len(M[str(q)]["dist_gt_m"]) for q in Q["query_id"])
        V = np.full((n, K), -np.inf); A = np.full((n, K), np.nan); D = np.full((n, K), np.inf)
        for i, q in enumerate(Q["query_id"]):
            m = M[str(q)]; k = len(m["dist_gt_m"])
            V[i, :k] = m[vc]; A[i, :k] = m[ac]; D[i, :k] = m["dist_gt_m"]
        return V, A, D

    def standardise_rows(X):
        ok = np.isfinite(X)
        cnt = ok.sum(1, keepdims=True)
        mu = np.where(ok, X, 0).sum(1, keepdims=True) / np.maximum(cnt, 1)
        var = (np.where(ok, X - mu, 0) ** 2).sum(1, keepdims=True) / np.maximum(cnt, 1)
        sd = np.sqrt(var); sd = np.where(sd > 1e-12, sd, 1.0)
        return np.where(ok, (X - mu) / sd, -np.inf)

    def relative_rows(Z):
        # each hypothesis against the logsumexp of the others, -inf padding ignored
        ok = np.isfinite(Z); Zf = np.where(ok, Z, -np.inf)
        m = Zf.max(1, keepdims=True); ex = np.where(ok, np.exp(Zf - m), 0.0)
        tot = ex.sum(1, keepdims=True)
        others = np.clip(tot - ex, 1e-300, None)
        return np.where(ok, Zf - (m + np.log(others)), -np.inf)

    def top_gap(X):
        s = -np.sort(-X, axis=1)
        return np.where(np.isfinite(s[:, 1]) if s.shape[1] > 1 else False,
                        s[:, 0] - (s[:, 1] if s.shape[1] > 1 else 0), np.inf)

    def prepare(QM, ve, ae):
        vc, ac = evidence_columns(ModeFusionConfig(vis_evidence=ve, ac_evidence=ae))
        V, A, D = pack(QM, vc, ac)
        zv = standardise_rows(V); za = standardise_rows(np.where(np.isfinite(V), A, np.nan))
        amb = top_gap(V)
        rel = relative_rows(za); disc = top_gap(rel)
        n_hyp = np.isfinite(V).sum(1)
        vb = np.argmax(V, axis=1)
        return dict(zv=zv, za=za, amb=amb, rel=rel, disc=disc, D=D, vb=vb, n=n_hyp)

    def score_prep(P, w, s, tv, ta, tr):
        gv = 1.0 / (1.0 + np.exp((P["amb"] - tv) / s))
        ga = 1.0 / (1.0 + np.exp(-(P["disc"] - ta) / s))
        alpha = w * gv * ga
        acoustic = P["rel"] if tr == "relative" else P["za"]
        k = np.argmax(P["zv"] + alpha[:, None] * acoustic, axis=1)
        k = np.where(P["n"] < 2, P["vb"], k)
        return P["D"][np.arange(len(k)), k]

    def score(QM, cfg):
        Q, M = QM
        vc, ac = evidence_columns(cfg)
        e = np.array([M[str(q)]["dist_gt_m"][choose(M[str(q)][vc], M[str(q)][ac], cfg)[0]]
                      for q in Q["query_id"]])
        return e

    def check_vectorised(QM):
        rng_ = np.random.default_rng(1)
        for _ in range(6):
            ve, ae = structures[rng_.integers(len(structures))]
            rule = list(rules)[rng_.integers(len(rules))]
            w, s, tv, ta, tr = rules[rule][rng_.integers(len(rules[rule]))]
            cfg = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="continuous", weight=w,
                                   sigmoid_scale=s, tau_v=tv, tau_a=ta, ac_transform=tr)
            a = score(QM, cfg); b = score_prep(prepare(QM, ve, ae), w, s, tv, ta, tr)
            if not np.allclose(a, b):
                raise RuntimeError(f"vectorised rule disagrees with choose() on {np.mean(a != b):.3f} of queries for {cfg}")

    out, js = [], {}
    W = out.append
    W(f"# Validation-selected configuration, tested once: {args.dataset}\n")
    W("Every discrete choice and every scalar is chosen on the validation rooms by recall at "
      "1 m; the test rooms are evaluated once with the chosen configuration and nothing is "
      "refit on them. `no projection` and `four-scalar rule` are the best configurations under "
      "the same validation-only selection with that part fixed, so each part's contribution "
      "is measured without the test rooms.\n")
    W("| backbone | selection | source | structure | rule | scalars | val @1m | test vision | test ours | gain | 95% CI |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")

    for bb in args.backbones:
        val = {s: load(cond, val_tag(bb, s, args.dataset)) for s in allowed_sources}
        val = {s: v for s, v in val.items() if v is not None}
        if not val:
            print(f"[{bb}] no validation tables yet"); continue
        best = {}   # variant -> (score, source, structure, rule, scalars, cfg)
        check_vectorised(next(iter(val.values())))
        for src, QM in val.items():
            for ve, ae in structures:
                P = prepare(QM, ve, ae)
                for rule, grid in rules.items():
                    for w, s, tv, ta, tr in grid:
                        cfg = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="continuous",
                                               weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta,
                                               ac_transform=tr)
                        r = float((score_prep(P, w, s, tv, ta, tr) < 1).mean())
                        # the selection score charges each extra part a margin, so a
                        # part survives only if it earns more than noise on validation
                        penalised = r - args.simplicity_margin * ((src is not None) + (rule == "full"))
                        if args.source != "val" and src is None:
                            penalised = -1.0      # the fixed protocol always carries W; 'no projection' is the ablation
                        for variant, ok, key in (("selected", True, penalised),
                                                 ("no projection", src is None, r),
                                                 ("four-scalar rule", rule == "full", r),
                                                 ("three-scalar rule", rule == "simple", r)):
                            if ok and (variant not in best or key > best[variant][0]):
                                best[variant] = (key, src, (ve, ae), rule, (w, s, tv, ta), cfg, r)
        js[bb] = {}
        for variant, (_, src, st, rule, sc, cfg, r) in best.items():
            tt = test_tag(args.dataset, bb, src)
            T = load(cond, tt)
            if T is None:
                W(f"| {LABEL[bb]} | {variant} | {src or 'none'} | {st[0]}/{st[1]} | {rule} | {sc} | "
                  f"{100*r:.1f} | (test table {tt} missing) | | | |")
                continue
            e = score(T, cfg); v = T[0]["e_vis"]
            d = (e < 1).astype(float) - (v < 1).astype(float)
            m = d[rng.integers(0, d.size, size=(args.boot, d.size))].mean(axis=1)
            lo, hi = np.percentile(m, [2.5, 97.5])
            W(f"| {LABEL[bb]} | {variant} | {src or 'none'} | {st[0]}/{st[1]} | {rule} | "
              f"w={sc[0]:g} s={sc[1]:g} τv={sc[2]:g}{'' if rule == 'simple' else f' τa={sc[3]:g}'} | "
              f"{100*r:.1f} | {100*(v<1).mean():.1f} | {100*(e<1).mean():.1f} | {100*d.mean():+.1f} | "
              f"[{100*lo:+.1f}, {100*hi:+.1f}] |")
            js[bb][variant] = dict(val=r, source=src, structure=st, rule=rule, scalars=sc,
                                   test_tag=tt, n=int(e.size), vision=float((v < 1).mean()),
                                   ours=float((e < 1).mean()), gain=float(d.mean()),
                                   ci=[float(lo), float(hi)],
                                   recalls={f"{t}m": float((e < t).mean()) for t in TH},
                                   vision_recalls={f"{t}m": float((v < t).mean()) for t in TH},
                                   median=float(np.median(e)), vision_median=float(np.median(v)))
        print(f"[{bb}] " + "; ".join(f"{k}: val {100*v[-1]:.1f} src {v[1]} {v[2]} {v[3]}" for k, v in best.items()))

    suffix = ("_fixed" if args.fixed else "") + ("" if args.source == "val" else f"_{args.source}")
    outp = args.out or REPO_ROOT / "feasible" / "results" / f"VAL_{args.dataset}{suffix}.md"
    outp.write_text("\n".join(out) + "\n")
    outp.with_suffix(".json").write_text(json.dumps(dict(dataset=args.dataset, results=js,
                                                         provenance=stamp()), indent=2, default=str))
    print("\n".join(out)); print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
