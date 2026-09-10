#!/usr/bin/env python3
"""Does the acoustic evidence come from geometry the camera cannot see?

This is the analysis the paper's motivating sentence stands or falls on. The
claim is that sound resolves visual ambiguity because it carries information
about geometry outside the camera's field of view. Every other experiment we
have is consistent with a weaker reading: that acoustic consistency happens to
discriminate hypotheses, for reasons that might be entirely inside the field of
view. Separating the two requires measuring the geometry directly.

The floorplan already provides it. The DESDF stores, at every cell, the free
distance along each of 36 yaw bins. The 11 bins spanned by the camera are what
vision sees; the remaining 25 are what it does not. For two competing
hypotheses, that gives two independent quantities:

    visible similarity   how alike the two places look to the camera. High means
                         vision has little to go on, which is why the pair is
                         ambiguous in the first place.
    hidden difference    how much the two places differ outside the camera cone.
                         High means there is something to hear that cannot be
                         seen.

The prediction that distinguishes the strong claim from the weak one is
specific: **acoustic discrimination should improve with hidden difference, and
that improvement should survive when visible similarity is held high.** If
accuracy tracks only visible similarity, the acoustic score is re-deriving what
the camera already had, and the honest claim is the weaker one.

One choice deserves stating rather than burying. Both hypotheses are evaluated
at the ground-truth heading. This is an analysis of geometry, not a prediction,
and using one heading for both places is the symmetric comparison; using each
backbone's own predicted heading would fold that backbone's yaw errors into a
measurement that is supposed to be about the room. The alternative was checked
to move nothing qualitatively.

    python analysis/beyond_fov.py --backbone unlocSTFT
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
    p.add_argument("--dataset-root", default="/root/storage/echoloc_dataset")
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--policy", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "unified_policy.json")
    p.add_argument("--backbone", default="unlocSTFT")
    p.add_argument("--also", nargs="*", default=["f3STFT", "discoID"])
    p.add_argument("--fit-collection", default="replica_f")
    p.add_argument("--condition", default="raw_scan_open",
                   choices=["raw_scan_open", "floorplan_closed"],
                   help="floorplan_closed removes the furniture gap and is the "
                        "control that says whether the effect exists at all")
    p.add_argument("--fov-bins", type=int, default=11,
                   help="yaw bins the camera spans; 11 of 36 is about 110 degrees")
    p.add_argument("--depth-clip", type=float, default=10.0)
    p.add_argument("--n-tiles", type=int, default=3)
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
    from track1_core.likelihood.mode_fusion import evidence_columns, ModeFusionConfig
    from track1_core.provenance import stamp

    pol = json.loads(args.policy.read_text())["policy"]
    root = Path(args.dataset_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # DESDF per scene, clipped exactly as the localiser clips it so that the
    # geometry this analysis reads is the geometry the method reads
    desdfs: dict[str, np.ndarray] = {}

    def desdf_of(scene: str) -> np.ndarray:
        if scene not in desdfs:
            d = np.load(root / "desdf" / scene / "desdf.npy", allow_pickle=True).item()
            v = np.array(d["desdf"], dtype=np.float64)
            v[v > args.depth_clip] = args.depth_clip
            desdfs[scene] = v
        return desdfs[scene]

    def split_geometry(scene, x, y, yaw_rad):
        """Free distances inside and outside the camera cone at one hypothesis."""
        D = desdf_of(scene)
        O = D.shape[2]
        yi = int(np.round(yaw_rad % (2 * np.pi) / (2 * np.pi) * O)) % O
        half = args.fov_bins // 2
        idx = np.arange(yi - half, yi - half + args.fov_bins) % O
        vis = D[int(y), int(x), idx]
        hid = D[int(y), int(x), np.setdiff1d(np.arange(O), idx)]
        return vis, hid

    def collect(tag):
        Q = read(args.analysis_dir / f"queries_{args.condition}_{tag}.csv")
        M = group(read(args.analysis_dir / f"modes_{args.condition}_{tag}.csv"))
        rep = Q["collection"] != args.fit_collection
        KEYS = ("vis_evidence", "ac_evidence", "rule", "weight",
                "sigmoid_scale", "tau_v", "tau_a")
        cfg = ModeFusionConfig(**{k: (-np.inf if pol[tag][k] is None else pol[tag][k])
                                  for k in KEYS})
        vc, ac = evidence_columns(cfg)
        rows = []
        qid = Q["query_id"][rep]
        scene = Q["scene"][rep]
        gyaw = Q["gt_yaw"][rep]
        for q, s, yw in zip(qid, scene, gyaw):
            m = M[str(q)]
            d = m["dist_gt_m"][:2]
            if d.size < 2 or (d < 1).sum() != 1:
                continue                       # only decidable pairs
            right = int(np.argmin(d))
            v1, h1 = split_geometry(s, m["x"][0], m["y"][0], yw)
            v2, h2 = split_geometry(s, m["x"][1], m["y"][1], yw)
            rows.append(dict(
                scene=str(s),
                # high means the camera cone looks the same at both places
                vis_sim=float(-np.mean(np.abs(v1 - v2))),
                # high means the rest of the room differs between them
                hid_diff=float(np.mean(np.abs(h1 - h2))),
                ac_ok=bool(int(np.argmax(m[ac][:2])) == right),
                vis_ok=bool(int(np.argmax(m[vc][:2])) == right),
                sep=float(m["dist_to_best_mode_m"][1])))
        return rows

    out: list[str] = []
    W = out.append
    js: dict = {}
    W("# C. Does the acoustic evidence come from beyond the field of view?\n")
    W(f"Condition `{args.condition}`. Decidable pairs only: vision's two strongest hypotheses, exactly one of "
      f"which is within 1 m of the true pose, so chance is 50%. Geometry is read "
      f"from the same DESDF the localiser uses, split into the "
      f"{args.fov_bins} yaw bins the camera spans and the "
      f"{36-args.fov_bins} it does not.\n")

    rows = collect(args.backbone)
    if not rows:
        print("no decidable pairs")
        return 1
    vs = np.array([r["vis_sim"] for r in rows])
    hd = np.array([r["hid_diff"] for r in rows])
    ok = np.array([r["ac_ok"] for r in rows])
    vok = np.array([r["vis_ok"] for r in rows])
    js["n"] = len(rows)

    # ---- one dimension at a time -----------------------------------------
    W(f"\n## Marginal effect of each quantity ({args.backbone}, {len(rows)} pairs)\n")
    W("| quantity | tercile | n | acoustic correct | 95% CI | vision correct |")
    W("|---|---|---|---|---|---|")
    for name, q in (("visible similarity", vs), ("hidden difference", hd)):
        edges = np.quantile(q, np.linspace(0, 1, args.n_tiles + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        for i in range(args.n_tiles):
            k = (q > edges[i]) & (q <= edges[i + 1])
            if not k.any():
                continue
            lo, hi = wilson(int(ok[k].sum()), int(k.sum()))
            lbl = ("low", "mid", "high")[i] if args.n_tiles == 3 else str(i + 1)
            W(f"| {name} | {lbl} | {int(k.sum())} | **{100*ok[k].mean():.1f}%** | "
              f"[{lo:.1f}, {hi:.1f}] | {100*vok[k].mean():.1f}% |")
            js.setdefault(name, {})[lbl] = dict(n=int(k.sum()), acc=float(ok[k].mean()),
                                                ci=[lo, hi], vis=float(vok[k].mean()))

    # ---- the joint statement ---------------------------------------------
    W("\n## The joint statement\n")
    W("Rows are hidden difference, columns visible similarity. The cell of "
      "interest is the top right: the camera sees the same thing at both places "
      "and the rest of the room differs. If acoustic accuracy is highest there, "
      "the acoustic evidence is coming from geometry the camera cannot see.\n")
    ev = np.quantile(vs, np.linspace(0, 1, args.n_tiles + 1)); ev[0] -= 1e-9; ev[-1] += 1e-9
    eh = np.quantile(hd, np.linspace(0, 1, args.n_tiles + 1)); eh[0] -= 1e-9; eh[-1] += 1e-9
    labs = ("low", "mid", "high") if args.n_tiles == 3 else tuple(map(str, range(args.n_tiles)))
    W("| hidden diff \\ visible sim | " + " | ".join(labs) + " |")
    W("|---" * (args.n_tiles + 1) + "|")
    G = np.full((args.n_tiles, args.n_tiles), np.nan)
    N = np.zeros_like(G)
    for a in range(args.n_tiles):
        cells = []
        for b in range(args.n_tiles):
            k = ((hd > eh[a]) & (hd <= eh[a + 1]) & (vs > ev[b]) & (vs <= ev[b + 1]))
            if k.sum() < 5:
                cells.append(f"n={int(k.sum())}")
                continue
            G[a, b] = 100 * ok[k].mean(); N[a, b] = k.sum()
            cells.append(f"**{G[a,b]:.0f}%** (n={int(k.sum())})")
        W(f"| {labs[a]} | " + " | ".join(cells) + " |")
    js["joint"] = dict(grid=G.tolist(), n=N.tolist())

    # the decisive contrast, stated as a number with an interval
    hi_v = vs > ev[args.n_tiles - 1]
    k_hi = hi_v & (hd > eh[args.n_tiles - 1])
    k_lo = hi_v & (hd <= eh[1])
    W("\n### The decisive contrast\n")
    W("Both groups below have *visually similar* competing hypotheses, so vision "
      "has little to work with in either. They differ only in whether the "
      "unseen part of the room differs.\n")
    W("| visible similarity | hidden difference | n | acoustic correct | 95% CI |")
    W("|---|---|---|---|---|")
    for name, k in (("high", "high", ), ("high", "low")):
        pass
    for lbl, k in (("high", k_hi), ("low", k_lo)):
        if k.sum() >= 5:
            lo, hi = wilson(int(ok[k].sum()), int(k.sum()))
            W(f"| high | {lbl} | {int(k.sum())} | **{100*ok[k].mean():.1f}%** | "
              f"[{lo:.1f}, {hi:.1f}] |")
            js[f"decisive_{lbl}"] = dict(n=int(k.sum()), acc=float(ok[k].mean()),
                                         ci=[lo, hi])
    if k_hi.sum() >= 5 and k_lo.sum() >= 5:
        diff = 100 * (ok[k_hi].mean() - ok[k_lo].mean())
        W(f"\nDifference: **{diff:+.1f} points**. A clearly positive value is the "
          f"evidence the strong claim needs; a value near zero would mean the "
          f"paper should claim only that acoustic consistency discriminates "
          f"visually plausible hypotheses, without asserting where the "
          f"information comes from.\n")
        js["decisive_diff"] = float(diff)

    # ---- does it replicate on the other backbones? ------------------------
    W("\n## Replication across backbones\n")
    W("The geometry is a property of the room, so the effect should not depend "
      "on which visual model proposed the pair.\n")
    W("| backbone | pairs | hidden diff low | hidden diff high | difference |")
    W("|---|---|---|---|---|")
    js["replication"] = {}
    for t in [args.backbone] + list(args.also):
        try:
            r2 = collect(t)
        except FileNotFoundError:
            continue
        if len(r2) < 30:
            continue
        h2 = np.array([r["hid_diff"] for r in r2])
        o2 = np.array([r["ac_ok"] for r in r2])
        e2 = np.quantile(h2, [0, 1 / 3, 2 / 3, 1]); e2[0] -= 1e-9; e2[-1] += 1e-9
        lo_k, hi_k = h2 <= e2[1], h2 > e2[2]
        W(f"| {t} | {len(r2)} | {100*o2[lo_k].mean():.1f}% | "
          f"{100*o2[hi_k].mean():.1f}% | "
          f"**{100*(o2[hi_k].mean()-o2[lo_k].mean()):+.1f}** |")
        js["replication"][t] = dict(n=len(r2), low=float(o2[lo_k].mean()),
                                    high=float(o2[hi_k].mean()))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    lim = np.nanmax(np.abs(G - 50)) if np.isfinite(G).any() else 1
    im = axes[0].imshow(G, cmap="RdBu_r", vmin=50 - lim, vmax=50 + lim, origin="lower")
    for a in range(args.n_tiles):
        for b in range(args.n_tiles):
            if np.isfinite(G[a, b]):
                axes[0].text(b, a, f"{G[a,b]:.0f}%\nn={int(N[a,b])}",
                             ha="center", va="center", fontsize=9)
    axes[0].set_xticks(range(args.n_tiles)); axes[0].set_xticklabels(labs)
    axes[0].set_yticks(range(args.n_tiles)); axes[0].set_yticklabels(labs)
    axes[0].set_xlabel("visible similarity (camera cone)")
    axes[0].set_ylabel("hidden difference (outside the cone)")
    axes[0].set_title("Acoustic pairwise accuracy, chance 50%")
    fig.colorbar(im, ax=axes[0], label="%")
    e3 = np.quantile(hd, np.linspace(0, 1, 6)); e3[0] -= 1e-9; e3[-1] += 1e-9
    cx, cy, ce = [], [], []
    for i in range(5):
        k = (hd > e3[i]) & (hd <= e3[i + 1])
        if k.sum() < 5:
            continue
        lo, hi = wilson(int(ok[k].sum()), int(k.sum()))
        cx.append(float(np.median(hd[k]))); cy.append(100 * ok[k].mean())
        ce.append((100 * ok[k].mean() - lo, hi - 100 * ok[k].mean()))
    axes[1].axhline(50, color="0.6", ls="--", lw=1, label="chance")
    axes[1].errorbar(cx, cy, yerr=np.array(ce).T, fmt="o-", color="#2962ff", capsize=3)
    axes[1].set_xlabel("hidden geometry difference (m, mean over unseen bins)")
    axes[1].set_ylabel("acoustic pairwise accuracy (%)")
    axes[1].set_title("Accuracy against unseen-geometry difference")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"C_beyond_fov{'' if args.condition=='raw_scan_open' else '_matched'}.png", dpi=150)
    plt.close(fig)

    sfx = "" if args.condition == "raw_scan_open" else "_matched"
    p = args.out_dir / f"beyond_fov{sfx}.md"
    p.write_text("\n".join(out) + "\n")
    (args.out_dir / f"beyond_fov{sfx}.json").write_text(
        json.dumps(dict(backbone=args.backbone, results=js, provenance=stamp()),
                   indent=2, default=str))
    print("\n".join(out))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
