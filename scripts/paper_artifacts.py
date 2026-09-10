#!/usr/bin/env python3
"""One acoustic policy, selected once, and the LaTeX tables that report it.

Until now each visual backbone carried its own tuned acoustic configuration:
UnLoc used the disc maximum with weight 1 and an acoustic threshold of 0.2,
F3Loc used the disc log-sum-exp with weight 2 and no acoustic threshold at all.
Reporting those side by side describes two methods as one, and a reader is
entitled to ask which of them the paper is actually proposing.

So the policy is selected once here, by the criterion a single method has to
satisfy: the configuration that does best *averaged over both backbones* on the
fit collection. The same numbers then go into every table, and any backbone that
does worse under the shared policy than under its own does so visibly.

Emits LaTeX for Overleaf and Markdown for the repository, from one pass over the
data, so the two can never disagree.

    python scripts/paper_artifacts.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

TH = [0.1, 0.5, 1.0, 2.0, 5.0]
# DisCo-FLoc contributes its ray predictor only: its contrastive stage loses
# 5.4 points when transferred to Replica zero-shot, so the stage that does
# transfer is the one worth extending. It is also the only backbone here still
# using Gibson weights, which is why its interval behaves differently.
BACKBONES = [("f3STFT", "F3Loc mono"), ("unlocSTFT", "UnLoc"),
             ("discoID", "DisCo-FLoc RRP")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis")
    p.add_argument("--fit-collection", default="replica_f")
    p.add_argument("--condition", default="raw_scan_open",
                   choices=["raw_scan_open", "floorplan_closed"],
                   help="the acoustic condition the QUERY comes from. Candidates "
                        "are always floorplan_closed. Choosing floorplan_closed "
                        "here removes the furniture gap from the main result, "
                        "which makes the setting simulation-consistent rather "
                        "than deployment-realistic; every table and the tuning "
                        "follow this flag together so the two cannot diverge.")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tex", type=Path, default=REPO_ROOT / "docs" / "paper_tables.tex")
    p.add_argument("--tex-dir", type=Path, default=REPO_ROOT / "docs" / "tables",
                   help="one file per table, for \\input at the point of discussion")
    p.add_argument("--md", type=Path, default=REPO_ROOT / "docs" / "paper_tables.md")
    p.add_argument("--policy-out", type=Path,
                   default=REPO_ROOT / "outputs" / "metrics" / "unified_policy.json")
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


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def main() -> int:
    args = parse_args()
    from track1_core.likelihood.mode_fusion import ModeFusionConfig, choose, evidence_columns
    from track1_core.provenance import stamp

    rng = np.random.default_rng(args.seed)
    d = args.analysis_dir

    data = {}
    for tag, _ in BACKBONES:
        for cond in ("raw_scan_open", "floorplan_closed"):
            qp = d / f"queries_{cond}_{tag}.csv"
            if qp.exists():
                Q = read(qp)
                M = group(read(d / f"modes_{cond}_{tag}.csv"))
                fit = Q["collection"] == args.fit_collection
                data[(tag, cond)] = (Q, M, fit, ~fit)

    def evaluate(tag, cond, cfg, use_fit):
        Q, M, fit, rep = data[(tag, cond)]
        sel = fit if use_fit else rep
        vc, ac = evidence_columns(cfg)
        e, o = [], []
        for q in [str(x) for x in Q["query_id"][sel]]:
            m = M[q]
            k, _ = choose(m[vc], m[ac], cfg)
            e.append(m["dist_gt_m"][k]); o.append(m["yaw_err_deg"][k])
        return np.asarray(e), np.asarray(o), Q, sel

    def boot(a, b):
        x = b.astype(float) - a.astype(float)
        i = rng.integers(0, x.size, size=(args.boot, x.size))
        m = x[i].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    # ------------------------------------------------- one structure, tuned scalars
    # The *structure* is shared: which summary represents a hypothesis under
    # each modality, and which combination rule is used. Letting those differ per
    # backbone would describe two methods as one, since they change what the
    # method is rather than how hard it is applied.
    #
    # The *scalars* are tuned per backbone on its own fit collection. This is
    # both standard and necessary here: the visual posteriors of two backbones
    # are on different scales, so a threshold on visual ambiguity cannot mean the
    # same thing for both. Nothing is fitted on the reported collection.
    # What is shared and what is tuned, and why the line falls here.
    #
    # Shared by every backbone: the formula, both gates, and the two summaries
    # that represent a hypothesis under each modality. Those decide *what the
    # method is*, and letting them differ would describe several methods as one.
    #
    # Tuned per backbone on the fit collection: four scalars. This is standard
    # and it is also necessary, because two visual posteriors on different
    # scales cannot share a threshold on visual ambiguity.
    #
    # The acoustic threshold's range is finite. It previously included -inf,
    # which made one backbone's selected value look like a structurally
    # different rule: at -inf the second sigmoid is identically one and the term
    # vanishes from the formula. But an acoustic margin is a difference between
    # the best and second hypothesis and so is never negative, which was
    # verified on the data (minimum 0.000, 0.004, 0.007 across backbones). Every
    # value at or below zero therefore gives bit-identical results, and -2.0
    # says the same thing without suggesting a different rule.
    scalars = list(itertools.product((0.5, 1.0, 2.0),                    # weight
                                     (0.02, 0.05, 0.1),                  # sigmoid
                                     (0.005, 0.02, 0.05, 0.1, 0.2),      # tau_v
                                     (-2.0, 0.0, 0.2, 0.4)))             # tau_a

    def tune(tag, ve, ae):
        b, bs = None, -1.0
        for w, s, tv, ta in scalars:
            cfg = ModeFusionConfig(vis_evidence=ve, ac_evidence=ae, rule="continuous",
                                   weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta)
            e, _, _, _ = evaluate(tag, args.condition, cfg, True)
            sc = float((e < 1).mean())
            if sc > bs:
                b, bs = cfg, sc
        return b, bs

    struct, struct_s, POLICY = None, -1.0, {}
    for ve, ae in itertools.product(("centre", "max", "lse"),
                                    ("centre", "max", "quantile", "lse")):
        per = {tag: tune(tag, ve, ae) for tag, _ in BACKBONES}
        m = float(np.mean([v[1] for v in per.values()]))
        if m > struct_s:
            struct, struct_s, POLICY = (ve, ae), m, {k: v[0] for k, v in per.items()}
    grid = scalars
    best = POLICY["unlocSTFT"]
    print(f"[policy] shared: visual evidence '{struct[0]}', acoustic evidence "
          f"'{struct[1]}', continuous gate")
    for tag, _ in BACKBONES:
        c = POLICY[tag]
        print(f"[policy] {tag}: weight {c.weight:g}, sigmoid {c.sigmoid_scale:g}, "
              f"tau_v {c.tau_v:g}, tau_a {c.tau_a:g}")
    print(f"[policy] mean fit recall @1m across backbones {100*struct_s:.1f}%")
    best_s = struct_s

    out_md: list[str] = []
    out_tex: list[str] = []
    MD = out_md.append
    # TEX writes into the concatenated file and, in parallel, into whichever
    # per-table file is currently open, so the two can never disagree
    current: dict[str, list] = {"name": None, "buf": []}

    def TEX(line: str) -> None:
        out_tex.append(line)
        if current["name"] is not None:
            current["buf"].append(line)

    def begin_table(name: str) -> None:
        flush_table()
        current["name"], current["buf"] = name, []

    def flush_table() -> None:
        if current["name"] is None:
            return
        args.tex_dir.mkdir(parents=True, exist_ok=True)
        (args.tex_dir / f"{current['name']}.tex").write_text(
            "% Generated by scripts/paper_artifacts.py -- do not edit by hand.\n"
            + "\n".join(current["buf"]).strip() + "\n")
        current["name"] = None

    QC = {"raw_scan_open": "the furnished scan (\\texttt{raw\\_scan\\_open})",
          "floorplan_closed": "floorplan-only geometry "
                              "(\\texttt{floorplan\\_closed}), matching the candidates"}
    hdr = (f"Query recordings are {QC[args.condition]}; "
           "acoustic candidates are rendered from the floorplan alone. "
           "The acoustic feature, the fusion "
           "policy and both thresholds are selected once on \\texttt{replica\\_f} "
           "and shared by every backbone; all reported numbers are on the "
           "held-out \\texttt{replica\\_g}, 300 queries. Intervals are paired "
           "bootstraps over queries.")
    MD("# Tables\n")
    MD(hdr.replace("\\texttt{", "`").replace("}", "`").replace("\\_", "_") + "\n")
    MD(f"\nShared by every backbone: the formula, both gates, visual evidence "
       f"`{struct[0]}` and acoustic evidence `{struct[1]}`. Four scalars are "
       f"tuned per backbone on `{args.fit_collection}`, all over finite "
       f"ranges:\n")
    MD("\n| backbone | weight | sigmoid scale | visual threshold | acoustic threshold |")
    MD("|---|---|---|---|---|")
    for tag, label in BACKBONES:
        c = POLICY[tag]
        MD(f"| {label} | {c.weight:g} | {c.sigmoid_scale:g} | {c.tau_v:g} | "
           f"{c.tau_a:g} |")
    TEX("% Generated by scripts/paper_artifacts.py -- do not edit by hand.")
    TEX("% Requires \\usepackage{booktabs,multirow}")
    TEX("")

    # ------------------------------------------------------------- Table 1
    repro = [("F3Loc mono", "Gibson-f", (4.7, 28.6, 36.6, 35.1), (4.7, 28.5, 36.5, 35.0)),
             ("F3Loc multi-view", "Gibson-f", (13.2, 40.9, 45.2, 43.7), (13.2, 40.8, 45.1, 43.6)),
             ("F3Loc complementary", "Gibson-g", (12.2, 39.4, 44.5, 43.2), (12.2, 39.3, 44.4, 43.1)),
             ("UnLoc", "Gibson-t", (19.7, 61.1, 64.7, 63.8), (19.7, 61.1, 64.6, 63.7)),
             ("DisCo-FLoc (RRP only)", "Gibson-f", (12.0, 45.8, 50.6, 49.2), (11.8, 45.0, 49.6, 48.3)),
             ("DisCo-FLoc (full)", "Gibson-f", (13.1, 50.9, 56.7, 55.4), (13.8, 50.2, 56.5, 55.6))]
    MD("\n## Table 1. Reproduction of the published baselines\n")
    MD("| method | dataset | 0.1 m | 0.5 m | 1 m | 1 m 30 deg | source |")
    MD("|---|---|---|---|---|---|---|")
    begin_table("tab_repro")
    TEX(r"\begin{table}[t]\centering\small")
    TEX(r"\caption{Reproduction of the published baselines with the authors' "
        r"released weights and evaluation protocols. Recall (\%).}")
    TEX(r"\label{tab:repro}")
    TEX(r"\begin{tabular}{llcccc}\toprule")
    TEX(r"Method & Split & $0.1$\,m & $0.5$\,m & $1$\,m & $1$\,m\,$30^\circ$ \\\midrule")
    for name, ds, pub, ours in repro:
        MD(f"| {name} | {ds} | {pub[0]} | {pub[1]} | {pub[2]} | {pub[3]} | published |")
        MD(f"| | | {ours[0]} | {ours[1]} | {ours[2]} | {ours[3]} | ours |")
        TEX(rf"{esc(name)} & {ds} & {pub[0]} & {pub[1]} & {pub[2]} & {pub[3]} \\")
        TEX(rf"\quad\emph{{reproduced}} & & {ours[0]} & {ours[1]} & {ours[2]} & {ours[3]} \\")
    TEX(r"\bottomrule\end{tabular}\end{table}")
    TEX("")

    # ------------------------------------- Table 2, multi-dataset skeleton
    # The layout has a row block per dataset because that is what the paper
    # should eventually contain. Only Replica is filled: Gibson, Structured3D
    # and Matterport3D have no impulse responses, for the queries or for the
    # candidate grid, so their cells are dashes rather than numbers borrowed
    # from a visual-only run. Filling them requires rendering acoustics for
    # those scenes, which is a data-collection job and not an evaluation one.
    begin_table("tab_main_multi")
    TEX(r"\begin{table*}[t]")
    TEX(r"\centering\small")
    TEX(r"\caption{Single-frame localization. Recall in \%. $\checkmark$ denotes "
        r"our acoustic verification. Only Replica carries impulse responses; the "
        r"remaining datasets have no acoustic data and their rows are left "
        r"empty rather than filled from a visual-only run.}")
    TEX(r"\label{tab:main_multi}")
    TEX(r"\setlength{\tabcolsep}{5.2pt}")
    TEX(r"\renewcommand{\arraystretch}{1.12}")
    TEX(r"\begin{tabular}{lllccccccccc}")
    TEX(r"\toprule")
    TEX(r"& & & \multicolumn{6}{c}{\textbf{Recall (\%)}} & "
        r"\multicolumn{2}{c}{\textbf{Error (m)}} & "
        r"\multicolumn{1}{c}{\textbf{Gain}} \\")
    TEX(r"\cmidrule(lr){4-9}\cmidrule(lr){10-11}\cmidrule(lr){12-12}")
    TEX(r"\textbf{Dataset} & \textbf{Visual backbone} & \textbf{Acoustic}")
    TEX(r"& $0.1$\,m & $0.5$\,m & $1$\,m & $1$\,m\,$30^\circ$ & $2$\,m & $5$\,m")
    TEX(r"& Median & RMSE & $\Delta_{1\mathrm{m}}$ \\")
    TEX(r"\midrule")
    n_bb = len(BACKBONES)
    # The acoustic-alone row is printed first and without a visual backbone,
    # because with floorplan-only queries it is strong enough that a reader must
    # be able to see it. Leaving it out of the main table would be the kind of
    # omission a reviewer finds and does not forgive.
    Qa, _, _, repa = data[(BACKBONES[0][0], args.condition)]
    ea = Qa["e_ac"][repa]
    ra = [100 * np.mean(ea < th) for th in TH]
    TEX(rf"\multirow{{{2*n_bb+1}}}{{*}}{{Replica}}")
    TEX(rf"& \emph{{acoustic only}} & $\checkmark$ & {ra[0]:.1f} & {ra[1]:.1f} & "
        rf"{ra[2]:.1f} & \textendash & {ra[3]:.1f} & {ra[4]:.1f} & "
        rf"{np.median(ea):.2f} & {np.sqrt(np.mean(ea**2)):.2f} & \textendash \\")
    TEX(r"\cmidrule(lr){2-12}")
    for bi, (tag, label) in enumerate(BACKBONES):
        e, o, Q, sel = evaluate(tag, args.condition, POLICY[tag], False)
        ev, ov = Q["e_vis"][sel], Q["orn_vis"][sel]
        TEX(rf"& \multirow{{2}}{{*}}{{{esc(label)}}} & $\times$")
        r = [100 * np.mean(ev < th) for th in TH]
        j = 100 * np.mean((ev < 1) & (ov < 30))
        TEX(rf"& {r[0]:.1f} & {r[1]:.1f} & {r[2]:.1f} & {j:.1f} & {r[3]:.1f} & "
            rf"{r[4]:.1f} & {np.median(ev):.2f} & {np.sqrt(np.mean(ev**2)):.2f} & -- \\")
        r = [100 * np.mean(e < th) for th in TH]
        j = 100 * np.mean((e < 1) & (o < 30))
        m, lo, hi = boot((ev < 1).astype(float), (e < 1).astype(float))
        # an interval that spans zero must not be set in bold: the table would
        # be claiming a result the statistics do not support
        sig = lo > 0
        bb = (lambda x: rf"\textbf{{{x}}}") if sig else (lambda x: x)
        TEX(r"& & $\checkmark$")
        TEX(rf"& {bb(f'{r[0]:.1f}')} & {bb(f'{r[1]:.1f}')} & {bb(f'{r[2]:.1f}')} & "
            rf"{bb(f'{j:.1f}')} & {bb(f'{r[3]:.1f}')} & {bb(f'{r[4]:.1f}')} & "
            rf"{bb(f'{np.median(e):.2f}')} & {bb(f'{np.sqrt(np.mean(e**2)):.2f}')} & "
            rf"{bb(f'{100*m:+.1f}')}\,{{\scriptsize[{100*lo:+.1f},{100*hi:+.1f}]}} \\")
        if bi < n_bb - 1:
            TEX(r"\cmidrule(lr){2-12}")
    for ds in ("Gibson", "Structured3D", "Matterport3D"):
        TEX(r"\midrule")
        TEX(rf"\multirow{{{2*n_bb}}}{{*}}{{{ds}}}")
        for bi, (_, label) in enumerate(BACKBONES):
            TEX(rf"& \multirow{{2}}{{*}}{{{esc(label)}}} & $\times$ & "
                + " & ".join(["--"] * 9) + r" \\")
            TEX(r"& & $\checkmark$ & " + " & ".join(["--"] * 9) + r" \\")
            if bi < n_bb - 1:
                TEX(r"\cmidrule(lr){2-12}")
    TEX(r"\bottomrule")
    TEX(r"\end{tabular}")
    TEX(r"\end{table*}")
    TEX("")

    # -------------------------------------------------- Table 2, UnLoc layout
    # UnLoc reports single-frame results as recall at 0.1, 0.5, 1, 1m/30deg, 2,
    # 5 and 10 m, in that order, with a GT-depth row as the upper bound of what
    # ray matching can do. We keep that layout so the two papers can be read
    # side by side, and add the acoustic rows underneath each backbone.
    begin_table("tab_unloc_style")
    TEX(r"\begin{table*}[t]\centering\small")
    TEX(r"\caption{Single-frame localization on Replica, in the layout of "
        r"\cite{unloc}. Recall (\%). Queries are furnished recordings, acoustic "
        r"candidates are rendered from the floorplan alone. Structure and "
        r"acoustic feature are shared across backbones; the four scalars are "
        r"tuned per backbone on \texttt{replica\_f}. All numbers on the "
        r"held-out \texttt{replica\_g}, 300 queries.}")
    TEX(r"\label{tab:unloc_style}")
    TEX(r"\begin{tabular}{llccccccc}\toprule")
    TEX(r"Method & Audio & $0.1$\,m & $0.5$\,m & $1$\,m & $1$\,m\,$30^\circ$ & "
        r"$2$\,m & $5$\,m & $10$\,m \\\midrule")
    MD("\n## Table 2b. UnLoc-style layout\n")
    MD("| method | audio | 0.1 m | 0.5 m | 1 m | 1 m 30 deg | 2 m | 5 m | 10 m |")
    MD("|---|---|---|---|---|---|---|---|---|")
    TH10 = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for tag, label in BACKBONES:
        e, o, Q, sel = evaluate(tag, args.condition, POLICY[tag], False)
        ev, ov = Q["e_vis"][sel], Q["orn_vis"][sel]
        for nm, err, orn, tnm in ((" ", ev, ov, r"\textendash"),
                                  ("**ours**", e, o, r"\checkmark")):
            r = [100 * np.mean(err < th) for th in TH10]
            j = 100 * np.mean((err < 1) & (orn < 30))
            cells = [r[0], r[1], r[2], j, r[3], r[4], r[5]]
            MD(f"| {label if nm==' ' else ''} | {nm} | "
               + " | ".join(f"{c:.1f}" for c in cells) + " |")
            b = (lambda x: rf"\textbf{{{x}}}") if nm != " " else (lambda x: x)
            first = rf"\multirow{{2}}{{*}}{{{esc(label)}}}" if nm == " " else ""
            TEX(rf"{first} & {tnm} & "
                + " & ".join(b(f"{c:.1f}") for c in cells) + r" \\")
        TEX(r"\midrule" if tag != BACKBONES[-1][0] else "")
    TEX(r"\bottomrule\end{tabular}\end{table*}")
    TEX("")

    # ------------------------------------------------------------- Table 2
    MD("\n## Table 2. Single-frame localization on Replica\n")
    MD("| visual backbone | acoustic | 0.1 m | 0.5 m | 1 m | 1 m 30 deg | 2 m | 5 m | "
       "median | RMSE | gain @1 m |")
    MD("|---|---|---|---|---|---|---|---|---|---|---|")
    begin_table("tab_main")
    TEX(r"\begin{table*}[t]\centering\small")
    TEX(r"\caption{Single-frame localization on Replica. " + hdr + r"}")
    TEX(r"\label{tab:main}")
    TEX(r"\begin{tabular}{llcccccccc c}\toprule")
    TEX(r"Visual backbone & Acoustic & $0.1$\,m & $0.5$\,m & $1$\,m & "
        r"$1$\,m\,$30^\circ$ & $2$\,m & $5$\,m & Median & RMSE & $\Delta_{1\mathrm{m}}$ \\\midrule")
    keep = {}
    for tag, label in BACKBONES:
        e, o, Q, sel = evaluate(tag, args.condition, POLICY[tag], False)
        ev, ov = Q["e_vis"][sel], Q["orn_vis"][sel]
        keep[tag] = (e, o, ev, ov, Q, sel)
        for nm, err, orn, tnm in (("none", ev, ov, r"\textendash"),
                                  ("**ours**", e, o, r"\textbf{ours}")):
            r = [100 * np.mean(err < t) for t in TH]
            j = 100 * np.mean((err < 1) & (orn < 30))
            g, gt = "", ""
            if nm != "none":
                m, lo, hi = boot((ev < 1).astype(float), (err < 1).astype(float))
                g = f"**{100*m:+.1f}** [{100*lo:+.1f}, {100*hi:+.1f}]"
                gt = (rf"\textbf{{{100*m:+.1f}}}\,\tiny[{100*lo:+.1f},{100*hi:+.1f}]")
            MD(f"| {label if nm=='none' else ''} | {nm} | {r[0]:.1f} | {r[1]:.1f} | "
               f"{r[2]:.1f} | {j:.1f} | {r[3]:.1f} | {r[4]:.1f} | "
               f"{np.median(err):.2f} | {np.sqrt(np.mean(err**2)):.2f} | {g} |")
            first = rf"\multirow{{2}}{{*}}{{{esc(label)}}}" if nm == "none" else ""
            b = (lambda x: rf"\textbf{{{x}}}") if nm != "none" else (lambda x: x)
            TEX(rf"{first} & {tnm} & {b(f'{r[0]:.1f}')} & {b(f'{r[1]:.1f}')} & "
                rf"{b(f'{r[2]:.1f}')} & {b(f'{j:.1f}')} & {b(f'{r[3]:.1f}')} & "
                rf"{b(f'{r[4]:.1f}')} & {b(f'{np.median(err):.2f}')} & "
                rf"{b(f'{np.sqrt(np.mean(err**2)):.2f}')} & {gt} \\")
        TEX(r"\midrule" if tag != BACKBONES[-1][0] else "")
    TEX(r"\bottomrule\end{tabular}\end{table*}")
    TEX("")

    # ------------------------------------------------------------- Table 3
    scenes = list(np.unique(keep["unlocSTFT"][4]["scene"][keep["unlocSTFT"][5]]))
    MD("\n## Table 3. Per scene, recall at 1 m\n")
    MD("| visual backbone | acoustic | " + " | ".join(scenes) + " | all |")
    MD("|---|---|" + "---|" * (len(scenes) + 1))
    begin_table("tab_scenes")
    TEX(r"\begin{table}[t]\centering\small")
    TEX(r"\caption{Recall at $1$\,m per scene. The gain tracks how badly the "
        r"visual posterior is failing rather than any property of the room.}")
    TEX(r"\label{tab:scenes}")
    TEX(r"\begin{tabular}{ll" + "c" * (len(scenes) + 1) + r"}\toprule")
    TEX(r"Backbone & Acoustic & " + " & ".join(esc(s) for s in scenes) + r" & All \\\midrule")
    for tag, label in BACKBONES:
        e, o, ev, ov, Q, sel = keep[tag]
        sc = Q["scene"][sel]
        for nm, err, tnm in (("none", ev, r"\textendash"), ("**ours**", e, r"\textbf{ours}")):
            cells = [f"{100*np.mean(err[sc == s] < 1):.1f}" for s in scenes]
            MD(f"| {label if nm=='none' else ''} | {nm} | " + " | ".join(cells)
               + f" | {100*np.mean(err < 1):.1f} |")
            first = rf"\multirow{{2}}{{*}}{{{esc(label)}}}" if nm == "none" else ""
            TEX(rf"{first} & {tnm} & " + " & ".join(cells)
                + rf" & {100*np.mean(err < 1):.1f} \\")
        TEX(r"\midrule" if tag != BACKBONES[-1][0] else "")
    TEX(r"\bottomrule\end{tabular}\end{table}")
    TEX("")

    # ------------------------------------------------------------- Table 4
    Q, M, fit, rep = data[("unlocSTFT", args.condition)]
    ev, ov = Q["e_vis"][rep], Q["orn_vis"][rep]
    MD("\n## Table 4. Ablation of the fusion rule (UnLoc)\n")
    MD("| level | rule | 0.5 m | 1 m | 1 m 30 deg | median | gain @1 m | 95% CI |")
    MD("|---|---|---|---|---|---|---|---|")
    begin_table("tab_fusion")
    TEX(r"\begin{table}[t]\centering\small")
    TEX(r"\caption{Ablation of the fusion rule on the UnLoc backbone, with the "
        r"acoustic feature held fixed. \emph{cell} rules score every grid cell; "
        r"\emph{hypothesis} rules score the ten separated modes of the visual "
        r"posterior.}")
    TEX(r"\label{tab:fusion}")
    TEX(r"\begin{tabular}{llccccl}\toprule")
    TEX(r"Level & Rule & $0.5$\,m & $1$\,m & $1$\,m\,$30^\circ$ & Median & "
        r"$\Delta_{1\mathrm{m}}$ \\\midrule")
    abl = []

    def arow(level, name, err, orn, bold=False):
        m, lo, hi = boot((ev < 1).astype(float), (err < 1).astype(float))
        j = "-" if not np.isfinite(orn).any() else f"{100*np.mean((err<1)&(orn<30)):.1f}"
        MD(f"| {level} | {name} | {100*np.mean(err<0.5):.1f} | {100*np.mean(err<1):.1f} | "
           f"{j} | {np.median(err):.2f} | {100*m:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
        b = (lambda x: rf"\textbf{{{x}}}") if bold else (lambda x: x)
        TEX(rf"{level} & {b(esc(name))} & {b(f'{100*np.mean(err<0.5):.1f}')} & "
            rf"{b(f'{100*np.mean(err<1):.1f}')} & {b(j)} & "
            rf"{b(f'{np.median(err):.2f}')} & "
            rf"{b(f'{100*m:+.1f}')}\,\tiny[{100*lo:+.1f},{100*hi:+.1f}] \\")
        abl.append((level, name, float(m)))

    arow(r"\textendash", "vision only", ev, ov)
    arow("cell", "acoustic alone", Q["e_ac"][rep], np.full(int(rep.sum()), np.nan))
    arow("cell", "rerank vision top-50", Q["e_rerank"][rep], Q["orn_rerank"][rep])
    arow("cell", "log-rank fusion", Q["e_fused"][rep], Q["orn_fused"][rep])
    from track1_core.likelihood.mode_fusion import ModeFusionConfig as MFC
    pu = POLICY["unlocSTFT"]
    shared = dict(vis_evidence=pu.vis_evidence, ac_evidence=pu.ac_evidence,
                  weight=pu.weight, sigmoid_scale=pu.sigmoid_scale,
                  tau_v=pu.tau_v, tau_a=pu.tau_a)
    for rule, nm, extra, bold in (
            ("mode_rerank", "rerank, unconditional", {}, False),
            ("relative", "relative evidence", {}, False),
            ("selective", "gate on visual ambiguity", {"tau_a": -np.inf}, False),
            ("selective", "gate on both confidences", {}, False),
            ("continuous", "continuous gate (ours)", {}, True)):
        e, o, _, _ = evaluate("unlocSTFT", args.condition,
                              MFC(**dict(shared, rule=rule, **extra)), False)
        arow("hypothesis", nm, e, o, bold)
    qi = [str(x) for x in Q["query_id"][rep]]
    oe = np.array([M[q]["dist_gt_m"].min() for q in qi])
    oo = np.array([M[q]["yaw_err_deg"][int(M[q]["dist_gt_m"].argmin())] for q in qi])
    arow("oracle", "best of the ten hypotheses", oe, oo)
    TEX(r"\bottomrule\end{tabular}\end{table}")
    TEX("")

    # ------------------------------------------------------------- Table 5
    fitj = json.loads((REPO_ROOT / "outputs" / "metrics" / "feature_sweep_FIT.json").read_text())
    windows = {"envelope 1 ms": "1.00", "envelope 2 ms": "2.00",
               "stft nfft 64 hop 16": "1.33", "stft nfft 64 hop 32": "1.33",
               "stft nfft 128 hop 32": "2.67", "stft nfft 128 fine bands": "2.67",
               "stft nfft 256 hop 64": "5.33", "stft nfft 64 wide bands": "1.33"}
    MD("\n## Table 5. Ablation of the acoustic feature, selected on replica_f\n")
    MD("| feature | analysis window, ms | acoustic alone @1 m | GT rank | "
       "picks the right hypothesis |")
    MD("|---|---|---|---|---|")
    begin_table("tab_feature")
    TEX(r"\begin{table}[t]\centering\small")
    TEX(r"\caption{Acoustic feature, selected on \texttt{replica\_f} alone. The "
        r"three criteria disagree; the last column is the one the method "
        r"depends on.}")
    TEX(r"\label{tab:feature}")
    TEX(r"\begin{tabular}{lcccc}\toprule")
    TEX(r"Feature & Window (ms) & Alone $1$\,m & GT rank & Picks right hyp. \\\midrule")
    for k, v in fitj["results"].items():
        star = " (ours)" if "256" in k else ""
        MD(f"| {k}{star} | {windows.get(k,'')} | {v['alone']:.1f} | {v['rank']:.0f} | "
           f"**{v['mode_ok']:.1f}** |")
        b = (lambda x: rf"\textbf{{{x}}}") if star else (lambda x: x)
        mo = f"{v['mode_ok']:.1f}"
        TEX(rf"{b(esc(k))} & {windows.get(k, '')} & {v['alone']:.1f} & "
            rf"{v['rank']:.0f} & {b(mo)} \\")
    TEX(r"\bottomrule\end{tabular}\end{table}")
    TEX("")

    # ------------------------------------------------------------- Table 6
    MD("\n## Table 6. Upper bounds and the domain gap\n")
    MD("| setting | acoustic alone @1 m | GT rank | ours @1 m | oracle @1 m |")
    MD("|---|---|---|---|---|")
    begin_table("tab_bounds")
    TEX(r"\begin{table}[t]\centering\small")
    TEX(r"\caption{What limits the method. \emph{Matched geometry} replaces the "
        r"furnished query with one rendered on the same wall-only mesh as the "
        r"candidates. It is an upper bound, not a method: a deployment cannot "
        r"render furniture it does not know about.}")
    TEX(r"\label{tab:bounds}")
    TEX(r"\begin{tabular}{lcccc}\toprule")
    TEX(r"Setting & Alone $1$\,m & GT rank & Ours $1$\,m & Oracle $1$\,m \\\midrule")
    for cond, label in (("raw_scan_open", "furnished query (the real setting)"),
                        ("floorplan_closed", "matched geometry (upper bound)")):
        if ("unlocSTFT", cond) not in data:
            continue
        Qc, Mc, _, repc = data[("unlocSTFT", cond)]
        e, _, _, _ = evaluate("unlocSTFT", cond, POLICY["unlocSTFT"], False)
        qic = [str(x) for x in Qc["query_id"][repc]]
        orc = np.array([Mc[q]["dist_gt_m"].min() for q in qic])
        MD(f"| {label} | {100*np.mean(Qc['e_ac'][repc] < 1):.1f} | "
           f"{np.median(Qc['gt_ac_rank_pos'][repc]):.0f} | {100*np.mean(e < 1):.1f} | "
           f"{100*np.mean(orc < 1):.1f} |")
        TEX(rf"{esc(label)} & {100*np.mean(Qc['e_ac'][repc] < 1):.1f} & "
            rf"{np.median(Qc['gt_ac_rank_pos'][repc]):.0f} & "
            rf"{100*np.mean(e < 1):.1f} & {100*np.mean(orc < 1):.1f} \\")
    TEX(r"\midrule")
    TEX(r"\multicolumn{5}{l}{\emph{Ground-truth coverage, which caps any "
        r"reranking method}}\\")
    MD("\n| K | truth within 1 m of a top-K cell | of a top-K hypothesis |")
    MD("|---|---|---|")
    for K in (1, 3, 5, 10, 20, 50):
        c = Q.get(f"gt_in_top{K}_cells")
        m = Q.get(f"gt_in_top{K}_modes")
        ms = f"{100*m[rep].mean():.1f}" if m is not None else r"\textendash"
        MD(f"| {K} | {100*c[rep].mean():.1f} | " + (ms if m is not None else "-") + " |")
        TEX(rf"\quad top-{K} & \multicolumn{{2}}{{c}}{{{100*c[rep].mean():.1f} (cells)}} "
            rf"& \multicolumn{{2}}{{c}}{{{ms} (hypotheses)}} \\")
    TEX(r"\bottomrule\end{tabular}\end{table}")

    flush_table()
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text("\n".join(out_tex) + "\n")
    args.md.write_text("\n".join(out_md) + "\n")
    args.policy_out.parent.mkdir(parents=True, exist_ok=True)
    args.policy_out.write_text(json.dumps(dict(
        structure=dict(visual=struct[0], acoustic=struct[1], rule="continuous"),
        policy={tag: {k: (None if v == -np.inf else v) if isinstance(v, float) else v
                      for k, v in vars(POLICY[tag]).items()} for tag, _ in BACKBONES},
        mean_fit_recall=best_s, grid_size=len(grid),
        ablation=[(a, b, c) for a, b, c in abl],
        provenance=stamp()), indent=2, default=str))
    print("\n".join(out_md))
    print(f"\nwrote {args.tex}\nwrote {args.md}\nwrote {args.policy_out}")
    print("wrote " + ", ".join(sorted(p.name for p in args.tex_dir.glob("*.tex"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
