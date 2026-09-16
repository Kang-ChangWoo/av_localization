"""Shared loading for the analysis folders under analysis/exp*.

Every experiment reads the same test tables of the headline protocol
(projection trained on the benchmark's own training rooms, fixed structure,
three scalars chosen on validation rooms, test rooms evaluated once) and the
configuration those rooms chose. This module turns one (benchmark, backbone)
into a list of per-sample records so the experiments only have to say what
they compute.

Definitions used by every experiment
    hypotheses   h_1..h_K, K = 10, spatial hypotheses after non-maximum
                 suppression at 1.5 m on pi(c) = max_o p_v(c, o); heading is
                 collapsed by max and plays no part in anything below
    v_k          log pi(h_k), the backbone's log posterior at the hypothesis
    alpha_k      the hypothesis' acoustic evidence alone: 0.9 quantile of the
                 projected acoustic score over the cells within 0.5 m of it
    m_v          v_(1) - v_(2), the visual ambiguity the gate reads
    correct      a hypothesis within 1 m of the truth
    sample       one query; every interval is a bootstrap over samples
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "feasible"))
from scripts.room_cv_eval import read, group          # noqa: E402

AN = ROOT / "outputs" / "analysis"
RES = ROOT / "feasible" / "results"
COND = {"replica": "raw_scan_open", "mp3d": "raw_scan_open", "s3d": "floorplan_closed"}
TEST_TAG = {("replica", "unloc"): "unloc_uproj", ("mp3d", "unloc"): "unloc_mp3d12proj", ("s3d", "unloc"): "unloc_s3dprojB",
            ("replica", "f3loc"): "f3loc_mono_proj", ("mp3d", "f3loc"): "f3loc_mono_mp3d12proj", ("s3d", "f3loc"): "f3loc_mono_s3dprojB",
            ("replica", "disco"): "disco_rrpprojR", ("mp3d", "disco"): "disco_rrp_mp3d12proj", ("s3d", "disco"): "disco_rrp_s3dprojB"}
DS = [("replica", "Replica"), ("mp3d", "Matterport3D"), ("s3d", "Structured3D")]
BB = [("f3loc", "F3Loc mono"), ("unloc", "UnLoc"), ("disco", "DisCo-FLoc RRP")]
NAME = dict(DS + BB)
C = {"vision": "#0072B2", "acoustic": "#E69F00", "fused": "#009E73", "bad": "#D55E00", "grey": "#8C8C8C", "oracle": "#7F7F7F"}


def config(ds, bb):
    """The rule the headline table used for this benchmark and backbone."""
    from track1_core.likelihood.mode_fusion import ModeFusionConfig
    sel = json.loads((RES / f"VAL_{ds}_fixed_indomain.json").read_text())["results"][bb]["selected"]
    w, s, tv, ta = sel["scalars"]
    return ModeFusionConfig(vis_evidence=sel["structure"][0], ac_evidence=sel["structure"][1], rule="continuous",
                            weight=w, sigmoid_scale=s, tau_v=tv, tau_a=ta,
                            ac_transform="standard" if sel["rule"] == "simple" else "relative")


def load_rows(ds, bb):
    """One record per test sample, or None when the tables are not on disk."""
    from track1_core.likelihood.mode_fusion import choose, evidence_columns
    tag = TEST_TAG[(ds, bb)]
    qp = AN / f"queries_{COND[ds]}_{tag}.csv"
    if not qp.exists():
        return None, None
    cfg = config(ds, bb); vc, ac = evidence_columns(cfg)
    Q = read(qp); M = group(read(AN / f"modes_{COND[ds]}_{tag}.csv"))
    rows = []
    for i, q in enumerate(Q["query_id"]):
        m = M[str(q)]; v = np.asarray(m[vc], float); a = np.asarray(m[ac], float); d = np.asarray(m["dist_gt_m"], float)
        k, acted = choose(v, a, cfg); kv = int(np.argmax(v)); top = np.sort(v)[::-1]
        rows.append(dict(scene=str(Q["scene"][i]), d=d, v=v, a=a, k_vis=kv, k_fused=int(k),
                         e_vis=float(d[kv]), e_ours=float(d[k]), e_ac=float(Q["e_ac"][i]),
                         m_v=float(top[0] - top[1]) if top.size > 1 else np.inf, acted=bool(acted),
                         n_cells=int(Q["n_cells"][i])))
    return rows, cfg


def quintile_of(values, bins=5):
    """Equal-count groups by rank (stable sort); 1 = smallest values."""
    order = np.argsort(np.asarray(values), kind="stable")
    q = np.empty(len(values), int)
    for i, idx in enumerate(np.array_split(order, bins)):
        q[idx] = i + 1
    return q


def boot_mean(x, n=5000, seed=0):
    """Mean and 95% bootstrap interval over samples."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if x.size == 0:
        return float("nan"), [float("nan"), float("nan")]
    b = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    return float(x.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_diff(a, b, n=5000, seed=0):
    """Mean of b - a with a paired bootstrap over samples."""
    return boot_mean(np.asarray(b, float) - np.asarray(a, float), n, seed)


def pct(x):
    return f"{100*x:.1f}"


def ci(m, lo_hi):
    return f"{100*m:.1f} [{100*lo_hi[0]:.1f}, {100*lo_hi[1]:.1f}]"


def write_md(target: Path, title: str, table_lines: list[str]):
    """Regenerate the table block of a markdown file, keeping the prose after it."""
    start, end = "<!-- tables:start -->", "<!-- tables:end -->"
    block = "\n".join(table_lines).rstrip() + "\n\n"
    if target.exists() and start in target.read_text() and end in target.read_text():
        t = target.read_text(); t = t[: t.index(start) + len(start)] + "\n" + block + t[t.index(end):]
    else:
        t = f"# {title}\n\n{start}\n{block}{end}\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(t)
