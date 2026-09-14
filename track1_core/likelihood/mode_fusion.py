"""Fusion at the level of spatial hypotheses rather than grid cells.

The cell-wise rule this replaces combines ``log`` rank percentiles over every
valid cell. Measured on 300 queries, among the fifty cells vision shortlists the
visual term spans 0.009 in log units and the acoustic term spans 0.67, a factor
of seventy. The visual posterior is therefore not participating in its own
fusion: rank saturates, because the shortlisted cells are all above the 99.9th
percentile of several thousand. The rule is a reranker wearing the clothes of a
weighted combination, and its failure mode follows directly, since nothing stops
a weak acoustic score from overruling a confident visual pick.

Three changes follow from that diagnosis, and each is a separate switch here so
the contribution of each can be measured.

**Hypotheses, not cells.** Vision's top fifty cells are usually two or three
*places* sampled many times each. Scoring places once each stops a broad mode
from outvoting a sharp one by sheer cell count.

**Evidence relative to the competing hypotheses.** The acoustic module is asked
"among the places vision proposes, which is most consistent", not "where am I in
the building". A global percentile answers the second question, which sound
cannot do here: alone it reaches 23.7% against vision's 49.7%.

**Audio applies only when both sides agree it should.** Vision must be ambiguous
*and* sound must be discriminative among these particular hypotheses. The
existing gate tests only the first, which is why it is a patch rather than part
of the rule.

Every function here takes per-hypothesis arrays for one query and returns the
index of the chosen hypothesis. The chosen hypothesis's own centre supplies the
final position and yaw, so sound never moves the answer to a place vision did
not propose, and never touches yaw, about which it has no information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..modes import logsumexp, relative_evidence, softmax


@dataclass(frozen=True)
class ModeFusionConfig:
    """Switches for the mode-level rules.

    vis_evidence   which summary of the visual posterior represents a mode:
                   'centre' the mode cell itself, 'max' the strongest cell in
                   its disc, 'lse' the log probability mass of the disc.
    ac_evidence    the same choice for the acoustic score. 'quantile' is the
                   robust version of 'max' and is the default because the
                   acoustic field is spatially rough.
    rule           'vision', 'mode_rerank', 'relative', 'selective',
                   'continuous', or 'contradiction'. See ``choose``.
    weight         how much relative acoustic evidence counts against the
                   standardised visual evidence, for the continuous rules.
    tau_v          audio is considered only when the visual log-odds between the
                   best and second hypothesis fall below this.
    tau_a          audio is applied only when its relative evidence for the best
                   hypothesis exceeds this.
    sigmoid_scale  softness of the continuous gate, in the same units as tau.
    lam            weight of the contradiction penalty.
    ac_transform   how the acoustic summaries become per-hypothesis evidence:
                   'relative' scores each against the logsumexp of the others,
                   'standard' only standardises them. The two order the
                   hypotheses identically; they differ in scale, which is what
                   the weight multiplies. 'standard' is the ablation.
    """

    vis_evidence: str = "lse"
    ac_evidence: str = "quantile"
    rule: str = "selective"
    weight: float = 1.0
    tau_v: float = 1.0
    tau_a: float = 0.0
    sigmoid_scale: float = 0.5
    lam: float = 1.0
    ac_transform: str = "relative"


def standardise(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit spread across the hypotheses of one query.

    The acoustic score is an unnormalised negative L1 whose offset and scale vary
    from query to query, so nothing can be compared across queries until it is
    standardised. Doing it per query, over the hypotheses only, is also what
    makes the thresholds below transferable.
    """
    x = np.asarray(x, dtype=np.float64)
    s = float(np.std(x))
    return (x - float(np.mean(x))) / (s if s > 1e-12 else 1.0)


def visual_ambiguity(vis_ev: np.ndarray) -> float:
    """Log-odds between the best and second-best hypothesis. Small is ambiguous.

    ``vis_ev`` is already in log units, so this is a difference. It is the same
    quantity the existing margin gate uses, computed over hypotheses instead of
    over an arbitrary suppression radius, which makes it consistent with how the
    hypotheses were extracted in the first place.
    """
    v = np.sort(np.asarray(vis_ev, dtype=np.float64))[::-1]
    return float(v[0] - v[1]) if v.size > 1 else float("inf")


def acoustic_evidence(ac_ev: np.ndarray) -> np.ndarray:
    """Relative acoustic evidence per hypothesis, in nats.

    Each hypothesis is scored against the aggregate of all the others, so the
    result says how much this place is preferred over the alternatives rather
    than where it sits in a global ranking.
    """
    return relative_evidence(standardise(ac_ev))


def acoustic_discriminativeness(ac_ev: np.ndarray) -> float:
    """How sharply sound separates the best hypothesis from the rest.

    This is the quantity the second half of the gate tests. It is deliberately
    the *margin*, not the score: a query where sound likes every hypothesis
    equally carries no information even if all its scores are high.
    """
    r = np.sort(acoustic_evidence(ac_ev))[::-1]
    return float(r[0] - r[1]) if r.size > 1 else float("inf")


def contradiction(ac_ev: np.ndarray) -> np.ndarray:
    """Per-hypothesis evidence *against*, and zero where there is none.

    The asymmetry is the intent. Furniture adds arrivals the floorplan does not
    predict, so a hypothesis whose predicted response the recording lacks is
    genuinely implausible, whereas one with extra energy is merely furnished. A
    penalty that only ever subtracts is the conservative use of a score too weak
    to be trusted positively.

    It does not work on this score, and ``contradiction_is_degenerate`` says why.
    """
    r = acoustic_evidence(ac_ev)
    return np.clip(-r, 0.0, None)


def contradiction_is_degenerate(ac_ev: np.ndarray) -> bool:
    """True when the contradiction penalty is algebraically the relative rule.

    ``relative_evidence`` compares each hypothesis against the logsumexp of the
    others, and with several hypotheses that aggregate almost always exceeds any
    single one, so every entry is negative. Then ``clip(-r, 0)`` is just ``-r``
    and subtracting it is adding ``r``: the two rules cannot differ. Measured on
    standard normal draws this happens 4% of the time at three hypotheses, 49%
    at five and 92% at ten, and on the real tables at ten hypotheses it is the
    normal case.

    Recording this matters more than the rule did. The brief asked for a
    conservative negative-evidence variant and said not to invent one if the
    current score cannot support it. It cannot: with a score that has no
    calibrated zero, "this hypothesis is contradicted" and "this hypothesis is
    less preferred than the others" are the same statement.
    """
    return bool((acoustic_evidence(ac_ev) < 0).all())


def choose(vis_ev: np.ndarray, ac_ev: np.ndarray,
           config: ModeFusionConfig) -> tuple[int, bool]:
    """Index of the chosen hypothesis, and whether audio was allowed to act.

    ``vis_ev`` must be in log units; ``ac_ev`` may be any monotone score, since
    it is standardised here.
    """
    vis_ev = np.asarray(vis_ev, dtype=np.float64)
    ac_ev = np.asarray(ac_ev, dtype=np.float64)
    vb = int(vis_ev.argmax())
    if config.rule == "vision" or vis_ev.size < 2:
        return vb, False

    rel = (acoustic_evidence(ac_ev) if config.ac_transform == "relative"
           else standardise(ac_ev))
    amb = visual_ambiguity(vis_ev)
    disc = acoustic_discriminativeness(ac_ev)

    if config.rule == "mode_rerank":
        return int(ac_ev.argmax()), True

    if config.rule == "relative":
        return int((standardise(vis_ev) + config.weight * rel).argmax()), True

    if config.rule == "selective":
        if amb < config.tau_v and disc > config.tau_a:
            return int(rel.argmax()), True
        return vb, False

    if config.rule == "continuous":
        # both gates as soft factors, so a query near either threshold is not
        # decided by which side of it it happens to fall
        gv = 1.0 / (1.0 + np.exp((amb - config.tau_v) / config.sigmoid_scale))
        ga = 1.0 / (1.0 + np.exp(-(disc - config.tau_a) / config.sigmoid_scale))
        alpha = config.weight * gv * ga
        return int((standardise(vis_ev) + alpha * rel).argmax()), bool(alpha > 0.05)

    if config.rule == "contradiction":
        pen = contradiction(ac_ev)
        return int((standardise(vis_ev) - config.lam * pen).argmax()), True

    raise ValueError(f"unknown rule {config.rule!r}")


def evidence_columns(config: ModeFusionConfig) -> tuple[str, str]:
    """Names of the table columns the configured evidence choices correspond to."""
    vis = {"centre": "vis_log_centre", "max": "vis_log_max", "lse": "vis_log_lse"}
    ac = {"centre": "ac_centre", "max": "ac_max",
          "quantile": "ac_quantile", "lse": "ac_lse"}
    if config.vis_evidence not in vis:
        raise ValueError(f"unknown vis_evidence {config.vis_evidence!r}")
    if config.ac_evidence not in ac:
        raise ValueError(f"unknown ac_evidence {config.ac_evidence!r}")
    return vis[config.vis_evidence], ac[config.ac_evidence]
