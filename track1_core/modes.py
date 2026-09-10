"""Spatially separated hypotheses of a visual posterior, and how to score them.

The visual posterior over a floorplan is not a blob with a peak. It is a handful
of separated places that explain the image about equally well, plus a large flat
remainder. Every analysis in this project that treats it as a ranked list of
cells inherits a defect from that: the top fifty cells are usually the top two or
three *places*, each represented many times, so a "top-50 rerank" is really a
choice among three options with the votes miscounted.

This module extracts the places. Greedy non-maximum suppression in XY at a fixed
radius, taking the strongest remaining cell and suppressing its neighbourhood, is
the same rule the confidence margin already uses to find "the strongest
competitor somewhere else", so mode extraction and the margin stay consistent.

The local aggregates exist because a mode is a region, not a point. A cell-centre
score is the noisiest possible summary of a region, and the acoustic score in
particular is spatially rough: it can identify the right region while being wrong
about the exact cell. Aggregating over a small disc around the centre is the
cheapest way to ask "is this *region* plausible" instead of "is this cell".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeConfig:
    """How visual hypotheses are extracted and summarised.

    nms_radius_m     two modes must be at least this far apart. 1.5 m matches the
                     separation the confidence margin uses.
    n_modes          how many to keep. Ten is enough that ground truth is
                     represented whenever it is represented at all; see the
                     coverage table.
    local_radius_m   radius of the disc a mode's local aggregates are taken over.
                     Smaller than the NMS radius so that discs do not overlap.
    quantile         the upper quantile used by the robust local aggregate.
    """

    nms_radius_m: float = 1.5
    n_modes: int = 10
    local_radius_m: float = 0.5
    quantile: float = 0.9


def extract_modes(vis: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                  res_m: float, config: ModeConfig) -> np.ndarray:
    """Indices of the separated modes of ``vis``, strongest first.

    ``vis`` is indexed the same way as ``rows`` and ``cols``: one entry per valid
    cell, in the order ``np.nonzero(mask)`` produced.
    """
    order = np.argsort(-vis)
    picked: list[int] = []
    taken = np.zeros(len(vis), dtype=bool)
    for i in order:
        if taken[i]:
            continue
        picked.append(int(i))
        taken |= (np.hypot(cols - cols[i], rows - rows[i]) * res_m
                  <= config.nms_radius_m)
        if len(picked) == config.n_modes:
            break
    return np.asarray(picked, dtype=np.int64)


def local_discs(modes: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                res_m: float, radius_m: float) -> list[np.ndarray]:
    """Index arrays for the cells inside each mode's local disc."""
    return [np.nonzero(np.hypot(cols - cols[m], rows - rows[m]) * res_m <= radius_m)[0]
            for m in modes]


def logsumexp(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def aggregate(field: np.ndarray, modes: np.ndarray, discs: list[np.ndarray],
              config: ModeConfig) -> dict[str, np.ndarray]:
    """Four summaries of ``field`` per mode: centre, max, upper quantile, logsumexp.

    ``field`` is expected in a log-like or at least monotone-comparable scale;
    ``logsumexp`` is only meaningful for the former, so callers pass log scores
    when they want probability mass and raw scores when they want the rest.
    """
    return {
        "centre": field[modes].astype(np.float64),
        "max": np.array([field[d].max() if d.size else -np.inf for d in discs]),
        "quantile": np.array([np.quantile(field[d], config.quantile) if d.size else -np.inf
                              for d in discs]),
        "lse": np.array([logsumexp(field[d]) for d in discs]),
    }


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(temperature, 1e-12)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, None)
    return float(-(p * np.log(p)).sum())


def relative_evidence(scores: np.ndarray) -> np.ndarray:
    """Each mode's score against the logsumexp of every *other* mode.

    This is the quantity the acoustic module should produce. A global percentile
    answers "where am I in the building", which sound cannot do here; this
    answers "among the places vision proposes, which is more consistent", which
    is the only question sound is being asked. It is also invariant to a constant
    offset in the score, which the global percentile is not.
    """
    s = np.asarray(scores, dtype=np.float64)
    out = np.empty_like(s)
    for k in range(len(s)):
        others = np.delete(s, k)
        out[k] = s[k] - logsumexp(others) if others.size else 0.0
    return out


def margins(scores: np.ndarray) -> tuple[float, float]:
    """(best minus second, best over second) for a score vector, in that order.

    The second is returned as a plain ratio and is only meaningful for positive
    scores; callers working in log space want the first.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if s.size < 2:
        return float("inf"), float("inf")
    ratio = float(s[0] / s[1]) if s[1] > 0 else float("inf")
    return float(s[0] - s[1]), ratio
