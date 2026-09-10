"""Pins for the mode-level fusion, and for the two ways it can quietly do nothing.

The cell-wise scoring in ``grid_score.py`` had two degeneracies that were each
measured before they were noticed, and both are now asserted there. The
mode-level rules have their own, of the same shape: an option that looks like a
distinct method and is algebraically another one. These tests exist so that the
next person to sweep over ``rule`` cannot report two identical columns as two
results.
"""

from __future__ import annotations

import numpy as np
import pytest

from track1_core.likelihood.mode_fusion import (
    ModeFusionConfig, acoustic_evidence, acoustic_discriminativeness, choose,
    contradiction, contradiction_is_degenerate, evidence_columns, standardise,
    visual_ambiguity,
)
from track1_core.modes import (
    ModeConfig, aggregate, entropy, extract_modes, local_discs, relative_evidence,
    softmax,
)


def grid(n=40):
    """A small square of cells with the (rows, cols) convention used everywhere."""
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return r.ravel(), c.ravel()


def test_nms_separates_modes_by_at_least_the_radius():
    rows, cols = grid()
    vis = np.zeros(len(rows))
    # three peaks, two of them closer together than the radius
    for (y, x, h) in ((5, 5, 3.0), (6, 7, 2.9), (30, 30, 2.0)):
        vis += h * np.exp(-((rows - y) ** 2 + (cols - x) ** 2) / 4.0)
    cfg = ModeConfig(nms_radius_m=1.5, n_modes=5)
    m = extract_modes(vis, rows, cols, 0.1, cfg)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            d = np.hypot(cols[m[i]] - cols[m[j]], rows[m[i]] - rows[m[j]]) * 0.1
            assert d > cfg.nms_radius_m


def test_modes_are_returned_strongest_first():
    rows, cols = grid()
    vis = np.zeros(len(rows))
    for (y, x, h) in ((5, 5, 1.0), (20, 20, 3.0), (35, 35, 2.0)):
        vis += h * np.exp(-((rows - y) ** 2 + (cols - x) ** 2) / 4.0)
    m = extract_modes(vis, rows, cols, 0.1, ModeConfig(n_modes=3))
    assert list(np.argsort(-vis[m])) == [0, 1, 2]


def test_local_disc_respects_its_radius():
    rows, cols = grid()
    m = np.array([len(rows) // 2])
    d = local_discs(m, rows, cols, 0.1, 0.5)[0]
    far = np.hypot(cols[d] - cols[m[0]], rows[d] - rows[m[0]]) * 0.1
    assert far.max() <= 0.5 + 1e-9
    assert d.size > 1


def test_relative_evidence_is_shift_invariant():
    """A constant offset is exactly what a global percentile is not immune to."""
    s = np.array([1.0, 0.4, -0.2, -1.1])
    assert np.allclose(relative_evidence(s), relative_evidence(s + 7.3))


def test_standardise_makes_the_score_scale_free():
    a = np.array([3.0, 1.0, -2.0, 0.5])
    assert np.allclose(standardise(a), standardise(5 * a + 2))


# ---------------------------------------------------------------- degeneracies
def test_contradiction_is_degenerate_with_many_hypotheses():
    """With ten hypotheses the clip never fires, so the rule is the relative one."""
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(200):
        ac = rng.standard_normal(10)
        if contradiction_is_degenerate(ac):
            hits += 1
            # and when it is degenerate the two rules must agree exactly
            assert np.allclose(contradiction(ac), -acoustic_evidence(ac))
    assert hits > 150, "expected the degenerate case to dominate at ten hypotheses"


def test_contradiction_can_be_non_degenerate_with_two_hypotheses():
    """The guard must not be vacuously true, or it would assert nothing."""
    ac = np.array([2.0, -2.0])
    assert not contradiction_is_degenerate(ac)
    assert contradiction(ac)[0] == 0.0


def test_contradiction_and_relative_give_the_same_choice_when_degenerate():
    rng = np.random.default_rng(3)
    for _ in range(100):
        vis = np.sort(rng.standard_normal(10))[::-1]
        ac = rng.standard_normal(10)
        if not contradiction_is_degenerate(ac):
            continue
        a, _ = choose(vis, ac, ModeFusionConfig(rule="relative", weight=1.0))
        b, _ = choose(vis, ac, ModeFusionConfig(rule="contradiction", lam=1.0))
        assert a == b


# ---------------------------------------------------------------- the rules
def test_vision_rule_never_lets_audio_act():
    vis = np.array([0.0, -1.0, -2.0])
    ac = np.array([-10.0, 5.0, 5.0])
    k, used = choose(vis, ac, ModeFusionConfig(rule="vision"))
    assert k == 0 and not used


def test_selective_rule_respects_both_thresholds():
    vis = np.array([0.0, -0.001, -3.0])      # ambiguous
    ac = np.array([-5.0, 5.0, -5.0])         # sound strongly prefers hypothesis 1
    loose = ModeFusionConfig(rule="selective", tau_v=1.0, tau_a=-np.inf)
    assert choose(vis, ac, loose)[0] == 1

    # a confident visual posterior must block audio whatever sound says
    confident = np.array([0.0, -9.0, -12.0])
    assert choose(confident, ac, loose)[0] == 0
    # and an acoustic threshold above what this query offers must block it too
    strict = ModeFusionConfig(rule="selective", tau_v=1.0, tau_a=1e9)
    assert choose(vis, ac, strict)[0] == 0


def test_audio_can_only_choose_among_the_given_hypotheses():
    """The whole point of mode-level fusion: no relocation off the shortlist."""
    rng = np.random.default_rng(7)
    for rule in ("mode_rerank", "relative", "selective", "continuous", "contradiction"):
        for _ in range(50):
            vis = rng.standard_normal(6)
            ac = rng.standard_normal(6)
            k, _ = choose(vis, ac, ModeFusionConfig(rule=rule, tau_v=1e9, tau_a=-np.inf))
            assert 0 <= k < 6


def test_continuous_gate_reduces_to_vision_at_zero_weight():
    rng = np.random.default_rng(11)
    for _ in range(50):
        vis = rng.standard_normal(8)
        ac = rng.standard_normal(8)
        k, _ = choose(vis, ac, ModeFusionConfig(rule="continuous", weight=0.0))
        assert k == int(vis.argmax())


def test_visual_ambiguity_is_the_gap_between_the_first_two():
    assert visual_ambiguity(np.array([2.0, 0.5, -3.0])) == pytest.approx(1.5)
    assert np.isinf(visual_ambiguity(np.array([1.0])))


def test_acoustic_discriminativeness_is_zero_when_sound_is_indifferent():
    assert acoustic_discriminativeness(np.zeros(5)) == pytest.approx(0.0)


def test_evidence_columns_reject_unknown_choices():
    with pytest.raises(ValueError):
        evidence_columns(ModeFusionConfig(vis_evidence="nope"))
    with pytest.raises(ValueError):
        evidence_columns(ModeFusionConfig(ac_evidence="nope"))


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        choose(np.array([1.0, 0.0]), np.array([0.0, 1.0]),
               ModeFusionConfig(rule="telepathy"))


def test_aggregate_orders_are_consistent():
    """max is never below the quantile, which is never below the centre's own disc min."""
    rows, cols = grid()
    field = np.random.default_rng(1).standard_normal(len(rows))
    m = extract_modes(field, rows, cols, 0.1, ModeConfig(n_modes=4))
    d = local_discs(m, rows, cols, 0.1, 0.5)
    a = aggregate(field, m, d, ModeConfig())
    assert np.all(a["max"] >= a["quantile"] - 1e-9)
    assert np.all(a["max"] >= a["centre"] - 1e-9)


def test_softmax_and_entropy_agree_on_the_uniform_case():
    p = softmax(np.zeros(4))
    assert np.allclose(p, 0.25)
    assert entropy(p) == pytest.approx(np.log(4))
