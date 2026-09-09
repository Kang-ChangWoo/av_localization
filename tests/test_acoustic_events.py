"""Contract tests for one-sided acoustic evidence matching.

Deterministic and synthetic throughout: no Gibson, Replica, or SoundSpaces
asset is touched, so these run anywhere and pin the behaviour the research
claim depends on. The central one is ``test_clutter_robustness``, which is the
hypothesis itself stated as an assertion: adding echoes a 2D floorplan cannot
predict must not move the one-sided score, and does move the symmetric one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from track1_core.likelihood.events import (  # noqa: E402
    PATH_TYPES,
    EventMatchConfig,
    EventSet,
    asymmetric_energy_score,
    event_match_score,
    merge_predicted,
    observed_events,
    pad_event_sets,
    pooled_observed_profile,
)

C = 343.0
FS = 8000


def _cfg(**kw) -> EventMatchConfig:
    return EventMatchConfig(**kw)


def _rir_from_delays(delay_ms, amps=None, n_channels=6, n_samples=1200,
                     config: EventMatchConfig | None = None):
    """Synthetic RIR: a unit direct peak then one impulse per requested delay.

    Delays are relative to the end of the direct guard, which is the same time
    origin the extractor uses, so a delay put in comes back out.
    """
    config = config or _cfg()
    rir = np.zeros((n_channels, n_samples))
    direct = 40
    rir[:, direct] = 10.0
    amps = np.ones(len(delay_ms)) if amps is None else np.asarray(amps, float)
    for t, a in zip(delay_ms, amps):
        s = direct + config.direct_guard_samples + int(round(t / 1000.0 * FS))
        if 0 <= s < n_samples:
            rir[:, s] += a
    return rir


def _symmetric_l1(a: np.ndarray, b: np.ndarray) -> float:
    """The existing baseline comparison: normalise both sides, take L1."""
    a = a / max(a.sum(), 1e-30)
    b = b / max(b.sum(), 1e-30)
    return float(np.abs(a - b).sum())


# --------------------------------------------------------------------------
# Test 1: a known wall distance produces the expected round-trip delay
# --------------------------------------------------------------------------
@pytest.mark.parametrize("d_m", [1.0, 2.5, 4.0, 6.0])
def test_round_trip_delay_matches_geometry(d_m):
    """For the co-located radial proxy the arrival must land at 2d/c."""
    cfg = _cfg()
    expected_ms = 2 * d_m / C * 1000.0
    rir = _rir_from_delays([expected_ms], config=cfg)
    ev = observed_events(rir, cfg)
    assert len(ev) >= 1
    nearest = ev.delay_ms[np.argmin(np.abs(ev.delay_ms - expected_ms))]
    # tolerance is one profile bin, which is what the binning can resolve
    assert abs(nearest - expected_ms) <= cfg.profile_resolution_ms * 1.5


def test_time_origin_convention():
    """Predicted and observed arrivals must be on the same clock.

    The RIR clock starts at emission; the direct sound peaks after crossing the
    5 cm ring, and the reflection window starts ``direct_guard_samples`` later.
    A predicted absolute path time therefore has to lose r/c + guard/fs before
    it can be compared. That gap is 2.15 ms, more than twice the matching
    tolerance, so getting it wrong makes the two sides disjoint rather than
    merely noisy. This test is the reason the units are worth writing down.
    """
    cfg = _cfg()
    assert cfg.window_offset_ms == pytest.approx(0.05 / C * 1000 + 16 / FS * 1000, rel=1e-9)
    assert cfg.window_offset_ms > cfg.tau_tolerance_ms

    d_m = 3.0                                   # a wall 3 m away
    tau_window = cfg.path_m_to_ms(2 * d_m)      # what the matcher should use
    rir = _rir_from_delays([tau_window], config=cfg)
    ev = observed_events(rir, cfg)
    nearest = ev.delay_ms[np.argmin(np.abs(ev.delay_ms - tau_window))]
    assert abs(nearest - tau_window) <= cfg.profile_resolution_ms * 1.5
    # and the round trip really is 2d/c in absolute time
    assert tau_window + cfg.window_offset_ms == pytest.approx(2 * d_m / C * 1000, rel=1e-9)


def test_predicted_merge_collapses_a_flat_wall():
    """Many rays returning from one wall are one event, with summed weight."""
    cfg = _cfg()
    delays = np.full(30, 11.6) + np.random.default_rng(0).normal(0, 0.05, 30)
    w = np.full(30, 1 / 72)
    d, wm, _ = merge_predicted(delays, w, np.zeros(30, dtype=int), cfg)
    assert d.shape[0] == 1
    assert wm[0] == pytest.approx(w.sum())


# --------------------------------------------------------------------------
# Test 2: extra clutter echoes -- the core research hypothesis
# --------------------------------------------------------------------------
def test_clutter_robustness():
    """Echoes the floorplan cannot predict must not penalise the candidate.

    The observation keeps every correct arrival and gains six it does not
    explain, which is what furniture does. The symmetric envelope distance has
    to degrade; the one-sided score must not.
    """
    cfg = _cfg()
    true_ms = np.array([5.8, 11.7, 17.5, 23.3])
    clean = _rir_from_delays(true_ms, config=cfg)
    clutter_ms = np.array([7.9, 9.4, 14.2, 19.6, 21.1, 26.8])
    cluttered = _rir_from_delays(np.concatenate([true_ms, clutter_ms]),
                                 amps=np.concatenate([np.ones(4), np.full(6, 0.8)]),
                                 config=cfg)

    pred_d = true_ms[None, :].copy()
    pred_w = np.full((1, 4), 0.25)

    s_clean = event_match_score(pred_d, pred_w, observed_events(clean, cfg), cfg)[0]
    s_clut = event_match_score(pred_d, pred_w, observed_events(cluttered, cfg), cfg)[0]

    from track1_core.likelihood.events import observed_profile
    l1_clean = _symmetric_l1(observed_profile(clean, cfg)[1], observed_profile(clean, cfg)[1])
    l1_clut = _symmetric_l1(observed_profile(clean, cfg)[1], observed_profile(cluttered, cfg)[1])

    assert s_clean > 0.9, f"clean score should be near perfect, got {s_clean}"
    assert s_clut > 0.9 * s_clean, f"one-sided score collapsed under clutter: {s_clean} -> {s_clut}"
    assert l1_clut > l1_clean + 0.3, "symmetric L1 should degrade under clutter"


# --------------------------------------------------------------------------
# Test 3: missing predicted paths degrade gracefully under TopFraction
# --------------------------------------------------------------------------
def test_missing_paths_degrade_gracefully():
    """Occluded wall returns should cost less with a robust aggregation."""
    true_ms = np.array([5.8, 11.7, 17.5, 23.3, 29.1])
    observed_partial = _rir_from_delays(true_ms[:3])       # two predicted paths missing
    pred_d = true_ms[None, :].copy()
    pred_w = np.full((1, 5), 0.2)

    full = _cfg(top_fraction=1.0)
    robust = _cfg(top_fraction=0.6)
    s_full = event_match_score(pred_d, pred_w, observed_events(observed_partial, full), full)[0]
    s_rob = event_match_score(pred_d, pred_w, observed_events(observed_partial, robust), robust)[0]

    assert s_rob > s_full, f"TopFraction should be more forgiving: {s_rob} vs {s_full}"
    assert s_full < 0.9, "requiring every path should notice two are missing"
    assert s_rob > 0.85, "keeping the best 60% should survive losing 40%"


# --------------------------------------------------------------------------
# Test 4: a geometrically wrong candidate ranks below ground truth
# --------------------------------------------------------------------------
def test_wrong_candidate_ranks_below_truth():
    cfg = _cfg()
    true_ms = np.array([5.8, 11.7, 17.5, 23.3])
    obs = observed_events(_rir_from_delays(true_ms, config=cfg), cfg)
    wrong_ms = true_ms + 4.0          # 4 ms = 1.37 m of path, far past tolerance
    pred_d = np.stack([true_ms, wrong_ms])
    pred_w = np.full((2, 4), 0.25)
    s = event_match_score(pred_d, pred_w, obs, cfg)
    assert s[0] > s[1], f"truth {s[0]} did not beat the wrong candidate {s[1]}"
    assert s[0] - s[1] > 0.3, f"margin too small to be useful: {s[0] - s[1]}"


# --------------------------------------------------------------------------
# Test 5: temporal tolerance degrades smoothly
# --------------------------------------------------------------------------
def test_temporal_tolerance_is_smooth():
    """A small delay error should cost a little, not everything."""
    cfg = _cfg(tau_tolerance_ms=1.0)
    true_ms = np.array([5.8, 11.7, 17.5, 23.3])
    obs = observed_events(_rir_from_delays(true_ms, config=cfg), cfg)
    offsets = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    scores = [event_match_score((true_ms + o)[None, :], np.full((1, 4), 0.25), obs, cfg)[0]
              for o in offsets]
    assert all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1)), scores
    # one sigma should still retain most of the score, not collapse it
    assert scores[3] > 0.5 * scores[0], f"one sigma cost too much: {scores}"
    assert scores[-1] < 0.2 * scores[0], f"four sigma should be rejected: {scores}"


# --------------------------------------------------------------------------
# Test 6: delay-only scoring is yaw-invariant by construction
# --------------------------------------------------------------------------
def test_delay_only_score_is_yaw_invariant():
    """Source and receiver are co-located, so the delay-only score is a
    function of position alone and must broadcast unchanged across yaw."""
    cfg = _cfg()
    true_ms = np.array([5.8, 11.7, 17.5])
    obs = observed_events(_rir_from_delays(true_ms, config=cfg), cfg)
    pred_d = np.repeat(true_ms[None, :], 36, axis=0)      # same position, 36 yaws
    pred_w = np.full((36, 3), 1 / 3)
    s = event_match_score(pred_d, pred_w, obs, cfg)
    assert np.allclose(s, s[0]), "delay-only score must not depend on yaw"


# --------------------------------------------------------------------------
# uniqueness, aggregation, and the dense variant
# --------------------------------------------------------------------------
def test_uniqueness_limits_one_peak_explaining_many():
    """Ten predicted events piled onto one observed peak must not score as
    highly as ten events each with their own peak."""
    cfg_none = _cfg(uniqueness="none", top_fraction=1.0)
    cfg_soft = _cfg(uniqueness="soft", top_fraction=1.0)
    obs = EventSet(delay_ms=np.array([10.0]), weight=np.array([1.0]),
                   kind=np.zeros(1, dtype=int))
    piled = np.full((1, 10), 10.0)
    w = np.full((1, 10), 0.1)
    assert event_match_score(piled, w, obs, cfg_none)[0] > 0.99
    assert event_match_score(piled, w, obs, cfg_soft)[0] < 0.5

    obs_many = EventSet(delay_ms=np.arange(10) * 3.0 + 5.0, weight=np.ones(10),
                        kind=np.zeros(10, dtype=int))
    spread = (np.arange(10) * 3.0 + 5.0)[None, :]
    assert event_match_score(spread, w, obs_many, cfg_soft)[0] > 0.9


def test_greedy_and_soft_agree_on_a_clean_case():
    obs = EventSet(delay_ms=np.array([5.0, 12.0, 20.0]), weight=np.ones(3),
                   kind=np.zeros(3, dtype=int))
    pred = np.array([[5.0, 12.0, 20.0]])
    w = np.full((1, 3), 1 / 3)
    a = event_match_score(pred, w, obs, _cfg(uniqueness="soft"))[0]
    b = event_match_score(pred, w, obs, _cfg(uniqueness="greedy"))[0]
    assert abs(a - b) < 0.05, (a, b)


def test_asymmetric_dense_matches_sparse_on_clean_input():
    cfg = _cfg()
    true_ms = np.array([5.8, 11.7, 17.5, 23.3])
    rir = _rir_from_delays(true_ms, config=cfg)
    t, pooled = pooled_observed_profile(rir, cfg)
    dense = asymmetric_energy_score(true_ms[None, :], np.full((1, 4), 0.25), t, pooled, cfg)[0]
    sparse = event_match_score(true_ms[None, :], np.full((1, 4), 0.25),
                               observed_events(rir, cfg), cfg)[0]
    assert dense > 0.5 and sparse > 0.9
    # offset by roughly half the arrival spacing, so the shifted predictions
    # fall between observed peaks rather than onto the next one
    wrong = asymmetric_energy_score((true_ms + 2.9)[None, :], np.full((1, 4), 0.25), t, pooled, cfg)[0]
    assert dense > wrong + 0.5, (dense, wrong)


def test_padding_contributes_nothing():
    cfg = _cfg()
    obs = observed_events(_rir_from_delays([5.8, 11.7], config=cfg), cfg)
    a = EventSet(np.array([5.8, 11.7]), np.array([0.5, 0.5]), np.zeros(2, dtype=int))
    b = EventSet(np.array([5.8]), np.array([1.0]), np.zeros(1, dtype=int))
    d, w, k = pad_event_sets([a, b])
    assert d.shape == (2, 2) and np.isnan(d[1, 1]) and w[1, 1] == 0.0
    s = event_match_score(d, w, obs, cfg)
    assert np.isfinite(s).all() and (s > 0.9).all()


# --------------------------------------------------------------------------
# Test 7: corner NLOS paths appear only where the geometry calls for them
# --------------------------------------------------------------------------
def test_corner_paths_only_in_l_shaped_geometry():
    """In an L-shaped room a remote wall is invisible from one arm, but a
    one-corner path to it exists. In an open rectangle there is nothing to
    diffract around, so far fewer corner paths should be produced."""
    pytest.importorskip("cv2")
    from track1_core.likelihood.corners import (
        CornerConfig, corner_events, corner_remote_distances, extract_corners,
    )

    res = 0.05
    cfg = _cfg()
    ccfg = CornerConfig(max_corners=32, n_corner_rays=32)

    # L-shaped free space, 200x200 pixels at 5 cm = 10 m x 10 m
    L = np.zeros((200, 200), dtype=bool)
    L[20:180, 20:100] = True          # vertical arm
    L[100:180, 20:180] = True         # horizontal arm
    corners = extract_corners(L, res, ccfg)
    assert corners.shape[0] > 0, "an L-shaped room must expose at least one corner"
    rho, ang = corner_remote_distances(L, corners, res, ccfg)
    # a candidate high in the vertical arm cannot see the far end of the other arm
    origin = np.array([[60.0, 40.0]])
    d, w, k = corner_events(L, origin, corners, rho, ang, res, ccfg, cfg)
    n_L = int(np.isfinite(d).sum())
    assert n_L > 0, "no one-corner path found in an L-shaped room"
    assert (k[np.isfinite(d)] == PATH_TYPES["corner_nlos"]).all()

    # open rectangle of comparable area: everything is directly visible
    R = np.zeros((200, 200), dtype=bool)
    R[20:180, 20:180] = True
    rc = extract_corners(R, res, ccfg)
    rrho, rang = corner_remote_distances(R, rc, res, ccfg)
    d2, _, _ = corner_events(R, np.array([[100.0, 100.0]]), rc, rrho, rang, res, ccfg, cfg)
    n_R = int(np.isfinite(d2).sum())
    assert n_L > n_R, f"L-shape should yield more NLOS paths than an open box: {n_L} vs {n_R}"


def test_corner_delay_is_the_broken_path_length():
    """The predicted delay must equal 2*(|src-corner| + rho)/c."""
    pytest.importorskip("cv2")
    from track1_core.likelihood.corners import CornerConfig, corner_events

    res = 0.05
    cfg = _cfg()
    ccfg = CornerConfig(n_corner_rays=4, occlusion_margin_deg=0.0)
    free = np.zeros((200, 200), dtype=bool)
    free[20:180, 20:180] = True
    corners = np.array([[100.0, 60.0]])
    ang = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    # the only remote surface sits at 3 m along 3*pi/2, which continues away
    # from the source past the corner; the other three directions face back
    # toward it and are excluded as not being occluded evidence
    rho = np.array([[np.nan, np.nan, np.nan, 3.0]])
    origin = np.array([[100.0, 100.0]])          # 2.0 m from the corner
    d, w, _ = corner_events(free, origin, corners, rho, ang, res, ccfg, cfg)
    finite = d[np.isfinite(d)]
    assert finite.size == 1
    # window time, so the same offset the radial family carries
    expected = 2 * (2.0 + 3.0) / C * 1000.0 - cfg.window_offset_ms
    assert abs(finite[0] - expected) < 0.2, (finite[0], expected)
