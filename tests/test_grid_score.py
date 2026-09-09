"""Contract tests for the shared candidate-grid scoring.

The one that matters is ``test_sum_normalisation_kills_the_asymmetry``. That
degeneracy was implemented and measured twice in this project before anyone
noticed it was an identity, so it is asserted here rather than rediscovered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from track1_core.likelihood.grid_score import (  # noqa: E402
    GridScoreConfig, asymmetry_is_degenerate, band_energy, featurise,
    normalise, score, shape,
)


def _rir(delays_ms, amps=None, n=1200, guard=16, fs=8000):
    r = np.zeros((6, n))
    r[:, 40] = 10.0
    amps = np.ones(len(delays_ms)) if amps is None else np.asarray(amps, float)
    for t, a in zip(delays_ms, amps):
        s = 40 + guard + int(round(t / 1000 * fs))
        if 0 <= s < n:
            r[:, s] += a
    return r


def test_sum_normalisation_kills_the_asymmetry():
    """Under sum normalisation the clipped penalty is half the symmetric one,
    so the two rank candidates identically and the asymmetry is a no-op."""
    sym = GridScoreConfig(normalise="sum", distance="symmetric")
    one = GridScoreConfig(normalise="sum", distance="one_sided")
    assert asymmetry_is_degenerate(one) and not asymmetry_is_degenerate(sym)

    rng = np.random.default_rng(0)
    cand = np.abs(rng.normal(size=(40, 6, 30))) + 0.01
    cand = cand / cand.sum(axis=(1, 2), keepdims=True)
    obs = np.abs(rng.normal(size=(6, 30))) + 0.01
    obs = obs / obs.sum()
    present = np.ones(40, dtype=bool)
    a = score(cand, obs, present, sym)
    b = score(cand, obs, present, one)
    assert np.allclose(b, a / 2, atol=1e-12)
    assert np.array_equal(np.argsort(a), np.argsort(b))


@pytest.mark.parametrize("norm", ["peak", "quantile"])
def test_other_normalisations_let_the_asymmetry_exist(norm):
    """Peak and quantile normalisation keep a scale, so the one-sided penalty
    is no longer a fixed fraction of the symmetric one and can reorder."""
    rng = np.random.default_rng(1)
    cand = np.abs(rng.normal(size=(60, 6, 30))) + 0.01
    obs = np.abs(rng.normal(size=(6, 30))) + 0.01
    present = np.ones(60, dtype=bool)
    sym = GridScoreConfig(normalise=norm, distance="symmetric", shape_feature=False)
    one = GridScoreConfig(normalise=norm, distance="one_sided", shape_feature=False)
    cs, os_ = normalise(cand, sym), normalise(obs, sym)
    a = score(cs, os_, present, sym)
    b = score(cs, os_, present, one)
    assert not np.allclose(b, a / 2)
    assert not np.array_equal(np.argsort(a), np.argsort(b))


def test_clutter_costs_nothing_under_one_sided_peak():
    """Adding arrivals the model does not predict must not change a one-sided
    score, which is the property the whole idea rests on."""
    cfg = GridScoreConfig(normalise="peak", distance="one_sided", shape_feature=False)
    model = np.zeros((1, 6, 30)); model[:, :, [3, 9, 15]] = 1.0
    obs = np.zeros((6, 30)); obs[:, [3, 9, 15]] = 1.0
    cluttered = obs.copy(); cluttered[:, [5, 11, 20, 26]] = 0.8
    present = np.ones(1, dtype=bool)
    clean = score(normalise(model, cfg), normalise(obs, cfg), present, cfg)[0]
    dirty = score(normalise(model, cfg), normalise(cluttered, cfg), present, cfg)[0]
    assert clean == pytest.approx(dirty, abs=1e-9)


def test_window_and_band_slicing():
    cfg = GridScoreConfig(feature="stft_band", bands=((0, 1000),), first_frame=15)
    e = band_energy(_rir([5.0, 40.0]), cfg)
    assert e.shape[0] == 6 * 1
    assert e.shape[1] == cfg.n_frames - 15
    full = band_energy(_rir([5.0, 40.0]),
                       GridScoreConfig(feature="stft_band", bands=((0, 1000),)))
    assert np.allclose(e, full[:, 15:])


def test_frame_timing_matches_the_feature():
    """The two features have different time resolution, and that difference is
    the whole reason the envelope works and the banded STFT does not."""
    env = GridScoreConfig(feature="envelope", window_ms=0.5)
    assert env.ms_per_frame == pytest.approx(0.5)
    assert env.frame_of_ms(30.0) == 60
    assert env.n_frames == 1024 // 4

    stft = GridScoreConfig(feature="stft_band")
    assert stft.ms_per_frame == pytest.approx(2.0)   # the hop
    assert stft.frame_of_ms(30.0) == 15
    # but the analysis window is nfft samples, so each frame really covers
    # 8 ms, which is 2.7 m of path and 27 cells of a 0.1 m pose grid
    assert 1000 * stft.nfft / stft.sample_rate_hz == pytest.approx(8.0)


def test_envelope_windows_do_not_leak():
    """A disjoint energy window confines an arrival to one frame, which is what
    the STFT's 8 ms analysis window destroys."""
    cfg = GridScoreConfig(feature="envelope", window_ms=0.5)
    e = band_energy(_rir([10.0]), cfg)
    hot = np.flatnonzero(e[0] > 0)
    assert hot.size == 1, f"one arrival should occupy one window, got {hot.size}"
    assert hot[0] == pytest.approx(round(10.0 / cfg.window_ms), abs=1)


def test_missing_candidates_rank_last():
    cfg = GridScoreConfig()
    cand = np.abs(np.random.default_rng(2).normal(size=(10, 6, 20))) + 0.01
    obs = np.abs(np.random.default_rng(3).normal(size=(6, 20))) + 0.01
    present = np.array([True] * 7 + [False] * 3)
    s = score(cand, obs, present, cfg)
    assert s[~present].max() < s[present].min()


def test_shape_feature_also_kills_the_asymmetry():
    """The second route to the same dead end: shape_feature normalises every
    row to sum one, so the one-sided penalty is again half the symmetric one
    whatever normalisation ran before it, and the normalise choice stops
    mattering at all."""
    from track1_core.likelihood.grid_score import normalisation_is_a_noop
    for norm in ("sum", "peak", "quantile"):
        cfg = GridScoreConfig(normalise=norm, distance="one_sided", shape_feature=True)
        assert asymmetry_is_degenerate(cfg)
        assert normalisation_is_a_noop(cfg)

    rng = np.random.default_rng(7)
    cand = np.abs(rng.normal(size=(30, 6, 25))) + 0.01
    obs = np.abs(rng.normal(size=(6, 25))) + 0.01
    present = np.ones(30, dtype=bool)
    scores = []
    for norm in ("sum", "peak", "quantile"):
        c = GridScoreConfig(normalise=norm, distance="symmetric", shape_feature=True)
        scores.append(score(featurise(cand, c), featurise(obs, c), present, c))
    assert np.allclose(scores[0], scores[1]) and np.allclose(scores[0], scores[2])

    # with shape_feature off, the normalisation finally does something
    a = GridScoreConfig(normalise="sum", distance="symmetric", shape_feature=False)
    b = GridScoreConfig(normalise="peak", distance="symmetric", shape_feature=False)
    sa = score(featurise(cand, a), featurise(obs, a), present, a)
    sb = score(featurise(cand, b), featurise(obs, b), present, b)
    assert not np.array_equal(np.argsort(sa), np.argsort(sb))
