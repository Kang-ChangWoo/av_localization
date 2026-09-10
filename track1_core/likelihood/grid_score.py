"""One implementation of scoring a rendered candidate grid against a recording.

Four scripts in this project grew their own copy of this scoring, each slightly
different, and the differences were invisible from the outside: two of them
reported the same quantity under the same name and disagreed by ten points. A
single implementation removes that class of error, and makes the axes that
actually matter explicit parameters rather than choices buried in a script.

Three axes turn out to matter, and they interact.

**Normalisation decides whether asymmetry exists at all.** Dividing both sides
by their total energy forces ``sum(model - obs) = 0``, so the one-sided penalty
``sum(max(model - obs, 0))`` is exactly half the symmetric one and produces an
identical ranking. That identity was hit twice in this project before it was
noticed. Normalising by the strongest arrival, or by a high quantile, breaks it
and lets the asymmetry do something.

**The asymmetry is the right model for clutter.** Furniture adds arrivals to the
recording and can occlude the wall returns the floorplan predicts, but it cannot
invent them. So energy the model predicts and the recording lacks is evidence
against a candidate, while energy the recording has and the model lacks is
expected and should be free.

**The window separates line-of-sight from what vision cannot see.** An arrival
before about 30 ms is a first wall return, which is close to what the camera
already measures. Anything that has visited a region outside line of sight has
reflected at least twice and arrives later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SOUND_SPEED_M_S = 343.0


@dataclass(frozen=True)
class GridScoreConfig:
    """How a candidate grid is compared with a recording.

    bands            frequency band edges in Hz. Measured in the matched domain,
                     0-1 kHz reaches 98-100% recall against 30-38% for 2-4 kHz,
                     because a 69 cm wavelength diffracts around furniture and
                     an 11 cm one scatters off it.
    first_frame      frames before this are dropped. Frame f covers
                     ``f * hop / sample_rate`` seconds of window time, so at the
                     defaults frame 15 is 30 ms, which is where non-line-of-sight
                     paths start to dominate.
    last_frame       None keeps the rest of the window.
    normalise        'sum' divides by total energy and makes one_sided a no-op.
                     'peak' divides by the strongest arrival, which both sides
                     share, and keeps the asymmetry alive. 'quantile' is the
                     robust version of 'peak', unaffected by a single outlier.
    distance         'symmetric' charges for any mismatch. 'one_sided' charges
                     only for model energy the recording lacks.
    shape_feature    True appends the per-row energy split, which is what
                     carries direction on the microphone ring.
    feature          'envelope' sums squared samples inside disjoint windows, so
                     a window of 0.5 ms is 17 cm of path and nothing leaks past
                     it. 'stft_band' buys a frequency axis by analysing with an
                     nfft-sample Hann window, which at nfft 64 smears every
                     frame over 8 ms, or 2.7 m of path, regardless of the hop.
                     On a 0.1 m pose grid that is 27 cells of blur, and it
                     costs almost everything: measured on office_4 the envelope
                     reaches GT percentile 96.6 and the banded STFT about 50,
                     which is chance.
    window_ms        envelope window. 0.5 ms measured best; 2.0 ms already
                     drops the percentile from 96.6 to 91.8.
    """

    sample_rate_hz: int = 8000
    direct_guard_samples: int = 16
    usable_samples: int = 1024
    nfft: int = 64
    hop: int = 16
    bands: tuple = ((0, 500), (500, 1500), (1500, 4000))
    first_frame: int = 0
    last_frame: int | None = None
    normalise: str = "sum"
    distance: str = "symmetric"
    shape_feature: bool = True
    quantile: float = 0.9
    feature: str = "envelope"
    window_ms: float = 0.5

    @property
    def window_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.window_ms / 1000.0)))

    @property
    def n_frames(self) -> int:
        if self.feature == "envelope":
            return self.usable_samples // self.window_samples
        return 1 + (self.usable_samples - self.nfft) // self.hop

    @property
    def ms_per_frame(self) -> float:
        if self.feature == "envelope":
            return self.window_ms
        return 1000.0 * self.hop / self.sample_rate_hz

    def frame_of_ms(self, ms: float) -> int:
        return int(round(ms / self.ms_per_frame))


def to_config_rate(rir: np.ndarray, rate_hz: int, config: GridScoreConfig) -> np.ndarray:
    """Bring a recording to the rate the candidate grid was rendered at.

    The dataset was re-rendered at 48 kHz while the candidate grid stayed at
    8 kHz, and the guard and window are stored as sample counts rather than
    times. Reading a 48 kHz file with an 8 kHz sample count silently takes a
    2 ms guard as 0.33 ms and a 128 ms window as 21 ms, which is most of the
    reflections thrown away. Decimating first keeps the physical convention
    intact whatever rate the file arrives at.
    """
    if rate_hz == config.sample_rate_hz:
        return np.asarray(rir)
    if rate_hz % config.sample_rate_hz:
        raise ValueError(f"{rate_hz} Hz is not an integer multiple of "
                         f"{config.sample_rate_hz} Hz")
    from scipy.signal import decimate
    return decimate(np.asarray(rir, dtype=np.float64),
                    rate_hz // config.sample_rate_hz, axis=-1, ftype="fir")


def band_energy(rir: np.ndarray, config: GridScoreConfig,
                rate_hz: int | None = None) -> np.ndarray:
    """Energy per (channel x band, frame), aligned on the direct peak.

    Both the candidates and the recording are aligned the same way, so a
    constant engine gain and the direct sound both drop out before any
    normalisation choice is made. ``rate_hz`` is the recording's own sample
    rate; anything above the configured one is decimated first.
    """
    rir = np.asarray(rir)
    if rate_hz is not None:
        rir = to_config_rate(rir, rate_hz, config)
    peak = int(np.abs(rir).max(axis=0).argmax())
    s = peak + config.direct_guard_samples
    seg = rir[:, s: s + config.usable_samples]
    if seg.shape[1] < config.usable_samples:
        seg = np.pad(seg, ((0, 0), (0, config.usable_samples - seg.shape[1])))
    if config.feature == "envelope":
        # disjoint windows: no analysis window, so no leakage past the boundary
        w = config.window_samples
        n = config.usable_samples // w
        e = (seg[:, : n * w] ** 2).reshape(seg.shape[0], n, w).sum(-1)
        return e[:, config.first_frame: config.last_frame].astype(np.float32)

    freqs = np.fft.rfftfreq(config.nfft, 1.0 / config.sample_rate_hz)
    sel = [(freqs >= lo) & (freqs < hi) for lo, hi in config.bands]
    w = np.hanning(config.nfft)
    # One batched transform rather than a Python loop over frames. At 48 kHz a
    # 128 ms window is 381 frames, and the loop made a whole-grid featurisation
    # cost minutes per configuration, which is enough to stop a sweep being run
    # at all. `test_stft_vectorisation_matches_the_loop` pins the equivalence.
    from numpy.lib.stride_tricks import sliding_window_view
    frames = sliding_window_view(seg, config.nfft, axis=-1)[:, :: config.hop]
    frames = frames[:, : config.n_frames]                      # (C, T, nfft)
    power = np.abs(np.fft.rfft(frames * w, axis=-1)) ** 2      # (C, T, F)
    out = np.stack([power[:, :, m].sum(-1) for m in sel], axis=1)  # (C, B, T)
    e = out.reshape(seg.shape[0] * len(config.bands), config.n_frames)
    return e[:, config.first_frame: config.last_frame].astype(np.float32)


def normalise(e: np.ndarray, config: GridScoreConfig) -> np.ndarray:
    """Put a candidate and a recording on a common scale.

    They come from separate renders, so a gain has to be fixed somehow. Which
    way is not cosmetic: 'sum' is the only one that forces the two sides to have
    equal total energy, and that is precisely the condition under which a
    one-sided penalty carries no information the symmetric one does not.
    """
    axes = tuple(range(e.ndim))[-2:]
    if config.normalise == "sum":
        d = e.sum(axis=axes, keepdims=True)
    elif config.normalise == "peak":
        d = e.max(axis=axes, keepdims=True)
    elif config.normalise == "quantile":
        flat = e.reshape(*e.shape[:-2], -1)
        d = np.quantile(flat, config.quantile, axis=-1)[..., None, None]
    else:
        raise ValueError(f"unknown normalise {config.normalise!r}")
    return e / np.clip(d, 1e-20, None)


def shape(e: np.ndarray) -> np.ndarray:
    """Per-row temporal shape plus the energy split across rows."""
    per = e / np.clip(e.sum(axis=-1, keepdims=True), 1e-20, None)
    total = e.sum(axis=(-1, -2))
    split = e.sum(axis=-1) / np.clip(total[..., None], 1e-20, None)
    return np.concatenate([per, np.repeat(split[..., None], 4, axis=-1)], axis=-1)


def featurise(e: np.ndarray, config: GridScoreConfig) -> np.ndarray:
    n = normalise(e, config)
    return shape(n) if config.shape_feature else n


def candidate_features(grid_npz, rows, cols, mask_shape, config: GridScoreConfig):
    """Features for every valid cell, plus which cells have a rendered candidate.

    Returns ``(features, present)`` where ``features`` is aligned to the
    ``(rows, cols)`` order the caller uses everywhere else.
    """
    blob = np.load(grid_npz)
    e = np.stack([band_energy(c, config) for c in blob["rir"]])
    feat = featurise(e, config)
    idx = blob["index"]
    lut = -np.ones(mask_shape, dtype=int)
    lut[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
    pick = lut[rows, cols]
    out = np.zeros((len(rows), feat.shape[1], feat.shape[2]), dtype=np.float32)
    out[pick >= 0] = feat[pick[pick >= 0]]
    return out, pick >= 0


def observation_feature(rir: np.ndarray, config: GridScoreConfig,
                        rate_hz: int | None = None) -> np.ndarray:
    return featurise(band_energy(rir, config, rate_hz), config)


def observation_rate(rir_path) -> int:
    """The recording's sample rate, from the metadata beside it.

    Defaults to the configured rate when no metadata is present, which is the
    case for the older renders.
    """
    import json
    m = Path(rir_path).parent / "rir_metadata.json"
    if m.exists():
        try:
            return int(json.loads(m.read_text())["sample_rate_hz"])
        except Exception:
            pass
    return 8000


def score(cand: np.ndarray, obs: np.ndarray, present: np.ndarray,
          config: GridScoreConfig) -> np.ndarray:
    """Per-candidate score, higher is better.

    Cells with no rendered candidate are pushed below every real one rather
    than dropped, so the returned array stays aligned with the caller's cells.
    """
    nw = min(cand.shape[2], obs.shape[1])
    diff = cand[:, :, :nw] - obs[None, :, :nw]
    if config.distance == "symmetric":
        d = np.abs(diff)
    elif config.distance == "one_sided":
        d = np.clip(diff, 0, None)
    else:
        raise ValueError(f"unknown distance {config.distance!r}")
    s = -d.sum(axis=(1, 2))
    return np.where(present, s, s.min() - 1.0)


def asymmetry_is_degenerate(config: GridScoreConfig) -> bool:
    """True when one_sided provably cannot differ from symmetric.

    Two separate ways to reach the same dead end, both of which were hit by
    measurement in this project before either was noticed.

    Sum normalisation makes both sides integrate to one, so ``sum(m - o) = 0``
    and ``sum(max(m - o, 0))`` is exactly half of ``sum|m - o|``.

    ``shape_feature`` does it again and more thoroughly: it divides every row by
    its own sum, so each row of both sides integrates to one whatever
    normalisation ran before it. That also makes the ``normalise`` choice itself
    a no-op, which is why a sweep over three normalisations once returned three
    identical columns.
    """
    if config.distance != "one_sided":
        return False
    return config.normalise == "sum" or config.shape_feature


def normalisation_is_a_noop(config: GridScoreConfig) -> bool:
    """``shape_feature`` is scale-free, so it erases whatever normalise did."""
    return config.shape_feature
