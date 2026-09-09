"""One-sided acoustic evidence matching between a 2D floorplan and a real room.

The existing score in ``acoustic.py`` normalises a synthesised proxy envelope and
a measured envelope and takes their symmetric L1. That asks the floorplan to
*equal* the room. It cannot: a real room contains furniture reflections, object
scattering, floor and ceiling paths, and clutter that a 2D floorplan has no way
to predict, and every one of those costs the candidate under a symmetric
distance.

What the floorplan can support is the weaker, checkable claim

    predicted geometric paths  are a subset of  real acoustic arrivals

so the score here only asks whether each floorplan-predicted arrival is
*supported* by the observation. Observed energy with no predicted counterpart is
free. That is the whole idea, and it is what makes the score robust to clutter
rather than merely tolerant of it.

Three things follow from taking that seriously.

**Delay carries the geometry, amplitude does not.** A path length is fixed by
the map; its amplitude depends on material, source gain, microphone response,
and everything the map omits. So matching is on arrival time, and predicted
amplitude enters only as an optional weak confidence (see ``confidence_mode``).

**Timing resolution has to beat the array.** The 6-microphone ring has radius
5 cm, so the largest inter-microphone path difference is about 10 cm, which at
8 kHz is 2.3 samples. The 2 ms (16-sample) binning used by the envelope score
destroys that outright. The default here is 0.25 ms.

**Clutter removes predicted paths too.** Furniture occludes wall returns as well
as adding its own, so requiring every predicted path to be supported is too
strong. Aggregation keeps the best ``top_fraction`` of predicted events, which
lets a minority go missing without condemning the candidate.

**The ring cannot contribute through energy.** ``event_match_score`` accepts a
six-channel agreement term, but a predicted one is not built, and the reason is
physical rather than pending work. An arrival reaching a 5 cm ring from 3 m away
is a near-plane wave, so the six microphone path lengths differ by at most 10 cm
and their ``1/r^2`` energies by about 3 per cent. Any predicted ring vector is
therefore nearly uniform, and its cosine similarity against an observed one is
nearly 1 for every candidate. What does carry direction is the sub-sample
*delay* across the ring, which at 8 kHz spans 2.3 samples and would need
cross-channel time-difference estimation rather than an energy comparison. The
hook is kept for that; the energy version is not worth reporting.

Nothing is learned, fitted, or tuned against ground truth. Every parameter is in
physical units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOUND_SPEED_M_S = 343.0


@dataclass(frozen=True)
class EventMatchConfig:
    """Parameters for one-sided acoustic evidence matching, all in physical units.

    sample_rate_hz          sampling rate of the observation
    sound_speed_m_s         speed of sound used for every delay conversion
    direct_guard_samples    samples skipped after the direct peak. The direct
                            sound is orders of magnitude louder than any
                            reflection and carries no room geometry.
    usable_samples          length of the reflection window that follows
    profile_resolution_ms   bin width of the dense observed profile. 0.25 ms is
                            2 samples at 8 kHz, just under the 2.3-sample
                            inter-microphone maximum, so ring timing survives.
    smoothing_ms            width of the Hann kernel applied to the sample-rate
                            energy before peak picking
    min_peak_separation_ms  minimum spacing between observed peaks. 0.5 ms is
                            17 cm of path, below which two arrivals are not
                            separable at this bandwidth anyway.
    peak_dynamic_range_db   observed peaks weaker than this far below the
                            strongest reflection are discarded. This is the
                            parameter that decides whether the observation is
                            *sparse*: a floorplan predicts on the order of ten
                            distinct wall returns, so retaining forty observed
                            peaks over the same span leaves a mean spacing near
                            the matching tolerance, at which point every
                            candidate is supported and the score carries no
                            information. 10 dB is one order of magnitude in
                            energy and is set from that argument, not from
                            ranking performance; sensitivity is reported.
    decay_window_ms         width of the running mean that estimates the local
                            reverberant decay. Energy falls by orders of
                            magnitude across the window, so a threshold taken
                            against the global maximum keeps only early
                            arrivals and silently truncates the observation
                            long before the floorplan stops predicting paths.
                            Dividing by the local trend first makes a peak's
                            strength mean "prominent against its own decay",
                            which is comparable at 5 ms and at 50 ms. Measured
                            result: it does recover the late arrivals, and the
                            observed set grows from 14 events to 38, at which
                            point the density failure returns and the ranking
                            gets worse rather than better. Off by default and
                            reported as an ablation; 6.0 enables it.
    max_observed_events     hard cap on retained observed peaks
    noise_quantile          quantile of the reflection window used as the robust
                            level reference, instead of the maximum, so one loud
                            arrival cannot set the scale for everything else
    min_delay_ms            arrivals earlier than this are dropped from *both*
                            sides. Source and receiver sit at a fixed mounting
                            height in a room of unknown height, so the floor and
                            ceiling bounces are the earliest and strongest
                            arrivals in the room, yet a 2D floorplan cannot
                            predict them and their delay hardly changes with
                            position. Excluding the window they occupy removes a
                            nuisance the map cannot explain, at the cost of also
                            discarding genuine returns from walls closer than
                            ``c * min_delay_ms / 2``. See
                            ``vertical_nuisance_cutoff_ms``.
    tau_tolerance_ms        sigma of the temporal compatibility kernel. 1 ms is
                            34 cm of path length; a 0.1 m pose grid moves a
                            round-trip path by about 0.2 m = 0.6 ms between
                            neighbouring cells, so this tolerates rasterisation
                            and wall-thickness error without blurring the grid.
    min_predicted_separation_ms
                            predicted arrivals closer than this are merged into
                            one event. A flat wall returns many rays at nearly
                            the same delay; those are one arrival, not many.
    top_fraction            fraction of predicted events retained by the robust
                            aggregation. Below about 0.4 a candidate can be
                            carried by a few accidental coincidences.
    confidence_mode         'solid_angle' weights a predicted event by how many
                            rays produced it, which is geometric. 'uniform'
                            ignores weighting entirely. 'inverse_square' adds
                            1/r^2 spreading and is the amplitude ablation.
    uniqueness              'none' lets one observed peak support any number of
                            predicted events. 'soft' divides its support by the
                            total demand on it, vectorised and cheap. 'greedy'
                            is exact one-to-one and is used for diagnostics.
    """

    sample_rate_hz: int = 8000
    sound_speed_m_s: float = SOUND_SPEED_M_S
    direct_guard_samples: int = 16
    usable_samples: int = 1024
    profile_resolution_ms: float = 0.25
    smoothing_ms: float = 0.25
    min_peak_separation_ms: float = 0.5
    peak_dynamic_range_db: float = 10.0
    max_observed_events: int = 40
    use_observed_confidence: bool = True
    decay_window_ms: float = 0.0
    receiver_radius_m: float = 0.05
    noise_quantile: float = 0.9
    tau_tolerance_ms: float = 1.0
    min_delay_ms: float = 0.0
    min_predicted_separation_ms: float = 0.5
    top_fraction: float = 0.6
    confidence_mode: str = "solid_angle"
    uniqueness: str = "soft"

    @property
    def profile_bin_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.profile_resolution_ms / 1000.0)))

    @property
    def window_ms(self) -> float:
        """Duration of the reflection window, in milliseconds."""
        return 1000.0 * self.usable_samples / self.sample_rate_hz

    @property
    def window_offset_ms(self) -> float:
        """Gap between absolute path time and time within the reflection window.

        The RIR clock starts at emission, so the direct sound peaks after it has
        crossed the microphone ring radius, and the reflection window then starts
        ``direct_guard_samples`` later still:

            window time = absolute path time - (r/c + guard/fs)

        For a 5 cm ring at 8 kHz with a 16-sample guard this is 2.15 ms. It is
        larger than ``tau_tolerance_ms``, so predicted and observed arrivals do
        not overlap at all unless it is applied.
        """
        return (self.receiver_radius_m / self.sound_speed_m_s
                + self.direct_guard_samples / self.sample_rate_hz) * 1000.0

    def path_m_to_ms(self, path_m):
        """Absolute path length in metres to time within the reflection window."""
        return np.asarray(path_m) / self.sound_speed_m_s * 1000.0 - self.window_offset_ms


def vertical_nuisance_cutoff_ms(source_height_m: float = 1.25,
                                ceiling_range_m: tuple[float, float] = (2.3, 3.2),
                                sound_speed_m_s: float = SOUND_SPEED_M_S,
                                window_offset_ms: float = 2.146,
                                margin_ms: float = 0.5) -> float:
    """Latest first-order floor/ceiling arrival over a plausible ceiling range.

    The floor bounce is a round trip of ``2h`` and the ceiling bounce ``2(H-h)``.
    Neither depends on the floorplan, both are among the closest surfaces in any
    room, and the mounting height ``h`` is a property of the rig rather than of
    the scene. Taking the worst case over a range of ordinary ceiling heights
    gives a cutoff that needs no per-scene knowledge:

        cutoff = max(2h, 2(H_max - h)) / c

    For h = 1.25 m and ceilings between 2.3 and 3.2 m this is about 11.4 ms,
    which corresponds to walls within 1.95 m. Returns from nearer walls are lost
    with the nuisance; that is the price of not knowing the room height.
    """
    lo, hi = ceiling_range_m
    longest_m = max(2.0 * source_height_m, 2.0 * (hi - source_height_m))
    # returned in window time, the same clock the matching uses
    return longest_m / sound_speed_m_s * 1000.0 - window_offset_ms + margin_ms


@dataclass(frozen=True)
class EventSet:
    """Sparse acoustic arrivals.

    delay_ms    arrival time relative to the direct peak, milliseconds
    weight      relative energy (observed) or geometric confidence (predicted)
    kind        integer path-type label, see ``PATH_TYPES``
    ring        optional ``(n, 6)`` per-channel energy at the arrival, used only
                by the ring-consistency ablation
    """

    delay_ms: np.ndarray
    weight: np.ndarray
    kind: np.ndarray | None = None
    ring: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.delay_ms.shape[0])


# Path-type labels. Kept explicit so an ablation can drop one family.
PATH_TYPES = {"radial": 0, "higher_order": 1, "corner_nlos": 2, "vertical": 3}


# --------------------------------------------------------------------------
# observation side
# --------------------------------------------------------------------------
def align_reflection_window(rir: np.ndarray, config: EventMatchConfig) -> np.ndarray:
    """Reflection window of a measured RIR, ``(6, usable_samples)``.

    The window starts ``direct_guard_samples`` after the direct peak, which is
    the convention ``dataset_meta.json`` fixes and the one the existing envelope
    score already uses, so both scores see identical samples.
    """
    rir = np.asarray(rir, dtype=np.float64)
    if rir.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {rir.shape}")
    peak = int(np.abs(rir).max(axis=0).argmax())
    start = peak + config.direct_guard_samples
    seg = rir[:, start: start + config.usable_samples]
    if seg.shape[1] < config.usable_samples:
        seg = np.pad(seg, ((0, 0), (0, config.usable_samples - seg.shape[1])))
    return seg


def observed_profile(rir: np.ndarray, config: EventMatchConfig) -> tuple[np.ndarray, np.ndarray]:
    """Dense channel-summed energy profile, as ``(delay_ms, energy)``.

    Channel-summed because the primary formulation is delay-only: the source
    sits at the array centre, so arrival time is a function of position alone
    and yaw is left to vision. The profile is smoothed by a short Hann kernel
    before binning so that peak picking is not decided by a single sample.
    """
    seg = align_reflection_window(rir, config)
    energy = (seg ** 2).sum(axis=0)

    k = max(1, int(round(config.sample_rate_hz * config.smoothing_ms / 1000.0)))
    if k > 1:
        kernel = np.hanning(k + 2)[1:-1]
        energy = np.convolve(energy, kernel / kernel.sum(), mode="same")

    b = config.profile_bin_samples
    n = energy.shape[0] // b
    binned = energy[: n * b].reshape(n, b).sum(axis=1)
    times = (np.arange(n) + 0.5) * b / config.sample_rate_hz * 1000.0
    return times, binned


def _decay_compensate(energy: np.ndarray, config: EventMatchConfig) -> np.ndarray:
    """Normalise the profile against its own reverberant decay.

    A room's energy decays roughly exponentially, so an arrival at 40 ms can be
    30 dB below one at 6 ms and still be the clearest feature of its own
    neighbourhood. Thresholding against the global maximum therefore does not
    select the strongest *arrivals*, it selects the earliest ones, and the
    observed set stops long before the floorplan stops predicting paths. The
    running mean below is the local trend; dividing by it makes prominence
    comparable across the whole window.
    """
    if config.decay_window_ms <= 0:
        ref = np.quantile(energy[energy > 0], config.noise_quantile) if (energy > 0).any() else 1.0
        return energy / max(float(ref), 1e-30)
    k = max(3, int(round(config.decay_window_ms / config.profile_resolution_ms)) | 1)
    pad = np.pad(energy, (k // 2, k // 2), mode="edge")
    trend = np.convolve(pad, np.ones(k) / k, mode="valid")[: energy.shape[0]]
    floor = np.quantile(energy[energy > 0], 0.25) if (energy > 0).any() else 1e-30
    return energy / np.clip(trend, max(float(floor), 1e-30), None)


def observed_events(rir: np.ndarray, config: EventMatchConfig,
                    with_ring: bool = False) -> EventSet:
    """Extract sparse arrivals from a measured RIR by robust local maxima.

    No learning and no absolute-amplitude threshold: the level reference is a
    high quantile of the window rather than its maximum, so a single loud
    arrival cannot suppress everything after it, and peaks are taken as local
    maxima subject to a minimum separation set by what the bandwidth can
    actually resolve.
    """
    times, energy = observed_profile(rir, config)
    if energy.size == 0 or not np.isfinite(energy).any():
        return EventSet(np.zeros(0), np.zeros(0), np.zeros(0, dtype=int))

    norm = _decay_compensate(energy, config)

    interior = np.zeros(norm.shape[0], dtype=bool)
    interior[1:-1] = (norm[1:-1] >= norm[:-2]) & (norm[1:-1] >= norm[2:])
    cand = np.flatnonzero(interior & (norm > 0))
    if cand.size == 0:
        cand = np.array([int(norm.argmax())])

    sep_bins = max(1, int(round(config.min_peak_separation_ms / config.profile_resolution_ms)))
    if config.min_delay_ms > 0:
        cand = cand[times[cand] >= config.min_delay_ms]
        if cand.size == 0:
            return EventSet(np.zeros(0), np.zeros(0), np.zeros(0, dtype=int))
    ranked = cand[np.argsort(-norm[cand])]
    # keep only arrivals within the configured dynamic range of the strongest
    # reflection: the rest are late diffuse tail, and retaining them makes the
    # observed set dense enough that any predicted delay finds a neighbour
    floor = norm[ranked[0]] * 10.0 ** (-config.peak_dynamic_range_db / 10.0)
    keep: list[int] = []
    for i in ranked:
        if norm[i] < floor:
            break
        if all(abs(int(i) - j) >= sep_bins for j in keep):
            keep.append(int(i))
        if len(keep) >= config.max_observed_events:
            break
    keep = sorted(keep)

    ring = None
    if with_ring:
        seg = align_reflection_window(rir, config)
        b = config.profile_bin_samples
        n = seg.shape[1] // b
        per = (seg[:, : n * b] ** 2).reshape(seg.shape[0], n, b).sum(axis=2)   # (6, n)
        ring = per[:, keep].T                                                  # (n_keep, 6)
        ring = ring / np.clip(ring.sum(axis=1, keepdims=True), 1e-30, None)

    return EventSet(delay_ms=times[keep], weight=norm[keep],
                    kind=np.zeros(len(keep), dtype=int), ring=ring)


def pooled_observed_profile(rir: np.ndarray, config: EventMatchConfig) -> tuple[np.ndarray, np.ndarray]:
    """Observed profile after a max-pool of +/- ``tau_tolerance_ms``.

    This is the dense counterpart of the sparse matching: reading the pooled
    profile at a predicted delay answers "is there observed energy within
    tolerance of where the map says an arrival should be", which is exactly
    ``max_j K_ij`` with a boxcar kernel instead of a Gaussian one.
    """
    times, energy = observed_profile(rir, config)
    norm = _decay_compensate(energy, config)
    half = int(round(config.tau_tolerance_ms / config.profile_resolution_ms))
    if half > 0:
        pad = np.pad(norm, (half, half), mode="constant", constant_values=0.0)
        norm = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1).max(axis=-1)
    # Scale to [0, 1] against this observation's own strongest arrival, here and
    # not in the score: normalising inside the score would divide by the maximum
    # over whatever candidates happened to be in the batch, so a single-candidate
    # call would always self-normalise to a perfect 1.0.
    return times, norm / max(float(norm.max()), 1e-30)


# --------------------------------------------------------------------------
# prediction side
# --------------------------------------------------------------------------
def merge_predicted(delay_ms: np.ndarray, weight: np.ndarray, kind: np.ndarray,
                    config: EventMatchConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse near-duplicate predicted arrivals into single events.

    A flat wall facing the source returns many rays within a fraction of a
    millisecond of each other. Those are one arrival. Merging them keeps the
    event count proportional to how much distinct structure the map predicts,
    rather than to the ray count, and the merged weight is the summed weight,
    so a wall subtending a larger solid angle stays more important.
    """
    valid = np.isfinite(delay_ms)
    if not valid.any():
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    d, w, k = delay_ms[valid], weight[valid], kind[valid]
    order = np.argsort(d)
    d, w, k = d[order], w[order], k[order]
    # a new event starts wherever the gap to the previous arrival exceeds the
    # separation the bandwidth can resolve, evaluated per path type
    new = np.ones(d.shape[0], dtype=bool)
    new[1:] = (np.diff(d) >= config.min_predicted_separation_ms) | (k[1:] != k[:-1])
    gid = np.cumsum(new) - 1
    n = int(gid[-1]) + 1
    wsum = np.bincount(gid, weights=w, minlength=n)
    dsum = np.bincount(gid, weights=w * d, minlength=n)
    centre = dsum / np.clip(wsum, 1e-30, None)
    first = np.zeros(n, dtype=int)
    first[gid[::-1]] = np.arange(d.shape[0])[::-1]
    return centre, wsum, k[first]


def predicted_events_from_echoes(echoes: dict, config: EventMatchConfig,
                                 origins_px: np.ndarray, map_resolution_m: float,
                                 max_order: int | None = None) -> list[EventSet]:
    """Turn traced floorplan echoes into per-candidate sparse path tokens.

    ``echoes`` is what ``acoustic.trace_echoes`` returns: ``path_m``, ``hit_x``,
    ``hit_y`` and ``alive``, each ``(N, R, K)``. Source and receiver are
    co-located at the array centre, so the closing leg is the straight line from
    the hit back to the origin and the total path is position-only. That is what
    makes the score yaw-invariant and cheap to broadcast.

    Order 1 is the round trip to the first wall, ``2d``. This is a radial
    wall-return proxy, *not* an exact specular path: only the ray striking a
    surface at normal incidence returns to a co-located receiver, so the closing
    leg here is a diffuse-return assumption. It is deliberately that, because
    the target is map topology rather than material-faithful acoustics.
    """
    path = np.asarray(echoes["path_m"])
    hx, hy = np.asarray(echoes["hit_x"]), np.asarray(echoes["hit_y"])
    alive = np.asarray(echoes["alive"])
    N, R, K = path.shape
    if max_order is not None:
        K = min(K, max_order)
    origins = np.asarray(origins_px, dtype=np.float64)

    out: list[EventSet] = []
    for i in range(N):
        d_all, w_all, k_all = [], [], []
        for k in range(K):
            ok = alive[i, :, k] & np.isfinite(path[i, :, k])
            if not ok.any():
                continue
            close = np.hypot(hx[i, ok, k] - origins[i, 0],
                             hy[i, ok, k] - origins[i, 1]) * map_resolution_m
            total_m = path[i, ok, k] + close
            if config.confidence_mode == "uniform":
                w = np.ones(total_m.shape[0])
            elif config.confidence_mode == "inverse_square":
                w = 1.0 / np.clip(total_m, 0.1, None) ** 2
            else:                                   # solid_angle
                w = np.full(total_m.shape[0], 1.0 / R)
            d_all.append(config.path_m_to_ms(total_m))
            w_all.append(w)
            k_all.append(np.full(total_m.shape[0],
                                 PATH_TYPES["radial"] if k == 0 else PATH_TYPES["higher_order"]))
        if not d_all:
            out.append(EventSet(np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)))
            continue
        d, w, kk = merge_predicted(np.concatenate(d_all), np.concatenate(w_all),
                                   np.concatenate(k_all), config)
        inside = (d < config.window_ms) & (d >= config.min_delay_ms)
        out.append(EventSet(d[inside], w[inside], kk[inside]))
    return out


def pad_event_sets(events: list[EventSet]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack ragged per-candidate event sets into ``(N, P)`` arrays.

    Padding is NaN in delay and zero in weight, so padded slots contribute
    nothing to either the kernel or the aggregation.
    """
    P = max((len(e) for e in events), default=0)
    N = len(events)
    delay = np.full((N, max(P, 1)), np.nan)
    weight = np.zeros((N, max(P, 1)))
    kind = np.full((N, max(P, 1)), -1, dtype=int)
    for i, e in enumerate(events):
        n = len(e)
        if n:
            delay[i, :n] = e.delay_ms
            weight[i, :n] = e.weight
            kind[i, :n] = e.kind if e.kind is not None else 0
    return delay, weight, kind


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _robust_aggregate(support: np.ndarray, weight: np.ndarray,
                      top_fraction: float) -> np.ndarray:
    """Weighted mean of the best ``top_fraction`` of predicted events.

    Unknown clutter both adds arrivals and *removes* predicted ones by occluding
    the surface that would have produced them. The one-sided kernel handles the
    additions; this handles the removals, by not requiring every predicted event
    to be supported. Weights are the geometric confidences, so dropping a event
    that subtended a large solid angle still costs more than dropping a sliver.
    """
    n_valid = (weight > 0).sum(axis=1)
    masked = np.where(weight > 0, support, -np.inf)
    order = np.argsort(-masked, axis=1)
    s = np.take_along_axis(support, order, axis=1)
    w = np.take_along_axis(weight, order, axis=1)
    rank = np.arange(s.shape[1])[None, :]
    keep_n = np.maximum(1, np.ceil(top_fraction * n_valid)).astype(int)[:, None]
    m = (rank < keep_n) & (w > 0)
    wm = np.where(m, w, 0.0)
    denom = wm.sum(axis=1)
    return np.where(denom > 0, (wm * np.where(m, s, 0.0)).sum(axis=1) / np.clip(denom, 1e-30, None), 0.0)


def _apply_uniqueness(K: np.ndarray, mode: str) -> np.ndarray:
    """Stop one observed arrival from explaining unlimited predicted events.

    'soft' divides each observed event's contribution by the total demand placed
    on it, which is one Sinkhorn half-step: fully vectorised, no sorting, and it
    degrades smoothly rather than making a hard assignment that a millisecond of
    jitter could flip.
    """
    if mode == "none":
        return K
    if mode == "soft":
        demand = K.sum(axis=1, keepdims=True)          # (N, 1, J)
        return K / np.clip(demand, 1.0, None)
    if mode == "greedy":
        out = np.zeros_like(K)
        for n in range(K.shape[0]):
            k = K[n]
            pi, oj = np.unravel_index(np.argsort(-k, axis=None), k.shape)
            used_p, used_o = set(), set()
            for a, b in zip(pi, oj):
                if k[a, b] <= 0:
                    break
                if a in used_p or b in used_o:
                    continue
                out[n, a, b] = k[a, b]
                used_p.add(int(a)); used_o.add(int(b))
        return out
    raise ValueError(f"unknown uniqueness mode {mode!r}")


def event_match_score(pred_delay: np.ndarray, pred_weight: np.ndarray,
                      obs: EventSet, config: EventMatchConfig,
                      pred_ring: np.ndarray | None = None,
                      use_ring: bool = False, chunk: int = 4096) -> np.ndarray:
    """One-sided sparse matching score for ``(N, P)`` predicted events.

    For predicted event ``i`` and observed event ``j``

        K_ij = exp(-0.5 ((tau_i - tau_j) / sigma_tau)^2)

    optionally multiplied by a scale-free six-channel agreement term, then

        m_i   = max_j K_ij          (after the uniqueness constraint)
        score = weighted mean of the best ``top_fraction`` of {m_i}

    Only predicted events enter the aggregation, so observed arrivals the map
    cannot explain cost nothing. Higher is better; the range is 0 to 1.
    """
    N = pred_delay.shape[0]
    if len(obs) == 0:
        return np.zeros(N)
    sigma = max(config.tau_tolerance_ms, 1e-6)
    obs_t = obs.delay_ms[None, None, :]
    # matching a strong arrival is better evidence than matching a weak one, so
    # each observed event can offer at most its own relative strength as support
    conf = np.ones_like(obs.delay_ms)
    if config.use_observed_confidence and obs.weight.size and obs.weight.max() > 0:
        conf = np.clip(obs.weight / obs.weight.max(), 0.0, 1.0)
    out = np.zeros(N)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        dt = (np.nan_to_num(pred_delay[s:e, :, None], nan=1e9) - obs_t) / sigma
        K = np.exp(-0.5 * dt ** 2) * conf[None, None, :]
        if use_ring and pred_ring is not None and obs.ring is not None:
            # scale-free directional agreement, cosine between normalised
            # six-channel energy vectors at the two arrivals
            a = pred_ring[s:e]                                     # (n, P, 6)
            a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-30, None)
            b = obs.ring / np.clip(np.linalg.norm(obs.ring, axis=-1, keepdims=True), 1e-30, None)
            K = K * np.clip(np.einsum("npc,jc->npj", a, b), 0.0, None)
        K = _apply_uniqueness(K, config.uniqueness)
        out[s:e] = _robust_aggregate(K.max(axis=2), pred_weight[s:e], config.top_fraction)
    return out


def asymmetric_energy_score(pred_delay: np.ndarray, pred_weight: np.ndarray,
                            profile_times: np.ndarray, pooled: np.ndarray,
                            config: EventMatchConfig) -> np.ndarray:
    """One-sided score against the dense max-pooled observed profile.

    Same asymmetry as ``event_match_score`` without the sparsification: each
    predicted arrival is looked up in an observation that has been dilated by
    the temporal tolerance, so the question is still "is there observed energy
    where the map predicts an arrival", and extra observed energy is still free.
    """
    res = config.profile_resolution_ms
    idx = np.clip(np.round((np.nan_to_num(pred_delay, nan=-1.0) - profile_times[0]) / res),
                  0, pooled.shape[0] - 1).astype(int)
    support = np.where(np.isfinite(pred_delay) & (pred_delay >= config.min_delay_ms),
                       pooled[idx], 0.0)
    return _robust_aggregate(support, pred_weight, config.top_fraction)
