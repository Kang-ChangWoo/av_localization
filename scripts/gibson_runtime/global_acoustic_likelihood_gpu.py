"""CUDA kernels for global F3Loc planar acoustic likelihood evaluation."""

from __future__ import annotations

from typing import Any

import torch

from planar_acoustic_likelihood import ECHOSCAN_RING_ANGLES_RAD, PlanarRirConfig


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Global acoustic likelihood requires torch CUDA.")


def _add_fractional_impulses(signal: torch.Tensor, samples: torch.Tensor, amplitudes: torch.Tensor) -> None:
    """In-place splat of `[B,O,C,R]` impulses into `[B,O,C,S]` signals."""
    samples = samples.clamp(0.0, signal.shape[-1] - 1.000001)
    lower = samples.floor().long()
    fraction = samples - lower
    signal.scatter_add_(-1, lower, amplitudes * (1.0 - fraction))
    upper = (lower + 1).clamp_max(signal.shape[-1] - 1)
    signal.scatter_add_(-1, upper, amplitudes * fraction)


def _energy_feature(rir: torch.Tensor, config: PlanarRirConfig) -> torch.Tensor:
    window = config.energy_window_samples
    samples = min(rir.shape[-1], config.sample_count)
    samples -= samples % window
    if samples <= 0:
        raise ValueError("RIR is shorter than one configured energy window.")
    energy = rir[..., :samples].reshape(*rir.shape[:-1], -1, window).square().sum(dim=-1)
    peak = energy.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    logged = torch.log1p(energy / peak)
    return logged / torch.linalg.vector_norm(logged, dim=-1, keepdim=True).clamp_min(1e-12)


def _peak_time_distance(proxy_energy: torch.Tensor, observed_energy: torch.Tensor, peak_count: int) -> torch.Tensor:
    """Return `[observations, batch, yaw]` normalized early-peak distance."""
    proxy_peaks = proxy_energy.topk(k=min(peak_count, proxy_energy.shape[-1]), dim=-1).indices
    observed_peaks = observed_energy.topk(k=min(peak_count, observed_energy.shape[-1]), dim=-1).indices
    differences = (proxy_peaks[None, :, :, :, :, None] - observed_peaks[:, None, None, :, None, :]).abs()
    return differences.min(dim=-1).values.float().mean(dim=(-1, -2)) / max(proxy_energy.shape[-1] - 1, 1)


def score_wall_distance_batch_cuda(
    wall_distances_m: torch.Tensor,
    observed_rirs: torch.Tensor,
    config: PlanarRirConfig,
) -> dict[str, torch.Tensor]:
    """Score `[B,R]` wall-distance profiles against `[N,6,S]` observations on CUDA.

    Candidate position cancels from the source-centred planar path geometry; its
    first-wall distance profile is expanded across the 36 F3Loc yaw bins and
    the established six-channel ring here.
    """
    _require_cuda()
    distances = wall_distances_m.to(device="cuda", dtype=torch.float32)
    observations = observed_rirs.to(device="cuda", dtype=torch.float32)
    if distances.ndim != 2 or distances.shape[1] != config.ray_count:
        raise ValueError("Wall distances must have shape `(batch, configured ray count)`.")
    if observations.ndim != 3 or observations.shape[1] != 6:
        raise ValueError("Observed RIRs must have shape `(observations, 6, samples)`.")
    if not torch.isfinite(observations).all():
        raise ValueError("Observed RIR inputs must be finite.")
    wall_present = torch.isfinite(distances)
    distances = torch.nan_to_num(distances, nan=0.0)

    device = distances.device
    batch_size, ray_count = distances.shape
    yaw_count = 36
    sample_count = config.sample_count
    ray_angles = torch.arange(ray_count, device=device, dtype=torch.float32) * (2.0 * torch.pi / ray_count)
    ray_directions = torch.stack((ray_angles.cos(), ray_angles.sin()), dim=-1)
    yaw_angles = torch.arange(yaw_count, device=device, dtype=torch.float32) * (2.0 * torch.pi / yaw_count)
    ring_angles = torch.as_tensor(ECHOSCAN_RING_ANGLES_RAD, device=device, dtype=torch.float32)
    microphone_angles = yaw_angles[:, None] + ring_angles[None, :]
    offsets = config.receiver_radius_m * torch.stack((microphone_angles.cos(), microphone_angles.sin()), dim=-1)

    wall_relative = distances[..., None] * ray_directions[None, :, :]
    return_leg = torch.linalg.vector_norm(wall_relative[:, :, None, None, :] - offsets[None, None, :, :, :], dim=-1)
    path_distance = distances[:, :, None, None] + return_leg
    path_samples = path_distance.permute(0, 2, 3, 1) * (config.sample_rate_hz / config.sound_speed_m_s)
    amplitudes = (config.wall_reflection_gain / path_distance.clamp_min(0.1).sqrt() / ray_count).permute(0, 2, 3, 1)
    amplitudes = amplitudes * wall_present[:, None, None, :]
    rir = torch.zeros((batch_size, yaw_count, 6, sample_count), device=device, dtype=torch.float32)
    _add_fractional_impulses(rir, path_samples, amplitudes)

    direct_samples = config.receiver_radius_m / config.sound_speed_m_s * config.sample_rate_hz
    direct_lower = int(direct_samples)
    direct_fraction = direct_samples - direct_lower
    rir[..., direct_lower] += config.direct_gain * (1.0 - direct_fraction)
    if direct_lower + 1 < sample_count:
        rir[..., direct_lower + 1] += config.direct_gain * direct_fraction

    proxy_energy_raw = rir[..., : config.sample_count].reshape(
        batch_size, yaw_count, 6, -1, config.energy_window_samples
    ).square().sum(dim=-1)
    observed_energy_raw = observations[..., : config.sample_count].reshape(
        observations.shape[0], 6, -1, config.energy_window_samples
    ).square().sum(dim=-1)
    proxy_feature = _energy_feature(rir, config)
    observed_feature = _energy_feature(observations, config)
    envelope_mse = (proxy_feature[None] - observed_feature[:, None, None]).square().mean(dim=(-1, -2))
    peak_distance = _peak_time_distance(proxy_energy_raw, observed_energy_raw, config.peak_count)
    mismatch = 0.8 * envelope_mse + 0.2 * peak_distance
    return {
        "mismatch": mismatch,
        "likelihood": torch.exp(-mismatch),
        "envelope_mse": envelope_mse,
        "peak_time_distance": peak_distance,
    }
