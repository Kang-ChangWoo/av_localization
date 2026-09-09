"""Geometric acoustic likelihood over the shared pose grid.

The observation is an actively emitted, co-located 6-channel ring recording.
Because the source sits at the array centre, the room response depends on the
candidate *position* only; yaw merely decides where each microphone sits on the
5 cm ring. Echo geometry is therefore traced once per candidate cell and reused
for all yaw bins, which is what makes a full grid sweep affordable.

Reflection order is a parameter, not a constant:

* order 1 is the first-wall fan. It carries the same quantity as the visual
  DESDF cache, but over the full 360 degrees rather than the camera's ~106
  degree field of view.
* order >= 2 adds specular multi-bounce paths, which encode room shape beyond
  line of sight and are the part with no visual counterpart.

Nothing here is learned or fitted; it is a map-topology model, not a
material-faithful room simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from track1_core.floorplan.pose_grid import PoseGrid

SOUND_SPEED_M_S = 343.0
# EchoScan channel order, matching dataset_meta.json acoustics.ring_angles_rad
RING_ANGLES_RAD = np.array(
    [np.pi, 4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi, np.pi / 3, 2 * np.pi / 3],
    dtype=np.float64,
)


@dataclass(frozen=True)
class AcousticProxyConfig:
    """Synthesis and scoring parameters for the geometric proxy."""

    sample_rate_hz: int = 8000
    sound_speed_m_s: float = SOUND_SPEED_M_S
    n_rays: int = 72
    max_order: int = 1
    reflection_gain: float = 0.7      # per bounce
    receiver_radius_m: float = 0.05
    max_time_ms: float = 100.0
    energy_window_ms: float = 2.0
    max_path_m: float = 20.0
    step_px: float = 0.7              # ray march step, in map pixels
    # dataset_meta.json defines the usable window as rir[:, peak+16 : peak+16+1024]:
    # the direct sound is orders of magnitude louder than the reflections and
    # carries no geometry, so it is cut rather than normalised against.
    direct_guard_samples: int = 16
    usable_samples: int = 1024
    # 2.5D extension: the proxy is otherwise purely horizontal, but the floor
    # and ceiling are the closest surfaces of all, so their reflections are the
    # earliest and strongest in the room. Vertical walls plus flat floor and
    # ceiling make the height axis separable, so their image sources combine
    # exactly with the traced horizontal paths.
    vertical_order: int = 0           # 0 reproduces the flat 2D proxy
    room_height_m: float = 2.761
    source_height_m: float = 1.25

    @property
    def n_samples(self) -> int:
        return min(self.usable_samples, int(round(self.sample_rate_hz * self.max_time_ms / 1000.0)))

    @property
    def window_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.energy_window_ms / 1000.0)))


def vertical_images(config: AcousticProxyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Height-axis image sources for a slab, as (vertical offset, bounce count).

    Source and receiver share height ``h`` in a room of height ``H``. The image
    heights are ``2nH + h`` (an even number of reflections, ``2|n|``) and
    ``2nH - h`` (an odd number, ``|2n - 1|``). The offset from the receiver is
    what lengthens a horizontal path; the bounce count is what attenuates it.

    The first two entries are the floor bounce at ``-2h`` and the ceiling bounce
    at ``2(H - h)`` -- the closest surfaces in the room, and the reflections a
    purely horizontal proxy omits entirely.
    """
    if config.vertical_order <= 0:
        return np.zeros(1), np.zeros(1, dtype=int)
    H, h = config.room_height_m, config.source_height_m
    seen: dict[float, int] = {0.0: 0}
    span = config.vertical_order + 1
    for n in range(-span, span + 1):
        for z, k in ((2 * n * H + h, abs(2 * n)), (2 * n * H - h, abs(2 * n - 1))):
            if k == 0 or k > config.vertical_order:
                continue
            dz = round(z - h, 6)
            if dz == 0.0:
                continue
            if dz not in seen or k < seen[dz]:
                seen[dz] = k
    items = sorted(seen.items(), key=lambda kv: (kv[1], abs(kv[0])))
    return np.array([d for d, _ in items]), np.array([k for _, k in items], dtype=int)


def ring_offsets_m(yaw_rad: np.ndarray, radius_m: float) -> np.ndarray:
    """Microphone ring offsets in floorplan metres, shape ``(..., 6, 2)``."""
    yaw = np.asarray(yaw_rad, dtype=np.float64)[..., None]
    angles = RING_ANGLES_RAD + yaw
    return np.stack([radius_m * np.cos(angles), radius_m * np.sin(angles)], axis=-1)


# --------------------------------------------------------------------------
# echo tracing
# --------------------------------------------------------------------------
def _surface_normals(free: torch.Tensor) -> torch.Tensor:
    """Outward normals of the obstacle field, ``(H, W, 2)`` as ``(nx, ny)``.

    Estimated from the gradient of the blurred free-space indicator, so a
    rasterised wall gives a stable normal instead of a staircase of axis flips.
    """
    f = free.to(torch.float32)[None, None]
    k = torch.ones(1, 1, 5, 5, device=free.device) / 25.0
    smooth = torch.nn.functional.conv2d(f, k, padding=2)[0, 0]
    gy, gx = torch.gradient(smooth)
    n = torch.stack([gx, gy], dim=-1)
    norm = n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return n / norm


def trace_echoes(
    free: torch.Tensor,
    origins_px: torch.Tensor,
    config: AcousticProxyConfig,
    map_resolution_m: float,
) -> dict[str, torch.Tensor]:
    """Trace ``n_rays`` specular paths from each origin up to ``max_order`` bounces.

    free        ``(Hmap, Wmap)`` bool, True where sound propagates
    origins_px  ``(N, 2)`` map pixels as ``(col, row)``

    Returns per-order tensors of shape ``(N, R, order)``:
        path_m   total source->hit path length in metres
        hit_x/hit_y  hit position in map pixels
        alive    whether the path existed at that order
    """
    device = free.device
    N = origins_px.shape[0]
    R = config.n_rays
    H, W = free.shape
    normals = _surface_normals(free)

    ang = torch.arange(R, device=device, dtype=torch.float32) * (2 * np.pi / R)
    d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)[None].expand(N, R, 2).contiguous()
    p = origins_px[:, None, :].expand(N, R, 2).to(torch.float32).contiguous()

    max_steps = int(config.max_path_m / map_resolution_m / config.step_px)
    travelled = torch.zeros(N, R, device=device)
    alive = torch.ones(N, R, dtype=torch.bool, device=device)

    path_out, hitx_out, hity_out, alive_out = [], [], [], []

    for _order in range(config.max_order):
        hit = torch.zeros(N, R, dtype=torch.bool, device=device)
        for _ in range(max_steps):
            step = alive & ~hit
            if not step.any():
                break
            p = torch.where(step[..., None], p + d * config.step_px, p)
            travelled = torch.where(step, travelled + config.step_px * map_resolution_m, travelled)
            c = p[..., 0].round().long().clamp(0, W - 1)
            r = p[..., 1].round().long().clamp(0, H - 1)
            oob = (p[..., 0] < 0) | (p[..., 0] >= W) | (p[..., 1] < 0) | (p[..., 1] >= H)
            blocked = ~free[r, c]
            hit = hit | (step & blocked)
            alive = alive & ~(step & oob)
            alive = alive & (travelled < config.max_path_m)

        ok = hit & alive
        path_out.append(torch.where(ok, travelled, torch.full_like(travelled, float("nan"))))
        hitx_out.append(p[..., 0].clone())
        hity_out.append(p[..., 1].clone())
        alive_out.append(ok.clone())

        if _order + 1 >= config.max_order:
            break
        # specular reflection, then step off the wall so the march restarts free
        c = p[..., 0].round().long().clamp(0, W - 1)
        r = p[..., 1].round().long().clamp(0, H - 1)
        n = normals[r, c]
        d = d - 2 * (d * n).sum(-1, keepdim=True) * n
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # step off the wall so the next march does not re-hit it; the offset is
        # real path length and has to be accounted for, not silently dropped
        step_off_px = 2.0
        p = p + d * step_off_px
        travelled = travelled + step_off_px * map_resolution_m
        alive = ok

    return {
        "path_m": torch.stack(path_out, dim=-1),
        "hit_x": torch.stack(hitx_out, dim=-1),
        "hit_y": torch.stack(hity_out, dim=-1),
        "alive": torch.stack(alive_out, dim=-1),
    }


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------
def synthesize_proxy(
    echoes: dict[str, torch.Tensor],
    origins_px: torch.Tensor,
    yaws: torch.Tensor,
    grid: PoseGrid,
    config: AcousticProxyConfig,
) -> torch.Tensor:
    """Proxy RIR energy envelopes, ``(N, O, 6, n_windows)``.

    Only the envelope is built, never the waveform: the score compares 2 ms
    energy windows, and going through a full impulse response would cost memory
    for no extra information.
    """
    device = origins_px.device
    N, R, K = echoes["path_m"].shape
    O = yaws.shape[0]
    res = grid.map_resolution_m
    fs, c = config.sample_rate_hz, config.sound_speed_m_s
    win = config.window_samples
    n_win = config.n_samples // win

    offs = torch.tensor(ring_offsets_m(yaws.cpu().numpy(), config.receiver_radius_m),
                        dtype=torch.float32, device=device)          # (O, 6, 2)
    env = torch.zeros(N, O, 6, n_win, device=device)
    # time origin is the direct arrival; the guard cuts the direct sound out of
    # both the proxy and the observation, so only reflection geometry is scored
    guard_s = config.direct_guard_samples + config.receiver_radius_m / c * fs
    dz_list, v_orders = vertical_images(config)

    origin_m = torch.tensor(grid.grid_to_metric(grid.map_to_grid(origins_px.cpu().numpy())),
                            dtype=torch.float32, device=device)      # (N, 2)

    # Direct path (source -> microphone, no wall) with its vertical images.
    # dz = 0 is the direct sound itself and falls inside the guard, so only the
    # floor and ceiling bounces survive -- the reflections a flat 2D proxy has
    # no way to produce at all.
    if len(dz_list) > 1:
        direct = offs.norm(dim=-1)[None, :, None, :].expand(N, O, 1, 6)   # (N, O, 1, 6)
        ones = torch.ones(N, 1, dtype=torch.bool, device=device)
        _scatter(env, direct, 1.0, ones, dz_list, v_orders, config,
                 guard_s, 1.0, N, O, n_win, win, fs, c)

    for k in range(K):
        alive = echoes["alive"][..., k]
        if not alive.any():
            continue
        base = echoes["path_m"][..., k]
        hit_px = torch.stack([echoes["hit_x"][..., k], echoes["hit_y"][..., k]], dim=-1)
        hit_m = torch.tensor(
            grid.grid_to_metric(grid.map_to_grid(hit_px.reshape(-1, 2).cpu().numpy())),
            dtype=torch.float32, device=device,
        ).reshape(N, R, 2)

        # hit -> microphone leg, per orientation and channel
        rel = hit_m[:, None, :, None, :] - (origin_m[:, None, None, None, :] + offs[None, :, None, :, :])
        leg = rel.norm(dim=-1)                                        # (N, O, R, 6)
        total = base[:, None, :, None] + leg
        _scatter(env, total, config.reflection_gain ** (k + 1), alive,
                 dz_list, v_orders, config, guard_s, R, N, O, n_win, win, fs, c)

    return env


def _scatter(env, total, gain, alive, dz_list, v_orders, config,
             guard_s, norm, N, O, n_win, win, fs, c) -> None:
    """Accumulate one horizontal path family, with its vertical images."""
    for dz, v_order in zip(dz_list, v_orders):
        path3d = total if dz == 0 else (total ** 2 + float(dz) ** 2).sqrt()
        # spherical spreading: pressure ~ 1/r, so the energy scattered below is 1/r^2
        amp_v = (gain * config.reflection_gain ** int(v_order)) / path3d.clamp_min(0.1) / norm
        sample = path3d / c * fs - guard_s
        idx = (sample / win).long().clamp(0, n_win - 1)              # (N, O, R, 6)
        keep = (alive[:, None, :, None].expand_as(idx)
                & (sample >= 0) & (sample < config.n_samples))
        energy = torch.where(keep, amp_v ** 2, torch.zeros_like(amp_v))

        # (N, O, R, 6) -> (N*O*6, R) so one scatter_add fills every channel
        rays = idx.shape[2]
        flat_idx = idx.permute(0, 1, 3, 2).reshape(N * O * 6, rays)
        flat_amp = energy.permute(0, 1, 3, 2).reshape(N * O * 6, rays)
        env.reshape(N * O * 6, n_win).scatter_add_(1, flat_idx, flat_amp)


def observed_envelope(rir: np.ndarray, config: AcousticProxyConfig) -> np.ndarray:
    """Energy envelope of the reflection window of a measured RIR.

    Starts ``direct_guard_samples`` after the direct peak, matching
    ``dataset_meta.json``'s ``usable_window``, so the direct sound -- which is
    far louder than every reflection and carries no geometry -- is excluded.
    """
    rir = np.asarray(rir, dtype=np.float64)
    peak = int(np.abs(rir).max(axis=0).argmax())
    start = peak + config.direct_guard_samples
    seg = rir[:, start:start + config.n_samples]
    if seg.shape[1] < config.n_samples:
        seg = np.pad(seg, ((0, 0), (0, config.n_samples - seg.shape[1])))
    win = config.window_samples
    n_win = config.n_samples // win
    return (seg[:, : n_win * win] ** 2).reshape(6, n_win, win).sum(-1)


def _normalize(env: torch.Tensor) -> torch.Tensor:
    """Normalise total energy, jointly over channels and time.

    Normalising each channel separately would divide out the relative level
    between microphones, which -- together with the sub-window inter-channel
    delays -- is the only thing on the ring that encodes direction. Scale is
    removed once, globally, so orientation information survives.
    """
    return env / env.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)


def score_envelopes(proxy: torch.Tensor, observed: torch.Tensor, temperature: float) -> torch.Tensor:
    """Exponential negative-L1 score between normalised envelopes."""
    p = _normalize(proxy)
    o = _normalize(observed)[None, None, :, :]
    l1 = (p - o).abs().sum(dim=(-1, -2))          # (N, O)
    return torch.exp(-l1 / temperature)
