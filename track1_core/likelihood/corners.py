"""One-corner NLOS path hypotheses from a 2D occupancy map.

Radial wall returns only describe what the candidate can see. The visual
likelihood is already biased toward what is visible, so a purely radial acoustic
proxy is correlated with the very failure it is supposed to fix: facing a wall,
standing in a repeated corridor, or having the disambiguating structure around a
corner leaves both modalities equally blind.

This adds one extra path family, and only one. A convex corner of the obstacle
field re-radiates into the region behind it, so a round trip

    source -> corner -> remote surface -> corner -> source

has total length

    L = 2 * (||source - corner|| + rho)

where ``rho`` is a first-wall distance measured *from the corner*, into
directions the source cannot see directly. The delay is ``L / c``.

This is a geometry-derived path *hypothesis*, not a UTD diffraction
calculation: no edge-diffraction coefficient is computed and no amplitude is
claimed. That is deliberate. The evidence being tested is whether the arrival
time predicted by map topology is supported by the recording, and an amplitude
model would only add unknown material dependence to a quantity the method
already refuses to rely on.

The factorisation is what makes it affordable. ``rho`` depends on the corner and
a direction, never on the candidate, so it is computed once per scene. Per
candidate only two cheap things remain: is the corner visible, and which of the
corner's directions are hidden from the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from track1_core.likelihood.events import PATH_TYPES, EventMatchConfig


@dataclass(frozen=True)
class CornerConfig:
    """Corner extraction and NLOS path parameters, in physical units.

    contour_epsilon_m       polygon simplification tolerance for obstacle
                            outlines. 0.15 m keeps doorframes and wall ends
                            while discarding rasterisation staircases.
    min_corner_spacing_m    corners closer than this are collapsed to one
    max_corners             cap per scene, strongest first
    n_corner_rays           directions probed outward from each corner
    max_remote_m            longest remote-surface distance retained
    min_turn_deg            how sharply the outline must turn for a vertex to
                            count as a diffracting edge. Below about 25 degrees
                            it is rasterisation noise, not a corner.
    occlusion_margin_deg    a corner direction counts as hidden from the
                            candidate only if it lies this far past the
                            grazing line, which keeps nearly-visible
                            directions out of the NLOS family
    """

    contour_epsilon_m: float = 0.15
    min_corner_spacing_m: float = 0.4
    max_corners: int = 64
    n_corner_rays: int = 32
    max_remote_m: float = 12.0
    min_turn_deg: float = 25.0
    occlusion_margin_deg: float = 10.0


def extract_corners(free: np.ndarray, map_resolution_m: float,
                    config: CornerConfig) -> np.ndarray:
    """Convex corners of the obstacle field, as ``(n, 2)`` map pixels ``(col, row)``.

    Obstacle outlines are traced and simplified, and a vertex is kept when the
    outline turns through more than ``min_turn_deg``. Working from contours
    rather than a corner detector means the result is a point on an actual
    wall edge, which is what a diffracting edge has to be.
    """
    import cv2

    obstacle = (~np.asarray(free, dtype=bool)).astype(np.uint8)
    contours, _ = cv2.findContours(obstacle, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    eps_px = max(1.0, config.contour_epsilon_m / map_resolution_m)

    pts, sharp = [], []
    for c in contours:
        if len(c) < 4:
            continue
        poly = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2).astype(np.float64)
        if poly.shape[0] < 3:
            continue
        prev = np.roll(poly, 1, axis=0)
        nxt = np.roll(poly, -1, axis=0)
        a = poly - prev
        b = nxt - poly
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        ok = (na > 1e-6) & (nb > 1e-6)
        cosang = np.clip((a * b).sum(1) / np.clip(na * nb, 1e-9, None), -1, 1)
        turn = np.degrees(np.arccos(cosang))
        sel = ok & (turn >= config.min_turn_deg)
        pts.append(poly[sel])
        sharp.append(turn[sel])
    if not pts or sum(p.shape[0] for p in pts) == 0:
        return np.zeros((0, 2))

    pts = np.concatenate(pts, axis=0)
    sharp = np.concatenate(sharp, axis=0)
    order = np.argsort(-sharp)
    spacing_px = config.min_corner_spacing_m / map_resolution_m
    keep: list[int] = []
    for i in order:
        p = pts[i]
        if all(np.hypot(*(p - pts[j])) >= spacing_px for j in keep):
            keep.append(int(i))
        if len(keep) >= config.max_corners:
            break
    return pts[keep]


def _march(free: np.ndarray, start: np.ndarray, direction: np.ndarray,
           max_px: float, step_px: float = 0.7) -> np.ndarray:
    """First-obstacle distance in map pixels along each ray, NaN if none.

    ``start`` is ``(n, 2)`` and ``direction`` is ``(n, 2)``; both are ``(col, row)``.
    """
    H, W = free.shape
    p = start.astype(np.float64).copy()
    travelled = np.zeros(p.shape[0])
    out = np.full(p.shape[0], np.nan)
    live = np.ones(p.shape[0], dtype=bool)
    for _ in range(int(max_px / step_px)):
        if not live.any():
            break
        p[live] += direction[live] * step_px
        travelled[live] += step_px
        c = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
        r = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        oob = (p[:, 0] < 0) | (p[:, 0] >= W) | (p[:, 1] < 0) | (p[:, 1] >= H)
        hit = live & ~free[r, c]
        out[hit] = travelled[hit]
        live = live & ~hit & ~oob
    return out


def corner_remote_distances(free: np.ndarray, corners: np.ndarray,
                            map_resolution_m: float,
                            config: CornerConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-corner first-wall distances, ``(n_corners, n_rays)`` in metres, and the ray angles.

    Computed once per scene: this depends on the floorplan and the corner only,
    never on the candidate position, which is what keeps the per-candidate cost
    to arithmetic.
    """
    if corners.shape[0] == 0:
        return np.zeros((0, config.n_corner_rays)), np.zeros(config.n_corner_rays)
    ang = np.arange(config.n_corner_rays) * (2 * np.pi / config.n_corner_rays)
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)                  # (R, 2)
    n_c, n_r = corners.shape[0], config.n_corner_rays
    start = np.repeat(corners, n_r, axis=0)
    direction = np.tile(d, (n_c, 1))
    # start just off the wall so the march does not immediately re-hit the corner
    start = start + direction * 2.0
    max_px = config.max_remote_m / map_resolution_m
    rho_px = _march(free, start, direction, max_px)
    rho_m = (rho_px + 2.0) * map_resolution_m
    return rho_m.reshape(n_c, n_r), ang


def corner_events(free: np.ndarray, origins_px: np.ndarray, corners: np.ndarray,
                  rho_m: np.ndarray, ray_angles: np.ndarray,
                  map_resolution_m: float, config: CornerConfig,
                  event_config: EventMatchConfig, chunk: int = 512
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-corner NLOS path delays per candidate, as padded ``(N, P)`` arrays.

    A corner contributes only when the candidate can see it, and only for those
    of its directions that the candidate cannot see past it. The second test is
    what makes these paths carry information the radial fan does not already
    have: a direction the candidate could reach directly is not NLOS evidence.
    """
    N = origins_px.shape[0]
    if corners.shape[0] == 0:
        return np.full((N, 1), np.nan), np.zeros((N, 1)), np.full((N, 1), -1, dtype=int)

    n_c, n_r = rho_m.shape
    delays: list[np.ndarray] = []
    margin = np.radians(config.occlusion_margin_deg)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        o = origins_px[s:e].astype(np.float64)                        # (n, 2)
        rel = corners[None, :, :] - o[:, None, :]                     # (n, C, 2)
        dist_px = np.linalg.norm(rel, axis=-1)
        unit = rel / np.clip(dist_px, 1e-9, None)[..., None]

        # source -> corner visibility, marched to just short of the corner
        flat_start = np.repeat(o, n_c, axis=0)
        flat_dir = unit.reshape(-1, 2)
        flat_max = dist_px.reshape(-1) - 2.0
        hit = _march(free, flat_start, flat_dir, float(max(flat_max.max(), 1.0)))
        visible = (~np.isfinite(hit)) | (hit >= flat_max)
        visible = visible.reshape(e - s, n_c) & (dist_px > 2.0)

        # a corner direction is NLOS evidence only if it points away from the
        # source, i.e. past the grazing line at the corner
        incoming = np.arctan2(unit[..., 1], unit[..., 0])              # (n, C)
        delta = np.abs(np.angle(np.exp(1j * (ray_angles[None, None, :] - incoming[..., None]))))
        hidden = delta < (np.pi / 2 - margin)

        d_src_m = dist_px * map_resolution_m
        total = 2.0 * (d_src_m[..., None] + rho_m[None, :, :])         # (n, C, R)
        ok = visible[..., None] & hidden & np.isfinite(rho_m)[None, :, :]
        delays.append(np.where(ok, event_config.path_m_to_ms(total), np.nan).reshape(e - s, -1))

    d = np.concatenate(delays, axis=0)
    inside = np.isfinite(d) & (d < event_config.window_ms)
    d = np.where(inside, d, np.nan)

    # merge per candidate so a wall seen through one corner from many directions
    # counts once, matching how the radial family is treated
    P = max(int(inside.sum(axis=1).max()), 1)
    out_d = np.full((N, P), np.nan)
    out_w = np.zeros((N, P))
    from track1_core.likelihood.events import merge_predicted
    for i in range(N):
        row = d[i][np.isfinite(d[i])]
        if row.size == 0:
            continue
        md, mw, _ = merge_predicted(row, np.full(row.shape[0], 1.0 / max(n_r, 1)),
                                    np.zeros(row.shape[0], dtype=int), event_config)
        n = min(md.shape[0], P)
        out_d[i, :n] = md[:n]
        out_w[i, :n] = mw[:n]
    out_k = np.where(out_w > 0, PATH_TYPES["corner_nlos"], -1)
    return out_d, out_w, out_k
