# Decision Log

## 2026-07-11 — Public repository boundary

**Decision:** Publish the project-owned `track1_core` package and design documents only.

**Reason:** External baseline source, datasets, DESDF caches, checkpoints, generated figures, and machine paths are not necessary to discuss or implement the project method and should not become accidental public artifacts.

## 2026-07-11 — Geometry cache dtype

**Decision:** New directional floorplan geometry caches use `float32`.

**Reason:** The pose-grid resolution is 0.1 m/cell and relevant ranges are tens of meters at most. Float32 rounding is negligible at this scale while halving cache storage and avoiding mixed float64/float32 runtime operations.

## 2026-07-11 — Candidate validity

**Decision:** AV-FPLoc will introduce one explicit `valid_pose_mask` shared by visual likelihood, acoustic likelihood, fusion, and evaluation.

**Reason:** A wall-only floorplan range field alone scores wall and otherwise invalid cells. Without a common mask, apparent acoustic gains can be caused by invalid candidate locations rather than structural complementarity.

## 2026-07-11 — Research sequencing

**Decision:** Implement an acoustic geometry oracle before a learned or simulated acoustic observation model.

**Reason:** The oracle establishes the maximum structural information available from the intended acoustic representation and separates representation limits from acoustic perception errors.

## Open decisions

- Acoustic sensing representation: beam/range profile, impulse-response-derived geometry features, or another compact signature.
- Acoustic field of view, beam count, yaw-zero convention, and invalid-ray policy.
- Likelihood calibration: fixed temperature, learned temperature, or confidence-conditioned fusion.
- Valid-pose-mask construction: occupancy erosion radius, navigability source, and boundary policy.
