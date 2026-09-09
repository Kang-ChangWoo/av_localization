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

## 2026-09-07 — Vendor the F3Loc baseline

**Decision:** Copy upstream F3Loc `modules/` and `utils/` into `third_party/f3loc`, pinned to `9e8027d`, and reverse the earlier "baseline source is not vendored here" boundary.

**Reason:** Upstream released no training script and no reachable checkpoints, so the visual observation models have to be trained from scratch inside this project. An external-path adapter would make every training run depend on an unversioned checkout, which defeats reproducibility for the one component the acoustic work is measured against. The vendored tree keeps upstream's imports and file layout unchanged so it stays diffable against the pinned revision, and all project-owned adapter code stays in `track1_core/`.

## 2026-09-07 — Visual training protocol

**Decision:** Train `mono` and `mv` on the union of the `gibson_f` and `gibson_g` train splits without roll/pitch augmentation, then fit only the `comp` selector on top of the frozen pair. Treat every frame as a monocular sample.

**Reason:** The paper reports a single set of checkpoints evaluated on both collections, so training on both matches the reported protocol. The Gibson collections are upright and upstream evaluation passes no gravity-alignment mask, so augmenting roll/pitch would train for a condition the evaluation never presents. `depth40` carries one target row per frame, so restricting monocular training to the evaluation reference frame would discard three quarters of the available supervision at no I/O saving. `comp_d_net` freezes the observation networks itself; pinning them to `eval()` additionally prevents BatchNorm statistics from drifting during selector training.

**Caveat:** The paper's optimisation hyperparameters are in unavailable supplementary material. Learning rate, batch size, epoch count, precision, and the shape-loss weight are AV-FPLoc choices and are labelled as such in README.md.

## Open decisions

- Acoustic sensing representation: beam/range profile, impulse-response-derived geometry features, or another compact signature.
- Acoustic field of view, beam count, yaw-zero convention, and invalid-ray policy.
- Likelihood calibration: fixed temperature, learned temperature, or confidence-conditioned fusion.
- Valid-pose-mask construction: occupancy erosion radius, navigability source, and boundary policy.
