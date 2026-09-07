# Data Contracts

## Common pose grid

The canonical pose volume is `float32` with shape `(H, W, O)`.

| Field | Contract |
| --- | --- |
| Spatial axes | Array index order `(grid_y, grid_x, yaw_bin)`; pose vectors use `(x, y, yaw)` |
| Spatial resolution | `0.1 m/cell` unless an experiment explicitly declares another value |
| Orientation axis | `O=36`, counter-clockwise, `yaw_bin * 2π/O`; therefore 10°/bin |
| Units | Floorplan ranges are radial distances in meters; yaw is radians |
| Storage dtype | `float32` for newly created geometry caches and all runtime likelihoods |

Each scene must provide explicit metric-to-grid and grid-to-metric functions. Offset, map resolution, crop origin, handedness, and yaw-zero convention are metadata, not implicit constants.

## Valid-pose support

`valid_pose_mask` has shape `(H, W)` and dtype `bool`. It represents navigable candidate locations after the map is rasterized and downsampled. Every visual likelihood, acoustic likelihood, fused posterior, normalization, argmax, and diagnostic rank computation must use the same mask. Invalid cells must be assigned zero probability or negative-infinite log score before normalization.

## Observations

An observation adapter must declare the following fields before it can produce a likelihood:

```text
ray_values:       float32, shape (V,)
ray_order:        ordered angular samples, stated explicitly
ray_angles_rad:   float32, shape (V,)
ray_units:        radial range in meters or camera-forward depth in meters
field_of_view:    angular support in radians
invalid_ray_mask: bool, shape (V,)
```

The adapter must convert non-radial depth to radial range before direct comparison with directional floorplan geometry. It must not silently use invalid rays in an L1 or correlation score.

## Likelihood and posterior

```text
visual_likelihood:   float32, (H, W, O)
acoustic_likelihood: float32, (H, W, O)
posterior:           float32, (H, W, O)
```

Likelihoods are observation scores, not automatically normalized posteriors. Fusion must calibrate modality temperatures or log-score weights on the valid support; beam count and arbitrary numerical scale must not become an accidental modality weight.

## Evaluation

Predicted pose is `(x, y, yaw)` in grid coordinates. Translation errors are reported in meters after grid-to-metric conversion; orientation errors use the wrapped minimum angular difference. At minimum, each experiment reports recall at 0.5 m and 1 m, 1 m+30°, GT likelihood rank/percentile, and posterior entropy or ambiguity diagnostics.
