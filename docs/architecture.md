# AV-FPLoc Architecture

## Purpose

AV-FPLoc studies whether acoustic geometry can reduce floorplan-pose ambiguity when a visual observation has insufficient structural information. The project owns a method implementation; F3Loc is an external baseline and protocol reference.

## Target data flow

```text
dataset adapter
  ├── visual observation ──> visual likelihood ──┐
  ├── acoustic observation ─> acoustic likelihood ├─> calibrated fusion ─> posterior ─> evaluation
  └── floorplan + GT pose ──> pose grid/mask ────┘
```

All likelihoods and the posterior share one pose-grid contract. Ground-truth pose is used to construct a sensor observation and to evaluate the output, never to select a candidate location during an observation-only experiment.

## Module ownership

| Module | Owns | Does not own |
| --- | --- | --- |
| `datasets` | Source-specific inputs and pose conversion into shared contracts | Floorplan scoring policy |
| `floorplan` | Occupancy map, pose grid, directional geometry cache, valid-pose mask | Learned observation model |
| `likelihood` | Visual/acoustic observation-to-likelihood adapters | Cross-modality weighting |
| `fusion` | Likelihood calibration and posterior combination | Sensor simulation |
| `evaluation` | Metrics and fixed protocols | Training or model selection |
| `visualization` | Qualitative figures derived from numerical outputs | Numerical inference |
| `configs` | Reproducible experiment choices | Machine-specific locations |

## Implementation order

1. Make the floorplan, pose-grid, coordinate conversion, and valid-mask contract executable.
2. Implement an acoustic geometry oracle against that contract.
3. Add a visual-likelihood adapter with the same output contract.
4. Add calibrated fusion and evaluation diagnostics.
5. Replace oracle acoustic observations with simulated or learned acoustic observations.

This order separates geometric upper-bound evidence from sensor-model performance.
