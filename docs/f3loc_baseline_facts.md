# F3Loc Baseline Facts

## Provenance

- External repository: `https://github.com/felix-ch/f3loc`
- Reviewed source revision: `9e8027d9219ca505078283ebfd925580f476ab97`
- Baseline source is not vendored here.

## Relevant behavior

F3Loc creates a directional floorplan range volume commonly named DESDF. Its candidate pose representation is `(H, W, 36)`: row/y, column/x, and 36 counter-clockwise yaw bins. The default output spatial resolution is 0.1 m/cell.

At runtime the visual model predicts a 1D depth profile. The localization utility samples 11 rays, converts image-forward depth into radial range, reverses the ray order to align it with the DESDF orientation axis, and computes an exponential L1 matching score over all candidate positions and yaw bins.

`depth40` and `depth160` are per-observation training targets with 40 and 160 image-column samples; they are not the number of candidate-pose rays in the DESDF. The runtime localizer uses 11 rays by default.

## Baseline limitations relevant to AV-FPLoc

- The reviewed baseline does not construct or apply a valid-pose mask in localization.
- Its existing coordinate conversion helpers are dataset-specific and partly inline.
- Some multi-view and filtering evaluation paths use ground-truth relative poses or transitions; AV-FPLoc protocols must declare this oracle condition explicitly.
- Structured3D offline depth-ray and runtime localization conventions require independent validation before comparison.

## Deliberate project changes

AV-FPLoc will use float32 geometry caches, explicit coordinate metadata, explicit valid-pose support, and modality-specific likelihood calibration. These are method contracts, not claims about F3Loc performance.
