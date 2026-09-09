# F3Loc Baseline Facts

## Provenance

- External repository: `https://github.com/felix-ch/f3loc`
- Reviewed source revision: `9e8027d9219ca505078283ebfd925580f476ab97`
- Since 2026-09-07 the baseline networks and localization utilities **are** vendored, at `third_party/f3loc`, unchanged from that revision. See the decision log for why the earlier no-vendoring boundary was reversed.

## Training gaps in the released code

Upstream released evaluation code and Lightning wrappers but no training script; its README states the training pipeline was Azure ML specific. Three mismatches block `trainer.fit()` on the released code as-is:

- `depth_net_pl.training_step` reads `batch["img"]` and `batch["gt_rays"]`; `GridSeqDataset` emits `ref_img` and `ref_depth`.
- `mv_depth_net.forward` indexes `x["ref_mask"]` and `x["src_mask"]` unconditionally, while the dataset omits those keys when roll/pitch augmentation is off, and `None` cannot survive `default_collate`.
- `comp_d_net` receives already-trained `mv_net` and `mono_net` instances and freezes them, so the three networks must be trained in sequence and only the selector MLP is fit in the last stage.

`mv_depth_net_pl` also defaults to `d_hyp=1.0` while `eval_observation.py` constructs the multi-view net with `d_hyp=-0.2`; training must use the evaluation value.

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
