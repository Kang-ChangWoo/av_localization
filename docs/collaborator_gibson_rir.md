# Collaborator Gibson RIR Workflow

## One-time setup

```bash
git clone https://github.com/LedersonLee/AV-FPLoc.git
cd AV-FPLoc
export AVFPLOC_DATA_ROOT=/file2/jeongeon/AV-FPLoc
```

Use the existing SoundSpaces container/environment. The shared assets are licensed data and live on `/file2`; they are not downloaded by Git.

If `import habitat_sim` terminates with `free(): invalid pointer`, start the SoundSpaces container with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. This is required for the legacy audio binding in the validated Track1 image; it is already set in Track1's SoundSpaces Compose service.

## Verify shared assets and SoundSpaces

```bash
python scripts/verify_collaborator_env.py --scene Springhill
```

The expected result is `"ready_for_rir": true`.

## Render RIRs

```bash
python scripts/gibson_runtime/render_f3loc_echoscan_rirs.py \
  --scene Springhill --pose-index 0 \
  --dataset-root "$AVFPLOC_DATA_ROOT/datasets/f3loc" \
  --mesh-root "$AVFPLOC_DATA_ROOT/meshes/gibson_v2" \
  --stage-root "$AVFPLOC_DATA_ROOT/alignment" \
  --output-root "$AVFPLOC_DATA_ROOT/rendered_rirs" \
  --sample-rate 8000 --indirect-rays 4096 --indirect-ray-depth 6 \
  --source-rays 512 --source-ray-depth 6 --no-diffraction --no-transmission
```

The source and six-mic array center are the released F3Loc pose at `1.25 m`; microphones form a `0.05 m` radius ring. Every output metadata JSON records the complete ray and channel configuration.

## Acoustic likelihood

After rendering poses `0`, `141`, and `283`, run:

```bash
AVFPLOC_DATA_ROOT="$AVFPLOC_DATA_ROOT" python scripts/gibson_runtime/run_global_acoustic_likelihood.py --scene Springhill
```

The first run creates a CPU wall-distance cache. CUDA scores all `9,677 x 36` candidate states afterwards. Open `reports/global_acoustic_likelihood/Springhill/global_acoustic_likelihood_report.html` from the shared root.

## Optional F3Loc visual baseline

F3Loc remains an external pinned baseline. It requires the *full* F3Loc `gibson_g` data and `comp.ckpt`, which are not part of the initial Springhill RIR bundle.

```bash
export F3LOC_ROOT=/path/to/f3loc
export F3LOC_DATASET_ROOT=/file2/path/to/full-f3loc-data
export F3LOC_CHECKPOINT_ROOT=/file2/path/to/f3loc-checkpoints
bash scripts/run_f3loc_visual_likelihood.sh
```

The wrapper checks the required F3Loc commit before running upstream evaluation.
