# Collaborative Gibson RIR Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task.

**Goal:** Let a collaborator with SoundSpaces installed clone AV-FPLoc, point it at `/file2/jeongeon/AV-FPLoc`, render an aligned Gibson RIR, and run an acoustic likelihood diagnostic without accessing the original Track1 workspace.

**Architecture:** AV-FPLoc owns portable configuration, asset preflight/staging, renderer and diagnostics. Licensed Gibson/F3Loc assets stay under `/file2`; F3Loc visual inference stays an external pinned adapter, not vendored source.

**Constraints:** Source and six-mic ring center are the F3Loc pose at `1.25 m`; ring radius is `0.05 m`; renderer is `8 kHz`; EchoScan-ready output is `[6,1024]` after direct-peak plus 16-sample crop; meshes/data/RIRs/checkpoints/reports remain Git-ignored.

### Task 1: Shared data contract
- [ ] Add `avfploc.paths.DataRoots` using `AVFPLOC_DATA_ROOT` with default `/file2/jeongeon/AV-FPLoc`.
- [ ] Add a YAML config declaring Gibson roots, scene, audio geometry, and renderer settings.
- [ ] Add a staging script that copies only a requested scene's map, poses, DESDF, mesh/MTL/textures, and alignment evidence to the shared root with SHA-256 manifest.
- [ ] Test root resolution and dry-run/staged asset checks.

### Task 2: Portable renderer
- [ ] Add standalone coordinate conversion, ring geometry, raw/closed-shell selection, SoundSpaces mono sensor renderer, and EchoScan preprocessing modules.
- [ ] Record all ray/material/source/channel settings in every render metadata JSON.
- [ ] Add a renderer CLI and tests for finite `[6,2049]` raw and `[6,1024]` prepared RIR contracts.

### Task 3: Acoustic diagnostic
- [ ] Add an explicit first-order CUDA likelihood path over F3Loc `(H,W,36)` and a labelled direct-cropped comparison report.
- [ ] Apply a shared valid spatial mask before argmax/ranking and distinguish raw scores from display normalization.
- [ ] Test direct crop, invalid mask exclusion, output shapes, and HTML links.

### Task 4: External F3Loc visual adapter
- [ ] Add `F3LOC_ROOT` preflight requiring commit `9e8027d9219ca505078283ebfd925580f476ab97`.
- [ ] Document visual-likelihood prerequisites and output contract; do not copy upstream F3Loc source or checkpoints.

### Task 5: Collaborator guide
- [ ] Add read-only preflight checking shared assets, Habitat audio API, GPU, and optional F3Loc revision.
- [ ] Document clone, environment activation, data-root export, RIR render, acoustic likelihood, and optional visual baseline commands.
- [ ] Run repository tests and one real Springhill render/preflight before publishing code-only changes.
