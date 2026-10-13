# Headline configuration: the setting behind `tab_main_multi.tex`

Written 2026-10-13. One place that says exactly which checkpoints, projection,
rule and scalars produced the best reported numbers, how to regenerate them,
and which upgrades are measured but not yet adopted. Companion to
`docs/tables/tab_main_multi_checkpoints.md` (paths only).

## 1. The protocol (fixed in advance, same everywhere)

| part | setting | where |
|---|---|---|
| visual posterior | backbone's own rays matched against DESDF, λ=40, 36 yaw bins, 11 rays (S3D: 7 rays, F_W 0.5959), eval-mode BatchNorm, batch 1 | `scripts/extract_mode_table.py` |
| hypotheses | K=10 NMS modes of π at 1.5 m, 0.5 m disc per mode | `track1_core/modes.py` |
| summaries | visual: log π at the centre; acoustic: 0.9 quantile of a(c) over the disc (`centre/quantile`) | `ModeConfig`, `val_select.py --fixed` |
| acoustic feature | `stft_band`, nfft 256, hop 64, 48 kHz, 3 bands, 2 ms guard then 128 ms window, −L1 | `track1_core/likelihood/grid_score.py` |
| projection W | linear 1746→128, InfoNCE on (furnished, wall-only) pairs of the benchmark's own training rooms; source per benchmark R/M/G, pooled B for S3D and ZInD | `feasible/run_invariant_projection.py`, `outputs/metrics/acoustic_projection*.npz` |
| rule | k* = argmax zv + γ·za, γ = w·σ((τv − m_v)/s); three scalars (w, s, τv) per backbone chosen on validation rooms by recall@1 m; τa = −2 and `standard` transform are fixed (three-scalar "simple" rule) | `track1_core/likelihood/mode_fusion.py`, `scripts/val_select.py --fixed --source indomain` |
| queries | per scene `np.linspace` over sorted pose dirs: Replica 100, MP3D 40, Gibson 40, S3D 20, ZInD 20 per collection | `extract_mode_table.py --n-poses` |
| test rooms | Replica 3, MP3D 12, Gibson 68 of 69 (Sargents_f2 excluded), S3D 30 of 250, ZInD 252 | `run_projection_matrix.py`, `run_gibson_pipeline.py`, `run_zind_pipeline.py` |

**Backbone checkpoint selection is *not* uniform, and the earlier claim that it was has been
withdrawn** (audited 2026-10-14, see `experiments/echoloc_cvpr/reports/AUDIT.md`). F3Loc mono
and UnLoc are `loss-valid` argmin by hardcoded monitor; DisCo RRP is `val_action_loss` argmin
on `val_split: val`; **SemRayLoc's semantic net is chosen by ray accuracy**
(`run_srl_pipeline.best_checkpoint` globs `acc_rays_val` first and falls back to `loss-valid`
only when no accuracy checkpoint exists — its docstring gives the reason: accuracy "is what
the localiser consumes"). Replica and MP3D take the accuracy branch; **S3D took the fallback**
because `save_top_k=2` on a loss-monitored run did not keep its accuracy-best epoch 32. No
selection anywhere touches test.

Re-selecting Replica's semantic net on `loss-valid` was measured rather than assumed
(`EXECUTION_LOG.md`, AUD-04-FIX) and **rejected**: it scores 41.3 against 46.7 on validation
recall@1 m, the very metric this protocol uses for every other choice. That run also showed
**`ΔR@1m` is not invariant to the backbone checkpoint** — the gain rose +7.3 → +9.7 while
vision fell 35.0 → 29.0 — so never quote a gain without the vision column beside it.

## 2. Every headline cell

Checkpoint paths are in `docs/tables/tab_main_multi_checkpoints.md`. Recall at 1 m in %, gain with 95% paired bootstrap CI over queries, median error in m. "gain w/o W" is the same validation-selected rule without the projection (the ablation row in `feasible/results/VAL_<ds>_fixed_indomain*.md`).

| dataset | backbone | W | scalars (w, s, τv) | val@1m | test vision | test ours | gain [CI] | median | gain w/o W | n |
|---|---|---|---|---|---|---|---|---|---|---|
| Replica | F3Loc mono | R | 2, 0.05, 0.02 | 48.8 | 38.3 | 48.8 | +10.5 [+7.0, +14.0] | 1.07 | +6.2 | 600 |
| Replica | UnLoc | R | 1, 0.1, 0.1 | 55.7 | 50.7 | 60.0 | +9.3 [+6.3, +12.5] | 0.39 | +5.2 | 600 |
| Replica | DisCo RRP | R | 1, 0.1, 0.2 | 50.7 | 37.0 | 51.3 | +14.3 [+10.7, +18.0] | 0.87 | +8.2 | 600 |
| Replica | SemRayLoc | R | 2, 0.05, 0.02 | 46.7 | 35.0 | 42.3 | +7.3 [+4.0, +10.5] | 1.47 | +2.8 | 600 |
| MP3D | F3Loc mono | M | 2, 0.05, 0.05 | 38.1 | 33.3 | 40.1 | +6.8 [+4.1, +9.6] | 2.91 | +2.0 | 960 |
| MP3D | UnLoc | M | 0.5, 0.1, 0.05 | 49.2 | 45.3 | 50.0 | +4.7 [+3.2, +6.2] | 1.01 | +4.3 | 960 |
| MP3D | DisCo RRP | M | 2, 0.02, 0.05 | 26.5 | 21.0 | 31.5 | +10.4 [+7.8, +13.0] | 4.46 | +4.2 | 960 |
| MP3D | SemRayLoc | M | 2, 0.02, 0.05 | 30.0 | 22.8 | 28.6 | +5.8 [+3.2, +8.4] | 4.70 | +3.4 | 960 |
| Gibson | F3Loc mono | G | 2, 0.02, 0.02 | 67.3 | 47.0 | 59.7 | +12.7 [+11.5, +13.9] | 0.48 | +2.5 | 5440 |
| Gibson | UnLoc | G | 2, 0.02, 0.05 | 68.5 | 51.7 | 64.0 | +12.3 [+11.2, +13.4] | 0.33 | +3.8 | 5440 |
| Gibson | DisCo RRP | G | 1, 0.02, 0.1 | 58.1 | 38.5 | 52.1 | +13.5 [+12.3, +14.7] | 0.84 | +3.0 | 5440 |
| S3D | F3Loc mono | B | 2, 0.02, 0.05 | 47.7 | 24.2 | 48.6 | +24.4 [+20.3, +28.7] | 1.11 | +22.2 | 558 |
| S3D | UnLoc | B | 2, 0.02, 0.2 | 53.4 | 30.1 | 57.9 | +27.8 [+23.7, +31.9] | 0.58 | +28.3 | 558 |
| S3D | DisCo RRP | B | 2, 0.02, 0.1 | 46.6 | 22.6 | 45.7 | +23.1 [+19.2, +27.1] | 1.33 | +19.4 | 558 |
| S3D | SemRayLoc | B | 2, 0.05, 0.05 | 57.5 | 32.6 | 55.4 | +22.8 [+18.8, +26.7] | 0.80 | +21.0 | 558 |
| ZInD | F3Loc mono | B | 2, 0.02, 0.1 | 73.5 | 44.0 | 68.1 | +24.1 [+22.7, +25.5] | 0.35 | +25.1 | 4080 |
| ZInD | UnLoc | B | 2, 0.02, 0.1 | 74.4 | 48.4 | 71.1 | +22.7 [+21.3, +24.1] | 0.27 | +24.2 | 4080 |
| ZInD | DisCo RRP | B | 2, 0.02, 0.1 | 71.8 | 39.1 | 65.6 | +26.4 [+25.0, +27.9] | 0.44 | +27.0 | 4080 |
| ZInD | SemRayLoc | B | 2, 0.02, 0.1 | 80.3 | 46.6 | 69.7 | +23.1 [+21.7, +24.5] | 0.37 | +23.2 | 4080 |

Source JSONs: `feasible/results/VAL_<ds>_fixed_indomain.json` (three backbones) and `VAL_<ds>_fixed_indomain_srl.json` (SemRayLoc), key `results.<bb>.selected`.

## 3. Is this the best-performing variant?

`docs/tables/tab_protocol.tex` compares five ways of fixing the method on the same test rooms, each with scalars chosen on validation. On the furnished benchmarks (Replica, MP3D, Gibson) the headline "own W" is the best or within noise of the best in every cell, and clearly above "none" (e.g. Gibson +12.7 vs +2.5). On the matched-geometry benchmarks (S3D, ZInD) the projection is unnecessary: "none" or "searched" is equal or better (S3D UnLoc none +28.3 vs own +27.8; ZInD every backbone none > own by 0.6–1.5). The headline keeps "own W" there for protocol uniformity, and the caption should say so.

Two facts to keep in mind when calling this "best":

- The structure (`centre/quantile`, three-scalar rule) was fixed after room-CV on the Replica *test* rooms (2026-09-14, `feasible/results/H_*.json`), and the `indomain` source map was fixed after per-source test numbers existed (commit 303f5fa). The validation-searched protocol gives lower Replica gains (+8.8 / +7.5 / +9.5). See `docs/decision_log.md` for the disclosure to add.
- The projection supplies half to three quarters of the gain on furnished benchmarks; the prose in `docs/method.tex` and `docs/experiments.tex` still describes the no-projection method and older numbers.

## 4. Measured upgrades not yet adopted

| cell | current checkpoint | candidate | validation evidence | status |
|---|---|---|---|---|
| Replica DisCo RRP | `rrp_replica_f_val_20260920_074725/.../epoch=02` (val_action_loss tie 0.28 with epochs 368, 386) | same run, `epoch=368` | vision recall@1m on 3 val rooms 41.7% vs 38.7% (epoch 386: 41.2%) | test not re-extracted; expect vision ↑, gain (+14.3) ↓ |
| MP3D SemRayLoc | `outputs/srl/mp3d/semantic/epoch=02` (batch 8, 4 epochs) | `outputs/srl/mp3d_retrain_b16_e50/semantic/semantic_net-epoch=05-loss-valid=0.5990.ckpt` (batch 16, stopped at 16) | loss-valid 0.599 vs 0.656, ray acc 0.797 vs 0.769 | test not re-extracted |
| Gibson DisCo RRP | `rrp_gibson_val_20260920_124720/.../epoch=03` (3 epochs, lr 1e-3) | `gibsonretrain_rrp_lr1e4_e50_*/.../epoch=12` (50 epochs, lr 1e-4) | val recall@1m 42.5 vs 42.1, @0.5m 34.6 vs 37.5 (480 queries): tie | keep current; retrain is evidence that longer training does not help |

To adopt one: point the checkpoint in `feasible/run_projection_matrix.py` BASE (or the SRL entry) at the candidate, extract the four tables (val and test, with and without the in-domain W) via `feasible/run_disco_reextract.py`-style commands, then

    /opt/conda/envs/f3loc/bin/python scripts/val_select.py --dataset <ds> --fixed --source indomain
    /opt/conda/envs/f3loc/bin/python scripts/val_tables.py

and update `docs/tables/tab_main_multi_checkpoints.md` and the coauthor bundle
`$ECHOLOC_ASSETS/checkpoints/` + `MANIFEST.tsv` (`$ECHOLOC_ASSETS` is the shared
assets directory beside the dataset tree; see `CLAUDE.md` §5).

## 5. Regenerating the table from what is on disk

    /opt/conda/envs/f3loc/bin/python scripts/val_select.py --dataset replica --fixed --source indomain
    ... (mp3d, gibson, s3d, zind; add --backbones srl --out feasible/results/VAL_<ds>_fixed_indomain_srl.md for SemRayLoc)
    /opt/conda/envs/f3loc/bin/python scripts/val_tables.py      # writes docs/tables/tab_main_multi.tex

Verified 2026-10-08: re-running the selection reproduces every committed JSON exactly.
