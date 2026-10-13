# DISCREPANCIES

Conflicts between the paper tables, the committed result JSONs, the generated LaTeX and the
per-query artifacts. Per execution principle §0.9, nothing here is silently overwritten.
Each entry records the exact values, which file holds which, whether the cause is
established or `[미확인]`, and the one action that resolves it.

Opened 2026-10-14 at repo SHA `08ed5c1`.

---

## D-01 — Replica DisCo RRP: the paper table disagrees with its own source JSON

**Status: OPEN. Cause [미확인]. Blocks REP-02/REP-03 for this cell.**

`docs/headline_config.md` §2 names `feasible/results/VAL_<ds>_fixed_indomain.json` as the
source of every headline cell, and `scripts/val_tables.py` generates
`docs/tables/tab_main_multi.tex` from those JSONs. For Replica × DisCo-FLoc RRP the three
artifacts do not agree:

| artifact | tracked? | vision @1m | ours @1m | gain |
|---|---|---|---|---|
| `feasible/results/VAL_replica_fixed_indomain.json` | tracked, clean | **0.3917 (39.2)** | **0.5317 (53.2)** | **+0.1400** |
| `feasible/results/VAL_replica_fixed_indomain.disco_epoch02.json` | **untracked**, written 2026-10-13 14:58 | 0.3700 (37.0) | 0.5133 (51.3) | +0.1433 |
| `docs/tables/tab_main_multi.tex:16` (the paper table) | tracked | 37.0 | — | — |
| `docs/headline_config.md` §2 | tracked | 37.0 | 51.3 | +14.3 |

So the paper table and `headline_config.md` both carry **37.0**, which matches the
*untracked* `.disco_epoch02.json`, while the committed JSON they are supposed to be
generated from carries **39.2**. Regenerating `tab_main_multi.tex` from the current
committed JSON would change the published Replica DisCo row.

Compounding it: `headline_config.md` §4 lists this cell under "measured upgrades **not yet
adopted**" and says "test not re-extracted", predicting "vision ↑, gain (+14.3) ↓". The
committed JSON shows vision ↑ (37.0 → 39.2) and gain ↓ (+14.3 → +14.0) — exactly the
predicted direction. Something was re-extracted; the documentation says it was not.

Also in flight, uncommitted on the node: `feasible/run_projection_matrix.py` repoints this
cell's checkpoint from `epoch=02-val_action_loss=0.28` to
`epoch=368-val_action_loss=0.28` (same run, same val loss, chosen on validation-room vision
recall 41.7 % vs 38.7 %).

What is **not** established: which checkpoint produced 0.3917. The committed JSON's last
commit is `a4d454b`, which predates the epoch-368 work, so the simple story "the JSON was
regenerated with epoch 368" is not supported by the git history. Do not assert a cause.

**Resolving action.** Re-run the selection for Replica and diff, without touching the
committed artifacts:

    /opt/conda/envs/f3loc/bin/python scripts/val_select.py --dataset replica --fixed \
        --source indomain --out <scratch>/VAL_replica_recheck.md

then compare `results.disco.selected` against both JSONs above. Whichever it reproduces
identifies the checkpoint actually behind the committed numbers. **Until that is done, the
Replica DisCo row must not be quoted as reproduced**, and `tab_main_multi.tex` must not be
regenerated — doing so would move a published number with no record of why.

---

## D-02 — MP3D SemRayLoc: the headline cell has no per-query evidence on disk

**Status: OPEN. Cause established. REP-03 is `BLOCKED` for this cell.**

`feasible/results/VAL_mp3d_fixed_indomain_srl.json` reports the headline MP3D SemRayLoc
cell — n = 960, vision 0.2281 (22.8), ours 0.2865 (28.6), gain +0.0583 — and its own
provenance block names `test_tag = semrayloc_mp3d12projM`.

**Neither that table nor its identity counterpart exists.** A glob over
`outputs/analysis/` for any filename containing both `semrayloc` and `mp3d` returns
**zero files**, while every other SemRayLoc row's tables are present
(`queries_raw_scan_open_semrayloc{,projR}.csv` for Replica,
`queries_floorplan_closed_semrayloc_s3d{,projB}.csv` for S3D,
`..._semrayloc_zind{,projB}.csv` for ZInD).

Consequences:

- The MP3D SemRayLoc row cannot be recomputed, re-bootstrapped, or re-paired from this
  checkout. Only the aggregate survives.
- Every per-query analysis the programme needs for that cell — paired CI, scene-clustered
  bootstrap, W→C / C→W taxonomy (HYP-04), ambiguity stratification (HYP-02) — is
  impossible for it until the extraction is re-run.
- It is one of the 15 rows of the paper's Table 1, and it is the cell whose checkpoint
  `headline_config.md` §4 also proposes to replace (batch-16 retrain, loss-valid 0.599 vs
  0.656). Re-extracting for the upgrade would regenerate the missing tables anyway.

**Reproducibility caveat affecting all four SemRayLoc rows.** Every `*_srl.json`
provenance block records `"dirty": true` — MP3D at commit `07da6a7c0a6a` with 419 diff
lines, S3D at `dd41338f9efa` with 408. The exact code that produced the SemRayLoc rows is
identified only by a `diff_sha1`, not by a commit. The three non-SRL backbones should be
checked for the same condition before Table 1 is called reproduced.

**Resolving action.** Re-extract the MP3D SemRayLoc cell (needs one GPU), which both
restores the per-query tables and lets the checkpoint upgrade be evaluated in the same
pass. Queue it as the first GPU job when a card is confirmed ours.

---

## Verified consistent — recorded so it is not re-litigated

Checked 2026-10-14, no discrepancy found:

- **All 19 headline cells are internally consistent.** In every
  `VAL_<ds>_fixed_indomain{,_srl}.json`, `gain == ours − vision` and
  `recalls["1.0m"] == ours` to 5e-4.
- **Every comparison the headline table makes is genuinely paired.** For all 18 cells whose
  tables resolve, the projected table and the identity (no-projection) table carry the
  **same `query_id` set in the same order**, and within each table the vision and fused
  arms have equal n. The "gain w/o W" column therefore subtracts across matched queries,
  satisfying §0.6. (The 19th is D-02, which has no tables.)
- **V vs V+A pairing is structural, not a join.** `extract_mode_table.py:640-641` writes
  `e_vis` and `e_fused` into the same CSV row for the same `query_id`, so the two arms
  cannot drift apart by construction.
- The per-query CSV's `e_fused` column is **not** the reported fused error: it is the
  extraction-time fusion over cells, whereas `val_select.py:187-188` recomputes the error
  with the validation-selected scalars over modes. The two differ by design and the CSV
  column must not be used to cross-check `ours`. `val_select.py:202` already guards the
  selected path with a vectorised-vs-`choose()` equivalence check that raises on
  disagreement.
