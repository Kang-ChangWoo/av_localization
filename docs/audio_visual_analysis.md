# Acoustic reranking for visual floorplan localization: full analysis

Written 2026-09-10 as a self-contained brief for an outside reader who has not
seen this codebase. It states the problem, the exact algorithm, every number
measured, and the specific things that are wrong or unverified. Nothing here is
rounded in the method's favour, and the section headed *Known weaknesses* is the
one to read first if you are deciding whether to believe the rest.

---

## 1. The problem

**Visual floorplan localization.** Given a 2D floorplan of a building and a
single RGB image taken somewhere inside it, estimate the camera's pose
`(x, y, yaw)`. The floorplan has walls only: no furniture, no texture, no
semantics. This is the setting of F3Loc (CVPR 2024), UnLoc (2025) and DisCo-FLoc
(2026).

**How the visual methods work.** All three share one structure:

1. A network maps the image to `V = 11` **depth rays**, one per angular bin over
   the horizontal field of view. This is the only use of the image.
2. Offline, the floorplan is converted to a **DESDF** (directional Euclidean
   signed distance field), an array of shape `(H, W, O)` with `H, W` the pose
   grid at 0.1 m per cell and `O = 36` yaw bins. `DESDF[y, x, o]` is the
   distance from cell `(x, y)` to the first wall along yaw bin `o`.
3. The posterior is a sliding L1 match of the predicted rays against the DESDF
   at every pose and yaw:

   ```
   prob_vol[y, x, o] = exp( -|| DESDF[y, x, o:o+V] - rays ||_1 / lambda ),  lambda = 40
   prob_dist[y, x]   = max_o prob_vol[y, x, o]        # what the metrics read
   ```

   UnLoc additionally predicts a per-ray Laplace scale and divides the residual
   by it, so rays it distrusts contribute less.

**The failure this project attacks.** Depth-to-wall is a weak descriptor. A
corridor looks like every other corridor; a corner two metres from two walls
looks like every other such corner. `prob_dist` is therefore strongly
**multi-modal**: several well-separated places in the building explain the image
about equally well, and the method has no way to choose between them. The
figures in `outputs/figures/` show this directly.

**The proposed fix.** A co-located speaker and microphone at the camera emit a
click and record the room impulse response. Late reflections have visited parts
of the room the camera cannot see, so the recording carries information that is
*not* a function of the visible depth profile. If the acoustic response can be
predicted for each candidate pose from the floorplan alone, comparing the
prediction with the recording can break the visual tie.

---

## 2. Data

`/root/storage/echoloc_dataset`, derived from Replica via Habitat and
SoundSpaces 2.0.

| item | value |
|---|---|
| scenes | 17, split 11 train / 3 val / 3 test |
| test scenes | `office_4`, `apartment_2`, `frl_apartment_5` |
| pose collections | `replica_f`, `replica_g`, two independent pose sets over the same rooms |
| images | 640x480 RGB, 4 yaw per position |
| floorplan | `map.png`, 0.01 m per pixel; pose grid is a 10x downsample of a crop |
| RIR | 6-microphone ring, radius 5 cm, source at the ring centre |
| RIR render | 48 kHz, 20000 rays, reflection depth 50, diffraction order 10 |
| conditions | `raw_scan_open` (full furnished mesh), `floorplan_closed` (walls only) |

Two acoustic conditions matter. `raw_scan_open` is the *query*: what a
microphone in the real furnished room would hear. `floorplan_closed` is the
*model*: what the floorplan alone predicts. The gap between them is furniture.

**The candidate grid.** For each test scene, an impulse response is rendered at
**every valid pose cell** on the floorplan-only mesh, giving `outputs/acoustic_grid_v2/<scene>.npz`
with `rir` of shape `(n_cells, 6, n_samples)` and `index` of shape `(n_cells, 2)`.
Cell counts: office_4 2734, apartment_2 5443, frl_apartment_5 5344. This is a
real deployment cost and is discussed under *Known weaknesses*.

---

## 3. The acoustic score

Implemented once in `track1_core/likelihood/grid_score.py`. Four scripts
previously had four subtly different copies that disagreed by ten points under
the same name; that is why it is centralised.

### 3.1 Feature

```python
peak = argmax over time of max over channels of |rir|      # direct sound
s    = peak + direct_guard_samples                         # guard = 2 ms
seg  = rir[:, s : s + usable_samples]                      # window = 128 ms
w    = round(sample_rate * window_ms / 1000)               # window_ms = 2.0
n    = usable_samples // w
E    = (seg[:, :n*w] ** 2).reshape(6, n, w).sum(-1)        # (6 channels, n frames)
```

Three details are load-bearing.

**Alignment on the direct peak** removes the engine's absolute gain and the
source-to-receiver delay, which are identical on both sides and carry no pose
information.

**The guard and the window are physical times, 2 ms and 128 ms, not sample
counts.** Storing them as sample counts is what broke this project once: the
dataset was re-rendered from 8 kHz to 48 kHz and the hard-coded counts silently
became a 0.33 ms guard and a 21 ms window. Every acoustic number collapsed to
chance while every visual number stayed identical. See
`docs/data_provenance_incident.md`.

**Disjoint windows, no analysis window.** An earlier version used a 64-sample
Hann STFT, which smears each frame over 8 ms regardless of hop. 8 ms of path is
2.7 m, or 27 cells of the 0.1 m pose grid. Measured on office_4 the envelope
reaches ground-truth percentile 96.6 and the banded STFT about 50, which is
chance. Time resolution is the single most important parameter.

Window length sweep, three test scenes, 600 poses, F3Loc mono as the visual
backbone (`scripts/envelope_rerank_eval.py`):

| feature | acoustic alone @1m | median GT rank of 4507 | gated recall @1m |
|---|---|---|---|
| envelope 0.125 ms | 12.3% | 640 | 37.7% |
| envelope 0.25 ms | 12.3% | 506 | 37.0% |
| envelope 0.5 ms | 15.3% | 248 | 41.2% |
| envelope 1 ms | 21.0% | 235 | 40.3% |
| **envelope 2 ms** | **23.7%** | 245 | **44.3%** |
| envelope 4 ms | 20.2% | 377 | 43.8% |
| STFT, 3 bands, nfft 64 | 14.5% | **210** | 41.5% |

Note that the best absolute ranker (STFT, rank 210) is *not* the best partner
for vision (envelope 2 ms, 44.3%). Absolute accuracy and complementarity to
vision are different properties.

### 3.2 Normalisation and distance

```python
F = E / E.sum()                        # 'sum' normalisation
F = shape(F)                           # per-row temporal shape + per-row energy split
score(cell) = -sum |F_cell - F_obs|    # symmetric L1, higher is better
```

Two degeneracies are asserted in code and pinned by tests
(`tests/test_grid_score.py`, `asymmetry_is_degenerate`):

- Under `sum` normalisation both sides integrate to one, so `sum(m - o) = 0` and
  the one-sided penalty `sum(max(m - o, 0))` is exactly half the symmetric one.
  It produces an identical ranking. This was measured twice before it was
  noticed.
- `shape_feature` divides every row by its own sum, which makes the preceding
  `normalise` choice a no-op. A sweep over three normalisations once returned
  three identical columns for this reason.

A one-sided distance is the *right* model for clutter, since furniture can add
arrivals but cannot remove predicted ones. Making it actually do something
requires `normalise='peak'` or `'quantile'` and `shape_feature=False`. That
variant was tried and did not help; the symmetric version is what all reported
numbers use.

### 3.3 Fusion

Visual and acoustic scores live on incomparable scales, so both are converted to
rank percentiles over the valid cells and combined in log space:

```python
rv = rank_norm(vis)        # (rank + 0.5) / n_cells
ra = rank_norm(acoustic)
fused = log(rv) + w * log(ra)
```

Three decision rules are reported everywhere:

| rule | definition |
|---|---|
| rerank | argmax of the acoustic score restricted to vision's top-50 cells |
| fused | argmax of `fused` over all valid cells |
| gated | `fused` when vision is unsure, vision's own pick otherwise |

The gate reads a **confidence margin** computable at inference time:

```python
best = argmax(vis)
far  = cells more than 1.5 m from best
margin = log( vis[best] / max(vis[far]) )
gate fires when margin < tau,  tau = 60th percentile of margins
```

This is not an oracle: it uses no ground truth.

---

## 4. Results

### 4.1 Reproduction of the three baselines

Released weights, published tables. `scripts/reproduction_audit.py`.

| | 0.1 m | 0.5 m | 1 m | 1m/30deg | verdict |
|---|---|---|---|---|---|
| F3Loc paper Ours_s, gibson_f | 4.7 | 28.6 | 36.6 | 35.1 | |
| reproduced | 4.7 | 28.5 | 36.5 | 35.0 | match |
| F3Loc paper Ours_m, gibson_f | 13.2 | 40.9 | 45.2 | 43.7 | |
| reproduced | 13.2 | 40.8 | 45.1 | 43.6 | match |
| F3Loc paper Ours_f, gibson_g | 12.2 | 39.4 | 44.5 | 43.2 | |
| reproduced | 12.2 | 39.3 | 44.4 | 43.1 | match |
| UnLoc paper, Gibson(t) single frame | 19.7 | 61.1 | 64.7 | 63.8 | |
| reproduced | 19.7 | 61.1 | 64.6 | 63.7 | match |
| DisCo paper, RRP only, Gibson(f) | 12.0 | 45.8 | 50.6 | 49.2 | |
| reproduced | 11.8 | 45.0 | 49.6 | 48.3 | 1.0 low |
| DisCo paper, RRP + DisCo | 13.1 | 50.9 | 56.7 | 55.4 | |
| reproduced | 13.8 | 50.2 | 56.5 | 55.6 | 0.7 |

UnLoc's sequential row also reproduces: 97.30% success at 1 m for T=100
trajectories against a published 97.3%.

**One trap.** F3Loc's own `eval_observation.py` calls `.eval()` on the
complementary net only and leaves the monocular and multi-view nets in **train
mode**, so BatchNorm uses batch statistics and the score depends on the
evaluation batch size. The monocular net reads 46.8% at batch 16 and 36.5% at
batch 1. Any comparison of two checkpoints is meaningless unless both used the
same batch size. `scripts/eval_visual.py` now forces batch 1 in that mode and
records the batch size in every metrics file.

**From-scratch training against released weights** (F3Loc only, matched batch):

| | released @1m | ours @1m | diff |
|---|---|---|---|
| mono gibson_f | 36.5 | 39.6 | +3.1 |
| mono gibson_g | 33.7 | 35.4 | +1.7 |
| mv gibson_f | 45.1 | 44.9 | -0.2 |
| mv gibson_g | 30.8 | 32.0 | +1.2 |
| comp gibson_f | 47.4 | 48.0 | +0.6 |
| comp gibson_g | 44.4 | 44.6 | +0.2 |

DisCo and UnLoc have no from-scratch Gibson run, so their trainability is
unverified.

### 4.2 The acoustic gain, held out

Fusion weight `w` and gate threshold `tau` are fitted on `replica_f` and the
table is computed on `replica_g`, 300 poses. Without this the knobs are chosen
by maximising the number being reported.

**F3Loc mono, trained on Replica, w=2, tau=0.010**

| | 0.1 m | 0.5 m | 1 m | 1m/30deg | median |
|---|---|---|---|---|---|
| vision only | 5.3% | 29.7% | 37.7% | 34.0% | 2.17 m |
| fused | 3.0% | 28.3% | 41.0% | 34.3% | 1.92 m |
| gated | 5.7% | 37.0% | 46.7% | 40.7% | 1.29 m |
| **gain** | +0.4 | +7.3 | **+9.0** | +6.7 | -0.88 m |

**DisCo RRP, Gibson weights zero-shot, w=4, tau=0.009**

| | 0.1 m | 0.5 m | 1 m | 1m/30deg | median |
|---|---|---|---|---|---|
| vision only | 1.3% | 16.7% | 33.0% | 31.0% | 2.08 m |
| fused | 3.3% | 25.0% | 39.3% | 36.0% | 2.07 m |
| gated | 2.7% | 20.7% | 37.7% | 34.0% | 1.79 m |
| **gain** | +2.0 | +8.3 | **+6.3** | +5.0 | -0.01 m |

**UnLoc, trained on Replica, w=0.5, tau=0.067**

| | 0.1 m | 0.5 m | 1 m | 1m/30deg | median |
|---|---|---|---|---|---|
| vision only | 9.0% | 44.7% | 49.7% | 49.0% | 1.12 m |
| fused | 13.3% | 46.3% | 52.0% | 50.3% | 0.71 m |
| gated | 10.7% | 48.7% | 54.0% | 52.3% | 0.62 m |
| **gain** | +1.7 | +4.0 | **+4.3** | +3.3 | -0.50 m |

Zero-shot UnLoc for comparison: vision 41.0%, gated 44.7%, gain +3.7. Training
UnLoc on Replica raised vision by 8.7 points and the acoustic gain did **not**
shrink.

Acoustic score alone, identical across backbones since it does not use the
image: 3.0 / 13.8 / 23.7 / — with a median error of 3.13 m.

### 4.3 Per scene

Replica-trained UnLoc, 200 poses per scene:

| scene | vision @1m | acoustic alone @1m | best @1m | gain |
|---|---|---|---|---|
| office_4 | 45.0% | 34.0% | 49.0% | +4.0 |
| apartment_2 | 29.5% | 27.0% | 42.5% | **+13.0** |
| frl_apartment_5 | 77.5% | 10.0% | 76.5% | -1.0 |

The same pattern holds across backbones within one room. In frl_apartment_5,
F3Loc mono reaches 76.5% and gains -1.5, while DisCo RRP reaches 29.0% off the
same images and gains +7.0. **The size of the gain tracks how badly the visual
posterior is failing, not the room and not the recording quality.**

### 4.4 What the fusion does, mechanically

`scripts/plot_decision_switch.py` reduces each pose to the decision among
spatially separated modes of the visual posterior, extracted by greedy
non-maximum suppression at 1.5 m. The result is blunt:

- Among the modes vision shortlists, **the visual rank percentile is saturated**:
  every one sits above the 99.9th percentile, so `log(rv)` is a few hundredths
  for all of them.
- The acoustic term `w * log(ra)` is whole units and varies by orders of
  magnitude across the same modes.
- Therefore **vision chooses the shortlist and sound orders it**. The fusion is
  much closer to reranking than the "weighted combination" framing suggests.

This explains three otherwise puzzling observations: why the gate is necessary
(nothing else stops sound from overruling a confident vision pick), why the best
`w` varies between backbones, and why fusion and reranking give similar numbers.

Counting on 300 poses with Replica-trained UnLoc: sound **repairs 40 poses and
breaks 33**. The net is +7 poses but recall rises 4.3 points because repairs are
large (6 to 9 m errors corrected to under 1 m) and regressions are the reverse.

### 4.5 What limits the method

Truth lies inside vision's top-50 only 83 to 92% of the time, depending on
backbone. Pure reranking cannot exceed that.

| backbone | truth in top-50 | achieved @1m |
|---|---|---|
| F3Loc mono | 82.8% | 46.7% |
| DisCo RRP | 83.8% | 39.3% |
| UnLoc zero-shot | 87.8% | 44.7% |
| UnLoc in-domain | 92.3% | 54.0% |

Roughly 38 points of headroom are unused. The bottleneck is the **quality of the
acoustic score**, not the fusion rule.

The reason is domain gap, and it is measurable. Rendering the candidates and the
query on the *same* mesh (`floorplan_closed` both sides) gives 98 to 100% recall
at 1 m in the 0 to 1 kHz band. Against real furnished-scan recordings the same
feature gives 4 to 26%. Furniture, not feature design, is the wall. Low-band
selection, sparse event matching, one-corner NLOS path hypotheses and a learned
contrastive encoder were all tried and all hit it.

**Caveat: those matched-domain numbers were measured on the pre-incident 8 kHz
grid and have not been re-run since the re-render.** The conclusion is very
unlikely to change but it is not currently verified.

---

## 5. Things that did not work

Recording these matters as much as the positives.

| attempt | result |
|---|---|
| banded STFT features | GT percentile ~50, chance; 8 ms analysis window destroys time resolution |
| single frequency band | best band differs per scene, so choosing one is overfitting |
| high-frequency only (2-4 kHz) | 30-38% even in the matched domain, worse than low band |
| one-sided asymmetric distance | provably identical to symmetric under the normalisation in use; the non-degenerate variant did not help |
| one-corner NLOS path hypotheses (`track1_core/likelihood/corners.py`) | GT rank worsened from 411 to 509 |
| learned contrastive acoustic encoder, 6600 paired samples | 13.3% validation top-1, but **2.7% on test scenes with GT rank 2402 against a chance of 2254**, i.e. worse than random |
| DisCo's own multiplicative rule `geo * exp(alpha*sim) * exp(w*ac)` | RRP alone 29.7%, +DisCo 24.3%, +DisCo+audio 29.8%; sound only undoes the damage DisCo's reranker does on Replica |
| DisCo contrastive stage trained from scratch on Replica | 98% train accuracy, **29% validation**; severe overfitting on 11 scenes |

The DisCo row deserves emphasis: **DisCo's published reranking stage, worth +6.9
on Gibson, is worth -5.4 on Replica zero-shot.** Its similarity model does not
transfer. So the DisCo comparison in section 4.2 uses only its ray predictor.

---

## 6. Known weaknesses

Ordered by how likely each is to invalidate the conclusion.

1. **Both sides of the acoustic comparison are simulated.** The query recordings
   and the candidate grid both come from SoundSpaces. The only gap between them
   is furniture. There is no evidence at all that this survives a real
   microphone in a real room. This is the single largest threat.
2. **Three test scenes, 300 held-out poses.** F3Loc's Gibson test set has 2007
   samples over roughly a hundred scenes. A 4-point difference on 300 paired
   poses is not backed by a significance test here.
3. **The held-out protocol holds out the pose collection, not the room.**
   `replica_f` and `replica_g` are different pose sets over the *same* three
   rooms. Whether `tau` and `w` transfer to an unseen room is untested.
4. **A candidate grid must be pre-rendered per scene.** office_4 alone took 12
   sharded render jobs. This is a genuine deployment cost.
5. **Sound supplies no heading.** The score is one number per cell, flat over the
   36 yaw bins, so it cannot move the per-cell heading argmax. Every reported
   heading is vision's own heading at whichever cell wins. The 1m/30deg gains are
   entirely a consequence of landing on a better cell.
6. **DisCo's RRP-only reproduction is 1.0 point low** and the cause is not yet
   identified; the `--all_imgs` variant was being tested to settle it.
7. **DisCo and UnLoc from-scratch reproduction is unverified.** Only F3Loc has
   been retrained on Gibson from scratch.
8. **Absorption was never varied.** The sweep that appeared to test it produced
   bit-identical renders because `acousticsConfig.enableMaterials = False`.

---

## 7. Reproducing everything

```bash
# reproduction audit: published tables, from-scratch training, Replica status
python scripts/reproduction_audit.py

# the headline table, all five rows, official metric set, per scene, held out
python scripts/replica_official_eval.py --visual f3loc_mono disco_rrp --per-scene

# UnLoc, which needs its own environment
/opt/conda/envs/unloc/bin/python scripts/unloc_replica_audio_eval.py \
    --checkpoint <replica ckpt> --per-scene

# acoustic feature sweep
python scripts/envelope_rerank_eval.py --grid-dir outputs/acoustic_grid_v2

# figures
/opt/conda/envs/unloc/bin/python scripts/plot_unloc_audio_maps.py --tag _indomain
/opt/conda/envs/unloc/bin/python scripts/plot_decision_switch.py --tag _indomain
```

Key source files:

| file | role |
|---|---|
| `track1_core/likelihood/grid_score.py` | the acoustic feature and score, single implementation |
| `track1_core/floorplan/pose_grid.py` | metric / grid / map-pixel conversions |
| `track1_core/provenance.py` | stamps every metrics file with argv, git state, input hashes |
| `scripts/replica_official_eval.py` | F3Loc and DisCo on the official metric set with audio |
| `scripts/unloc_replica_audio_eval.py` | the same for UnLoc |
| `scripts/disco_full_audio_eval.py` | DisCo's full two-stage pipeline plus audio |
| `scripts/soundspaces_render.py` | candidate grid rendering, sharded |
| `scripts/merge_grid_shards.py` | shard merge with config and duplicate checks |
| `docs/data_provenance_incident.md` | the sample-rate incident, in full |

Every metrics JSON carries a `provenance` block with the UTC timestamp, argv,
git commit, a SHA-1 of uncommitted changes, sampled content hashes of every
input file and library versions. Two identical runs produce byte-identical JSON.

---

## 8. Questions worth an outside opinion

1. Is "vision proposes, sound disposes" a strong enough contribution given that
   the shortlist coverage caps it at roughly 83 to 92%?
2. Given that the visual rank saturates among shortlisted modes (section 4.4),
   is log-rank fusion the right combiner, or should the visual term use a
   quantity with dynamic range among the top modes, such as the likelihood ratio
   to the best mode?
3. The gain does not shrink when the visual backbone improves by 8.7 points
   (UnLoc zero-shot to in-domain), yet within a room it tracks visual weakness.
   What reconciles these?
4. How damaging is the fully simulated acoustic evaluation, and is a control on
   a public measured-RIR dataset such as SoundCam or dEchorate enough to fix it
   without a full visual pipeline on that data?
5. Is the margin gate defensible as part of the method, or will it read as a
   tuned patch over a fusion rule that does not work unconditionally?
