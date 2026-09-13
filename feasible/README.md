# EchoLoc validation programme

Written against the paper as it actually stands, not as it is pitched. The
claim under test is narrow and should stay narrow:

> A floorplan-derived acoustic forward model is too inaccurate for global
> acoustic localization, yet still carries enough relative evidence to
> discriminate among a handful of visually plausible poses.

Everything below is organised by the reviewer objection it answers. An
experiment that does not reduce a specific rejection risk is not in this list.

## What is already known, and what it costs the paper

Three findings from the work so far constrain the programme, and two of them
are inconvenient.

**The headline number shrank twice under scrutiny.** Tuning the four fusion
scalars on the forward-motion collection and reporting on the general-motion
collection is not a holdout: on the Replica test rooms 73% of general poses have
a forward pose within one grid cell and 100% within the 1 m reporting
threshold, and the acoustic score depends on position alone. Pooling both
collections and holding out whole rooms instead moves F3Loc from +8.0 to +4.2,
UnLoc from +5.0 to +4.5 and DisCo from +5.3 to +2.7. Separately, the first
Matterport3D number was computed on the six smallest rendered rooms, which have
fewer candidate cells and therefore an easier ranking problem; on all twelve
rooms the three backbones give +2.3, +3.3 and +2.3. **The paper's honest
effect size is two to four points, not five to eight.**

**Resampling rooms rather than queries widens every Replica interval to span
zero.** Three test rooms is an effective sample size of three. Replica cannot
support a building-level generalisation claim on its own; Matterport3D's twelve
rooms and Structured3D's thirty can, and the programme below leans on them.

**The beyond-FoV story is not supported.** Splitting the post-direct response
into time ranges puts the 64--128 ms range *below chance* at ranking the true
pose, and the sign of the beyond-FoV effect flips across rooms under a
furnished query. Section O says what to do about that.

## A. Real-room validation

**Reviewer question.** Is this a simulation artefact (risk A)?

**Why necessary.** Every impulse response in the paper comes from one engine,
and the furnished and floorplan conditions differ only in the mesh handed to
that engine. Nothing currently rules out that the whole effect is a property of
the simulator's reflection model. This is the single largest rejection risk and
no amount of extra simulation reduces it.

**Protocol.** Two rooms, three if a third is cheap: one acoustically ordinary
office or meeting room, one harder room with soft furnishing. Thirty to forty
poses per room on an irregular path, not a grid, at a fixed camera height.
Source co-located with the array, matching the simulated setting exactly;
adding a separate fixed source changes the sensing assumption and should not be
mixed into the same table. One source orientation. A six-element circular array
of 5 cm radius if available, otherwise any compact array with a known geometry,
since the ring's channels saturate at three (Sec. L). Exponential sine sweep,
3--6 s, 100 Hz to 12 kHz, deconvolved to an impulse response; a sweep gives
20--30 dB more usable dynamic range than a click and costs nothing.

Direct-path alignment and gain are the two places a real recording will differ
from simulation in a way that is not interesting. Align on the direct peak and
discard the first 2 ms exactly as the simulated pipeline does, and normalise
each response by its own energy in the scoring window, which removes source
level and microphone sensitivity together. Candidates come only from the
floorplan extrusion, measured with a tape or taken from a building plan. No
learned domain adaptation.

**Fixed.** Backbone (F3Loc monocular, the weakest and therefore the clearest
signal), acoustic feature, fusion rule, and the four scalars, all transferred
from the simulated validation rooms without re-tuning. **Changed.** Only the
provenance of the recordings.

**Metrics.** Recall at 1 m, paired bootstrap over queries, and the per-room
breakdown. With 60--120 queries the interval will be roughly ±10 points.

**What would support the paper.** A positive point estimate in both rooms with
the pooled interval excluding zero, at any effect size. **What would weaken
it.** A negative estimate in either room, or acoustic-alone recall
indistinguishable from chance, which would mean the real measurement carries no
usable evidence at all rather than merely less.

**Interpretation.** This is the transfer claim, not the headline. Report it as
its own table and state the sample size in the caption.

**Placement.** Main paper. Tier 1.

## B. Geometry mismatch as a continuum

**Reviewer question.** How wrong can the forward model be before verification
stops helping (risk H)?

**Why necessary.** The paper currently has two points, matched and mismatched,
and a reviewer reads two points as an anecdote. A monotone curve turns the
domain gap from a caveat into a characterised property.

**Protocol.** Six levels, each a separate candidate grid rendered on the same
scenes: (0) query and candidate identical geometry; (1) wall-only candidate,
wall-only query; (2) wall-only candidate, furnished query, which is the
deployment setting; (3) wall positions perturbed by a Gaussian with 5, 10 and
20 cm standard deviation; (4) absorption coefficients randomised within the
plausible range for the material class; (5) ceiling height wrong by ±0.3 m.
Doors closed versus open is a sixth level worth having because a floorplan
rarely records door state.

**Fixed.** Queries, visual posterior, fusion rule, scalars. **Changed.** The
candidate grid only. **Cost.** Each level is a full re-render, about 4 s of
single-core simulation per cell; on the three Replica test rooms that is 15
core-hours per level. Budget for four levels, not nine.

**Metrics.** Two curves on one axis: acoustic-alone recall at 1 m, which will
collapse, and the fusion gain, which is the claim. The interesting result is
that the second curve is flat where the first is falling.

**What would support the paper.** The gain surviving to 10 cm wall error and
to randomised absorption. **What would weaken it.** The gain tracking
acoustic-alone recall exactly, which would mean verification is just weak
global localization and there is no separate phenomenon.

**Placement.** Main figure. Tier 1.

## C. Global localization against hypothesis verification

**Reviewer question.** Does restricting acoustics to a visual shortlist
actually change the problem, or is this bookkeeping (risks C, J, K)?

**Why necessary.** It is the paper's thesis and it is cheap: everything needed
is already in the extracted tables.

**Protocol.** Five settings on identical queries: acoustic-only over the whole
grid; acoustic reranking of all visual cells; acoustic reranking of the top-K
cells for K in 1, 3, 10, 50, 200; acoustic verification over spatial-NMS
hypotheses; and the proposed selective fusion. Report candidate-set size beside
each so the axis is interpretable.

**Metrics.** Ground-truth acoustic rank (median and interquartile range),
recall at 1 m, and the oracle over the same candidate set. **The figure** is
candidate-set size on a log x-axis against two curves, the method's recall and
the oracle's, with acoustic-only at the far right. The gap between them is what
better acoustics could still buy; the shape of the method's curve is the thesis.

**Placement.** Main figure. Tier 1, no new compute.

## D. Classical known-room acoustic localization

**Reviewer question.** If the floorplan is known, why not localize from echoes
directly (risk C)?

**Why necessary.** Echo-based localization in a known room predates this work
by two decades. A paper that does not implement one is asking to be rejected by
whoever wrote those papers.

**Baselines, in increasing order of faithfulness.** (i) Nearest-neighbour
retrieval over the same candidate grid using the full normalised response,
which is the paper's own acoustic-alone number and should be labelled as the
fingerprint baseline rather than presented as new. (ii) Time-of-arrival echo
matching: pick the first N prominent peaks after the direct path, convert to
path lengths, and match against the image-source path lengths predicted by the
floorplan at each candidate, scoring by a robust set distance. (iii) Image-source
labelling: for each candidate, predict the first-order image sources from the
floorplan and score the observed early response against a rendered
first-order-only response.

**Assumptions classical methods make and why the mismatch hurts them.** They
assume the reflectors are the ones in the model. Furniture adds early
reflections that belong to no wall, and (ii) and (iii) have no mechanism to
discount them, which is exactly the failure this paper is about. That makes the
comparison informative rather than a strawman.

**Required control.** Run all baselines under matched geometry first. If (ii)
and (iii) do not reach near-perfect localization there, they are misimplemented
and their mismatched numbers mean nothing. This control is not optional.

**Placement.** Main table, one row per baseline. Tier 1.

## E. Wider field of view and multi-view vision

**Reviewer question.** If the ambiguity is a field-of-view problem, use more
pixels (risk D).

**Why necessary.** It is the first thing a vision reviewer thinks, and the
assets to answer it already exist: F3Loc's multi-view and complementary
networks are trained on Replica and Matterport3D at both learning rates, and
have never been run with the acoustic term.

**Protocol.** Six rows: monocular vision; monocular plus acoustics; multi-view
vision; multi-view plus acoustics; complementary vision; complementary plus
acoustics. Same queries, same scalars, same rooms.

**The fair comparison is not "does acoustics beat four views".** It is whether
the acoustic gain survives on top of a stronger visual posterior. A method that
only helps a weak backbone is a method that will be obsoleted by the next
backbone.

**What would support the paper.** A positive gain on the complementary network,
even a smaller one. **What would weaken it.** The gain vanishing at multi-view,
which would mean acoustics is a substitute for visual evidence rather than a
complement, and the paper should then be reframed around the single-image
setting explicitly.

**Placement.** Main table. Tier 1, no new rendering.

## F. Unseen-room generalisation

**Reviewer question.** Were the scalars tuned on the rooms you report (risk G)?

**Protocol.** Rooms split three ways. Training rooms feed only the visual
backbones, as their original papers specify. Validation rooms choose K, the NMS
radius, the acoustic feature, the fusion weight, the gate thresholds and the
neighbourhood radius. Test rooms are untouched until the final run.

The honest complication is that DESDF caches and candidate grids currently
exist only for the test rooms, so "validation rooms" means rendering grids for
three more Replica rooms, about 15 core-hours, and a handful of Matterport3D
rooms. Until that is done, the leave-one-room-out cross-validation already in
place is the defensible fallback: scalars are refit on the other rooms in every
fold and every reported query comes from a room its scalars never saw.

**Report both the query-level and the room-clustered bootstrap.** The second is
wider and spans zero on Replica. Saying so in the paper is far cheaper than
having a reviewer discover it.

**Placement.** Main table protocol paragraph plus a supplementary table.
Tier 1.

## G. Cross-backbone validation

**Reviewer question.** Does this work for one posterior only (risk F)?

Three backbones are enough if they differ in kind, and these do: F3Loc's
monocular network, UnLoc's per-ray Laplace uncertainty, and DisCo's ray
regression predictor. The structure, meaning which summary represents a
hypothesis under each modality and which combination rule is used, must be
identical across backbones or the paper is describing three methods. The four
scalars may differ, and should, because two visual posteriors on different
scales cannot share a threshold on visual ambiguity; state this explicitly
rather than letting a reviewer find it.

**Consistency required.** All three point estimates positive. One interval
spanning zero is survivable if the other two do not; two is not.

**Placement.** Main table, already the row structure. Tier 1, done.

## H. Permutation and correspondence controls

**Reviewer question.** Is the gain from score statistics rather than
pose-dependent acoustic information (risk E)?

Four controls, each diagnosing something different, and no more:

| control | destroys | diagnoses |
|---|---|---|
| shuffle candidate scores within a query | the spatial map, keeps the distribution | whether any spatial structure is used |
| swap query acoustics between queries in the same room | the query-to-pose link, keeps room statistics | whether the evidence is pose-specific or room-specific |
| pair the query with another room's grid | both | the floor of the effect |
| rotate the query's channels | inter-channel structure only | whether the array's geometry matters at all |

The third is the weakest and can be dropped. The fourth is already measured:
rotating channels leaves the true cell first among 600 candidates, which says
the ring carries no directional information and is why the paper reports
orientation from vision alone.

**Placement.** Supplementary table, one line of prose in the main paper.
Tier 1, no new compute.

## I. Pairwise acoustic discrimination

**Reviewer question.** Acoustic-alone recall is weak, so is there any
information there at all (risk J)?

**Protocol.** Force a binary choice between the correct hypothesis and one
incorrect hypothesis, stratified four ways: visually ambiguous against visually
confident, and spatially close against spatially far. Chance is 50%. Report the
visual ordering as the competing baseline, with paired bootstrap intervals.

This is the experiment that rescues the paper if end-to-end gains stay at two
to four points: it shows the information exists even where the decision rule
cannot exploit it.

**Placement.** Main figure, four panels or one grouped bar chart. Tier 1.

## J. Visual shortlist oracle

**Reviewer question.** Is the shortlist or the verifier the bottleneck (risk K)?

Recall of the ground-truth region within the top K for K in 1, 2, 3, 5, 10, 20,
against the method and against an oracle selector over the same K. **Interpretation
is fixed in advance:** if the oracle is far above the method, the verifier is the
bottleneck and the paper is about acoustic discrimination; if the oracle is close
to the method, the shortlist is the bottleneck and the paper should propose more
hypotheses instead. Current evidence says the former, with an oracle near 94%
against a method near 55% on Replica, and that gap is the paper's own statement
of what is left to do.

**Placement.** Main table. Tier 1, done.

## K. Is the gate necessary

**Reviewer question.** Why should acoustics intervene on some queries only?

Compare vision alone, acoustics alone, unconditional reranking, fixed-weight
fusion, a gate on visual ambiguity, a gate on acoustic separability, the
product of the two, and the proposed continuous gate. Report repair rate and
break rate separately, not only the net, because the net hides the mechanism.

The known result is that unconditional reranking is far worse under a furnished
query and *better* under matched geometry, which says plainly that the gating
machinery compensates for the domain gap and would be harmful without it. That
is a more interesting sentence than "our gate is best" and it should be the one
in the paper.

**Placement.** Main table. Tier 1, done.

## L. Sensor perturbation

Ranked by how likely each is to be raised, not by how easy it is to run:

1. **Additive noise on the query.** Already measured: recall moves from 22.0%
   to 20.0% between clean and 0 dB SNR, because the feature integrates over
   128 ms and three bands. Report it and move on.
2. **Microphone and source gain mismatch.** Removed by the per-response energy
   normalisation, which should be stated rather than tested at length.
3. **Receiver height error, ±10 and ±20 cm.** Needs a re-render and is the
   perturbation most likely to matter, since Structured3D's own camera height
   varies by 36 cm within a scene.
4. **Channel rotation.** Done, no effect, see H.
5. Sampling-rate mismatch and microphone position jitter are not worth the
   compute.

**Placement.** Supplementary. Tier 2 except the first, which is Tier 1 and
already done.

## M. Floorplan uncertainty

Wall position error is level (3) of experiment B and should not be run twice.
The two additions worth the compute are missing interior walls, which a
floorplan of an open-plan space genuinely gets wrong, and door state. Report
graceful degradation, not robustness.

**Placement.** Supplementary, folded into the B figure. Tier 2.

## N. How to position Structured3D

Structured3D ships no furnished mesh, so its query is rendered on the same
floorplan extrusion as its candidates. Its acoustic-alone recall is 65.6% where
Replica's furnished query gives 20.7%. **It is a matched-geometry upper bound
and a sanity check, and must never be the main benchmark.**

A reviewer who sees +24 there beside +3 on Matterport3D will conclude one of
two things: either the authors do not understand their own setting, or the
Structured3D number is the real result and the Matterport3D number is the
failure. Both conclusions reject the paper. The way to prevent it is a column
in the main table naming the geometry each query was rendered on, which the
table generator now emits, plus a caption sentence saying the rows are not
comparable on equal terms.

Structured3D earns its place as the only benchmark with thirty test rooms,
which is what makes its room-clustered intervals tight. Use it for the
generalisation claim, not the effect-size claim.

## O. The beyond-FoV claim

**Be blunt: the current evidence does not support it.**

What correlational analysis shows is that the gain concentrates where the
visual posterior is ambiguous, which is consistent with beyond-FoV information
and equally consistent with acoustics simply being a second measurement.

What a causal experiment would need is a pair of scenes with identical visible
geometry and different hidden geometry, and a demonstration that the acoustic
score separates them. That is constructible: take a room, render the query from
a pose, then modify only the geometry outside the camera frustum and re-render.
If the acoustic score changes and the visual posterior does not, the claim is
proved. This is the only convincing version and it costs one extra render per
scene.

Path attribution inside the simulator would be stronger still but the engine
does not expose it.

Early-versus-late decomposition is *not* evidence. It has already been run and
the 64--128 ms range alone places the true pose below chance, which if anything
argues against the story.

**Recommended framing if the causal experiment is not run: "vision and
acoustics have different spatial support", stated as motivation, with no claim
that the measured gain comes from unseen geometry.** That framing costs the
paper nothing and removes a reviewer's easiest attack.

## Prioritisation

| tier | experiments | why |
|---|---|---|
| 1 | A real room; B mismatch curve; C global vs verification; D classical baseline; E multi-view; F room holdout; H controls; I pairwise | each removes a named rejection risk |
| 2 | L receiver height; M missing walls; O causal beyond-FoV | strengthen, do not rescue |
| 3 | binaural receiver; grid resolution; compute cost | supplementary interest |
| 4 | a learned acoustic network; more datasets; more fusion rules | adds complexity, answers nobody |

Tier-1 cost, in the labels requested:

| experiment | implementation | compute | data burden | scientific value | risk reduction |
|---|---|---|---|---|---|
| A real room | Medium | Low | **High** | **High** | **High** |
| B mismatch curve | Medium | **High** | Low | **High** | **High** |
| C global vs verification | Low | Low | Low | **High** | Medium |
| D classical baseline | **High** | Medium | Low | Medium | **High** |
| E multi-view | Low | Medium | Low | **High** | **High** |
| F room holdout | Low | Low | Low | Medium | **High** |
| H controls | Low | Low | Low | Medium | Medium |
| I pairwise | Low | Low | Low | **High** | Medium |

## Final package

**Main Table 1, dataset by backbone.** Rows: three datasets by three backbones
by audio on and off. Columns: recall at 0.1, 0.5, 1, 1 m with 30°, 2 and 5 m,
median, RMSE, and the gain with its interval. A column naming the query
geometry. *Takeaway: the gain is small, consistent across backbones, and larger
where the geometry matches, which the table lets the reader see rather than
hides.*

**Main Table 2, fusion and gating.** Rows: the eight rules of experiment K plus
both oracles. *Takeaway: gating is compensating for the domain gap, and under
matched geometry it would be harmful.*

**Main Table 3, real rooms.** Rows: two or three rooms plus pooled. *Takeaway:
the trend survives real acoustic mismatch.*

**Main Figure 3, the mismatch curve.** X: forward-model error, from matched
through wall perturbation to furnished. Y: two curves, acoustic-alone recall
and fusion gain. *Takeaway: global acoustic localization collapses long before
relative verification does.*

**Main Figure 4, candidate-set size.** X: candidate-set size, log scale, from
one hypothesis to the whole grid. Y: recall at 1 m for the method and for the
oracle. *Takeaway: the problem acoustics is asked to solve is only tractable at
small candidate-set size.*

**Main Figure 5, pairwise discrimination.** Grouped bars over four strata, with
chance and visual-ordering baselines. *Takeaway: acoustic evidence is
informative about the choice even where end-to-end recall moves little.*

**Supplementary.** Sensor perturbation, floorplan perturbation, the permutation
controls, the binaural receiver null result, compute cost, the dataset audit
sheets, and the ray-fan and protocol findings.

## The four answers requested

**Five highest-value experiments.** A real room; E multi-view; B mismatch
curve; D classical baseline; I pairwise discrimination.

**Three most likely to move a review from weak reject to weak accept.** A, E,
D. Each answers an objection that is currently unanswerable rather than merely
under-supported.

**If only one more experiment were possible: E, multi-view.** It costs no new
rendering, the checkpoints exist, and it decides whether the paper's
contribution survives a stronger visual backbone. If it fails, the paper needs
reframing before any other experiment is worth running.

**Biggest remaining rejection risk even if everything succeeds.** The effect
size. Two to four points with room-clustered intervals that graze zero, on a
method that is deliberately simple, invites the reviewer to write "interesting
observation, insufficient impact". The defence is not a bigger number; it is to
make the paper's contribution the *characterisation* of when an approximate
acoustic forward model is and is not useful, with the localization gain as one
consequence among several. A paper that claims a small effect and proves it
thoroughly survives review better than one that claims a large effect and
cannot.
