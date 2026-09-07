# AV-FPLoc

Research scaffold for active acoustic-visual floorplan localization under visual degradation.

## Status

The repository includes a collaborator-facing Gibson SoundSpaces RIR and acoustic-likelihood reproduction path. Fusion remains research work in progress; the visual F3Loc baseline remains an external pinned adapter.

## Research goal

Estimate a single-observation 2D floorplan pose posterior by combining visual structural cues with active acoustic geometry cues. The primary question is whether acoustic geometry reduces the structural ambiguity that remains when visual observations are low-information or visually degraded.

## Repository layout

```text
track1_core/  Future implementation package and module ownership boundaries.
docs/         Architecture, tensor contracts, baseline facts, and decisions.
```

## External baseline boundary

F3Loc is used as a baseline and protocol reference, not copied into this repository. The baseline inventory in [docs/f3loc_baseline_facts.md](docs/f3loc_baseline_facts.md) records the relevant behavior and pinned source revision.

## Data policy

Datasets, floorplan caches, checkpoints, generated figures, external repository clones, secrets, and machine-specific paths are deliberately excluded. Reproduction instructions may name required artifact formats, but do not publish the artifacts themselves.

## Design-first workflow

Use the documents in `docs/` as the source of truth for research design. Record accepted design choices in [docs/decision_log.md](docs/decision_log.md) before implementation changes are made in `track1_core/`.

## Gibson RIR Reproduction

With the shared licensed assets under `/file2/jeongeon/AV-FPLoc` and an existing SoundSpaces environment, run the read-only preflight:

```bash
export AVFPLOC_DATA_ROOT=/file2/jeongeon/AV-FPLoc
python scripts/verify_collaborator_env.py --scene Springhill
```

The full workflow is documented in [docs/collaborator_gibson_rir.md](docs/collaborator_gibson_rir.md).
