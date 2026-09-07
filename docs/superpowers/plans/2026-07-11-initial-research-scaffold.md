# AV-FPLoc Initial Research Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a small, reproducible repository skeleton for the Track1 acoustic-visual floorplan-localization method without publishing external baselines, datasets, caches, or checkpoints.

**Architecture:** The repository root describes scope and provenance; `track1_core/` is the future implementation package with stable module boundaries; `docs/` is the human and LLM-readable research contract. Data-bearing artifacts remain outside the repository and are referenced only by format and provenance.

**Tech Stack:** Python package skeleton, Markdown documentation, Git ignore rules.

## Global Constraints

- Include `track1_core/` source skeleton and research design documents only.
- Exclude datasets, `desdf.npy`, images, checkpoints, model outputs, external repository clones, secrets, and machine-specific paths.
- Treat the F3Loc repository as an external baseline; record its pinned commit and behavior, but do not copy its source into this repository.
- Do not claim that an acoustic method is implemented; this commit establishes contracts and planning context only.

---

### Task 1: Establish the publishable repository boundary

**Files:**
- Modify: `README.md`
- Create: `.gitignore`
- Test: repository-tree and ignore-rule shell checks

**Interfaces:**
- Consumes: initial one-line repository README.
- Produces: a public entry point that directs readers to the method package and design documents, plus ignore rules that prevent data/cache publication.

- [ ] **Step 1: Add a failing repository-boundary check**

Run:

```bash
test -f .gitignore && git check-ignore -q data/example.npy
```

Expected: FAIL because `.gitignore` does not exist.

- [ ] **Step 2: Add the minimal public README and ignore rules**

```text
README: research goal, current status, source layout, external-baseline boundary.
.gitignore: Python artifacts plus data/, outputs/, checkpoints/, desdf caches, external repos, and secrets.
```

- [ ] **Step 3: Re-run the boundary check**

Run:

```bash
test -f .gitignore && git check-ignore -q data/example.npy
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md .gitignore
git commit -m "docs: define public research repository boundary"
```

### Task 2: Publish the Track1 module skeleton

**Files:**
- Create: `track1_core/__init__.py`
- Create: `track1_core/README.md`
- Create: `track1_core/{configs,datasets,evaluation,floorplan,fusion,likelihood,visualization}/__init__.py`
- Create: `track1_core/{configs,datasets,evaluation,floorplan,fusion,likelihood,visualization}/README.md`
- Test: Python import check

**Interfaces:**
- Consumes: approved module ownership boundaries.
- Produces: importable `track1_core` package and module-level documents that state the future interface of each component.

- [ ] **Step 1: Add a failing import check**

Run:

```bash
python -c "import track1_core"
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Copy only the existing Track1 package skeleton**

```python
"""Common code for Track1 acoustic-visual floorplan localization."""
```

Each submodule receives an empty package initializer and a short responsibility README; no baseline code or runtime dependency is copied.

- [ ] **Step 3: Re-run the import check**

Run:

```bash
python -c "import track1_core; import track1_core.floorplan; import track1_core.likelihood"
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add track1_core
git commit -m "chore: add track1 method package skeleton"
```

### Task 3: Add the research design context pack

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/data_contracts.md`
- Create: `docs/f3loc_baseline_facts.md`
- Create: `docs/decision_log.md`
- Test: documentation heading and forbidden-artifact checks

**Interfaces:**
- Consumes: the F3Loc code inventory and current research objective.
- Produces: concise design context that can be uploaded to a ChatGPT Project or used as the repository's stable source of truth.

- [ ] **Step 1: Add a failing context-pack check**

Run:

```bash
test -f docs/architecture.md && test -f docs/data_contracts.md
```

Expected: FAIL because `docs/` does not exist.

- [ ] **Step 2: Write the four focused documents**

```text
architecture.md: component flow and ownership.
data_contracts.md: pose-grid, ray, likelihood, mask, dtype, and coordinate invariants.
f3loc_baseline_facts.md: pinned external baseline facts and non-transferable assumptions.
decision_log.md: dated decisions and open decisions.
```

- [ ] **Step 3: Re-run the context-pack check**

Run:

```bash
test -f docs/architecture.md && test -f docs/data_contracts.md && rg -n '^# ' docs/*.md
```

Expected: PASS with one top-level heading per document.

- [ ] **Step 4: Commit**

```bash
git add docs
git commit -m "docs: add acoustic-visual localization design context"
```

### Task 4: Verify the initial public export

**Files:**
- Verify: all tracked files

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a clean, inspectable tree ready for a user-authenticated push to `origin/main`.

- [ ] **Step 1: Verify package imports and public tree**

Run:

```bash
python -c "import track1_core, track1_core.floorplan, track1_core.likelihood, track1_core.fusion, track1_core.evaluation"
git ls-files
```

Expected: imports succeed; tracked files contain no datasets, cache, checkpoints, outputs, or external baseline source.

- [ ] **Step 2: Verify ignore rules and working tree**

Run:

```bash
git check-ignore -q data/example.npy
git check-ignore -q outputs/example.json
git status --short
```

Expected: both ignore checks succeed and the working tree is clean after commits.

- [ ] **Step 3: Hand off for authenticated push**

```bash
git push origin main
```

Expected: user runs this command in an authenticated terminal; no credential is supplied to an agent.
