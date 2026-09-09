# The observation data changed under the experiments

On 2026-09-09 between 09:14 and 11:27 the dataset producer re-rendered every
impulse response in `echoloc_dataset/rir/`. Nothing in this repository recorded
which render an experiment had read, so results measured before and after the
change sit in the same directory looking alike, and a headline number that no
longer reproduced took several hours to attribute.

## What changed

| | before | after |
|---|---|---|
| sample rate | 8 kHz | **48 kHz** |
| indirect rays | 4,096 | **20,000** |
| ray depth | 6 | **50** |
| diffraction | off | **on, to order 10** |
| direct-sound guard | 16 samples | 96 samples (**2 ms either way**) |
| usable window | 1,024 samples | 6,144 samples (**128 ms either way**) |

The physical convention was preserved. The producer even left
`guard_usable_at_8khz_spec: [16, 1024]` in the metadata, recording what the old
spec had been.

## Why it broke this project

This code hard-coded the guard and the window as **sample counts** rather than
times. Reading a 48 kHz recording with an 8 kHz sample count skips 0.33 ms
instead of 2 ms and keeps 21 ms instead of 128 ms, so most of the reflections
were discarded before any comparison happened. Every acoustic score collapsed to
chance while every visual score stayed identical, because vision reads `rgb/`.

The candidate grid was also stale in a second, independent way: it had been
rendered at 8 kHz with diffraction off, so even after the sample rates were
reconciled the model was a strictly simpler physics than the observation.

Resampling the observation to 8 kHz recovered most of it on office_4:
matched-domain GT rank went from 1474 back to 20, against 6 before the change.
Re-rendering the candidate grid at the new settings is the actual fix.

## What this cost

Several hours were spent looking in the wrong places: the acoustic feature, its
time resolution, the normalisation, the fusion rule, and a learned domain-bridge
encoder were each suspected and tested. All of them were fine. A headline result
was retracted that should not have been.

## What is in place so that it cannot happen the same way again

- `track1_core/provenance.py` stamps each result with the commit, a digest of
  any uncommitted diff, a content hash of every input file, and library
  versions. Two runs that disagree can now be attributed rather than guessed at.
- `track1_core/likelihood/grid_score.py` reads the recording's own sample rate
  from `rir_metadata.json` and decimates to the grid's rate.
- The four scattered copies of the acoustic scoring were merged into that one
  module, so two scripts can no longer report the same quantity and disagree.

## Reading older results

Anything in `outputs/metrics/` written before 2026-09-09 09:14 was measured
against the previous render and cannot be reproduced: that data was overwritten
in place and no backup or filesystem snapshot exists. Those files are kept for
reference but must not be compared against anything measured afterwards.
