# Superseded write-ups

These describe the project at earlier stages and are kept because they are the
record, not because they are current. Every one of them contains at least one
conclusion that later measurement overturned, which is precisely why deleting
them would be the wrong kind of tidying: a reader who finds an old number
quoted somewhere needs to be able to see where it came from and what replaced
it.

| file | what it was | what replaced it, and why |
|---|---|---|
| `acoustic_results.md` | the first results write-up, cell-wise fusion with a 2 ms energy envelope and a margin gate | `../mode_fusion_diagnosis.md`. Its gains were real but its intervals were never computed, and both cell-wise rules turn out to span zero. Its §2 tables also predate the 48 kHz re-render. |
| `acoustic_event_matching.md` | sparse arrival-event matching and one-corner non-line-of-sight path hypotheses | nothing: measured to hurt, ground-truth rank 411 to 509. Kept as a negative result. |
| `audio_visual_analysis.md` | a self-contained brief written for an outside reader before the mode-level formulation | `../mode_fusion_diagnosis.md`, which supersedes it entirely and adds the diagnosis that motivated the redesign. |

Anything still true in these files is restated in the current documents. Where
they disagree with a current document, the current document is right.
