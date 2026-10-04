# Rule: evidence strength — always produce, and always review for, the strongest evidence

Source: "PPR Expectations - 261 - Bb.pdf", slide 4 ("Strength of Evidence"), Weekly Blackboard
Week 7. Extracted verbatim on 2026-10-04.

| Rank | Level of evidence | Course note |
|---|---|---|
| 0 | Words (text-only explanation) | Never acceptable as evidence for design or prototype |
| 1 | Basic sketches, diagrams or flow diagram | Basic conceptual representation, lacks structure. Not direct proof of either design or prototype |
| 2 | Calculations & predictions | Supports design feasibility; not acceptable for prototype |
| 3 | Simulations & CAD model | Good digital representation; acceptable for design, not evidence for prototype |
| 4 | Picture of the demo (physical or simulation) | Visual proof but no motion or interaction; weak evidence for prototype |
| 5 | Video of the demo (physical or simulation) | Shows functionality but lacks physical interaction |
| 6 | Simulations & CAD model + data | Digital validation with data-backed results for reliability |
| 7 | Picture of the demo + data | Visual proof supported by recorded data, for credibility |
| 8 | Video of the demo + data | Demonstrates functionality with empirical data; highly verifiable |
| **9** | **Physical prototype or working simulation demo at the presentation** | **Strongest evidence; direct proof of functionality** |

## Rules for every agent that produces or reviews evidence

1. **Target Rank 9.** Every spec claim should be backed by something that runs live in front of
   the reviewer: a notebook cell or CLI command that recomputes the number from data on disk.
   A screenshot or a pasted number is not enough.
2. **Rank 6–8 is the fallback, never Rank 0–3.** If something can't run live (for example, a
   training run that takes hours), ship it as stored, reproducible data: a JSON/CSV report with
   a seed, a commit hash and the exact command. Pair it with a live re-check that loads that
   report and re-scores the saved predictions.
3. **Label provenance on every number.** Each number is one of: `[MEASURED on <dataset>]`,
   `[SYNTHETIC]`, `[CALCULATED]` or `[CITED: source]`. Public-dataset results are real
   measurements of *that* dataset's bearings, not of the team's rig. Say which.
4. **No fabricated or cherry-picked results.** Splits are by bearing/run. Test bearings are never
   seen in training or model selection. When a spec is not met, report FAIL with the measured
   number, explain the gap, and give the plan. Never quietly redefine the metric.
5. **A review agent scores every artifact** against this ladder. It reports the rank achieved,
   the rank achievable, and the concrete step that would raise it. If an artifact sits below
   Rank 6 and a stronger form is feasible before the deadline, the review fails.
