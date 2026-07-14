# Refinement Report

**Problem**: success-only VLA spatial-generalization evaluation hides basin-vs-grounding.
**Initial approach**: invert the reach-field diagnostic into a training engine (P2T).
**Date**: 2026-07-15 · **Rounds**: 4/5 · **Final score**: 9.1/10 · **Verdict**: READY.

## Output Files
- Review summary: `REVIEW_SUMMARY.md`
- Final proposal: `FINAL_PROPOSAL.md`
- Score history: `score-history.md`

## Score Evolution
| Round | ProbFid | MethSpec | Contrib | Frontier | Feas | ValFocus | Venue | Overall | Verdict |
|-------|--------|---------|--------|---------|-----|---------|------|--------|--------|
| 1 | 9   | 7   | 8   | 8   | 8   | 6   | 7   | 7.8 | REVISE |
| 2 | 9   | 8.5 | 8.5 | 8   | 7   | 8   | 8   | 8.3 | REVISE |
| 3 | 9.5 | 9   | 9   | 8.5 | 8   | 9   | 8.5 | 8.9 | REVISE |
| 4 | 9.5 | 9.2 | 9.1 | 8.7 | 9.1 | 9.2 | 9.0 | 9.1 | READY  |

## Method Evolution Highlights
1. Reframed from "P2T = better acquisition" (failed claim) to an evaluation/mechanism
   paper: the interventional reach readout + basin-not-grounding finding.
2. Added the decisive validity control: a competence-matched degradation null with a
   sign-correct directional statistic G = [u_fail−u_succ]_int − [u_fail−u_succ]_null,
   plus an orthogonal-dispersion guard — separating grounding from "worse policy,
   shifted survivors."
3. Pre-committed scope (Minimum Publishable Matrix) + locked feasible 2nd policy
   (Octo-Small, which also tests VLM-specificity), bounding the study to ~2–3 GPU-weeks.

## Pushback / Drift Log
No drift across rounds; all reviewer items accepted (each strengthened the anchored
claim rather than changing the problem). The one reviewer error we would have
inherited — a wrong-signed DiD — was itself flagged by the reviewer in round 2 and
fixed in round-2-refinement.

## Remaining Weaknesses
Execution risk only: run the pre-registered MPM, report all three nulls and all scoped
replication outcomes (including failures), keep everything beyond the MPM labeled
"extended."

## Next Steps
- READY → `/experiment-plan` to turn the MPM into an execution-ready roadmap, then
  `/run-experiment`.
- Then the paper track (`/paper-plan` → `/paper-write` → figures → `/paper-compile`).

## Raw Reviewer Responses
See `round-1-review.md`, `round-2-review.md`, `round-3-review.md` (each has a
`<details>` raw block); reviewer thread `019f616c-4d07-7e60-9ac1-f07be9539cc3`.
