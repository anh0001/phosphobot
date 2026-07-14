# Review Summary

**Problem**: success-only VLA spatial-generalization evaluation hides basin-vs-grounding.
**Initial approach**: invert the reach-field diagnostic into a training engine (P2T).
**Date**: 2026-07-15 · **Rounds**: 4/5 · **Final score**: 9.1/10 · **Verdict**: READY.

## Problem Anchor
Reported VLA spatial generalization is measured by success rate, which cannot
distinguish grounding+reaching the displaced object from a canonical motor program
whose grasp basin is wide enough to occasionally succeed. Need a cheap, model-agnostic,
interventional readout that dissociates the two. (Full anchor in FINAL_PROPOSAL.md.)

## Round-by-Round Resolution Log

| Round | Main reviewer concerns | What this round changed | Solved? | Remaining risk |
|-------|------------------------|-------------------------|---------|----------------|
| 1 (7.8) | degradation-vs-grounding defense underspecified; acquisition study dilutes focus | added matched-degradation null + DiD; demoted acquisition to motivation; one factorial; decodability probe | partial | null estimand not yet formal |
| 2 (8.3) | DiD SIGN ERROR (u=0 at object → grounding is negative); null matching/TOST/2nd-policy underspecified | corrected estimand to G=[u_fail−u_succ]_int−[..]_null>0; pre-registered null matching + TOST margins + clustering; named 2nd suite/policy | partial | feasibility of full matrix |
| 3 (8.9) | full matrix tight for 1 GPU; scope could look post-hoc | locked Octo-Small as 2nd policy; pre-registered Minimum Publishable Matrix | yes | none conceptual |
| 4 (9.1) | — | (verification round) | READY | ordinary execution risk only |

## Overall Evolution
- The paper became an **evaluation/mechanism** paper: dominant contribution = the
  interventional reach readout + the basin-not-grounding finding; the failed
  acquisition study is demoted to one motivating figure.
- The central validity threat (lower u = grounding vs degradation) is now defended by
  a formally correct, pre-registered, competence-matched degradation null with a
  directional statistic G and an orthogonal-dispersion guard.
- Scope is pre-committed via the Minimum Publishable Matrix, so results can't be
  selectively scoped after the fact.

## Final Status
- Anchor: preserved across all rounds.
- Focus: tight (one dominant + one supporting contribution).
- Modernity: appropriate (VLM as object of study; synthesis as MimicGen/DemoGen
  instrument; interventional attribution).
- Strongest parts: the readout + the matched-null causal design + pre-registered scope.
- Remaining weaknesses: execution risk only (run the MPM honestly, report all nulls
  and all scoped replication outcomes including failures).
