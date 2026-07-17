# Novelty Check Report — P2T "Basin vs Grounding" reach readout

**Date**: 2026-07-17 · **Search**: scite MCP + arXiv abstracts · **Cross-check**: gpt-5.5 xhigh (Codex).

## Proposed contribution
An interventional, displacement-normalized, success-conditioned **reach readout** that,
on VLA manipulation policies, dissociates object-**grounding** (reach steered by the
perceived object) from **basin-tolerance** (reach anchored to the canonical training
location but succeeding because the grasp basin is wide) — and the cross-regime finding
that the same counterfactual-FT intervention basin-widens in wide-basin tasks but is
forced to ground in tight-basin tasks (invisible to success rate).

## Core claims → novelty
| # | Claim | Novelty | Closest |
|---|---|---|---|
| 1 | Displacement-normalized, **success-conditioned** endpoint readout (u/ρ) separating basin-tolerance from grounding | **MEDIUM–HIGH** (no prior does exactly this) | Robust Skills, Brittle Grounding |
| 2 | Success-only spatial-gen eval is a blind spot | **LOW** (already argued) | LIBERO-X; Robust Skills |
| 3 | Same intervention → basin-widening (wide basin) vs grounding (tight basin) | **HIGH** (specific, unclaimed) | — |
| 4 | Defended-prior / gradient competition (boundary) | **LOW–MEDIUM** (LP-FT/DFR lineage; wide-basin-only) | LP-FT, DFR, WiSE-FT |

## Closest prior work
| Paper | arXiv | Type | Overlap | Delta (ours) |
|---|---|---|---|---|
| **Robust Skills, Brittle Grounding** | 2602.24143 | eval/diagnosis | controlled object-loc perturbations; decomposed success/grasp/**reach** metrics separating primitive vs grounding | they ask "did it reach the object as regularities are removed"; **we ask, on SUCCESSES, did the endpoint move WITH the object (u→0) or stay canonical (u→1) and survive via basin** — continuous displacement-normalized u/ρ + success/failure split + basin-vs-grounding interpretation |
| **Affordance Field Intervention (Memory Traps)** | 2512.07472 | method/fix | names "memory trap" = reach memorized trajectory not object = same anchoring phenomenon | they PROPOSE a fix; **no readout, no dissociation, no eval-blindspot** |
| **LIBERO-X: Robustness Litmus** | 2602.06556 | benchmark | argues success-only eval is misleading; spatial-gen capability axis | their instrument is progressive PERTURBATIONS + capability decomposition by **success**; **not a mechanistic where-does-the-reach-go readout** |
| **Do You Need Proprioceptive States?** | 2509.18644 | mechanism | proprio → shortcut on training trajectories, poor spatial gen | collides with our mechanistic story; **not a basin-vs-grounding endpoint readout** (relevant to our proprio-shortcut appendix) |
| **Causal Confusion in Imitation** | 1905.11979 | mechanism | policy latches nuisance correlates | foundational; no manipulation reach readout |
| One-Shot IL: Pose Estimation Perspective | 2310.12077 | method | IL as object-pose-conditioned trajectory transfer | not a VLA diagnostic, not success-conditioned |
| Mobi-π / N2M / ManiBox | 2505.23692 / 2509.18671 / 2411.01850 | analysis | spatial tolerance / success region / scaling | success-region, not grounding-vs-basin dissociation |

## Overall assessment
- **Score: 6.5 / 10** (Codex). **Recommendation: PROCEED WITH CAUTION.**
- **Key differentiator**: the readout that explains **successful** episodes (basin vs
  grounding), via displacement-normalized success-conditioned endpoints, + the
  cross-regime inversion. No prior does exactly this.
- **Rejection risk**: "known phenomenon (memory trap / causal confusion) + another
  success-only-eval critique (LIBERO-X)." Mitigate by NOT selling the phenomenon or the
  eval-critique as the novelty.

## Suggested positioning
> "We do not introduce the observation that VLA policies memorize spatial regularities.
> We introduce a readout for a different question: whether apparent spatial
> generalization — including successful rollouts — is caused by object-grounded steering
> or by task-level tolerance around a canonical reach."
- Title/abstract about the **endpoint readout** and **mechanisms of *successful*
  spatial generalization**, NOT "memory traps."
- Put Robust-Skills-Brittle-Grounding, AFI, LIBERO-X in related work EARLY; state
  exactly what each does not measure (where-the-successful-reach-goes).
- Lead with **diagnostic method + cross-regime empirical finding** (strong shape);
  avoid "another benchmark critique" (weak shape). Keep gradient-competition secondary
  unless the null is very strong.
- For NeurIPS D&B: frame the readout as a **reusable evaluation artifact**, not a
  one-off LIBERO critique.
