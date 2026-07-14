# Round 1 Refinement

## Problem Anchor (verbatim from round 0)
- **Bottom-line problem**: Reported VLA "spatial generalization" is measured by
  success rate, which cannot distinguish a policy that grounds+reaches the displaced
  object from one that still runs a canonical motor program whose grasp basin is wide
  enough to occasionally succeed.
- **Must-solve bottleneck**: No cheap, model-agnostic, interventional readout
  dissociates "reach steered by perceived object position" from "reach anchored to
  training location, succeeds via basin tolerance."
- **Non-goals**: not a new SOTA data-generation/acquisition method (the gain-targeted
  acquisition claim is dropped — it did not beat a trivial baseline); not solving
  spatial generalization; not a new benchmark suite.
- **Constraints**: 1× RTX 6000 Ada, sim-first (LIBERO), SmolVLA primary + one 2nd
  policy on same GPU; CoRL / NeurIPS D&B bar; honesty over hype (headline is a
  negative/partial mechanism result).
- **Success condition**: reviewer agrees (1) the readout is a valid cheap
  interventional measure success rate cannot substitute for; (2) on ≥2 policies and
  ≥2 suites, counterfactual-FT success gains are predominantly basin-widening with
  only partial directional grounding recovery; (3) the cause is localized (gradient
  competition, not frozen-feature limit).

## Anchor Check
- **Original bottleneck**: success rate hides basin-vs-grounding. Still the target.
- **Why the revised method still addresses it**: the dominant contribution stays the
  interventional u/ρ readout + the basin-not-grounding finding; the added
  degradation-null control *strengthens* exactly the anchored claim (that the u-shift
  is grounding, not a degraded policy's survivor subset). No drift.
- **Reviewer suggestions rejected as drift**: none — all accepted; none change the
  problem.

## Simplicity Check
- **Dominant contribution after revision**: (1) the readout; (2) the basin-not-
  grounding mechanism finding. Cause-localization is one supporting factorial.
- **Components removed/merged**: the 5-arm acquisition study is DEMOTED from a
  contribution to one motivating-negative paragraph ("even a diagnostic-targeted
  acquisition function does not beat trivial failure-targeting — success is the wrong
  objective"). Cause-localization collapses to a single factorial. State-masking,
  first-chunk intervention, and the decodability probe become *validity checks*, not
  claims.
- **Reviewer suggestions rejected as unnecessary complexity**: none.
- **Why the remaining mechanism is still smallest adequate**: one readout + one
  factorial + a matched null; no new trainable network.

## Changes Made

### 1. Add the clean-success-matched degradation null + difference-in-differences (CRITICAL)
- **Reviewer said**: success-conditioned u-split is strong but not bulletproof;
  M2/M3 also dropped clean success, leaving a survivor/selection concern.
- **Action**: add a **degradation-null** family — take the base policy and degrade it
  by (a) label-noise fine-tune, (b) action-noise, (c) early-stopped/corrupted
  fine-tune — tuned so its *clean AND displaced success match the intervention arm*
  (e.g. match M2 at 52% clean). Evaluate with identical success-conditioned endpoint
  statistics. Report the **cluster-bootstrap difference-in-differences**:
  DiD = ([u_succ − u_fail]_intervention) − ([u_succ − u_fail]_degradation-null),
  plus ρ and the *orthogonal* dispersion (|v| IQR) so a pure spread-inflation is
  distinguishable from a directional shift. **Grounding is claimed only if the
  intervention's success-specific u-shift exceeds the matched null's, with a
  positive DiD CI, WITHOUT larger |v| dispersion.**
- **Reasoning**: this is the exact control that separates "partial grounding" from
  "worse policy whose surviving successes are a shifted subset."
- **Impact**: converts the central claim from suggestive to defensible; becomes the
  paper's key validity figure.

### 2. Demote the 5-arm acquisition study to motivation
- **Reviewer said**: move acquisition out of the main contribution.
- **Action**: it appears only as motivating negative evidence (Fig: success rises,
  steering doesn't, across all 5 strategies) supporting "success is the wrong
  objective." No acquisition claim in the contributions.
- **Impact**: sharper single thesis; removes the weakest (already-failed) claim.

### 3. Collapse cause-localization into one factorial + add contact/lift-time endpoints
- **Reviewer said**: one factorial; add decodability probe; paired rollouts.
- **Action**: factorial = {counterfactual fraction ∈ 17/50/100%} × {canonical demos
  present vs absent at FIXED synthetic count (decouples fraction from count)} × {VLM
  frozen vs unfrozen}, ≥3 seeds per claim-critical cell, paired rollouts (identical
  seed/task/magnitude across arms). Endpoints measured at pregrasp AND
  gripper-object contact AND object-lift onset (from sim state).
- **Impact**: removes the fraction-vs-count confound flagged internally; contact-time
  endpoint fixes proj's weak tie to success (base successes ρ=2.29 at pregrasp).

### 4. Add a frozen-interface object-position decodability probe (validity check)
- **Reviewer said**: distinguish "features lack position" from "expert ignores it."
- **Action**: linear/shallow probe of the frozen SmolVLM features (and the action
  expert's conditioning inputs) for displaced object xy across the readout grid. If
  position IS decodable but u stays ≈1 → the expert ignores available signal (H2/
  representation refuted, gradient-competition supported); if NOT decodable → the
  features are the wall. Presented as a check, not a claim.
- **Impact**: makes the "cause = gradient competition, not representation" claim
  mechanistic rather than inferential.

## Revised Proposal

### Method Thesis
A cheap, interventional, displacement-normalized reach readout (u along −d, ρ
normalized error, success-conditioned, at pregrasp/contact/lift) dissociates
object-grounding from basin-tolerance in VLA policies; using it — and a
clean-success-matched degradation null — we show counterfactual-fine-tuning success
gains are predominantly basin-widening, with only partial directional grounding
recovery that appears solely when synthetic data displaces the canonical
demonstrations, localizing the cause to gradient competition (a frozen-feature
position-decodability probe rules out a representational wall).

### Contribution Focus
- **Dominant**: the readout (validated against a matched-degradation null) + the
  basin-not-grounding finding.
- **Supporting**: cause localization (gradient competition) via one factorial +
  decodability probe.
- **Non-contributions**: acquisition function (motivation only), new benchmark, new
  method.

### Claim-Driven Validation Sketch
- **Claim 1 (readout validity + basin/grounding dissociation)**: base policy, two
  regimes of similar success but opposite u; success-conditioned split. Metric: u
  (success-cond), ρ, fraction within ρ≤0.75. Expected: 50 mm successes anchored
  (u≈0.8–1.0) at 31% success.
- **Claim 2 (gains are basin-widening; grounding only partial + only by displacing
  canonical demos; cause = gradient competition)**: the factorial + the matched
  degradation null + DiD; decodability probe; ≥3 seeds; wild-cluster bootstrap +
  TOST for the "unchanged" (equivalence) claims. Expected: DiD positive for
  high-synthetic arms (directional u-shift beyond the null, no |v| inflation),
  monotone in canonical-fraction removal, unfreezing adds little, position decodable
  from frozen features.
- **Validity checks (not claims)**: state-masked/noised arm (proprio-shortcut),
  first-chunk intervention (continue-vs-snap-back), paired rollouts.

### Compute & Timeline
Readouts ~1–2 GPU-h; fine-tunes ~3 GPU-h. Claim-critical arms: factorial (~6 cells)
+ degradation null (~3) × ≥3 seeds × 2 policies, staged on the resumable
orchestrator; ~3 weeks on 1 GPU. 5 figures exist; ~3–4 weeks to a draft.
