# Research Proposal: Basin, Not Grounding — An Interventional Reach Readout Reveals That VLA Spatial-Generalization Gains Are Mostly Tolerance-Widening

## Problem Anchor
- **Bottom-line problem**: Reported "spatial generalization" improvements for VLA
  manipulation policies are measured almost entirely by task success rate. Success
  rate cannot distinguish a policy that actually *grounds and reaches the object at
  its new location* from one that still executes a canonical (memorized) motor
  program whose grasp basin happens to be wide enough to occasionally succeed. If a
  large fraction of "generalization" is the latter, the field is optimizing and
  reporting the wrong quantity.
- **Must-solve bottleneck**: There is no cheap, model-agnostic readout that
  *dissociates* "the reach is steered by the perceived object position" from "the
  reach is anchored to the training location but succeeds via basin tolerance."
  Without it, both data-centric (augmentation) and model-centric (fine-tuning)
  interventions are credited or dismissed on a metric that is blind to the
  mechanism they claim to fix.
- **Non-goals**: (a) Proposing a new state-of-the-art data-generation method or a
  better acquisition function — an internal 5-arm study already showed our
  gain-targeted acquisition did NOT beat a trivial failure-targeted baseline, and
  we explicitly drop that claim. (b) Solving spatial generalization. (c) A new
  benchmark suite. The contribution is a **readout + a mechanism finding**, not a
  method that wins a leaderboard.
- **Constraints**: Single lab, 1× RTX 6000 Ada (48 GB) primary. Sim-first
  (LIBERO). SmolVLA is the primary policy (frozen VLM + trainable action expert is
  the standard recipe); a second policy family must fit the same GPU. Reviewer
  bar: CoRL / NeurIPS Datasets & Benchmarks. Honesty over hype — the headline
  result is a *negative/partial* mechanism result and must be framed as such.
- **Success condition**: A reviewer agrees that (1) the reach readout is a valid,
  cheap, interventional measure that success rate cannot substitute for; (2) on ≥2
  policies and ≥2 task suites, measured success gains from counterfactual
  fine-tuning are shown to be *predominantly basin-widening*, with only *partial,
  directional* recovery of object-conditioned reaching; and (3) the cause is
  localized (gradient competition with canonical demonstrations, not a frozen-
  feature representational wall). "Yes — success-only evaluation was hiding this,
  and now I can see it."

## Technical Gap
Current VLA spatial-generalization work (MimicGen/DemoGen-style counterfactual
data generation; ManiBox/Mobi-π/N2M spatial-tolerance analysis; success-region
studies) reports success rate under object displacement. The failure mode we
document — "anchored-not-steered": under a 50 mm displacement a competent SmolVLA
(74% clean) reaches the *canonical* training location, not the object (reach
projection proj ≈ 1.0), yet still succeeds 31% of the time because the grasp basin
is wide — is invisible to success rate. Two policies with identical success curves
can have opposite grounding. Naive fixes do not close this: (a) more/uniform
counterfactual data (MimicGen-scale) raises success without moving the reach; (b)
success-region metrics measure *where* it works, not *why* (basin vs grounding).
The missing instrument is an **interventional, displacement-normalized reach
readout** that scores whether the pre-contact reach is *caused by* the perceived
object position, and a study that uses it to attribute success gains to
basin-widening vs grounding.

## Method Thesis
- **One-sentence thesis**: A cheap, interventional, displacement-normalized reach
  readout dissociates object-grounding from basin-tolerance in VLA policies, and
  using it we show that counterfactual fine-tuning's success gains are
  predominantly basin-widening — object-grounding recovers only partially and only
  when synthetic data displaces the canonical demonstrations, localizing the cause
  to gradient competition rather than a representational limit.
- **Why this is the smallest adequate intervention**: The readout reuses the
  existing rollout machinery (object displacement at reset + endpoint logging); the
  only new primitive is a normalized coordinate (u along −d: 0 = object, 1 =
  canonical; ρ = ‖endpoint − object‖/‖d‖) plus a success-conditioned split. No new
  trained component. The mechanism study is a controlled fine-tune sweep, not a new
  architecture.
- **Why timely**: VLAs are the frontier of robot learning; "does it actually use
  its inputs" is exactly the interventional-attribution question (cf. sanity-checks-
  for-saliency, causal confusion) that the field lacks for manipulation.

## Contribution Focus
- **Dominant contribution**: The **reach readout** — a validated, cheap,
  model-agnostic interventional metric (u/ρ, success-conditioned) that success rate
  cannot substitute for — plus the **mechanism finding** it enables: measured
  VLA spatial-generalization gains are mostly basin-widening, not grounding.
- **Optional supporting contribution**: **Cause localization** — a dose-response +
  freeze/unfreeze design showing the anchoring is *defended by gradient competition
  with canonical demonstrations* (removing them yields partial grounding; unfreezing
  the VLM adds little), refuting the "frozen features can't represent it" hypothesis.
- **Explicit non-contributions**: gain-targeted acquisition function (dropped —
  did not beat baseline); a new benchmark; a spatial-generalization method.

## Proposed Method
### Complexity Budget
- **Frozen / reused**: SmolVLA (frozen VLM + action expert), LIBERO sim, existing
  rollout harness, MimicGen/DemoGen-style servo-retargeted synthesis engine.
- **New**: (1) the reach readout (u/v/ρ, success-conditioned, contact/lift-time
  endpoints, with equivalence-test statistics); (2) a controlled counterfactual-
  fraction × plasticity fine-tune design. No new trainable network.
- **Intentionally excluded**: gain-targeted acquisition, new losses, new
  architecture, a new suite.

### System Overview
1. Base policy fine-tuned to competence (SmolVLA, 74% clean on LIBERO-spatial).
2. **Readout**: for each (task, seed), reset with the task object displaced by d
   (grid of magnitudes × directions); roll out; log endpoints at pregrasp, at
   gripper-object contact, and at object-lift onset. Compute u, v, ρ in the
   displacement-normalized frame; split by success/failure.
3. **Mechanism sweep**: synthesize counterfactual demos (object displaced,
   servo-retargeted, renderer-exact, success-filtered); fine-tune the base policy
   under a controlled grid varying (a) counterfactual fraction, (b) whether
   canonical demos are present at fixed synthetic *count*, (c) VLM frozen/unfrozen,
   (d) proprioceptive state masked/noised. Re-run the readout on each.
4. Attribute: does success rise while u stays ≈1 (basin) or does u shift toward 0
   with contained ρ (grounding)? Which knob moves u?

### Core Mechanism (the readout)
- **Input / output**: rollout endpoint + (object, canonical) positions → (u, v, ρ,
  success). **u** = component of endpoint-error along −d (0 = at object, 1 = at
  canonical); **ρ** = normalized distance from the object; success-conditioned.
- **Why this is the main novelty**: it is *interventional* (the object is actively
  displaced, not observed) and *displacement-normalized* (comparable across tasks
  and magnitudes), and the success-conditioned split is what exposes "successes are
  still anchored" — the exact dissociation success rate cannot make. It is the
  robot-manipulation analogue of sanity-checks / causal-attribution, which the VLA
  literature lacks.

### Modern Primitive Usage
- The **VLM** is the object of study (frozen vs unfrozen is a treatment, not a
  bolt-on). The synthesis engine uses **MimicGen/DemoGen-lineage** servo retargeting
  as a *controlled instrument*, explicitly not claimed as novel.

### Training Plan
- No new objective. Standard SmolVLA action-expert fine-tune (lr 1e-4, 20k steps,
  bf16). The design varies data composition and plasticity, ≥3 seeds per
  claim-critical arm. Contact/lift-time endpoints extracted from sim state.

### Failure Modes and Diagnostics
- **"Lower proj = degraded policy, not steering"** → success-conditioned u split +
  ρ concentration + a degradation control (a checkpoint made worse by label noise
  should lower success without the success-specific u-shift). This is the central
  validity threat and is designed against directly.
- **Servo-label proprioceptive shortcut** → state-masked/noised arm (M4) +
  first-chunk intervention (teacher-force k steps toward the object; does it
  continue tracking or snap back?).
- **Readout instability at few clusters (10 tasks)** → wild-cluster bootstrap +
  TOST equivalence bounds for the "unchanged" claims.

### Novelty and Elegance Argument
Closest work: causal confusion in imitation (de Haan et al.) names the shortcut but
gives no manipulation reach readout; MimicGen/DemoGen generate data but evaluate on
success; Mobi-π/N2M measure spatial tolerance but as success-regions, not as an
interventional grounding-vs-basin dissociation; LP-FT/DFR/WiSE-FT explain
fine-tuning geometry in classification, not in embodied reach. The irreducible new
thing: **a cheap interventional reach readout that separates basin from grounding,
and the resulting evidence that success-only VLA evaluation systematically
over-credits grounding.**

## Claim-Driven Validation Sketch
### Claim 1 (dominant): Success rate hides the basin/grounding distinction; the readout exposes it.
- **Minimal experiment**: on the base policy, show two regimes (small vs large
  displacement, or two tasks) with similar success but opposite u; show
  success-conditioned u split.
- **Baseline/ablation**: success rate alone (blind); reach readout (sighted).
- **Metric**: u (success-conditioned), ρ concentration, fraction within ρ≤0.75.
- **Expected**: successes at 50 mm are anchored (u≈0.8–1.0) despite 31% success.

### Claim 2 (supporting): Counterfactual-fine-tuning gains are mostly basin-widening; grounding recovers only partially, and only by displacing canonical demos.
- **Minimal experiment**: fine-tune at synthetic fraction {17,50,100}% at fixed
  total token budget; plus a 100%-synthetic + equal-count-canonical arm (decouples
  fraction from count); frozen vs unfrozen VLM; ≥3 seeds.
- **Baseline/ablation**: no-aug; state-masked arm; degradation control.
- **Metric**: Δu (successes) with wild-cluster-bootstrap CI + TOST; Δsuccess.
- **Expected**: success rises across arms; u shifts toward object only as canonical
  fraction drops (monotone in dose), never reaching ρ-convergence; unfreezing adds
  little → cause = gradient competition, not representation.

## Experiment Handoff Inputs
- **Must-prove claims**: readout validity (not a degradation artifact); basin-vs-
  grounding attribution; cause localization (gradient competition).
- **Must-run ablations**: fraction-vs-count decoupling; state-masked; degradation
  control; first-chunk intervention; contact/lift-time endpoints; 2nd policy; 2nd
  suite.
- **Critical datasets/metrics**: LIBERO-spatial + LIBERO-object; u/ρ success-
  conditioned; wild-cluster bootstrap + TOST.
- **Highest-risk assumptions**: (1) u-shift is grounding not dispersion; (2)
  findings transfer to a 2nd policy/suite; (3) servo labels aren't a proprio shortcut.

## Compute & Timeline Estimate
- Readout runs: ~1–2 GPU-h each. Fine-tunes: ~3 GPU-h each; ~8 claim-critical arms ×
  3 seeds × 2 policies ≈ manageable in ~2–3 weeks on 1 GPU with the existing
  resumable orchestrator. Data/annotation: none (sim). Timeline: ~3–4 weeks to a
  submittable draft (5 figures already exist).
