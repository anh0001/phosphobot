# Round 2 Refinement

## Problem Anchor (verbatim)
- **Bottom-line problem**: Reported VLA "spatial generalization" is measured by
  success rate, which cannot distinguish grounding+reaching the displaced object from
  a canonical motor program whose grasp basin is wide enough to occasionally succeed.
- **Must-solve bottleneck**: No cheap, model-agnostic, interventional readout
  dissociates "reach steered by perceived object position" from "reach anchored to
  training location, succeeds via basin tolerance."
- **Non-goals**: not a new SOTA data-generation/acquisition method (gain-targeted
  acquisition dropped); not solving spatial generalization; not a new benchmark.
- **Constraints**: 1× RTX 6000 Ada, sim-first (LIBERO), SmolVLA primary + one 2nd
  policy on same GPU; CoRL / NeurIPS D&B bar; honesty over hype.
- **Success condition**: reviewer agrees (1) readout is valid+cheap+interventional;
  (2) on ≥2 policies/≥2 suites, counterfactual-FT gains are predominantly basin-
  widening with partial directional grounding; (3) cause localized (gradient
  competition, not frozen-feature limit).

## Anchor Check
- Bottleneck unchanged. All round-2 fixes are statistical-rigor fixes to the SAME
  claim (correct estimand, pre-registration, named generality). No drift.

## Simplicity Check
- Dominant contribution unchanged (readout + basin-not-grounding). Round-2 only
  formalizes the null estimand and pre-registration; state-mask/first-chunk move to
  appendix validity checks; degradation-null = one pre-registered primary + robustness
  variants (not a family to sift through). Nothing added conceptually.

## Changes Made

### 1. Corrected, pre-registered grounding estimand (fixes the DiD sign error) — BLOCKING
- **Reviewer said**: with u=0 at object, grounding makes (u_succ − u_fail) more
  NEGATIVE, so a "positive DiD" is the wrong sign.
- **Action**: define the **directional success-specific grounding gain**
  **G = ([u_fail − u_succ]_intervention) − ([u_fail − u_succ]_degradation-null)**.
  Grounding is claimed iff **G > 0** with a task-cluster (wild-cluster) bootstrap CI
  excluding 0, AND the intervention does NOT show larger orthogonal dispersion
  (|v| IQR_intervention ≤ |v| IQR_null within a pre-declared tolerance). Intuition:
  in a genuinely grounding policy the *successful* reaches are pulled toward the
  object (low u) relative to its *failed* reaches, and by MORE than a merely-degraded
  policy matched to the same competence.
- **Reasoning**: correct sign + a matched null + a dispersion guard is exactly the
  estimand that separates grounding from "worse policy, shifted survivors."
- **Impact**: the key validity result is now formally stated and directionally correct.

### 2. Pre-registered degradation-null matching + reporting rule
- **Action**: the **primary degradation null** = an action-noise fine-tune from the
  base policy, its noise scale tuned so BOTH clean success (within ±3 pp) AND
  displaced success @50 mm (within ±3 pp) match the intervention arm being tested
  (e.g. M2 at ~52% clean), matched per (task, magnitude) where cell counts allow.
  **Robustness nulls**: label-noise FT and early-stopped FT, matched the same way.
  Reporting rule (anti-cherry-pick): the intervention must beat the **strongest**
  (smallest-G-defeating, i.e. most conservative) matched null; report G against ALL
  three nulls, not the easiest. Pre-registered before running.
- **Impact**: removes comparator-selection bias.

### 3. Pre-registered equivalence tests (TOST) for the "unchanged" claims
- **Action**: predeclare TOST margins — |Δu| < 0.15, |Δρ| < 0.25, |Δ|v|| < 0.20 —
  for every "steering unchanged" contrast (round-1 A–E arms). Contrast family
  enumerated; **Holm** correction across it; clustering unit = **task** (10 clusters)
  via **wild-cluster bootstrap** (percentile bootstrap undercovers at 10 clusters).
- **Impact**: "unchanged" becomes a positive equivalence result, not an eyeballed null.

### 4. Named 2nd policy, 2nd suite, and minimum replication criterion
- **Action**: **2nd suite = LIBERO-object** (already partially validated: base 68%
  clean; anchoring cos 0.97, TC success 3% in the pilot). **2nd policy = π0 (LoRA
  fine-tune)** if it fits 48 GB; **feasible fallback = Octo-Small (~27 M)** or a
  from-scratch **ACT** transformer — the ACT/Octo contrast additionally tests whether
  anchoring is VLM-specific or a general BC-under-narrow-placement phenomenon.
  **Minimum replication criterion**: the *basin-not-grounding dissociation* (Claim 1:
  displaced successes anchored, u high, on the base policy) must reproduce on the 2nd
  policy AND 2nd suite; the *cause-localization* (Claim 2) must reproduce on ≥1 of the
  two. **Scoping rule**: if a 2nd policy shows NO anchoring at all, that is reported as
  a boundary condition ("anchoring arises under narrow-placement BC with a strong
  canonical prior") and the claim is scoped to anchoring-exhibiting policies, not
  hidden.
- **Impact**: generality is concrete and falsifiable, with an honest failure rule.

## Revised Proposal (deltas from round 1; everything else unchanged)

### Method Thesis (unchanged wording, corrected estimand)
A cheap interventional displacement-normalized reach readout (u, ρ, |v|,
success-conditioned, at pregrasp/contact/lift) dissociates object-grounding from
basin-tolerance; against a clean-success-matched degradation null it shows
counterfactual-FT success gains are predominantly basin-widening, with partial
directional grounding (G = [u_fail−u_succ]_int − [u_fail−u_succ]_null > 0) appearing
only when synthetic data displaces the canonical demonstrations — localizing the
cause to gradient competition (a frozen-feature position-decodability probe rules out
a representational wall).

### Claim-Driven Validation Sketch (final)
- **Claim 1 — readout validity + dissociation**: base policy, similar-success/opposite-
  u regimes; success-conditioned split; replicate on 2nd policy + 2nd suite. Metric:
  u/ρ (success-cond), fraction within ρ≤0.75.
- **Claim 2 — gains are basin-widening; grounding partial + only by displacing
  canonical demos; cause = gradient competition**: factorial {fraction × canonical-
  presence-at-fixed-count × VLM plasticity}, ≥3 seeds, paired rollouts; **primary
  degradation null + 2 robustness nulls**, matched on clean+displaced success;
  statistic **G** with wild-cluster-bootstrap CI + the |v|-dispersion guard;
  decodability probe. TOST for all "unchanged" contrasts.
- **Validity checks (appendix)**: state-masked/noised arm; first-chunk intervention.

### Compute & Timeline
Factorial (~6 cells) + 3 nulls + decodability, × ≥3 seeds × 2 policies × 2 suites,
staged on the resumable orchestrator; ~3–4 weeks on 1 GPU (π0 LoRA is the cost driver;
Octo/ACT fallback is cheaper). 5 figures exist.
