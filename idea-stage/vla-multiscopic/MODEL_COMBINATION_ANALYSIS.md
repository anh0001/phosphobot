# Model Combination Analysis — VLA × Multi-Scopic, Re-attacked Post-Evidence

**Date**: 2026-06-10 · **Method**: thinking-model-combination skill (Innovation recipe → Adversarial/Bayesian) ·
**Cross-model**: GPT-5.5 (adversary) · **Prior art**: scite (210M corpus).

## Problem Characterization
Re-attack the original goal — a *clever, fundamental-method* fusion of SmolVLA with the multi-scopic paper's
**indigenous** primitives, LIBERO-verifiable — but now anchored on what our experiments proved: the failure
bottleneck is **object-pose distribution shift / perceptual brittleness**, NOT control timing (P5 preemption is
dead: E1 + crux control). Single model won't crack this; we need a lens that finds an angle that neither the
(dead) timing branch nor the (crowded) perception branch occupies.

## Combination Pattern
**Sequential (Innovation recipe) → Adversarial + Bayesian.** First-Principles/TRIZ/Via-Negativa *generate* and
*prune*; a hostile cross-model adversary + Bayesian estimate *validate*. Max 4 roles, explicit tiebreaker =
Bayesian EV.

| Model | Role | What it addresses |
|---|---|---|
| First Principles | strip to fundamentals (why it breaks; what the paper uniquely offers) | avoids re-using dead assumptions |
| TRIZ | name the contradiction the fusion must resolve | forces a non-obvious mechanism |
| Via Negativa | remove dead ends + already-claimed space | enforces novelty + feasibility |
| Adversarial (Codex) + Bayesian | hostile novelty/feasibility estimate, tiebreaker | prevents motivated reasoning |

## Analysis

### First Principles
- SmolVLA fails under pose shift because its visual representation **entangles task-relevant geometry with
  appearance/position nuisance** (LIBERO-Plus 2510.13626: 95%→<30%; our crux: 1/14 from scratch).
- Object-centric/slot representations *help* (STORM 2601.20381; Spotlighting 2601.21416) — but use **fixed-K,
  semantic, trained** slots.
- The multi-scopic paper's **un-transferred** primitives: feedback-driven **adaptive representational density**
  (DD-GNG δ), **embodiment-relative affordance** (P4), **topological/embodiment-aware map** (P6/P7).
- **Key insight:** the proven bottleneck is representational, so any real method must change the *representation*
  (not the controller) — which is precisely where the field has already converged in the last ~4 months.

### TRIZ
- **Contradiction:** need a structured, pose/nuisance-robust, *action-relevant* representation (robustness)
  **without** retraining the backbone or hand-engineering perception (keep VLA generality, ≤1 day GPU).
- Inventive principles: #3 Local Quality (fidelity only where action happens), #25 Self-service (use the model's
  own signals to gate representation), #1 Segmentation.
- **Candidate X (generated):** *uncertainty-adaptive, embodiment-relative representational density* — DD-GNG-style
  adaptive-K slots gated by the policy's own flow-head dispersion + action-relevance, on a frozen backbone.
- **Key insight:** the only non-occupied dimension is *adaptive density + self-uncertainty gate*; everything else
  is taken.

### Via Negativa (what we remove)
Dead/claimed → out: timing/preemption (killed); OOD-by-construction perturbations; object-pose-fragility claims
(LIBERO-Plus); fixed-K object-centric slots (STORM/Spotlighting); spatial-prior training (ST4VLA); affordance-
keypose conditioning (RT-Affordance); anything needing backbone retrain or >1 day GPU.
**Key insight:** Via Negativa eats almost the entire method space — what remains (Candidate X) is a *thin sliver*
adjacent to STORM, not a clearing.

### Adversarial + Bayesian (Codex, hostile)
- Candidate X = "STORM + adaptive-K + uncertainty gate" → **incremental**. P(novel-enough for top workshop) ≈
  **0.35**; P(positive on SmolVLA/LIBERO ≤1 day, no retrain) ≈ **0.15**. Flow dispersion is a *weak spatial
  allocator* (frame/global, stochastic); training-free variable-density slots unlikely to fix pose-OOD.
- No novel + feasible + likely-positive alternative recombination exists under the compute budget.
- The only other concrete option is a **privileged pose-token probe** (inject true object-pose-relative-to-eef as
  a conditioning token, LoRA the projection only) — but that's a **mechanistic control**, not a deployable method
  (too privileged; ST4VLA/RT-Affordance already gesture there).

## Synthesis

### Convergence (all lenses + cross-model agree)
There is **no novel, feasible, likely-positive VLA × multi-scopic *method*** on SmolVLA/LIBERO within ≤1 day GPU.
Both candidate branches are occupied: timing (we killed it) and perception-robustness (STORM/Spotlighting/ST4VLA/
RT-Affordance, all very recent). The genuinely defensible, *indigenous* output is the **mechanistic finding** our
own experiments produced.

### Divergence
Candidate X carries *some* novelty (adaptive-density DD-GNG-for-VLA) but fails on feasibility/payoff. Resolution
via the tiebreaker (Bayesian EV): EV(diagnostic) > EV(stop) ≳ EV(Candidate X) > EV(probe-as-method).

### Unique contributions
| Model | Unique insight |
|---|---|
| First Principles | the bottleneck is representational → method must touch representation, where the field already crowded |
| TRIZ | the only open axis is adaptive-density + self-uncertainty gate (thin) |
| Via Negativa | the dead+claimed set covers nearly the whole method space |
| Adversarial/Bayesian | quantifies X as low-EV; surfaces the privileged-pose probe as the keystone *evidence* (not a method) |

## Combined Conclusion
Solving the problem *indigenously* = recognizing, via the latticework, that the problem **as posed** (a novel
positive VLA×multi-scopic method on LIBERO in ≤1 day) has **no good solution** — and that the high-value move is
to convert the rigorous negative into evidence for the user's **existing "Unreliable by Default" reliability
paper**:

> **SmolVLA's failure under mid-rollout object displacement is a *representation* failure, not a *timing*
> failure: even solving from scratch after the same 5 cm pose shift collapses (1/14), so multi-rate control /
> preemption cannot repair a missing object-pose-generalization capability.**

**One concrete cheap experiment** that would make this airtight (the keystone control): the **privileged
pose-token probe** — feed true target-object-pose-relative-to-eef into SmolVLA's suffix/prefix, LoRA only the new
projection (short run, no backbone retrain), and show pose-OOD collapse *disappears*. That proves the missing
variable is object pose (representation), closing the argument. It is a control, not a product — keep it internal/
as a paragraph, not a standalone paper.

## Confidence
**High** that no ≤1-day novel positive method exists here (two independent lenses + cross-model + fresh prior art).
**Medium** that the privileged pose-token probe cleanly recovers success (it's the obvious next evidence; ~0.6).
What would change my mind: a *non-pose* indigenous primitive (e.g., genuine topological embodiment-map state)
shown feasible without backbone retrain — not found.

## Recommendation
1. **Stop opening new method directions** (confirmed twice, now via structured + adversarial analysis).
2. If the reliability paper wants it: run the **privileged pose-token probe** (~half day) as the keystone control
   + power the perturbed_start δ-sweep to n≥60 → one clean figure/paragraph for *Unreliable by Default*.
3. Otherwise: return to the active-learning project.
