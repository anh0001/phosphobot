# Novelty Dossier — P2T "Basin vs Grounding" reach readout

## Proposed contribution (evaluation/methodology paper, VLA spatial generalization)

**Primary**: an INTERVENTIONAL, displacement-normalized, SUCCESS-CONDITIONED **reach
readout** for VLA/imitation manipulation. Actively displace the target object at
episode start; roll out; measure WHERE the pre-grasp/contact/lift endpoint lands in a
displacement-normalized frame: u along −d (0 = object, 1 = canonical training
location), ρ = ‖endpoint − object‖/‖d‖; split by success vs failure. It DISSOCIATES
"reach steered by perceived object position (grounding)" from "reach anchored to the
canonical training location but succeeds because the grasp basin is wide
(basin-tolerance)".

**Claim 2 — evaluation blind spot**: success-rate-only spatial-generalization
evaluation cannot distinguish grounding from basin-tolerance, so reported
"generalization" gains can be basin-widening, not grounding; the readout exposes this.

**Claim 3 — cross-regime finding**: the SAME counterfactual-augmentation fine-tuning
produces basin-widening in wide-basin tasks (LIBERO-spatial: M2 u_succ 0.51, success up
via basin) but forced object-grounding in tight-basin tasks (LIBERO-object: M2 u_succ
0.21, disp 10→27%) — opposite mechanisms, invisible to success rate.

**Claim 4 — mechanistic (boundary/negative, scoped to wide-basin)**: "the anchoring
prior is defended by gradient competition with the canonical demos" (spatial M2−N1
+0.20; fails on object +0.03); a competence-matched degradation-null control with
statistic G = [u_fail−u_succ]_intervention − [..]_null.

## Candidates found (scite + arXiv, 2024–2026)

**Near-collisions (must differentiate):**
- **Affordance Field Intervention: Escape Memory Traps (arXiv 2512.07472)** — "memory
  trap" = VLA reproduces memorized trajectories instead of adapting to the updated
  scene = the SAME anchoring/canonical-prior PHENOMENON. BUT it PROPOSES A FIX
  (3D affordance field), gives NO interventional reach readout, no basin-vs-grounding
  dissociation, no eval-blindspot claim. Confirms the phenomenon is recognized; the
  readout is not theirs.
- **LIBERO-X: Robustness Litmus for VLA (arXiv 2602.06556)** — a robustness BENCHMARK;
  argues existing benchmarks give "misleading assessments due to insufficient
  evaluation protocols" (OVERLAPS the eval-blindspot thesis). BUT its instrument is
  hierarchical/progressive PERTURBATIONS + capability decomposition (spatial gen,
  object recog, instruction) measured by SUCCESS degradation — NOT a mechanistic reach
  readout of WHERE the arm goes, and no basin-vs-grounding separation.

**Phenomenon / prior lineage (cite):** Causal Confusion in Imitation (de Haan 2019,
1905.11979) + Object-Aware Regularization (2110.14118); "Robust Skills, Brittle
Grounding" (2602.24143, mislocalization diagnosis); "Shortcut Learning in Generalist
Robot Policies" (2508.06426, data-vs-model dichotomy). Attribution-standard genealogy:
Sanity Checks for Saliency Maps (Adebayo 2018). Fine-tuning geometry: LP-FT (Kumar
2022), DFR (Kirichenko 2023), WiSE-FT. Counterfactual data gen: MimicGen, DemoGen.
Spatial-tolerance/success-region: Mobi-π (2505.23692), N2M (2509.18671), ManiBox
(2411.01850). "PriorVLA: Prior-Preserving Adaptation" (2605.10925) — name overlaps
"prior" but is an adaptation method, not a diagnostic.

## Questions for the reviewer
1. Given "memory trap" (2512.07472) already names the anchoring phenomenon and
   LIBERO-X (2602.06556) already argues success-only eval is misleading, is the
   remaining novelty (the specific interventional displacement-normalized
   success-conditioned reach readout that DISSOCIATES basin-widening from grounding +
   the cross-regime finding) enough for a CoRL / NeurIPS D&B contribution, or is it
   "known phenomenon + another eval critique"?
2. What is the single strongest prior-work collision, and the exact defensible delta?
3. Is there an obvious paper we MISSED (a reach/endpoint/trajectory readout under
   object displacement; a basin-vs-grounding or tolerance-vs-grounding separation; a
   success-conditioned steering metric) — name it if so.
4. Overall novelty score /10 and positioning to maximize defensible novelty.
