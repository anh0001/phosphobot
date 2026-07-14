# Research Proposal: Basin, Not Grounding — An Interventional Reach Readout for VLA Spatial Generalization

**Status: READY (9.1/10, gpt-5.5 xhigh, 4 rounds).**

## Problem Anchor
- **Bottom-line problem**: Reported VLA "spatial generalization" is measured by
  success rate, which cannot distinguish a policy that grounds+reaches the displaced
  object from one that still runs a canonical (memorized) motor program whose grasp
  basin is wide enough to occasionally succeed.
- **Must-solve bottleneck**: No cheap, model-agnostic, interventional readout
  dissociates "reach steered by perceived object position" from "reach anchored to
  the training location, succeeds via basin tolerance."
- **Non-goals**: not a new data-generation/acquisition method; not solving spatial
  generalization; not a new benchmark.
- **Constraints**: 1× RTX 6000 Ada, sim-first (LIBERO), SmolVLA primary + Octo-Small
  2nd; CoRL / NeurIPS D&B bar; honesty over hype.
- **Success condition**: reviewer agrees the readout is valid+cheap+interventional;
  on ≥2 policies/≥2 suites, counterfactual-FT gains are predominantly basin-widening
  with only partial directional grounding; the cause is localized to gradient
  competition, not a frozen-feature limit.

## Thesis (one line)
An interventional, displacement-normalized, success-conditioned reach readout
(u along −d: 0=object, 1=canonical; ρ=‖e‖/‖d‖; |v|=orthogonal spread; at
pregrasp/contact/lift) dissociates object-grounding from grasp-basin tolerance in
VLA policies; validated against a competence-matched degradation null (directional
gain **G = [u_fail−u_succ]_int − [u_fail−u_succ]_null > 0**, no excess dispersion),
it shows counterfactual-fine-tuning success gains are predominantly basin-widening,
with only partial directional grounding that emerges solely as synthetic data
displaces the canonical demonstrations — localizing the cause to gradient
competition, not a representational wall (a frozen-feature position-decodability
probe confirms position is available but ignored).

## Contribution Focus
- **Dominant**: the readout (validated against a matched-degradation null) + the
  basin-not-grounding finding.
- **Supporting**: cause localization (gradient competition) via one factorial +
  a frozen-feature decodability probe.
- **Non-contributions** (explicit): gain-targeted acquisition (motivation only —
  it did not beat trivial failure-targeting); new benchmark; new method.

## Method
- **Readout**: reset with the task object displaced (grid of magnitudes × directions);
  roll out; log endpoints at pregrasp, gripper-object contact, and object-lift onset;
  compute u, v, ρ in the displacement-normalized frame; split by success/failure. New
  primitive = the normalized coordinate + success-conditioned split. No trained part.
- **Mechanism factorial** (SmolVLA, action-expert FT, 20k steps): {counterfactual
  fraction 17/50/100%} × {canonical demos present vs absent at FIXED synthetic count}
  × {VLM frozen vs unfrozen}, paired rollouts (identical seed/task/magnitude).
- **Degradation null** (the key validity control): base policy degraded by
  action-noise FT tuned so BOTH clean and displaced success match the intervention arm
  (±3 pp, per task/magnitude); robustness nulls = label-noise FT + early-stopped FT.
  Grounding claimed iff **G > 0** with wild-cluster-bootstrap CI excluding 0 AND
  |v|-dispersion not larger than the null; must beat the STRONGEST matched null;
  report G against all three.
- **Statistics**: wild-cluster bootstrap (task = cluster, 10 clusters); TOST
  equivalence for every "unchanged" contrast (margins |Δu|<0.15, |Δρ|<0.25,
  |Δ|v||<0.20), Holm-corrected over the enumerated family.
- **Decodability probe**: linear/shallow probe of frozen SmolVLM features (and the
  expert's conditioning inputs) for displaced object xy — "features lack position" vs
  "expert ignores position."
- **Validity checks (appendix)**: state-masked/noised arm (proprio-shortcut);
  first-chunk intervention (continue-vs-snap-back).

## Claim-Driven Validation
- **Claim 1 — readout validity + basin/grounding dissociation**: base policy shows
  similar-success/opposite-u regimes; displaced successes are anchored (u≈0.8–1.0) at
  ~31% success. REPLICATED on SmolVLA×LIBERO-object AND Octo-Small×LIBERO-spatial.
- **Claim 2 — gains are basin-widening; grounding partial + only by displacing
  canonical demos; cause = gradient competition**: factorial + matched null + G +
  decodability; ≥3 seeds on claim-critical cells. Expected: success rises across arms;
  u shifts toward object only as canonical fraction drops (monotone); G>0 vs the
  strongest null with no excess dispersion; position decodable from frozen features;
  unfreezing adds little.

## Minimum Publishable Matrix (pre-registered — scope fixed before any result)
- **Claim 1**: SmolVLA×LIBERO-spatial + SmolVLA×LIBERO-object + Octo-Small×LIBERO-spatial.
- **Claim 2** (SmolVLA×LIBERO-spatial): 3 claim-critical factorial cells {base,
  100%-syn, 100%-syn+equal-count-canonical} × frozen-VLM at **3 seeds**; other cells
  1 seed; primary action-noise null at 3 seeds + 2 robustness nulls at 1 seed;
  decodability probe 1 pass. Claim 2 must reproduce on **≥1** of {LIBERO-object,
  Octo-Small}; if only one, scope explicitly.
- **Extended (not required)**: full 6-cell factorial × 3 seeds × both policies × both
  suites; π0 LoRA; appendix probes.

## Compute & Timeline
~2–3 weeks on 1 GPU for the MPM (Octo-Small is cheap; SmolVLA fine-tunes ~3 GPU-h;
readouts ~1–2 GPU-h) on the existing resumable orchestrator. 5 figures already exist.

## Closest Work / Novelty
Causal confusion in imitation (de Haan 2019) names the shortcut but gives no
manipulation reach readout; MimicGen (Mandlekar 2023)/DemoGen generate data but
evaluate on success; Mobi-π/N2M/ManiBox measure spatial tolerance as success-regions,
not an interventional grounding-vs-basin dissociation; LP-FT (Kumar 2022)/DFR
(Kirichenko 2023)/WiSE-FT explain fine-tuning geometry in classification, not embodied
reach; sanity-checks-for-saliency (Adebayo 2018) is the attribution-standard genealogy.
Irreducible new thing: **a cheap interventional reach readout that separates basin from
grounding, and the evidence that success-only VLA evaluation systematically
over-credits grounding.**
