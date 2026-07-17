# Paper Plan

**Title (working)**: *Basin or Grounding? A Success-Conditioned Reach Readout for
Vision-Language-Action Spatial Generalization*
**One-sentence contribution**: An interventional, success-conditioned reach readout
shows that the same counterfactual-fine-tuning success gain can reflect basin-widening
in one LIBERO suite (successful reaches stay off-object) and object-near reaching in
another (successful reaches move to the object) — a distinction success rate cannot make.

**Positioning guardrails (from novelty check, enforce throughout)**: the anchoring
phenomenon and the "success-only eval is imperfect" critique are CROWDED — they live in
Related Work, never in contributions. Novelty = the success-conditioned readout + the
cross-regime data. Avoid "forced grounding" (say "successful rollouts become
object-near in the tight-basin suite") and "all 5 strategies raise success" (say
"five conditions leave steering anchored; success changes are not diagnostic of steering").
**Venue**: NeurIPS Datasets & Benchmarks (primary; readout = reusable eval artifact) /
CoRL (alt). **Type**: empirical / diagnostic (evaluation methodology).
**Date**: 2026-07-17 · **Page budget**: 9 pages (main body to Conclusion; refs/appendix excluded).
**Section count**: 6.

## Claims-Evidence Matrix
| Claim | Evidence | Status | Section |
|-------|----------|--------|---------|
| **C1** The reach readout (u/ρ, success-conditioned, pregrasp/contact/lift) dissociates object-grounding from grasp-basin tolerance — successful displaced rollouts can still be anchored | Base SmolVLA, both suites: spatial u_succ high; object u_succ 0.67@50 despite 10% success; endpoint clouds (Fig4); contact/lift shows pregrasp anchored u 0.51 → contact 0.22 (anchored planning, late correction) | **Supported** | §3, §4.1 |
| **C2** Success-rate-only evaluation is not diagnostic of steering — five fine-tuning conditions leave steering anchored while success changes | Five conditions: steering proj≈1.0 regardless of success (Fig1, Fig2); TOST equivalence on steering | **Supported** | §4.2 |
| **C3** The SAME counterfactual-FT success gain reflects basin-widening in the wide-basin suite (successes stay off-object) vs object-near reaching in the tight-basin suite | Spatial M2 u_succ 0.51 (off-object) vs object M2 u_succ 0.21, disp 10→27%; dose-response (Fig5); + a basin-tightness diagnostic (base displaced-success vs magnitude per suite) | **Supported** | §4.1, §4.3 |
| ~~C4~~ (DEMOTED — NOT a contribution; §5 boundary analysis only) Mechanistic account (gradient competition / grounding-beyond-degradation) | Spatial M2−N1 +0.20 [−0.01,+0.46]; object M2−N1 +0.03; object G −0.12 — **the readout REJECTS these as general mechanisms** | **Boundary/negative — appendix stats** | §5 |

## Structure

### §0 Abstract (~200 words)
- **What**: a success-conditioned interventional reach readout separating basin-tolerance from object-grounding in VLA policies.
- **Why hard**: success rate conflates the two; a policy can succeed on a displaced object while reaching the canonical training location, because the grasp basin is wide.
- **How**: displace the object at reset; measure the displacement-normalized endpoint (u: 0=object,1=canonical; ρ), split by success/failure, at pregrasp/contact/lift.
- **Result**: across 2 suites + 5 augmentation strategies, success gains are basin-widening in wide-basin tasks but forced grounding in tight-basin tasks — the same intervention, opposite mechanism, hidden by success rate.
- **Artifact**: the readout is a drop-in, model-agnostic evaluation tool.

### §1 Introduction (1.5p)
- **Hook**: VLA policies report rising "spatial generalization," measured by task success. But a policy can succeed on a moved object without ever grounding to it.
- **Gap**: the anchoring/"memory-trap" phenomenon is known; that success-only eval is imperfect is argued. What's missing: a readout that asks, *on the SUCCESSES*, whether the reach was object-grounded or canonical-and-tolerated — and a demonstration that this distinction changes conclusions.
- **One-sentence contribution** (above).
- **Contributions** (3, C4 removed): (1) the success-conditioned displacement-normalized reach readout (released tool); (2) the basin-vs-grounding dissociation on competent VLAs across 2 suites; (3) the cross-regime finding (same success gain → basin-widening vs object-near reaching). (The mechanistic boundary analysis in §5 is framed as "the readout rejects tempting mechanisms," not a contribution.)
- **Results preview**: Fig 1 (hero, data-driven arrows) + the spatial-vs-object inversion.
- **Scope stated up front**: SmolVLA on two LIBERO suites (spatial, object); the readout is model-agnostic by construction, additional policy validation pending (limitations).
- **Hero figure (Fig 1)**: see Figure Plan.
- **Key citations**: de Haan 2019; Robust Skills Brittle Grounding 2602.24143; LIBERO-X 2602.06556; Adebayo 2018; MimicGen.

### §2 Related Work (1p, positioning-first)
- **Anchoring / shortcut phenomenon**: causal confusion (de Haan 2019; Object-Aware Reg 2110.14118), memory traps / Affordance Field Intervention (2512.07472), proprio shortcuts (Do You Need Proprioceptive States 2509.18644). *We do not claim the phenomenon; we measure it differently.*
- **Spatial-generalization evaluation**: LIBERO-X (2602.06556), Robust Skills Brittle Grounding (2602.24143), Mobi-π/N2M/ManiBox. *These measure success-degradation or success-regions under perturbation; none measure, on the successes, whether the reach moved with the object vs stayed canonical.*
- **Attribution-as-standard**: sanity-checks-for-saliency (Adebayo 2018) — genealogy of interventional acceptance tests.
- **Counterfactual data gen & fine-tuning geometry**: MimicGen/DemoGen (instrument, not novel); LP-FT (Kumar 2022)/DFR (Kirichenko 2023)/WiSE-FT (the gradient-competition lineage, §5).

### §3 The Reach Readout (method/setup, 1.5p)
- Setup: SmolVLA, LIBERO-spatial & -object; displacement grid (mag × dir) at reset.
- Definitions: e = endpoint − object; u = ⟨e,−d̂⟩/‖d‖ (0=object,1=canonical); v ⊥; ρ = ‖e‖/‖d‖; endpoints at pregrasp / contact / lift; success-conditioned split.
- Why success-conditioned: the novel question is about *successful* rollouts.
- Competence-matched degradation null + statistic G (for §5).
- Reusability: model-agnostic (any policy exposing rollouts), suite-agnostic (E_SUITE), released as a tool.

### §4 Experiments (3p)
- **§4.1 The dissociation exists (C1)**: base policies, both suites — displaced successes anchored; endpoint clouds (Fig4); contact/lift decomposition (anchored planning + late correction).
- **§4.2 Success hides it (C2)**: 5 acquisition strategies raise success, steering unmoved (Fig1 success-by-condition, Fig2 mechanism success-vs-steering, Fig3 synthesis distributions); TOST equivalence on steering.
- **§4.3 Cross-regime inversion (C3)**: spatial (basin-widening) vs object (forced grounding); dose-response (Fig5); the same intervention, opposite mechanism.

### §5 Analysis / Mechanism & Boundaries (1p)
- Gradient-competition hypothesis (defended-prior); competence-matched null; **honest scope**: holds (near-sig) on wide-basin spatial, fails on tight-basin object; grounding-beyond-degradation is boundary/negative. Interpretation: the readout is robust; the mechanism is regime-dependent. ACT 2nd-policy integration failure noted as limitation/future work.

### §6 Conclusion (0.5p)
- Restate the readout + cross-regime finding; limitations (1 model family, sim, single-GPU seeds; 2nd policy pending); future work (more policies via the released tool, real-robot, contact-time metrics).

## Figure Plan
| ID | Type | Description | Data source | Priority |
|----|------|-------------|-------------|----------|
| **Fig 1 (HERO)** | schematic + DATA panel | LEFT (schematic): the readout — object displaced by d, endpoint, the u-axis (object=0 … canonical=1); a grounded reach (u→0) vs an anchored-but-succeeds reach (u→1) in a wide grasp basin. RIGHT (DATA, not concept): two arrows in (displaced-success, u_succ) space — spatial base→M2 (success 31→39%, u_succ 1.06→0.51, ρ_succ≈1.27, stays off-object) and object base→M2 (10→27%, u_succ 0.67→0.21, becomes object-near). Top strip "success improves in both"; readout strip "spatial stays off-object, object becomes object-near." NO "identical curves" claim. | manual + eval jsonl | HIGH |
| Fig 2 (MAIN) | endpoint clouds | displacement-normalized u/v, success/fail, base/M2 both suites + contact/lift panels | fig4_endpoint_clouds | HIGH |
| **Fig 3 (MAIN, NEW)** | line — basin diagnostic | base displaced-success vs displacement magnitude, per suite (spatial degrades slowly = wide basin; object collapses = tight basin) — makes "tight basin" a MEASURED claim, not post-hoc | reach_field jsonl | HIGH |
| Table 1 (MAIN) | comparison | per-suite per-arm: clean, disp, u_succ, ρ_succ, u_gap (spatial + object) | eval_*.jsonl | HIGH |
| **Table 2 (MAIN, surgical 3 rows)** | related-work delta | Robust-Skills-Brittle-Grounding / LIBERO-X / AFI × columns {Perturbs object? · Measures reach endpoint? · Conditioned on successes? · Separates basin-tolerance from grounding?}; ours = only "yes" on last two | manual (from P2T_NOVELTY.md) | HIGH |
| Fig A1–A4 (APPENDIX) | — | success-by-condition bars (fig1), success-vs-steering scatter (fig2), synthesis histograms (fig3), full dose-response (fig5), degradation-null/G/bootstrap stats, ACT integration note | fig1/2/3/5 | LOW |

## Citation Plan
- §1: de Haan 2019 [1905.11979]; Robust Skills Brittle Grounding [2602.24143]; LIBERO-X [2602.06556]; Adebayo 2018 [1810.03292 VERIFY]; MimicGen [2310.17596 VERIFY].
- §2: + Affordance Field Intervention [2512.07472]; Do You Need Proprioceptive States [2509.18644]; Object-Aware Reg [2110.14118]; Mobi-π [2505.23692]; N2M [2509.18671]; ManiBox [2411.01850]; DemoGen [VERIFY]; LP-FT Kumar 2022 [VERIFY]; DFR Kirichenko 2023 [VERIFY]; WiSE-FT [VERIFY]; LIBERO benchmark [VERIFY]; SmolVLA [VERIFY].
- §3: SmolVLA; LIBERO; relative-action equivariance (Wang 2505.13431 [VERIFY]).
- **All [VERIFY] resolved via citation-audit before submission — never from memory.**

## Reviewer Feedback (Codex gpt-5.5 xhigh — 6.8/10, REVISE; applied above)
- **Story**: lead §4 with the cross-regime inversion; sticky-prior = supporting, not spine. ✅ applied (C3 now spans §4.1/4.3).
- **Claim-evidence**: C2 reworded ("not all 5 raise success"); C3 detoned ("object-near" not "forced grounding"). ✅
- **C4 demoted**: removed from abstract + contributions; §5 = "boundary analysis: the readout rejects tempting mechanisms," full G/null stats to appendix. ✅
- **Hero Fig 1**: right panel now DATA (arrows in (success, u_succ) space with real numbers); dropped "identical curves." ✅
- **Table 2 surgical**: 3 rows (Robust-Skills/LIBERO-X/AFI), 4 yes/no columns. ✅
- **Basin-tightness diagnostic added** (Fig 3, MAIN): base displaced-success vs magnitude per suite — so "tight basin" is measured. ✅
- **Page budget**: main body = Fig1 hero + Fig2 endpoint + Fig3 basin + Table1 + Table2; everything else appendix. ✅
- **OPEN RISK (biggest, D&B)**: single policy family (SmolVLA). Cheapest fix = ONE second-policy base-only readout smoke (u_succ/ρ_succ/success, 1 suite, 50mm). ACT integration failed → do NOT use ACT; try a lerobot-native policy that loads cleanly (e.g. a diffusion policy or pi0 if it fits) for a base-only compatibility datapoint, OR scope explicitly as "a SmolVLA/LIBERO diagnostic study" (safer for CoRL than NeurIPS D&B). Decision needed before submission.

## Next Steps
- [ ] **Decide venue framing** given the single-policy risk: CoRL "SmolVLA/LIBERO diagnostic study" (safe) vs NeurIPS D&B "reusable artifact" (needs 1 more policy datapoint)
- [ ] /paper-figure — build Fig 1 hero + Fig 3 basin diagnostic (Fig 2 endpoint exists)
- [ ] /paper-write — draft LaTeX (main body tight per cut plan)
- [ ] /citation-audit — resolve all [VERIFY]
- [ ] /paper-compile — build PDF
