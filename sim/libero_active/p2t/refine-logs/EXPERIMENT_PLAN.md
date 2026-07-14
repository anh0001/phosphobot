# Experiment Plan — Basin, Not Grounding

**Problem**: success-only VLA spatial-generalization evaluation cannot distinguish
object-grounding from grasp-basin tolerance.
**Method Thesis**: an interventional, displacement-normalized, success-conditioned
reach readout (u/ρ/|v| at pregrasp/contact/lift) dissociates the two; against a
competence-matched degradation null it shows counterfactual-FT gains are mostly
basin-widening, grounding is partial and only appears as synthetic data displaces the
canonical demos → cause = gradient competition, not a frozen-feature limit.
**Date**: 2026-07-15 · **Governed by** the pre-registered Minimum Publishable Matrix in
`FINAL_PROPOSAL.md`.

## Claim Map
| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| **C1** readout validity + basin/grounding dissociation | success rate is the field's metric; if it conflates basin and grounding, results are misreported | on the base policy, displaced **successes are anchored** (u≈0.8–1.0, ρ high) despite non-trivial success; holds on 2 policies + 2 suites; success rate alone cannot show it | B1, B5 |
| **C2** FT gains are basin-widening; grounding partial + only by displacing canonical demos; cause = gradient competition | tells the field WHY augmentation "works" and what actually moves grounding | **G>0** vs the strongest competence-matched degradation null (no excess |v|); u shifts toward object monotone in canonical-fraction removal; position decodable from frozen features but ignored; unfreezing adds little | B2, B3, B4 |
| **Anti-claim to rule out** | — | "lower u = a degraded/dispersed policy, not grounding" (killed by B2's matched null + G + |v| guard); "grounding gain is just more synthetic data" (killed by B3's fixed-count decoupling) | B2, B3 |

## Paper Storyline
- **Main paper must prove**: C1 (the readout + dissociation) and C2 (basin-not-grounding
  + matched-null causal statistic + cause localization).
- **Appendix can support**: proprio-shortcut checks (state-mask, first-chunk intervention);
  robustness nulls; extended factorial.
- **Intentionally cut**: the P2T "gain-targeted acquisition function" claim — appears only
  as one motivating figure (success rises, steering doesn't, across all 5 strategies).

## Experiment Blocks

### Block B1 — Readout validity + basin/grounding dissociation (C1) · MUST-RUN
- **Why**: the foundational result; makes the readout necessary.
- **Task/data**: SmolVLA base × LIBERO-spatial, displacement grid (4 mag × 4 dir), the
  existing 15 clean-success pairs; **reuse** e1r reach-field logs.
- **Compared**: success rate (blind) vs reach readout (sighted); successes vs failures.
- **Metrics**: u, ρ, |v| (success-conditioned) at **pregrasp, contact, lift**; fraction
  within ρ≤0.75. Decisive: u_succ high while success non-trivial.
- **Build**: contact/lift-time endpoints — **extract OFFLINE** from existing
  `reach_field_logs*` npz (they log `obj_xyz` + `eef_xyz` per step): contact = first step
  gripper-object xy-dist < ε AND gripper closing; lift = first step object z − table > δ.
- **Success criterion**: at 50 mm, u_succ ≈ 0.8–1.0 with ≥30% success (anchored yet
  succeeding); contact/lift endpoints corroborate.
- **Failure interpretation**: if successes already track (u_succ≈0), the dissociation is
  absent → C1 weakened; report honestly.
- **Figure**: fig4 (endpoint clouds) extended with contact/lift panels + the success table.

### Block B2 — Competence-matched degradation null + statistic G (C2 validity) · MUST-RUN
- **Why**: THE decisive control — separates grounding from "worse policy, shifted survivors."
- **Compared**: intervention = M2 (100%-synthetic) vs **primary null** = action-noise FT
  from base, noise scale tuned so clean (±3pp of 52%) AND displaced@50 (±3pp of 37%) match
  M2; **robustness nulls** = label-noise FT + early-stopped FT, matched the same way.
- **Metric/stat**: **G = ([u_fail − u_succ]_M2) − ([u_fail − u_succ]_null)**, wild-cluster
  bootstrap CI (task = cluster, 10 clusters) excluding 0; **|v|-dispersion guard** (M2 |v|
  IQR ≤ null |v| IQR + tol). Report G vs ALL three nulls; must beat the strongest.
- **Build**: (a) noise-scale search harness (bisect noise to hit target success); (b) stats
  module: wild-cluster bootstrap + G + |v| guard (extend `analyze_conditions.py`).
- **Success criterion**: G > 0 vs the strongest null with CI excluding 0 and no larger |v|.
- **Failure interpretation**: G ≤ 0 or via |v| inflation → "grounding" is degradation-driven
  → C2's grounding sub-claim falls; the basin-widening + success-only-blindspot claims survive.
- **Table**: Table (main) — G and CI per null; the paper's key validity result.

### Block B3 — Mechanism factorial with fraction-vs-count decoupling (C2) · MUST-RUN
- **Why**: shows gains are basin-widening and grounding emerges only by displacing canonical
  demos, controlling for synthetic COUNT.
- **Compared** (SmolVLA×spatial, frozen): {base, 17% (reuse A/round-1), 50% (reuse M1), 100%
  (reuse M2)} + the **new critical cell N1 = 100%-synthetic + equal-count (352) canonical**
  (decouples fraction from count) + {unfrozen 100% = reuse M3}.
- **Seeds**: **3 seeds** on the 3 claim-critical cells {base, 100%-syn (M2), N1}; others 1 seed.
- **Metric/stat**: Δu (success-cond) with wild-cluster bootstrap; **TOST** for the "unchanged"
  round-1 A–E contrasts (margins |Δu|<0.15, |Δρ|<0.25, |Δ|v||<0.20, Holm over the family).
- **Build**: N1 dataset (reuse `make_mixed_dataset.py` — add 352 originals to the 352-synthetic
  pool); 3-seed reruns of base/M2/N1.
- **Success criterion**: u(N1) ≈ u(M2 at 50% mix) ≫ u(100%-pure) — i.e. adding canonical back
  at fixed synthetic count re-anchors → count is not the driver, canonical presence is.
- **Figure**: fig5 (dose-response) + a fraction-vs-count panel.

### Block B4 — Cause localization: frozen-feature decodability (C2 support) · MUST-RUN
- **Why**: distinguishes "features lack position" (representation wall) from "expert ignores
  available position" (gradient competition).
- **Compared**: linear/shallow probe of frozen SmolVLM features (and the action-expert's
  conditioning inputs) for displaced object xy, across the reach grid; base vs M2 vs M3.
- **Metric**: probe R²/MAE for object xy; contrast with u (ignored despite decodable).
- **Build**: feature-dump hook at the expert interface + a ridge/MLP probe (CPU-cheap).
- **Success criterion**: object xy decodable (R² high) from frozen features while base u≈1 →
  position available but ignored → gradient-competition supported, representation wall refuted.
- **Failure interpretation**: not decodable → the features are the wall (H3 revived); report.
- **Figure**: probe-R² vs u scatter.

### Block B5 — Generality: 2nd policy + 2nd suite (C1, C2 scoped) · MUST-RUN
- **Why**: the min-replication criterion; tests VLM-specificity.
- **Compared**: **C1** replicated on **SmolVLA×LIBERO-object** AND **Octo-Small×LIBERO-spatial**;
  **C2** critical cells (base, 100%-syn, N1, primary null) on **≥1** of the two.
- **Seeds**: 1 seed for C1 replication; 3 seeds on the C2 setting chosen.
- **Build**: **Octo-Small integration** (biggest new eng: pipeline builder + FT + reach field
  for Octo); LIBERO-object synthesis (reuse engine, swap suite) — base object already validated
  (68% clean).
- **Success criterion**: C1 dissociation reproduces on both; C2 reproduces on ≥1 (scope
  explicitly if only one). Octo (no VLM) also anchoring → phenomenon is general BC, not
  VLM-specific (stronger); Octo NOT anchoring → boundary condition, reported.
- **Table**: generality table (u_succ, G) across policy×suite.

### Block B6 — Proprio-shortcut checks (appendix) · NICE-TO-HAVE
- state-masked/noised FT arm (do servo labels teach a proprio copycat?) + first-chunk
  intervention (teacher-force k steps toward object; continue vs snap back). Appendix unless
  they overturn the main interpretation.

### Block B7 — Extended matrix (appendix) · NICE-TO-HAVE
- full 6-cell factorial × 3 seeds × both policies × both suites; π0 LoRA stretch arm.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision gate | Cost | Risk |
|---|---|---|---|---|---|
| **M0 sanity** | contact/lift endpoints offline from existing npz; stats module (G, wild-cluster bootstrap, TOST) reproduces round-1/2 numbers | R001–R003 | endpoints sane (contact after pregrasp, before lift); stats re-derive known u/proj | ~0 GPU (offline+CPU) | endpoint ε/δ thresholds — calibrate on 5 clean rollouts |
| **M1 build** | degradation-null noise-scale search (match M2 52/37%); Octo-Small integration smoke | R010–R014 | a null matches M2 competence within ±3pp; Octo trains + reach field runs | ~15 GPU-h | Octo API mismatch → fall back to ACT (declared) |
| **M2 decisive** | B2 (3 nulls) + B3 critical cell N1 + 3-seed base/M2/N1 | R020–R035 | **G>0 vs strongest null** (go); else pivot C2 to basin-only | ~45 GPU-h | G≤0 → grounding sub-claim falls (paper still stands on C1 + basin-widening) |
| **M3 generality** | B5: Octo×spatial + SmolVLA×object readouts + C2 on chosen setting; B4 decodability | R040–R052 | C1 reproduces on both; C2 on ≥1 | ~35 GPU-h | 2nd setting fails → scope claim per rule |
| **M4 polish** | B6 appendix checks; B7 extended if time | R060+ | — | ~20 GPU-h | non-blocking |

## Compute and Data Budget
- **Total (MPM, M0–M3)**: ≈ **95 GPU-h** ≈ 2 weeks at realistic duty on 1× RTX 6000 Ada.
- **Data prep**: N1 dataset (reuse packer); LIBERO-object synthesis (reuse engine); no annotation.
- **Biggest bottleneck**: Octo-Small integration (new policy in the harness) and the 3-seed
  reruns; both parallelize poorly on 1 GPU → stage on the resumable orchestrator.

## Risks and Mitigations
- **G ≤ 0 (grounding is degradation)** → the paper's spine (C1 + "gains are basin-widening,
  invisible to success-only eval") survives; C2 becomes "no measurable grounding recovery",
  still a clean result. Pre-committed framing.
- **Octo integration slips** → ACT (from-scratch) declared fallback; both test VLM-specificity.
- **10 clusters undercover** → wild-cluster bootstrap (already specified), report CI width.
- **Contact/lift endpoint noise** → calibrate ε/δ on clean rollouts; report all 3 endpoints.

## Final Checklist
- [x] Main tables covered (B1 dissociation, B2 G-null, B3 factorial, B5 generality)
- [x] Novelty isolated (readout necessity in B1; G-null in B2)
- [x] Simplicity defended (acquisition demoted to motivation; one factorial)
- [x] Frontier justified (VLM = object of study; decodability probe B4; synthesis = instrument)
- [x] Must-run (M0–M3) separated from nice-to-have (B6/B7, M4)
