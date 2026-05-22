# Phase 3 — Deep Novelty Check

## I1 — Conformal-Uncertainty Active Demo Query

**Closest prior work:**

| Paper | Key claim | What it does NOT do |
|-------|-----------|---------------------|
| **UPS** "When to Act, Ask, or Learn" (arXiv 2602.22474) | Conformal calibration over a VLM verifier choosing among {execute, ask, intervene} on VLAs | No sample-complexity study; no cheap real arm; treats interventions as cost, not as primary training data |
| **ConformalDAgger** (arXiv 2410.08852, ICLR'25) | IQT-based intermittent-label conformal prediction → active expert query, 7-DOF arm | BC-style policies, not VLAs; focuses on expert shift, not demo budgeting |
| **CRSAIL** (arXiv 2512.00453, Dec 2025) | kNN distance + conformal threshold → query control in active IL | MuJoCo only; BC; no real arm; no VLA |
| **Confidence Calibration in VLA** (arXiv 2507.17383) | Confidence calibration for VLA actions | Diagnostic, not query / data-efficiency focused |
| **ReconVLA** (arXiv 2604.16677) | CQR + SMD for runtime failure detection on VLA | Runtime safety, not demo collection |

**Novelty verdict: MEDIUM-HIGH** — building blocks exist but no work combines all of:
1. Modern VLA backbone (Pi0 / SmolVLA / ACT) — UPS does VLA but no efficiency study.
2. **Low-cost real 6-DOF arm** (Piper-scale, $4k) — none of these used it.
3. Conformal action uncertainty as a **demo-collection stopping criterion + targeting signal** (not runtime gating).
4. Multimodal teleop (leader arm) as the intervention modality.

**Required differentiation in framing:** "Conformal-uncertainty-driven *demo budgeting* for VLA fine-tuning on a low-cost arm — not gating, not safety; the metric of interest is `N_demos → success rate` and a learned stopping rule."

## I3 — HITL Correction-as-Data

**Closest prior work:**

| Paper | Key claim | Overlap |
|-------|-----------|---------|
| **Dual-Actor / Talk-and-Tweak** (arXiv 2509.13774) | HITL VLA refinement, 100 % success in 101 min | Direct competitor; uses language commands instead of raw teleop |
| **DexHiL** (arXiv 2603.09121) | HITL VLA post-training with lightweight teleop corrections | Direct competitor for dexterous setting |
| **Hi-ORS** (arXiv 2510.26406) | Online rejection sampling with HITL corrections, 1.5 h to fine-tune | Direct competitor |
| **HIL-SERL on SO-101** (Patil blog 2026) | 40 min on a $300 arm | Same hardware tier; RL though, not VLA fine-tuning |
| **ConRFT** (arXiv 2502.05450) | BC+Q offline → consistency-policy online + HITL, 96.3 % in 45–90 min | Direct competitor |

**Novelty verdict: LOW** — concept is saturated. Multiple 2025-2026 papers ship HITL VLA fine-tuning loops.

**Decision:** Merge I3 into I1 — use HITL teleop takeover as the **mechanism** for collecting demos that I1's uncertainty signal targeted. The novelty contribution is the *trigger* (calibrated conformal uncertainty), not the *correction loop* itself.

## I7 — Sim Pretrain + Active Query

**Closest prior work:**

| Paper | Key claim | Overlap |
|-------|-----------|---------|
| **SmolVLA** (arXiv 2506.01844) | Pretrains on ~23k community trajectories, beats ACT on SO-100/101 | Pretraining is part of the model release; PyBullet free-play not the source |
| **VLA-RFT** (arXiv 2510.00406) | World-model simulator → RL fine-tune in 400 steps | Isaac-grade sim, not PyBullet |
| **Grounding Sim-to-Real** (arXiv 2603.22876) | Empirical study of sim2real transfer for VLAs | Isaac / Mujoco focused |
| **UniVLA** (case study on Piper) | Cross-morphology learning, Piper as one embodiment | Different sim stack |

**Novelty verdict: LOW-MEDIUM** as a standalone idea — PyBullet-as-pretraining-source is the only differentiator and might not pay off given SmolVLA's existing pretraining.

**Decision:** Demote to **ablation axis** inside I1's proposal — "with vs. without PyBullet free-play warm-up" — rather than its own contribution.

## Consolidated Top Idea (carried into Phase 4)

**Working title:** *"Calibrated Demo Budgeting: Conformal-Uncertainty-Targeted Teleop for VLA Fine-Tuning on a Low-Cost 6-DOF Arm"*

**One-line pitch:** Tell the operator *which demonstration to collect next*, and *when to stop*, using calibrated conformal action uncertainty from a partially-fine-tuned VLA on a real Piper arm — and demonstrate a 2–5× reduction in real demos to reach 80 % task success vs. random demo collection.

**Differentiation matrix:**

| | VLA backbone | Real low-cost arm | Conformal calibration | Active demo *budget* (not gating) | Multimodal teleop |
|---|---|---|---|---|---|
| UPS | ✓ | ✗ | ✓ | ✗ (gating) | partial |
| ConformalDAgger | ✗ (BC) | partial (7-DOF) | ✓ | ✗ (expert shift) | ✗ |
| CRSAIL | ✗ (BC) | ✗ (MuJoCo) | ✓ | partial | ✗ |
| Talk-and-Tweak | ✓ | partial | ✗ | ✗ | ✓ (language) |
| Hi-ORS | ✓ | ✓ | ✗ | partial | ✓ |
| HIL-SERL SO-101 | ✗ (RL) | ✓ | ✗ | ✗ | ✓ |
| **Ours** | ✓ | ✓ (Piper) | ✓ | ✓ | ✓ (leader-arm) |

No prior work occupies the bottom row.
