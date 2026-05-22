# Final Proposal — Calibrated Demo Budgeting for VLA Fine-Tuning on a Low-Cost Piper Arm

**Date**: 2026-05-22 (rev. 2026-05-23: sim-first workflow + simulator choice)
**Author**: anh0001 (2× AgileX Piper for leader-follower teleop, phosphobot stack, consumer-GPU lab)

## Publication Strategy: Path A (sim-main)

This project follows **Path A**: main quantitative results in simulation, real Piper as
cross-embodiment transfer evidence. The contribution is reframed accordingly.

- **Contribution**: "sample-efficient active demonstration selection for VLAs" — NOT human-effort
  reduction. (In sim the expert is a costless oracle; a sim-main paper cannot honestly claim it
  saves human effort. Matching the claim to the evidence removes a near-automatic reject.)
- **Stage A — main results, LIBERO** (robosuite/MuJoCo, Franka). A recognized benchmark is
  required because sim now carries the paper; a custom task would read as "authors built the
  world where their method wins." Secondary generalization check on ManiSkill3 *built-in* tasks.
- **Stage B — real 2-Piper setup**: a small, clean experiment showing the conformal-beats-random
  trend survives on real hardware and a different embodiment (Franka-sim → Piper-real is an
  accepted cross-embodiment sanity check, not a weakness).
- **Target venues**: ICRA / IROS / RA-L / CoRL workshop.
- A future **Path B** paper (real-arm-main, human-effort claim, CoRL/RSS-tier) is a separate
  follow-up — not this submission.

Conformal thresholds do not transfer sim→real (visual domain gap breaks exchangeability):
recalibrate on real data per task/camera, split by episode, and recalibrate after every LoRA
update. See `EXPERIMENT_PLAN.md` Phase S for the LIBERO pipeline and sim exit gate.

## Problem Anchor (frozen)

**The Problem.** For a researcher with a single low-cost 6-DOF manipulator (Piper, SO-100/101, Koch), the central practical question when fine-tuning a modern VLA (SmolVLA, ACT, Pi0) on a new task is:

> *How many real teleop demonstrations do I need, and which ones should I collect, to reach an acceptable success rate?*

Existing literature gives qualitative guidance and shows HITL loops work, but no published work delivers a calibrated **stop-and-target** rule on a low-cost real arm with a modern VLA. Random demo collection is the de-facto baseline; everyone over-collects.

**The Anchor.** This proposal does *not* aim to invent a new VLA architecture, a new fine-tuning algorithm, or a new uncertainty estimator. It composes existing ingredients into a benchmarked, open-source pipeline that answers the question above with sample-complexity curves and a learned stopping rule.

## Method Thesis (one sentence)

Conformal action uncertainty from a partially-fine-tuned SmolVLA, deployed in a phosphobot rollout-with-teleop-takeover loop, lets a single operator collect 2–4× fewer real demonstrations to reach 80 % task success than random demo selection, and predicts a near-optimal stopping point on held-out tasks.

## Dominant Contribution

A **calibrated stopping-and-targeting rule** for real-world VLA demo collection, validated end-to-end on a single cheap arm, packaged as a phosphobot extension PR.

Secondary contributions:
- The first clean `N_demos → success_rate` curves for SmolVLA / ACT on AgileX Piper across 3 tasks × 3 seeds × 4 methods (a missing benchmark for the community).
- An empirical study of conformal coverage under DAgger-style demo drift (relevant to W1).

## Pipeline (system view)

```
┌──────────────────────────────────────────────────────────────────┐
│ seed_demos (N=5)  ──►  LoRA fine-tune SmolVLA-450M               │
│        ▲                       │                                 │
│        │                       ▼                                 │
│        │              phosphobot rollout                         │
│        │                       │                                 │
│        │            per-step conformal action uncertainty        │
│        │                       │                                 │
│        │           uncertainty > τ_α  ──► /auto/stop             │
│        │                       │                                 │
│        │           leader-arm teleop recovers segment            │
│        │                       │                                 │
│        │           write trajectory chunk to dataset             │
│        └───────────────────────┘                                 │
│                                                                  │
│ Stop when: predicted_marginal_gain(N+1) < ε on held-out probe    │
└──────────────────────────────────────────────────────────────────┘
```

Concrete components:
- **Backbone**: SmolVLA-450M with LoRA-rank-16 on the action expert (consumer 24 GB GPU friendly). ACT-50M ablation.
- **Uncertainty**: Conformal Quantile Regression (CQR) head over per-timestep action norms; calibration set = held-out 20 % of running buffer; IQT-style intermittent updates to handle DAgger drift.
- **Threshold τ_α**: target empirical coverage α=0.9; re-calibrated after every round.
- **Stopping rule**: regression of marginal gain `Δsuccess(N→N+1)` on dataset statistics (size, average uncertainty on probe, conformal interval width); halt when predicted gain < 2 pp.
- **Tasks**: 3 tabletop tasks on Piper —
  1. **Block-pick-and-place** (rigid, structured)
  2. **Towel-fold-corner** (deformable, partial)
  3. **Stack-cups** (multi-step, contact-rich)
  - **Held-out probe task**: **Drawer-open-and-place** (different topology — stopping-rule must transfer).
- **Operator**: anh0001 (single operator → acknowledged confound, mitigated by report of inter-trial variance over 3 days).

## Baselines (W3)

1. **B-Rand**: random demo collection until N=80.
2. **B-Ent**: entropy of action token logits as query signal.
3. **B-kNN** (CRSAIL-style): kNN distance to existing dataset in visual-feature space.
4. **Ours (B-Conf)**: conformal action uncertainty + IQT calibration.

Each method runs to N∈{5, 10, 20, 40, 80} demos collected, evaluated on 20 trials per task.

## Claims (testable)

| ID | Claim | Falsification criterion |
|----|-------|-------------------------|
| C1 | B-Conf beats B-Rand by ≥2× on demos-to-80%-success (median over tasks) | If average ratio < 1.3×, falsified |
| C2 | B-Conf beats B-Ent by ≥10 pp success at N=20 | If gap < 5 pp, falsified |
| C3 | Stopping rule trained on T1+T2 predicts halt within ±5 demos of empirical optimum on T3 (held-out) | If error > 10 demos, falsified |
| C4 | Empirical conformal coverage at α=0.9 stays in [0.85, 0.95] across rounds | If drifts outside [0.8, 1.0], reported as failure mode |
| C5 | Pipeline runs end-to-end with zero hardware safety incidents over ≥40 h of real-arm time | If any incident, reported transparently |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Conformal coverage breaks under heavy drift | Report empirical coverage every round; fall back to IQT; ablate non-calibrated variant |
| SmolVLA does not learn the deformable task well | Add ACT-50M as secondary backbone; deformable task is the hardest by design and a partial result is acceptable |
| Single-operator confound | Report 3-day inter-trial variance; openly position as case study; PR + dataset release lets the community replicate |
| Piper kinematic drift between sessions | Re-home before each trial; log proprio drift |
| Wall-clock blowout | Hard cap: 5 demos/round × 4 rounds × 4 methods × 3 tasks × 3 seeds = 720 demos = ~40 h of real-arm time + ~30 h GPU |
| Pi0 / Gr00t LoRA flakiness on consumer GPU | Make Pi0 a stretch backbone, only attempt one task if time permits |

## Why Phosphobot Specifically

- `piper.py` driver is mature; URDF + sim variants shipped.
- `/auto/start`, `/auto/stop`, `/recording/*` endpoints expose exactly the rollout-and-takeover hooks the pipeline needs.
- Leader-arm + Quest teleop are first-class.
- Modal remote-inference path means SmolVLA can sit on a cloud GPU while inference clients stream observations from the Piper.

The resulting pipeline is contributed back as a phosphobot extension PR (`phosphobot/endpoints/active_demo.py` + `phosphobot/uncertainty/`).

## Reviewer-Friendly Statement of Limitations

- One operator, one follower arm, three tasks + one held-out → small sample of method-validation matrix.
- SmolVLA's existing community-data pretraining is a confound vs. truly-from-scratch settings.
- The stopping-rule generalization claim (C3) only covers one held-out task — proof of concept, not a universal rule.

## Recommended Next Step

`/run-experiment` to launch Phase 1 of the experiment plan (next file).
