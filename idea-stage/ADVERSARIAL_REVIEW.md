# Phase 4 — Adversarial Review

**Subject:** *Calibrated Demo Budgeting: Conformal-Uncertainty-Targeted Teleop for VLA Fine-Tuning on a Low-Cost 6-DOF Arm*
**Reviewer persona:** Senior NeurIPS / CoRL / ICRA area chair, robot-learning track. Brutal but constructive.

## Summary of the Proposal
Authors propose a closed-loop pipeline on a single $4k AgileX Piper arm: (1) fine-tune a VLA (Pi0 / SmolVLA / ACT) from a tiny seed of teleop demos, (2) deploy on phosphobot; during rollout compute conformal action uncertainty per timestep, (3) when uncertainty exceeds a calibrated threshold the policy halts and the operator delivers a leader-arm teleop demo from the queried state, (4) the new demo is added to a LoRA fine-tuning buffer and the policy is updated periodically. The primary contribution claim is a learned, calibrated **stopping rule** + **targeting rule** for real-world demo budgeting on a cheap arm, validated by a sample-complexity curve.

## Scores (1–10)

| Axis | Score | Comment |
|------|-------|---------|
| Problem importance | 8 | "How many demos do I need?" is the #1 unanswered question for hobbyist & lab cheap-arm users |
| Novelty | 6 | All ingredients exist; the *combination on cheap real hardware* is new but incremental |
| Method elegance | 7 | Conformal calibration → stopping rule is clean; needs care for non-iid violation under DAgger-style demos |
| Empirical risk | 5 | One arm, one operator → confounds; needs careful baselines |
| Practical impact | 8 | If the result is "stop at N=K, gain ≥X %" with K small, it directly affects every Piper / SO-100 / Koch user |
| Overall | **6.8** | Borderline accept at top venue, solid accept at second-tier venue (CoRL workshop, IROS) |

## Critical Weaknesses

### W1. Non-iid breaks conformal coverage guarantee
Conformal prediction's coverage guarantee assumes exchangeable calibration data. As the policy updates and the operator is queried at uncertainty hotspots, the distribution drifts continuously. **Mitigation must be explicit**: either use IQT-style intermittent quantile tracking (ConformalDAgger) or report empirical coverage at deployment, not theoretical.

### W2. Single-operator confound
One person × one arm = one trajectory through method-space. Generalizability beyond the author's lab is unclear. **Mitigation:** at minimum add a second human teleoperator for one task and report variance; if not possible, position as a case study and own the limitation.

### W3. The "active" baseline is critical
Without comparing against (a) random demo, (b) entropy-based query (cheap baseline), and (c) coverage-based query (kNN distance — CRSAIL), the conformal contribution is unsupported. **Must include 3 baselines minimum.**

### W4. Stopping-rule validation needs unseen tasks
A stopping rule learned on the same tasks it's evaluated on is a curve fit, not a rule. **Need held-out task families.**

### W5. Result format must be a *curve*, not a single number
Single-number reporting ("we beat baseline by X %") fails to make the central claim. The claim is sample-complexity, so the artifact is `N_demos → success_rate` curves for ≥3 tasks × ≥3 seeds × ≥3 methods.

### W6. Compute budget reality check
LoRA fine-tuning of Pi0 (4B params) on a 24 GB GPU after every 5 demos is borderline; SmolVLA-450M (or 50M ACT) is more honest for the single-RTX class. **Recommend pivoting primary backbone to SmolVLA or ACT and treating Pi0 as the "stretch" backbone.**

### W7. Safety story is missing
Real-arm uncertainty-triggered halt: what if the arm halts mid-grasp and drops a 0.5 kg payload? Need explicit workspace limits, a torque-monitoring fallback, and reporting of safety incidents during data collection. Reviewers care about this on real hardware.

## Suggested Minimum Viable Improvements

- **MVI-1**: Add CRSAIL-style kNN baseline. Without it the conformal angle is unmotivated.
- **MVI-2**: Use **SmolVLA-450M** as primary backbone (consumer GPU-friendly, public LeRobot lineage). Ablate to ACT.
- **MVI-3**: Define the stopping rule by held-out probe-task success and report on 1 held-out task.
- **MVI-4**: Empirically report conformal coverage at each round and discuss drift.
- **MVI-5**: Cap real-arm experiments to **3 tasks × 3 seeds × 4 methods × N∈{5,10,20,40}** to keep wall-clock in 2 weeks.
- **MVI-6**: Open-source the demo-budget logger + uncertainty-query module as a phosphobot extension PR — this builds practical impact and credibility.
- **MVI-7**: Single-line safety story: hard joint-velocity limit + auto-halt on torque > τ; report 0 safety incidents from N hours of trials.

## Revised Claim (after MVIs)

> "On a single AgileX Piper arm with SmolVLA-450M, conformal-uncertainty-targeted teleop reduces the number of real demonstrations to reach 80 % success by **2–4×** across 3 tabletop tasks, vs. random, entropy, and kNN-coverage baselines. An empirically-validated stopping rule predicts the optimal halt-point on a held-out task to within ±5 demos."

## Decision

**Carry the consolidated idea into Phase 4.5 refinement** with MVI-1 through MVI-7 baked in. Risk-managed via SmolVLA pivot and explicit baselines.
