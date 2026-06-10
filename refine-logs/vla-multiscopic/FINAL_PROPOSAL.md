# Final Proposal — "Override, not Replan": a Preemptive Dual-Clock VLA

> **STATUS (2026-06-09): METHOD KILLED ON LIBERO.** The pre-registered E1 oracle test failed its kill gate at
> both 5cm and 10cm disturbances (`open_loop == replan_only`; `preempt_hold` gain +7.1/−7.1pp, sign-flips,
> within ±1-pair noise; stale-harm favors the stale chunk over hold). The bottleneck is the policy's recovery
> capability, not stale-action timing. See `EXPERIMENT_TRACKER.md` / `IDEA_REPORT.md` § E1. The method below is
> retained as the tested hypothesis; pivot options are in the report. C1/C2 = FAILED; C3–C5 moot as posed.

**Direction**: A (Preemption VLA), selected at the idea-discovery Phase-2 checkpoint.
**Date**: 2026-06-08 · **Branch**: `idea/vla-multiscopic-cognition`
**Substrate**: SmolVLA (`lerobot/smolvla_base`, LeRobot 0.5.x) on LIBERO. Flow-matching action expert,
`chunk_size=50`, `num_steps=10`, async RobotClient/PolicyServer; harness computes flow-head dispersion.

---

## Problem Anchor (frozen — prevents scope drift)
> Modern VLAs execute predicted action **chunks open-loop**: once a chunk is committed, the robot runs it for
> tens of control steps before the slow backbone re-infers. Every existing "reactivity" mechanism for chunked
> VLAs keeps the **slow policy as the sole author of every executed action** — they only change *when* it
> re-authors (RTC, async, Mixture-of-Horizons, AutoHorizon, Adaptive Action Chunking), *blend a residual onto*
> its chunk (A2C2), or *filter/resample its candidates before execution* (Pre-VLA). **None lets a fast loop
> seize actuation authority and substitute a corrective action mid-chunk, overriding the slow plan before the
> next authorized re-inference.** That is precisely primitive P5 of Saputra et al. 2022 (the Affordance-
> Effectivity-Fit *interrupt*): in legged locomotion, a fast reflex preempts the gait when an affordance
> suddenly demands it. The anchor question: **does discrete fast-loop preemption (authority handover) add
> anything to a VLA beyond replanning faster or correcting additively — and if so, when?**

## Method Thesis (one sentence)
Replace SmolVLA's fixed open-loop chunk-execution rule with a **priority-arbitrated dual-clock controller** in
which a cheap micro loop monitors the flow head's *own* disagreement and, when it crosses a calibrated trigger,
**preempts** the queued slow-loop action — holding / vetoing / substituting a fast corrective action — until the
slow VLA re-grounds.

## Reframed Research Question (after cross-model refine round 2 — the headline)
> **When should a chunked VLA cede actuation authority from the slow planner to a fast reflex, instead of
> rescheduling (replan) or blending a correction (residual)?**
This converts the closest competitor (A2C2 additive correction) from a novelty threat into the *main scientific
question*. The deliverable is a **phase diagram**: axes = disturbance magnitude × replan latency (or residual
authority budget), stratified by manipulation phase (transport / approach / contact), comparing {open-loop,
fixed-short-horizon, RTC, residual-correction, **authority-handover (override)**, full-replan} under **matched
compute/latency**. The scientific payload is the *boundary* where bounded residual correction and rescheduling
stop being enough and **vetoing the stale action** (handover) starts to win. "Preemption VLA" is the mechanism;
the phase diagram is the contribution.

## The Fundamental Method Change (exact SmolVLA surface)
- **What exists**: `SmolVLAPolicy.select_action` keeps an action queue of length `n_action_steps`;
  `_check_get_actions_condition` refills it (time/queue-empty) by calling `sample_actions` (re-encode prefix →
  `num_steps` flow integration → 50-step chunk). Reactivity = *when the queue refills*.
- **What we change**: insert a per-control-step **arbiter** between the queue and the actuator. Each step it
  computes a cheap trigger `s_t` and chooses a mode by a **priority hierarchy (reflex > plan > default)**:
  - `KEEP` (default): execute the queued slow action.
  - `OVERRIDE` (`s_t > τ_hold`): the **micro controller authors this step** — substitutes a fast corrective
    action and the queued chunk is suspended (authority handover). This is the new object.
  - `REPLAN` (`s_t > τ_replan`, or override persists ≥ k steps): trigger a slow re-inference (macro re-ground).
  This is the AEF interrupt mapped onto VLA inference: a switching controller, not a reschedule or a residual.
- **What the micro loop substitutes** (ablation axis — defines "override"):
  1. **Reflex** (model-free): damp/clamp/hold (zero-velocity hold or retract). Tests the pure value of *stopping
     a stale plan*. No new params.
  2. **Cheap fast expert**: one flow step from the **cached prefix KV** (no full re-encode) → a low-latency
     action. Reuses `use_cache=True`.
  3. **Tiny learned reflex head** (the AEF ANN analog): `proprio (+cached prefix) → corrective delta`, LoRA-
     light, trained on recovery transitions. Most faithful to P5.

## Novelty — exact differentiation from the nearest 2025–2026 work
| Prior work | What it does | Why we differ |
|---|---|---|
| RTC (2506.07339), async SmolVLA, MoH, AutoHorizon, Adaptive Action Chunking (2604.04161) | **Reschedule/shorten** slow re-inference; slow policy authors all actions | We let a **separate fast policy author** actions during the override window |
| **A2C2 / "Leave No Observation Behind"** (OpenReview y5SGBsndWv) | Always-on **additive residual** correction onto the chunk every step | We do **discrete gated handover** — *replace/veto*, not blend; switching vs residual control |
| **Pre-VLA** (2605.22446) | **Preemptive resampling/verification BEFORE execution** | We act **during** execution of an already-committed chunk |
| Brain-inspired **Fast Safety Reflex** (2601.14628) | Fast loop bypasses cortical loop on **tactile/force** for withdrawal | Triggered by the policy's **own flow-head disagreement** (no extra sensor); aims at **task recovery**, not withdrawal |
| GR00T N1 / Helix / π0.5 / DuoCore-FS dual-system | Fast loop **executes slow-authorized** latents | Fast loop can **override** the slow plan |

**Defensible novelty (narrow but real)**: a *discrete, internally-triggered, priority-arbitrated authority
handover* over an executing chunk, plus a **mid-rollout dynamic-disturbance LIBERO protocol** to make it bite.
LIBERO-Plus / LIBERO-PRO / Libero-V perturb at **episode start** (initial state / viewpoint / texture); none
inject a **mid-rollout** dynamic disturbance requiring online recovery — that protocol is itself a contribution.

## Claims (each falsifiable; result matters either way)
- **C0 (headroom, pilot — RESOLVED, NEGATIVE)**: clean-LIBERO headroom `succ(n=1) − succ(n=50)` measured at
  **−10 pp** (n=50→30/31%, n=8→32%, n=1→20% on libero_spatial). Re-inference every step *hurts* (flow-head
  re-sampling breaks chunk coherence). ⇒ (a) the P1 perturbation protocol is **mandatory**; (b) the contribution
  must be **selective** override — blanket replan (`n=1`) is empirically a **weak** baseline, not a strong one,
  which *strengthens* the override-vs-replan framing; (c) the override action must be **coherent** (hold/retract),
  not a fresh re-sample.
- **C1 (oracle handover > replan-only, perturbed)** — *the core, dispersion-free claim*: under a mid-rollout
  disturbance, with an **oracle trigger** and a fixed replan latency `L`, executing a hold/retract **override**
  during the latency window recovers more success than executing the **stale queued actions** during the same
  window (same trigger, same `L`, same macro-call count).
- **C2 (handover ≈/> full replan at lower compute)**: handover matches/beats full re-inference (`n=1`) recovery
  at a fraction of inferences/episode — the efficiency-under-latency case.
- **C3 (handover > BOUNDED residual for large/discrete disturbances)** — *formalized to be non-vacuous*: against
  an A2C2-style residual head with a **bounded** authority budget and **matched inputs/params/training data**,
  discrete handover wins once the stale queued action is **anti-aligned** with recovery (cosine(stale, oracle-
  fresh) < 0) beyond a deviation threshold — because the bounded residual must *fight* the base action while the
  switch *vetoes* it. Measured with action-authority metrics: anti-alignment cosine, residual authority budget,
  mode-ownership (fraction of steps the executed action is independent of the queued action).
- **C4 (trigger validity, separable)**: flow-head dispersion predicts "should-override" steps better than an
  equal-cost image/state-delta baseline (AUROC). Run **only after C1**; a weak trigger does not kill the
  mechanism (it stays trigger-agnostic) but feeds Direction B (dispersion audit) as a negative result.
- **C5 (phase diagram, the contribution)**: the win-region of handover vs {residual, RTC, replan, open-loop}
  forms a coherent boundary in (disturbance magnitude × latency/budget), stratified by phase — handover dominant
  only in the large-discrete-latency-sensitive corner.

## Verification (LIBERO / SmolVLA)
- **Pilot P0 (running)** — execution-horizon ablation `n_action_steps ∈ {50,8,1}` on `libero_spatial`, training-
  free, measures C0 headroom. Go/no-go for the clean-vs-perturbed framing.
- **Experiment E1 (the decisive next step — ORACLE handover, dispersion-free)**: isolate authority handover from
  the unproven trigger. On `libero_spatial`, paired seeds, one object-nudge magnitude, 4 arms at **matched macro
  calls + matched latency `L=8`**: (1) `open-loop n=50`; (2) `oracle replan-only` (perturbation→request macro
  replan, execute **stale queued** actions during the `L`-step latency); (3) `oracle preempt-hold/retract` (same
  trigger/`L`/macro count, execute **hold/retract override** during latency); (4) `n=1 full replan` (compute
  upper bound). **Kill criterion**: if (3) does not beat (2) by **≥ +10 pp conditional perturbed success** *or*
  reduce post-perturb stale-action harm by **≥ 25%** at identical macro calls → kill the preemption mechanism /
  reframe as safety-only. If (4) dominates massively → contribution is efficiency-under-latency, not recovery.
- **Perturbation protocol P1** — mid-rollout disturbance at a fixed pre-grasp progress fraction. **Primary =
  target-object qpos nudge** (lateral table-plane, no penetration, settle 2–3 sim steps); secondary = eef
  displacement; deprioritized = action dropout (tests actuator fault, gameable by hold). Genuinely distinct from
  LIBERO-Plus/PRO/V (episode-start shifts). Built on the `lerobot-eval` libero path.
- **Main baseline = A2C2** (additive real-time chunk correction) with a **bounded** authority budget and
  matched inputs/params/data — plus open-loop n=50, full replan n=1, RTC (`rtc_config`), scalar-dispersion
  replan (MoH-like). Not a side note.
- **Metrics (non-gameable)**: **conditional perturbed success** `P(success_perturbed | same seed succeeds clean)`;
  **recovery latency** (steps until gripper-target distance < δ and decreasing for `m` steps); **stale-action
  harm** = AUC over first `H=25` post-perturb steps of disagreement vs oracle fresh-replan action; **compute**
  (macro calls/episode, ms/step); **holding penalty** — holding counts only if final task success within the
  normal horizon.
- **Suites**: `libero_spatial` + `libero_object` (primary), `libero_long` (stretch). ≥3 seeds for headline
  C1–C3; the C5 phase diagram sweeps disturbance magnitude × latency, stratified by phase.

## Risks & mitigations
- **R1 LIBERO too static** (jury's main doubt) → P1 perturbation protocol creates the headroom; C0 quantifies it.
- **R2 Novelty too narrow vs A2C2/Pre-VLA** → frame on the *override-vs-replan-vs-correct* axis with C3 as the
  decisive contrast; the mid-rollout protocol is a second, independent contribution.
- **R3 Dispersion not a usable trigger** → C4 turns this into a publishable negative (Direction B); the override
  mechanism is trigger-agnostic (can use image-delta), so the method survives a weak dispersion result.
- **R4 Reflex hand-engineering** → start model-free (mode 1), escalate to learned head (mode 3) only if mode 1
  shows headroom.

## Next
- Fold pilot P0 numbers in → finalize `EXPERIMENT_PLAN.md` → implement arbiter (mode 1 first) → C1/C2 on
  perturbed `libero_spatial` → `/auto-review-loop`.
