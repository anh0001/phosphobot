# Cross-model refine round 2 — Direction A proposal
- Date: 2026-06-08 | backend: codex gpt-5.5 xhigh | threadId: 019ea5c6-a961-7760-ae2d-50138ab45e4f
- Input: refine-logs/vla-multiscopic/FINAL_PROPOSAL.md + new novelty (A2C2, Pre-VLA, Fast Safety Reflex, LIBERO-Plus/PRO/V)
- Verdict: 5/10 as-is. Key upgrades:
  1. C3 vacuous unless formalized as BOUNDED residual vs DISCRETE handover + action-authority metrics
     (anti-alignment cosine, residual authority budget, mode ownership), matched inputs/params/data. A2C2 = MAIN baseline.
  2. REFRAME to phase diagram: "when should a chunked VLA switch actuation authority vs reschedule vs correct?"
     x=disturbance magnitude, y=replan latency/residual budget, stratified by phase (transport/approach/contact).
  3. Perturbation: object qpos nudge (pre-grasp) most decisive; action dropout weakest. Metric = conditional
     perturbed success P(succ_pert | seed succeeds clean) + recovery latency + stale-action harm AUC + compute;
     holding != recovery unless final success.
  4. NEXT decisive exp = ORACLE-triggered handover (no dispersion yet): open-loop n=50 / oracle replan-only
     (execute stale during L=8 latency) / oracle preempt-hold (same trigger+latency+macro count, hold/retract) / n=1.
     KILL if preempt-hold doesn't beat replan-only by >=+10pp cond. success OR reduce stale-harm >=25% at equal macro calls.
  5. One-sentence framing: "We study when chunked VLAs should cede actuation authority from a slow planner to a
     fast reflex, and show under mid-rollout LIBERO disturbances that handover only helps in large, discrete,
     latency-sensitive regimes where residual correction and rescheduling keep executing anti-aligned stale actions."
