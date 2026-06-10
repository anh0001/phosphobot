# Cross-model plan — after recovery probe confirmed
- Date: 2026-06-10 | codex gpt-5.5 | threadId: 019ea5c6-a961-7760-ae2d-50138ab45e4f
- Input: recovery_probe (best-case 28.6% ≈ open_loop 21.4%, n=14) + RNG-confound finding.
- VERDICT: RUN option (1) — powered diagnostic — with a HARD STOP after it.
- CRUX (C): must PROVE the disturbance is recoverable-in-principle, else the negative is about the
  perturbation not the policy. Two controls:
  1. privileged scripted recovery ORACLE from perturbed states (retreat/lift/re-approach/grasp/place);
     if >80% succeed -> task physically recoverable.
  2. perturbed_start control: apply SAME 5cm displacement at EPISODE START, SmolVLA solves from reset.
     If ~= clean -> object pose isn't the problem -> mid-trajectory recovery IS -> "lacks recovery manifold" valid.
     If it also collapses -> claim downgrades to "lacks robustness to object pose shift" (weaker/known).
  + log visibility/sanity: object still in view, not penetrating/off-table, reachable, success predicate satisfiable.
- MATRIX (min submit-able): libero_spatial n_clean_success>=60 (~220 pairs); libero_object >=40-60;
  primary SmolVLA ckpt (+stronger if avail; OpenVLA/pi0 only if already runnable). Disturbance: object qpos nudge
  δ=5cm primary, 10cm stress. Arms: clean, open_loop, delayed_replan_L8, immediate_replan, preempt_hold(killed),
  perturbed_start, privileged_recovery_oracle(subset).
- STATS: paired binary -> exact McNemar on discordant pairs; paired risk diff + 95% CI via bootstrap over task IDs;
  equivalence margin ±10pp (only claim "no benefit" if CI within band, else "no detectable benefit").
- Top reviewer objection: "perturbation is unrecoverable/OOD" -> preempted by the oracle + perturbed_start controls.
- Venue: CoRL/ICRA workshop or short diagnostic; main-track only with 2nd policy family + clean harness release.
- Title: "Replanning Earlier Is Not Recovery: A Mid-Rollout LIBERO Diagnostic for Chunked VLA Policies"
