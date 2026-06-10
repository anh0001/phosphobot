# Cross-model final decision — STOP
- Date: 2026-06-10 | codex gpt-5.5 | threadId: 019ea5c6-a961-7760-ae2d-50138ab45e4f
- Input: crux control perturbed_start = 7.1% (1/14), worst arm -> displaced pose unsolvable from scratch (OOD).
- VERDICT: STOP. No non-obvious, likely-positive SmolVLA/LIBERO salvage within 1 day GPU. Multi-scopic preemption
  thesis has no clean remaining target on this substrate; diagnostic fallback confounded by object-pose OOD
  brittleness (already in LIBERO-Plus/PRO neighborhood).
- Small-δ rescue (1-3cm perturbed_start): ~10-15% prob of rescuing a NOVEL claim; cap 15 min, closure only, not a
  continuation bet. Stop unless clean gap (perturbed_start>=70% AND immediate_replan<=30% same pairs).
- KEEP (internal note / 1 paragraph in 'Unreliable by Default'): single takeaway — "In SmolVLA/LIBERO-Spatial,
  apparent mid-rollout recovery failures under object nudges are dominated by object-pose distribution shift: the
  same 5cm displacement at episode start collapses conditioned success to 1/14, so faster replanning / fast-loop
  override solves the wrong bottleneck." Also keep: the perturbation harness, the RNG-confound lesson (metrics
  calling predict_action_chunk must isolate RNG/model state), P0 (n_action_steps=1 hurts vs 50).
- DISCARD as paper claims: "preemption helps", "VLAs lack recovery manifold", "LIBERO mid-rollout recovery
  benchmark" (unless rebuilt around solvable perturbations + stronger policies).
- FINAL: STOP. Return to active-learning project; optional 15-min 1-3cm perturbed_start sweep as closure only.
