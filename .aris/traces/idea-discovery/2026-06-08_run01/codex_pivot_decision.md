# Cross-model pivot decision — after E1 negative
- Date: 2026-06-09 | backend: codex gpt-5.5 | threadId: 019ea5c6-a961-7760-ae2d-50138ab45e4f
- Input: E1 negative results (preempt_hold fails kill gate; open_loop==replan_only; recovery is bottleneck)
- VERDICT: Pick B (reframe as diagnostic/negative paper), STRICT 1-week cap. Not A (dispersion audit answers a
  less-important question — oracle trigger already showed trigger quality isn't the bottleneck).
- E1 NOT publishable as-is (n_clean_success=14, underpowered). Must-run to make credible:
  1. More conditioning mass: stronger checkpoint / more seeds until n_clean_success >= 60.
  2. RECOVERY-CAPABILITY PROBE (decisive): at perturbation discard queue, give best-case recovery (immediate
     fresh replan, no latency, n_action_steps=8, from perturbed state). If still fails -> bottleneck IS recovery
     capability (proves the thesis).
  3. Cross-suite/cross-policy: libero_object min; OpenVLA/pi0 ideal. (Skip eef/action-dropout for now.)
- Reframed thesis: "Closed-loop headroom in VLA manipulation benchmarks is often illusory: under mid-rollout
  object displacement, earlier replanning and fast override do not improve recovery because the learned policy
  lacks a recovery manifold." Contribution = mid-rollout perturbation protocol + timing-vs-capability separation
  + recovery-capability probe.
- Brutal bottom line: no novel POSITIVE multi-scopic VLA method likely to work on SmolVLA/LIBERO in 1 day GPU;
  stop method-chasing. B (compact diagnostic, 1-week cap) or C (highest top-venue ROI = drop).
