# Experiment Tracker — Preemptive Dual-Clock VLA (Direction A)

Branch `idea/vla-multiscopic-cognition`. Plan: `EXPERIMENT_PLAN.md`. Gates fail-fast (see kill criteria).

| Block | Claim | Status | Result | Gate |
|------|-------|--------|--------|------|
| P0 horizon ablation | C0 headroom | ✅ DONE | n=50→30/31%, n=8→32%, n=1→20% (libero_spatial). headroom=**−10pp (NEG)** | gap≳+8pp ⇒ clean headroom → **FAILED (negative): static LIBERO has no reactivity headroom; n=1 hurts** |
| E1 oracle handover | C1,C2 | ✅ DONE | δ-sweep 50 pairs/14 clean-succ. cond: open_loop=replan_only=21.4% both δ; preempt_hold 28.6%@5cm / 14.3%@10cm. **FAILED both**: cond_gain +7.1/−7.1pp (sign-flips, =±1 pair), stale-harm −35% (hold worse) | **NOT MET — preemption killed on LIBERO** |
| E2 handover vs bounded residual (A2C2) | C3 | ⬜ TODO | — | residual ties across magnitudes ⇒ C3 vacuous, lean on E4 |
| E3 dispersion trigger validity | C4 | ⬜ TODO | — | AUROC(dispersion) > image-Δ ⇒ end-to-end; else negative→Dir B |
| E4 phase diagram | C5 | ⬜ TODO | — | win-region boundary characterized |

## Log
- 2026-06-08 — idea-discovery run: branch created, ref paper summarized, 43-agent survey+gen, GPT-5.5 jury
  (Direction A selected at checkpoint), refine round 2 (reframe → phase diagram), proposal+plan written.
- 2026-06-08 15:05 — P0 launched; first run mis-sized (`--eval.n_episodes` is PER-TASK → 200/horizon). Kept
  h50@200=31% anchor, killed the rest.
- 2026-06-08 ~16:30 — P0 lean (`pilots/run_horizon_pilot_fast.sh`, 50 ep/horizon) DONE:
  **n=50→30%, n=8→32%, n=1→20%**. Clean headroom = −10pp (NEGATIVE): re-inference every step HURTS (flow-head
  re-sampling breaks chunk coherence). Implication: perturbation protocol P1 is mandatory; contribution must be
  SELECTIVE override (blanket replan `n=1` is a weak baseline); coherent hold/retract is the right first mechanism.
  → Proceed to build E1 harness (oracle handover under mid-rollout object-nudge).

- 2026-06-09 — E1 δ-sweep DONE (`preempt/e1_sweep_d005.json`, `e1_sweep_d010.json`). **Preemption mechanism
  FAILS its pre-registered kill gate at both 5cm and 10cm.** Decisive sub-finding: `open_loop == replan_only`
  (21.4% cond) at BOTH magnitudes → replanning 8 steps earlier gives ZERO benefit → the bottleneck is the
  policy's recovery capability, not stale-action execution timing → preemption addresses a non-bottleneck.
  `preempt_hold` cond_gain = +7.1pp@5cm / −7.1pp@10cm (sign-flips = ±1 conditioning pair at n=14 → no effect);
  stale-action harm −35% (HOLD diverges from the recovery action MORE than the stale chunk does). Consistent
  with P0 (n=1 hurts) and the GPT-5.5 jury's prediction (static manipulation sim lacks closed-loop headroom).
  **Verdict: Direction A's method is killed on LIBERO.** This is a valid negative/diagnostic result. Per the
  plan, pivot → Direction B (dispersion-trigger validity audit) or reframe as "why reactive override does not
  help VLA manipulation recovery on standard benchmarks." E2/E3/E4 are moot for the method as posed.

- 2026-06-10 — RECOVERY-CAPABILITY PROBE done (`preempt/e1_recovery_d005.json`, δ=5cm, 14 clean-success pairs).
  Best-case recovery (immediate replan from perturbed obs + 8-step cadence) = **28.6% (4/14)** vs open_loop
  **21.4% (3/14)** — within ±1 pair. preempt_hold 35.7% (5/14), replan_only 21.4%. Per-pair success is
  **uncorrelated with strategy** (noise signature). **THESIS CONFIRMED: bottleneck = recovery capability, not
  execution timing.** Caveats: (1) n=14 underpowered (need ≥60); (2) measurement-RNG confound — stale-harm metric
  advances the policy flow-RNG and shifts the primary outcome (preempt_hold 28.6% w/ harm-meas ON vs 35.7% OFF);
  use `--no-stale-harm` for clean numbers and decouple the metric RNG before publishing.
  Decision (Codex, 1-week cap): Option B diagnostic. Remaining must-runs: power to n≥60 seeds, add libero_object
  (+OpenVLA/π0 ideal), fix RNG confound.

- 2026-06-10 — CRUX CONTROL `perturbed_start` (`preempt/perturbed_start_control.json`): displace same object 5cm
  at EPISODE START, solve from scratch → conditional success **7.1% (1/14)**, LOWER than mid-rollout arms
  (open_loop 21.4%, recovery_probe 28.6%). **Framing gate FAILED**: the displaced pose is not solvable from
  scratch (5cm = OOD vs calibrated init states), so the preemption negative is CONFOUNDED by object-pose
  fragility — a known result (LIBERO-Plus). The novel "replanning isn't recovery" diagnostic does NOT hold at
  δ=5cm. Unifying explanation across P0/E1/recovery/crux: failure under displacement is perceptual/distributional
  brittleness, not control timing. RNG confound also fixed (save/restore around harm metric). Low-odds rescue =
  small-δ sweep (pose solvable-from-scratch but mid-rollout still fails); otherwise DIRECTION DONE.

## Notes
- Substrate: SmolVLA LoRA, libero_spatial ckpt ~30% base. Knobs verified: `--policy.n_action_steps`, `num_steps`,
  `rtc_config` (RTC baseline built-in).
- Traces: `.aris/traces/idea-discovery/2026-06-08_run01/` (jury + refine round 2).
