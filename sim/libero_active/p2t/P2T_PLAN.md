# P2T — Probe-to-Train (idea #4 pilot): pre-registered plan

**Date:** 2026-07-06 · **Branch:** `idea/vla-multiscopic-cognition` · **Hardware:** RTX 6000 Ada (GPU 1)

## Hypothesis

The anchored-not-steered study measured WHERE the policy ignores object position
(reach field: anchoring rises 0.63→0.94 over 25→100 mm; paste: vision does not
steer). **P2T inverts the probe into a training engine**: synthesize counterfactual
demonstrations exactly where the measured gain field says ∂action/∂object-position
is missing, and test whether *gain-targeted* placement beats *generic* placement
of the same synthesis budget.

**Paradigm claim under test:** a causal probe defines an acquisition function
(where to synthesize) — not just a diagnostic.

## Synthesis engine (DemoGen-in-sim, success-filtered)

For a source demo of task t with target object at canonical C:
1. Reset env to the demo's matched benchmark init state (mapping probe, below).
2. Kinematically displace the target object by d (the placement chosen by the
   acquisition policy).
3. Replay the demo's action sequence with geometric retargeting:
   reach segment (0..t_grasp): +d spread over xy action deltas;
   transport segment (t_grasp..t_release): −d spread (restores the place target);
   after release: unchanged. t_grasp/t_release from gripper open↔close crossings
   (same convention as reach_field.py / object_local_paste.py).
4. Record real rendered obs (agentview+wrist, 360→256), state, executed actions
   → a new LeRobot episode. **Keep only episodes with `is_success`.**

Renderer-exact by construction (every frame is a true sim render of the displaced
scene, physics real throughout — no pixel pasting in training data).

## Conditions (matched budget: same N_syn, same fine-tune steps, same recipe)

| | Condition | Placement of the N_syn synthetic episodes |
|---|---|---|
| A | **gain-targeted** | budget ∝ anchoring deficit per (task,dir,mag) cell from the reach field of the base ckpt |
| B | no-aug control | none — continue training on the 432 originals only (controls for extra steps) |
| C | uniform | same budget uniform over all (task,dir,mag) cells |
| D | failure-targeted (generic acquisition, IntervenGen-style) | budget ∝ (1 − success rate) per cell — same probe data, non-causal signal |
| E | sham (specificity control) | budget on the cells where gain is LEAST missing (lowest deficit) |

Acquisition maps computed from the **new** base checkpoint's reach field (the gain
field is policy-specific). A↔D map correlation reported *before* launch; if
ρ > 0.9 the contrast is declared underpowered and D is re-specified openly.

## Kill-gate (pre-registered)

Primary metric: displaced success rate @50 mm (10 tasks × 5 seeds × 4 dirs = 200
displaced rollouts per condition) + reach-field tracking τ / DVR cossim.

- **PASS**: A > C and A > D (and A > B, A > E) on displaced success, with A's
  advantage ≥ +5 pp over D and outside the seed-bootstrap 90% CI.
- **A ≈ D** → the causal gain field adds nothing over the obvious failure signal
  → the "measured gain field" delta collapses to a reframing. Report as negative.
- **A ≈ C** → placement doesn't matter at all; only synthesis volume does.
- **E ≈ A** → specificity failure; the effect is "any extra synthetic data".
- Secondary: in-distribution (clean) success must not degrade > 5 pp vs B.

## Budgets

- N_syn = 128 successful synthetic episodes per augmented condition (≈30% of 432).
- Fine-tune: 20k steps from the strong ckpt, batch 32, lr 1e-4 cosine→2.5e-6,
  identical for A/B/C/D/E (B trains on originals only).
- Eval per condition: 250 rollouts (1 clean + 4×50 mm per pair).

## Deviations log

- **2026-07-06**: `results/e0_full_expert_100k` (the 74%-clean strong ckpt) was
  missing on disk (wiped during June sweep churn; no backup found on nas/bak/
  remote). Retraining with the exact recorded recipe (`preempt/e0_full_expert.sh`,
  STEPS=100000, BATCH=32, seed 0) — running on GPU 1 at ~3.0 step/s (ETA ≈ 9.5 h),
  log `results/e0_retrain_100k.log`. All acquisition maps and baselines will be
  measured on the retrained ckpt; old e1r numbers are design references only.

- **2026-07-07 (pre-registered yield trigger fired)**: naive open-loop delta-spread
  retargeting collapsed at 75–100 mm (yields 16–26% < 30%; 90/128 cells exhausted
  at 100 mm; conditions imbalanced 71–91 eps). Orchestrator paused BEFORE any
  fine-tune. Fix: **servo retargeting** (MimicGen-style) — track the demo's
  recorded eef trajectory (HDF5 obs/ee_pos) shifted by a front-loaded ramp
  (p=0.5), xyz = feedforward+feedback commands, rot+gripper verbatim; labels =
  executed commands. Validation: 100 mm yield 0/8 → 5/8; pregrasp offsets back
  to the clean rim-grasp baseline (~4–7 cm). ALL conditions regenerated with the
  servo engine (open-loop v1 archived at p2t/staging_openloop_v1 — never mixed).
  Packer now equalizes per-condition counts to the minimum delivered (seeded).

## Risks / open checks

1. **Episode→init-state mapping** must be verified (dataset carries no init
   metadata): match first-frame `observation.state` to per-init derived state;
   require unique bimodal matches. Fidelity gate: replayed episodes reproduce
   `is_success` on ≥80% of a 20-episode sample.
2. **Retarget yield**: naive linear spread may miss grasps; oversample ×2 and
   success-filter. If yield < 30%, switch to ramped spread over the final
   approach segment only.
3. **fps semantics**: dataset nominal 10 fps vs env 20 Hz control — replay is
   1:1 by construction (policy trained on this dataset acts 1:1); fidelity gate
   catches any mismatch.
4. GPU 1 shared between training and rendering smokes (few hundred MB) — OK.
