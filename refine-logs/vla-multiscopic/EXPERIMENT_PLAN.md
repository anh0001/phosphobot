# Experiment Plan — Preemptive Dual-Clock VLA ("Override vs Replan vs Correct")

**Proposal**: `FINAL_PROPOSAL.md` · **Branch**: `idea/vla-multiscopic-cognition` · **Date**: 2026-06-08
**Machine**: kubotaserver2 (RTX A6000, ~35 GB free) · **Env**: `sim/libero_active/.venv` (py3.12, lerobot[libero])
**Policy/substrate**: SmolVLA LoRA ckpt `results/fixedbuf_random_N20/seed0/.../004000/pretrained_model`
(libero_spatial, ~30% base success). Knobs: `--policy.n_action_steps`, `--policy.num_steps`, `policy.rtc_config`.

Headline question (C5): *the win-region of authority-handover vs {residual, RTC, replan, open-loop} in
(disturbance magnitude × replan latency), stratified by phase.* Each block below maps to a claim and has a
**kill criterion** so we fail fast.

---

## Block 0 — Pilot P0: clean-LIBERO headroom (RUNNING)  → C0
- **Run**: `idea-stage/vla-multiscopic/pilots/run_horizon_pilot.sh` — `n_action_steps ∈ {50,8,1}`, libero_spatial,
  20 eps, seed 1000, training-free.
- **Read**: `succ(n=1) − succ(n=50)`.
- **Decision**: gap ≳ +8 pp ⇒ clean headroom exists, handover has something to capture even un-perturbed
  (strengthens efficiency story). gap ≈ 0 ⇒ reactivity is perturbation-only ⇒ Block 1's protocol is mandatory
  (expected, and fine). Either way E1 proceeds.
- **Budget**: < 1 GPU-h.

## Block 1 — E1: ORACLE authority-handover (decisive, dispersion-free)  → C1, C2
The cleanest test: isolate *handover* from the unproven trigger by using an **oracle trigger** (we know the
perturbation step). Build the custom rollout harness (see Implementation) and run 4 arms, **matched macro-call
count and matched replan latency `L=8`**, paired seeds, one object-nudge magnitude, libero_spatial:
1. `open-loop n=50` — no special recovery.
2. `oracle replan-only` — at perturbation: request macro replan; execute **stale queued** actions during the
   `L`-step latency window, then run the fresh chunk.
3. `oracle preempt-hold/retract` — same trigger, same `L`, same macro count; execute a **hold/retract override**
   during the latency window.
4. `n=1 full replan` — compute upper bound.
- **Metric**: conditional perturbed success + stale-action harm AUC (H=25) + recovery latency + macro calls.
- **n**: 20 paired eps if tight, 40 if eval speed allows.
- **KILL**: if (3) − (2) < **+10 pp** conditional success *and* stale-harm reduction < **25%** at equal macro
  calls ⇒ kill preemption-for-recovery; fall back to safety-only framing or pivot to Direction B.
  If (4) ⋙ all ⇒ reframe as *efficiency-under-latency*, not superior recovery.
- **Budget**: ~1–2 GPU-h.

## Block 2 — E2: handover vs BOUNDED residual (A2C2 main baseline)  → C3
- Reimplement an **A2C2-style** residual head: inputs = latest obs + base (queued) action + chunk index + base
  features → time-aware additive `delta`, with a **bounded** norm budget `‖delta‖ ≤ b`. Matched params/inputs/
  training data to the learned override head (Proposal mode 3).
- Compare {bounded-residual, override} across nudge magnitudes; log **anti-alignment cosine(stale, oracle-fresh)**,
  residual budget `b`, mode-ownership.
- **KILL/scope**: if bounded residual matches override across all magnitudes ⇒ C3 vacuous ⇒ the contribution is
  the **phase diagram + protocol** (Block 4), not "override ≠ correct". (Still publishable, per refine round 2.)
- **Budget**: ~1–2 GPU-h (+ short LoRA train for the heads).

## Block 3 — E3: dispersion-as-trigger validity (separable)  → C4
- Replace the oracle trigger with online signals; compute AUROC for predicting "should-override" steps (defined
  by E1's oracle): flow-head dispersion (K=2–4 samples) vs equal-cost image-Δ / state-Δ baseline.
- **Outcome**: dispersion wins ⇒ end-to-end preemptive VLA. dispersion loses ⇒ **negative result** for Direction
  B (dispersion audit); method stays trigger-agnostic.
- **Budget**: ~1 GPU-h (reuses E1 rollouts/logs where possible).

## Block 4 — E4: the phase diagram (the contribution)  → C5
- Sweep disturbance magnitude × replan latency `L` (and residual budget `b`), stratified by phase
  (transport/approach/contact via gripper-target distance + contact flags), suites libero_spatial + libero_object.
- Plot the win-region boundary of {open-loop, short-horizon, RTC, residual, override, full-replan} at matched
  compute. ≥3 seeds for headline cells.
- **Budget**: ~4–8 GPU-h (the bulk; only after Blocks 1–2 clear their kill gates).

---

## Implementation surface (what to build)
1. **Custom rollout harness** `sim/libero_active/preempt/rollout.py` (new, isolated from the active-learning code):
   - Build LIBERO env via the same factory `lerobot-eval` uses (`--env.type=libero`); reuse SmolVLA load via the
     `policy_runner._load_policy_and_meta` pattern (PreTrainedConfig + make_policy + make_pre_post_processors).
   - Drive the loop manually: maintain the action queue ourselves so we control re-inference timing, override,
     and latency `L`. This replaces the opaque `lerobot-eval` rollout for E1–E4 (P0 keeps using `lerobot-eval`).
2. **Perturbation hook** `preempt/perturb.py`: at a pre-grasp progress fraction, nudge the target object's qpos
   in the underlying MuJoCo sim (`env.sim` / robosuite handle), lateral table-plane, no penetration, settle 2–3
   steps. Param: magnitude. (Secondary: eef displacement.)
3. **Arbiter** `preempt/arbiter.py`: priority hierarchy KEEP/OVERRIDE/REPLAN; override modes = hold/retract
   (model-free), cheap-fast-expert (1 flow step from cached prefix), learned reflex head (LoRA). Trigger pluggable
   (oracle | dispersion | image-Δ).
4. **Residual baseline** `preempt/residual_a2c2.py`: bounded additive correction head.
5. **Metrics** `preempt/metrics.py`: conditional perturbed success, recovery latency, stale-action harm AUC,
   anti-alignment cosine, mode-ownership, compute counters.

## Run order & gates
P0 (running) → **E1 (kill gate)** → E2 (kill gate) → E3 → E4. Stop at any failed gate and report the negative
result (it is publishable: either "handover doesn't help on manipulation" or "dispersion isn't a valid trigger").

## Compute budget
P0 <1h · E1 1–2h · E2 1–2h · E3 ~1h · E4 4–8h. Headline (P0+E1+E2) ≤ ~5 GPU-h — fits a day on the A6000 around
the isaaclab jobs (35 GB free).
