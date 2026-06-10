# E1 Oracle-Handover Harness — Build + Smoke Report

**Question (E1).** SmolVLA executes 50-step action chunks open-loop. After a
mid-rollout disturbance makes the committed chunk STALE, is it better to **HOLD**
(override the stale actions with a brief freeze) than to **keep executing the stale
chunk** while a re-plan computes? This harness compares override-vs-replan at equal
macro-call budget, with a built-in kill criterion.

---

## 1. What was built

All code lives in `sim/libero_active/preempt/`.

| File | Role |
|------|------|
| `harness_lib.py` | Core library: pipeline builder, single in-process LIBERO env, manual chunk execution, the 5 rollout variants. |
| `perturb.py` | Mid-rollout disturbance: nudges the movable object nearest the eef laterally (`sim.data.qpos[adr]+=delta; sim.forward()`). |
| `e1_oracle_handover.py` | Main harness (argparse: `--tasks --seeds --delta --latency --tp-frac --out --smoke`). Runs the grid, aggregates metrics, writes JSON, prints the kill-criterion verdict. |
| `run_full_e1.sh` | One-shot launcher for the full 10x5 run. |
| `_stage1_probe.py` | Stage-1 de-risk script (plain chunked rollout). |
| `smoke_results.json` | Smoke output (spec timing, t_p=0.35). |
| `smoke_results_early_tp.json` | Smoke output (early timing, t_p=0.14) - see caveat below. |

### Pipeline fidelity
The harness mirrors `lerobot_eval.rollout` / `eval_main` step-for-step:
`make_policy` + `make_pre_post_processors(pretrained_path=...)` +
`make_env_pre_post_processors`, with the policy config loaded from the checkpoint
(`chunk_size=50, n_action_steps=50, num_steps=10`) and a `LiberoEnv` env config
matching the working CLI (`--env.type=libero --env.task=libero_spatial`,
obs 360x360; the policy preprocessor resizes to 256).

Per control step the canonical loop is reproduced exactly:
`preprocess_observation` -> set `observation["task"]` -> `env_preprocessor`
(LiberoProcessorStep) -> `preprocessor` (normalizer) -> chunk action ->
`postprocessor` -> `env_postprocessor` -> `env.step`. Instead of the opaque
`select_action` queue we call `policy.predict_action_chunk(batch)` -> `(1,50,7)`
normalized chunk and manage execution ourselves so we control re-inference timing,
staleness, and override.

**Single in-process (sync) env.** We instantiate `LiberoEnv` directly per `task_id`
and reach the MuJoCo sim via `env._env.env.sim`. A single env yields un-batched obs
(`images (H,W,C)`, `robot_state` arrays `(D,)`); we add a leading batch axis to the
`robot_state` arrays before `preprocess_observation` because
`LiberoProcessorStep._quat2axisangle` requires shape `(B,4)`. Images are
auto-unsqueezed by `preprocess_observation`. We read `info["is_success"]` and
`terminated` BEFORE `LiberoEnv.step` auto-resets on termination.

### The 5 variants (per `(task_id, seed)`)
`set_seed(seed)` is called before EACH rollout and the env init state is fixed by
`episode_index`, so pre-perturbation behavior is identical across variants (same
base chunk cadence: re-infer at steps 0, 50, 100, ...). Perturb at
`t_p = int(tp_frac * max_steps)`, latency `L`.

1. **clean** - no perturbation; normal chunked execution. -> `success_clean`.
2. **open_loop** - perturb at t_p; keep executing base chunks. -> `success_ol`.
3. **replan_only** - perturb at t_p; during `[t_p, t_p+L)` execute the STALE
   pre-perturbation chunk; recompute at `t_p+L`. -> `success_ro`.
4. **preempt_hold** - same as replan_only but during `[t_p, t_p+L)` execute
   HOLD = zeros on the 6 pose dims + last gripper command; recompute at `t_p+L`.
   Same macro-call count as replan_only (neither re-infers inside the window;
   both do exactly one replan at `t_p+L`). -> `success_ph`.
5. **full_replan** - re-infer EVERY step (first action of a fresh chunk); perturb at
   t_p. Upper-bound reference. -> `success_fr`.

### Metrics (written to JSON, per arm)
- **conditional_perturbed_success** (PRIMARY) = mean(success_arm over pairs where
  `success_clean==True`).
- raw_perturbed_success, n_pairs, n_clean_success, avg_macro_calls.
- **stale_action_harm** (secondary) = mean over the first 25 post-perturb steps of
  the L2 distance between the executed action and the first action of a fresh
  `predict_action_chunk` at that step.

### Kill criterion (reported)
preempt_hold is worth pursuing iff
`cond(preempt_hold) - cond(replan_only) >= +10pp` OR
`stale_harm reduction >= 25%` at equal macro calls.

---

## 2. Staged validation

- **Stage 1 (plumbing).** Plain chunked rollout on `task_id=0`, 4 episodes:
  2/4 success (50%), episodes terminating at 82-86 steps with 2 macro calls
  (and doomed ones at 280/6). Plausible vs the ~30% baseline -> the inference
  pipeline (obs preprocessing, normalization, chunk decoding, sim access) is
  correct. Determinism confirmed: identical outcomes across repeated runs of the
  same `(task, episode_index, seed)`.
- **Stage 2 (variants + perturbation).** Perturbation confirmed to move the
  task-relevant object (e.g. `akita_black_bowl_1_joint0`). With a mid-task t_p the
  variants diverge correctly: clean succeeds, the post-disturbance arms fail, and
  `replan_only`/`preempt_hold` carry equal macro calls.
- **Stage 3 (smoke + metrics + report).** Below.

---

## 3. Smoke results

### 3a. Spec timing (`t_p = int(0.35*280) = 98`), 2 tasks x 2 seeds
File: `preempt/smoke_results.json`

| arm | conditional succ | raw succ | avg macro | stale_harm |
|-----|-----------------:|---------:|----------:|-----------:|
| clean        | 100.0% | 75.0% | 3.0 | - |
| open_loop    | 100.0% | 75.0% | 3.0 | - |
| replan_only  | 100.0% | 75.0% | 3.0 | - |
| preempt_hold | 100.0% | 75.0% | 3.0 | - |
| full_replan  |   0.0% |  0.0% | 280.0 | - |

`n_clean_success = 3 / n_pairs = 4`.

**Kill criterion (spec timing): cond_gain = 0.0 pp, harm_reduction = n/a ->
DOES NOT PASS.**

**Why this smoke is degenerate (important):** every clean SUCCESS on
`libero_spatial` completes in ~76-98 steps, i.e. at or before t_p=98. So on
exactly the episodes that enter the conditional metric, the task is already done
when the disturbance would fire - the perturbation never fires on a
clean-success pair (`perturbed=False` for all stale-window arms there), and
replan_only/preempt_hold/open_loop trivially equal clean. The only pair where the
perturbation fired (task3/seed1004) was already a clean failure and is excluded by
the conditional definition. This is a property of the spec timing on this suite,
not a harness bug.

### 3b. Early timing (`t_p = int(0.14*280) = 39`), 2 tasks x 2 seeds
File: `preempt/smoke_results_early_tp.json`

This fires the disturbance mid-task (before typical completion) so the
override-vs-replan question is actually exercised end-to-end.

| arm | conditional succ | raw succ | avg macro | stale_harm |
|-----|-----------------:|---------:|----------:|-----------:|
| clean        | 100.0% | 75.0% |   3.0 | - |
| open_loop    |  33.3% | 25.0% |   5.0 | - |
| replan_only  |  33.3% | 25.0% |   5.0 | 0.845 |
| preempt_hold |  33.3% | 25.0% |   5.0 | 1.321 |
| full_replan  |  33.3% | 25.0% | 234.2 | - |

`n_clean_success = 3 / n_pairs = 4`. The disturbance now FIRES mid-task on all 3
clean-success pairs (`perturbed=True`). On 2 of the 3 every post-disturbance arm
fails; on 1 every arm recovers.

**Kill criterion (early timing): cond_gain = 0.0 pp, harm_reduction = -56.4%
(equal_macro=True) -> DOES NOT PASS** - and the harm signal points AGAINST
preempt_hold: HOLD (zeros + last gripper) diverges from the policy's intended
post-perturbation action MORE than the stale chunk does (preempt_hold harm 1.32 >
replan_only harm 0.84). On this checkpoint there is no evidence that overriding with
a freeze beats continuing the stale chunk; if anything the stale chunk is closer to
what a fresh re-plan would do.

---

## 4. Verdict on the smoke

The smoke is a plumbing check, far too small (<=4 pairs) to be statistically
conclusive - it does NOT decide E1. Under spec timing it is additionally degenerate
(section 3a). The early-tp smoke (section 3b) confirms the full metric +
kill-criterion machinery produces meaningful, differentiated numbers when the
disturbance actually lands mid-task.

**Smoke kill-criterion verdict: DOES NOT PASS** under either timing
(cond_gain = 0.0 pp; harm_reduction is 0/negative). On the early-tp smoke the
secondary signal actively favors replan_only over preempt_hold (preempt_hold has
~56% MORE stale-action harm), i.e. on this checkpoint HOLD looks worse than letting
the stale chunk ride. This is the opposite of the E1 hypothesis - worth taking
seriously, but on n=3 conditioning pairs it is only suggestive. A full 10x5 run at
an early t_p is needed before drawing any conclusion.

`full_replan` collapsing to ~0% is a genuine, reproducible finding: re-inferring a
fresh chunk every step and taking only its first action destabilizes this
checkpoint (it is trained for 50-step open-loop execution), so it is NOT a
useful upper bound for this policy as-is.

---

## 5. Correctness caveats

1. **t_p vs completion time (design issue, surface to PI).** On `libero_spatial`,
   successful episodes finish in ~30% of `max_steps`, and `t_p = 0.35*max_steps`
   lands after that. To make E1 measure anything on clean-success pairs, perturb
   before typical completion - recommend `--tp-frac ~ 0.12-0.18` (t_p ~ 35-50),
   or define t_p relative to a per-episode expected horizon rather than the global
   max-step cap. The harness exposes `--tp-frac` for exactly this.
2. **Conditional metric needs enough clean successes.** With ~30% clean success,
   10x5 gives ~15 conditioning pairs - usable but noisy. More seeds would tighten it.
3. **Macro-call equality** holds between replan_only and preempt_hold by
   construction (neither re-infers in the hold window; both replan once at t_p+L).
   open_loop may differ by +/-1 call depending on where t_p+L lands relative to a
   chunk boundary; the kill criterion only compares replan_only vs preempt_hold.
4. **HOLD definition** = zeros on the 6 pose dims + last gripper command, in
   NORMALIZED action space (the chunk space). After postprocessor/env_postprocessor
   un-normalization this is a near-stationary relative command (control_mode=relative).
5. **stale_action_harm** uses an extra `predict_action_chunk` per in-window step
   (only for replan_only/preempt_hold), which roughly doubles their per-step cost
   inside the measured horizon. Disable with `--no-stale-harm` for speed.
6. **Single in-process env**; results are not parallelized. Determinism relies on
   `set_seed` + fixed `episode_index`.

---

## 6. Exact command for the FULL E1 (10 tasks x 5 seeds)

Spec timing (matches the prompt; expect section 3a-style degeneracy on libero_spatial):
```bash
cd sim/libero_active
bash preempt/run_full_e1.sh
# equivalently:
source .venv/bin/activate
env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python preempt/e1_oracle_handover.py \
    --tasks 0,1,2,3,4,5,6,7,8,9 --seeds 1000,1001,1002,1003,1004 \
    --delta 0.05 --latency 8 --tp-frac 0.35 \
    --out preempt/e1_full_results.json
```

Recommended (perturbation actually lands mid-task - this is the run that
answers E1):
```bash
env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python preempt/e1_oracle_handover.py \
    --tasks 0,1,2,3,4,5,6,7,8,9 --seeds 1000,1001,1002,1003,1004 \
    --delta 0.05 --latency 8 --tp-frac 0.15 \
    --out preempt/e1_full_results.json
```

Runtime: ~720 s for the 20-rollout smoke (dominated by the 4 full_replan rollouts at
280 inferences each). The full 250-rollout grid is ~2.5-3 h; drop `full_replan` or
add `--no-stale-harm` to cut it substantially. The checkpoint under `results/` is
read-only and is never modified.
