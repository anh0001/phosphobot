# Experiment Tracker

Branch: `research/conformal-active-demo-budgeting`
Code: `sim/libero_active/`

## Phase S progress

| Step | Status | Evidence |
|------|--------|----------|
| PS.0 env setup | DONE | venv (py3.12, torch2.11+cu128, 2×RTX6000), lerobot 0.5.2, LIBERO — all import |
| PS.3 conformal core | DONE | 15/15 unit tests pass (`tests/test_core.py`) |
| PS.1 rollout harness | DONE | smoke test: SmolVLA rollouts on libero_spatial via `lerobot-eval`, success parsed |
| PS.2 oracle | DONE (code) | `oracle.py` pool-based selection; live integration validated within PS.5 |
| PS.4 offline sanity | **PASSED** | single-task gate, 30 demos × spatial task 0, 15k steps → **80.0%** ≥ 80% gate |
| PS.5 active loop | CODE WRITTEN | `active_loop.py` + `policy_runner.py`; needs live debug after PS.4 |
| PS.6 main sweep | PENDING | blocked on PS.4 gate + PS.5 live validation |

## Pipeline smoke test (PS.4 --smoke)

| Run | demos | steps | pc_success | verdict | note |
|-----|-------|-------|-----------|---------|------|
| ps4_smoke | 4 | 200 | 0.0% | pipeline OK | success not gated (200 steps only); train→eval→parse all execute |

## PS.4 gate runs

| Run | suite | task | demos | steps | pc_success | gate (≥80%) | note |
|-----|-------|------|-------|-------|-----------|-------------|------|
| v1  | libero_spatial | (all 10) | 40 | 20000 | 0.0% | FAIL | trained on libero_10 by accident (combined-dataset suite bug); train/eval scope mismatch |
| v2  | libero_spatial | 0 only | 30 | 15000 | **80.0%** | **PASS** | suite-aware episode selection + single-task scope (5h43m wall) |

## Integration bugs fixed during bring-up

1. lerobot 0.5.2 requires Python ≥3.12 (venv was 3.10)
2. `lerobot-train` refuses a pre-existing `--output_dir`
3. `--dataset.episodes` JSON must be space-free
4. `--policy.push_to_hub=false` needed (else demands a hub repo_id)
5. LIBERO first import blocks on interactive `input()` — pre-seed `~/.libero/config.yaml`
6. `num2words` is an undeclared SmolVLM-processor dependency
7. eval metrics live under `eval_info.json["overall"]`, not `["aggregated"]`

## PS.5 single-cell runs

| Method | Suite | Seed | Curve (n_demos -> pc_success) | Note |
|--------|-------|------|-------------------------------|------|
| random | libero_spatial | 0 | 5 -> 9.0%, 10 -> 22.5%, 15 -> 33.0%, 20 -> 35.0% | First real active-loop curve; monotonic, diminishing returns by N=20 |
| conformal | libero_spatial | 0 | 5 -> 9.0%, 10 -> 21.0%, 15 -> 24.5%, 20 -> 28.0% | First real conformal curve; monotonic but flatter than random seed 0 — single seed, not conclusive |
| random | libero_spatial | 1 | 5 -> 10.5%, 10 -> 31.0%, 15 -> 32.5%, 20 -> 40.5% | First attempt OOM'd at N=20 (GPU co-tenant); solo re-run completed cleanly. Second seed of the random baseline. |
| conformal | libero_spatial | 1 | 5 -> 10.5%, 10 -> 16.0%, 15 -> 18.5%, 20 -> 25.0% | Second conformal seed; **flatter than seed 0 and below random at every N>=10**. |
| entropy | libero_spatial | 0 | 5 -> 9.0%, 10 -> 15.5%, 15 -> 25.5%, 20 -> 29.0% | First entropy curve; mid-pack, beats conformal at N=15/20, still trails random. |

## Outstanding blockers for the conformal/entropy methods — RESOLVED

The SmolVLA 227-vs-178 attention-mask mismatch was rooted in two missing pieces
of LeRobot's training pipeline: `delta_timestamps` (so the dataset returns
50-step action chunks instead of single actions) and `lerobot_collate_fn` (so
language tokens are padded to the model's max length). Both are now wired into
`policy_runner.episode_signals`. Standalone verification against the PS.4
checkpoint produced reasonable per-episode losses (0.02-0.04). The conformal
smoke (c42bff21) passed `[smoke] active loop OK (rounds=2)`.

## In-flight runs

- None.

## Method comparison (libero_spatial, max_demos=20)

| N  | random s0 | random s1 | conformal s0 | conformal s1 | entropy s0 |
|----|-----------|-----------|--------------|--------------|------------|
| 5  | 9.0%      | 10.5%     | 9.0%         | 10.5%        | 9.0%       |
| 10 | 22.5%     | 31.0%     | 21.0%        | 16.0%        | 15.5%      |
| 15 | 33.0%     | 32.5%     | 24.5%        | 18.5%        | 25.5%      |
| 20 | 35.0%     | 40.5%     | 28.0%        | 25.0%        | 29.0%      |

Random (mean@N=20 = 37.75%) beats both uncertainty methods (conformal mean
26.5%, entropy 29.0%) by ~9-12 absolute success points. Same paired eval init
states across methods (seed = 1000+run_seed) so the gap is not eval noise.

## Conformal-scoring bug (found 2026-05-28)

External code review + offline selection-trace audit (per codex GPT-5.2's
"invariant check": rank should equal entropy's if conformal score is just
loss/qhat globally) revealed a real bug at `active_loop.py:77-82`:

```
elif method.name == "conformal":
    method.uncertainty.add_calibration(sig.loss_mean)
    raw = sig.loss_mean
    scores[idx] = method.uncertainty.normalized(raw) if method.uncertainty.is_calibrated else raw
```

Three compounding issues:
1. **Self-referential calibration.** Each candidate's own loss is added to the
   calibration buffer *before* it is scored. The candidate is normalized against
   statistics that already include itself.
2. **Buffer reset every round.** `ConformalQuery.reset_round` clears `_calib`.
   With `calib_min=20` and `max_candidates_per_round=30`, candidates 0..19 are
   returned as raw loss (typical 0.02-0.04) while candidates 20..29 are returned
   as `loss/qhat` (typical 0.5-1.5). The two scales differ by ~20-30x, so the
   top-5 selection is dominated by whichever candidates happen to be iterated
   after calibration kicks in.
3. **Iteration order = sorted episode index.** `rng.choice` is followed by
   `sorted(picked_idx)`. Combined with (2), conformal is structurally biased
   toward picking the *highest-indexed candidates in the random subsample*.

Empirical confirmation: seed-0 N=5->10, conformal new picks
{1394, 1447, 1458, 1493, 1523} vs entropy new picks
{1394, 1458, 1567, 1574, 1579} — overlap 2/5 despite both methods being driven
by the same `sig.loss_mean`. Rank invariant is violated.

Implication: the published methods comparison so far is between random vs
*broken conformal* and entropy vs broken conformal. The conformal cells need
to be re-run after the bug fix before any direction-claim is defensible.

## Codex critical-read (2026-05-28)

External GPT-5.2 review, given the table above:
- "Uncertainty-beats-random" is *not* a safe prior for VLA + LoRA + 5..20 demos.
  Random's main contribution is task/layout coverage; loss-based uncertainty
  selects outliers/long-trajectory/contact-heavy episodes that don't improve
  sample efficiency.
- Conformal calibration provides coverage guarantees for a chosen nonconformity
  score; it does not make the score useful for acquisition. For a flow-matching
  action head, teacher-forced loss is a weak acquisition signal; the more
  defensible signal is action-chunk dispersion, ideally constrained by task
  coverage or embedding diversity.
- Bet on the cleanest publishable direction: "naive loss uncertainty is
  anti-informative in low-demo VLA adaptation; need diversity + calibrated
  epistemic disagreement". *Not* "conformal once debugged" — fixing the bug
  will likely make conformal rank-equivalent to entropy.

## Real-Piper transfer (Stage B) — not started
