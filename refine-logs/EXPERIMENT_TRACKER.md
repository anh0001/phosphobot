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
| random | libero_spatial | 1 | 5 -> 10.5%, 10 -> 31.0%, 15 -> 32.5%, 20 -> NaN | N=20 OOM'd (co-tenant process consumed 45 GiB on the visible GPU); re-launching alone |

## Outstanding blockers for the conformal/entropy methods — RESOLVED

The SmolVLA 227-vs-178 attention-mask mismatch was rooted in two missing pieces
of LeRobot's training pipeline: `delta_timestamps` (so the dataset returns
50-step action chunks instead of single actions) and `lerobot_collate_fn` (so
language tokens are padded to the model's max length). Both are now wired into
`policy_runner.episode_signals`. Standalone verification against the PS.4
checkpoint produced reasonable per-episode losses (0.02-0.04). The conformal
smoke (c42bff21) passed `[smoke] active loop OK (rounds=2)`.

## In-flight runs

- GPU 0: PS.5 random seed 1 re-launch (the previous attempt lost the N=20 round
  to an OOM when the conformal run on a sibling GPU spilled memory; both GPUs
  are now idle so this is solo).

## Real-Piper transfer (Stage B) — not started
