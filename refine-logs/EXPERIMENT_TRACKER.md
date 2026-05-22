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
| PS.4 offline sanity | RUNNING | gate run: 40 demos, 20k steps, libero_spatial, GPU 0 — ~6 h ETA |
| PS.5 active loop | CODE WRITTEN | `active_loop.py` + `policy_runner.py`; needs live debug after PS.4 |
| PS.6 main sweep | PENDING | blocked on PS.4 gate + PS.5 live validation |

## Pipeline smoke test (PS.4 --smoke)

| Run | demos | steps | pc_success | verdict | note |
|-----|-------|-------|-----------|---------|------|
| ps4_smoke | 4 | 200 | 0.0% | pipeline OK | success not gated (200 steps only); train→eval→parse all execute |

## PS.4 gate run (in progress)

| Run | suite | demos | steps | pc_success | gate (≥80%) |
|-----|-------|-------|-------|-----------|-------------|
| ps4_offline_sanity | libero_spatial | 40 | 20000 | (running) | (pending) |

## Integration bugs fixed during bring-up

1. lerobot 0.5.2 requires Python ≥3.12 (venv was 3.10)
2. `lerobot-train` refuses a pre-existing `--output_dir`
3. `--dataset.episodes` JSON must be space-free
4. `--policy.push_to_hub=false` needed (else demands a hub repo_id)
5. LIBERO first import blocks on interactive `input()` — pre-seed `~/.libero/config.yaml`
6. `num2words` is an undeclared SmolVLM-processor dependency
7. eval metrics live under `eval_info.json["overall"]`, not `["aggregated"]`

## Real-Piper transfer (Stage B) — not started
