# Experiment Plan — Calibrated Demo Budgeting

**Linked proposal**: `refine-logs/FINAL_PROPOSAL.md`
**Hardware budget**: 2× AgileX Piper (leader + follower), 1× consumer GPU (RTX 4090/5090 or 24 GB equivalent), 1 operator.
**Wall-clock budget**: ~120 h (~25 h sim dev + 40 h real-arm + 40 h GPU + 15 h analysis).

## Workflow: Path A — Sim-Main Results, Real Piper as Transfer Evidence (read this first)

This project follows **Path A**: the main quantitative results are obtained in simulation; the
real Piper provides cross-embodiment transfer evidence. The contribution is framed as
**"sample-efficient active demonstration selection for VLAs"** — NOT human-effort reduction.

- In **sim**, corrective demos come from a scripted/oracle expert — no fatigue, no cost.
  Sim measures **active-query / sample efficiency**, which IS the (reframed) contribution.
- The **main results table** lives in Stage A on a recognized benchmark (LIBERO). Sim is the
  paper, so the benchmark must be reviewer-trusted — hence LIBERO, not a custom task.
- Stage B (real Piper) shows the **conformal-beats-random trend survives on hardware** — a
  small, clean cross-embodiment confirmation, not a full real-hardware matrix.

| Stage | Question answered | Role in the paper |
|-------|-------------------|-------------------|
| A (LIBERO sim) | Does conformal active-query beat random/entropy/kNN/human-gated on sample efficiency? | **Main results table** |
| B (real Piper) | Does the same trend survive on real hardware + a different embodiment? | Transfer-evidence section |

> **Why the human-effort claim was dropped:** in sim the expert is an oracle with no cost, so a
> sim-main paper cannot honestly claim "saves human effort." Reframing to "sample efficiency"
> makes the claim match the evidence and removes a near-automatic reject. (If you later want the
> human-effort claim, that is a separate, real-arm-main follow-up paper — Path B.)

**Conformal-transfer rule:** thresholds calibrated in sim do NOT transfer to real (visual domain
gap breaks exchangeability). On real hardware, recalibrate per task/camera, split calibration by
episode (not random frames), and recalibrate after every LoRA update. Report coverage/query-rate
drift over rounds.

**Trigger semantics:** calibrate the uncertainty trigger against *intervention need / failure
risk*, not raw action residual — high action variance can be harmless multimodality, low variance
can be confidently-wrong under OOD.

## Phase S — Simulation Stage (LIBERO primary) (~2.5 weeks, autonomous-friendly)

**Benchmark choice (revised — Path A requires reviewer-trusted sim):**

Because the **main quantitative results are now obtained in simulation**, the sim must be a
*recognized benchmark*, not a custom task suite. A custom Piper-in-ManiSkill task would read to
reviewers as "the authors built the world where their method wins."

- **PRIMARY: LIBERO** (robosuite / MuJoCo, Franka Panda). Standardized task suites, ships human
  demos, and SmolVLA is already benchmarked on it — a recognized anchor reviewers trust.
  Run via LeRobot (SmolVLA's native ecosystem).
- **SECONDARY: ManiSkill3 built-in tasks** (`PickCube-v1`, `StackCube-v1`, `PegInsertionSide-v1`,
  `PushCube-v1`) — shows the method is not overfit to LIBERO. Use the *standard* tasks, NOT a
  custom Piper suite.
- **APPENDIX only: custom Piper-in-ManiSkill** — engineering validation that the pipeline runs
  on the real embodiment; not core evidence.
- PyBullet stays a phosphobot smoke-test backend.

> **Embodiment note:** LIBERO uses a Franka arm; the real-transfer arm is a Piper. Codex confirms
> this is acceptable — the claim is "conformal uncertainty is a better active-query trigger,"
> which is embodiment-agnostic. Franka-sim + Piper-real is framed as a *cross-embodiment sanity
> check*, which is a strength, not a weakness, when stated honestly.

| # | Task | Acceptance | Where it lives |
|---|------|------------|----------------|
| PS.1 | Stand up LIBERO via LeRobot; get SmolVLA-450M running rollouts on a LIBERO suite (e.g. LIBERO-Object or LIBERO-Spatial) | SmolVLA produces actions; LIBERO eval harness reports success | `sim/libero_active/` (new) |
| PS.2 | Implement the corrective-expert. LIBERO is a live robosuite/MuJoCo sim, so corrections come from a scripted/planner oracle reset to the queried state; LIBERO's shipped human demos also serve as a demo pool for pool-based selection | Oracle/pool delivers a valid corrective segment from any queried state | `sim/libero_active/oracle.py` |
| PS.3 | Match observation/action schema to the LeRobot dataset format (also used by phosphobot for the real arm) | A LIBERO episode and a phosphobot episode load in the same training script | shared schema module |
| **PS.4** | **OFFLINE-PIPELINE SANITY (gate before any active-query work)**: collect demos → fine-tune SmolVLA-450M → deploy in LIBERO → evaluate. Proves *collect → train → predict → succeed* works at all. | **A SmolVLA fine-tuned on ~40 demos reaches ≥80 % success on one LIBERO suite over 20 eval rollouts.** If not, STOP — fix data format / training / inference first. | `sim/libero_active/offline_sanity.py` |
| PS.5 | Port the full active loop: rollout → conformal trigger → corrective-demo insert → resume → log | Loop runs unattended for 20-50 rollouts | reuse `phosphobot/uncertainty/`, `active_demo.py` |
| PS.6 | **Main-results sweep** — all 5 methods (random / entropy / kNN / human-gated-proxy / conformal) across ≥2 LIBERO suites, ≥3 seeds, shared query budget; end-to-end LoRA retrain | Sample-efficiency curves with conformal's advantage statistically visible | `sim/libero_active/run_sweep.py` |
| PS.7 | **Generalization check** — replicate the sweep on 2-3 ManiSkill3 *built-in* tasks | Trend holds on a second simulator → not LIBERO-overfit | `sim/maniskill_builtin/` |

**Phase S exit gate (all must hold):**
1. **PS.4 passed** — the offline pipeline produces a policy that *actually does the task* at
   ≥80 % success. (Core requirement: prove collect→train→predict works before hardware.)
2. The active-query loop runs **unattended for ≥20 rollouts** without manual restart.
3. **PS.6 main results** are reproducible: conformal beats random/entropy/kNN on ≥2 LIBERO
   suites with non-overlapping seed variance.
4. **PS.7** confirms the trend on ManiSkill built-in tasks (else: report LIBERO-only and note it).

Once green → proceed to Stage B (Phase 0). PS.1–PS.7 is the work an agentic coding tool can
drive autonomously. PS.6 *is the paper's main table* — treat it as such.

> **Honest scope of what Phase S buys you:** passing this gate de-risks the entire *software
> pipeline* (data format, training, inference, the loop, the math). It does NOT de-risk the
> *sim-to-real visual/dynamics gap* — that is irreducible and is handled in Stage B by real-data
> recalibration + a small real fine-tune. Sim removes most of the trouble, not all of it.

## Phase 0 — Engineering Prerequisites (Stage B, real arm) (3-5 days)

| # | Task | Acceptance | Where it lives |
|---|------|------------|----------------|
| P0.1 | Verify two-Piper leader-follower teleop end-to-end in phosphobot (`leader_follower.py`); the limp leader Piper has NO gravity compensation (only SO-100 supports it) — add a counterweight rig or a small Piper gravity-comp routine | Record + replay a 30 s trajectory cleanly; operator can teleop ≥5 min without excessive fatigue | optional `phosphobot/hardware/piper.py` gravity-comp PR |
| P0.2 | Stand up SmolVLA-450M LoRA fine-tune script (or use `huggingface/VLAb`) on the lab GPU | 5-demo fine-tune completes in < 20 min | `inference/smolvla_lora/` (new) |
| P0.3 | Wire SmolVLA inference into phosphobot's existing remote-inference path | `/auto/start` runs SmolVLA at ≥10 Hz | `inference/smolvla_lora/server.py` |
| P0.4 | Implement conformal action-uncertainty hook on top of SmolVLA action expert (flow-matching outputs → predicted variance via CQR head, or ensemble proxy) | Per-step scalar uncertainty stream exposed via WebSocket | `phosphobot/uncertainty/` (new) |
| P0.5 | Implement teleop-takeover trigger: WebSocket-driven `/auto/stop` when uncertainty > τ, then leader-arm dataset write | Hand-trigger a stop, recover, append, verify dataset file | `phosphobot/endpoints/active_demo.py` (new) |

**Pilot gate**: at end of Phase 0, run a 5-min closed-loop on Block-pick-and-place from a 5-demo seed. If the loop doesn't survive 5 minutes without manual restart, fix infra before proceeding. (PILOT_MAX_HOURS = 2.)

## Phase 1 — Seed Datasets and Backbone Sanity (1 day)

Run 1: collect **40 random demos** on Block-pick-and-place. Use this single seed set for:
- B-Rand reference run (subsample to N∈{5,10,20,40}).
- Sanity-check SmolVLA fine-tune curve.

Decision point: if SmolVLA at N=40 doesn't exceed 50 % success on Block-pick, escalate to ACT-50M and report finding. Don't proceed to harder tasks until Block works.

## Phase 2 — Method Sweep (8-10 days, 3 tasks × 3 methods × 3 seeds + B-Conf)

Run order (lowest-risk first → harder):

| Block | Task | Method | N targets | Seeds |
|-------|------|--------|-----------|-------|
| B1 | Block-pick-and-place | B-Rand | 5,10,20,40 | 3 |
| B2 | Block-pick-and-place | B-Ent | 5,10,20,40 | 3 |
| B3 | Block-pick-and-place | B-kNN | 5,10,20,40 | 3 |
| B4 | Block-pick-and-place | **B-Conf** | 5,10,20,40 | 3 |
| B5 | Stack-cups | B-Rand | 5,10,20,40 | 3 |
| B6 | Stack-cups | B-Conf | 5,10,20,40 | 3 |
| B7 | Towel-fold-corner | B-Rand | 5,10,20,40 | 3 |
| B8 | Towel-fold-corner | B-Conf | 5,10,20,40 | 3 |
| B9 | (stretch) Stack-cups | B-Ent, B-kNN | 5,10,20,40 | 2 |
| B10 | (stretch) Towel-fold | B-Ent, B-kNN | 5,10,20,40 | 2 |

Evaluation: 20 rollout trials per (method, task, N) cell.

**Add a 5th method — B-HumanGated**: operator intervenes whenever *they* judge it necessary
(no uncertainty trigger). This is the honest real-world competitor; run it on Block-pick + 1 more task.

## Phase 3 — Stopping-Rule Training and Held-Out Test (2 days)

- Use measurements from B1-B8 to train a small (~10-param) marginal-gain regressor on features `(N, mean_uncertainty_on_probe, mean_conformal_width)`.
- **Held-out evaluation**: Drawer-open-and-place. Run B-Conf, apply the trained stopping rule live during data collection. Compare against:
  - "stop at the empirical optimum" (oracle).
  - "stop at N=40" (collect-everything baseline).
- **Statistical-power caveat**: one held-out task is underpowered. Either evaluate
  leave-one-task-out across several task *variants* (object/position randomizations), or
  demote the stopping rule from a primary claim (C3) to a secondary qualitative result.

## Phase 4 — Required Ablations (2 days)

| Ablation | Tests | Cost |
|----------|-------|------|
| A1: Drop IQT (vanilla split conformal) | W4 robustness claim | 1 task × 2 seeds |
| A2: Drop calibration entirely (raw uncertainty threshold) | C4 calibration value | 1 task × 2 seeds |
| A3: ManiSkill sim warmup vs. cold-start | I7 sub-question | 1 task × 2 seeds |
| A4: ACT-50M backbone | Backbone robustness | 1 task × 2 seeds (Block only) |
| A5 (stretch): Pi0 backbone | Scale generalization | 1 task × 1 seed |

## Phase 5 — Safety & Reporting (1 day)

- Log every hardware event during Phase 2-3: stalls, collisions, e-stops.
- Joint-velocity hard limit: 1.0 rad/s; torque-trip threshold from `piper.py`.
- Report total real-arm hours, zero-incident verification (C5).

## Deliverables

1. `idea-stage/IDEA_REPORT.md` — final report (Phase 5 of workflow).
2. `refine-logs/EXPERIMENT_TRACKER.md` — live run log (to be filled by `/run-experiment`).
3. `phosphobot/uncertainty/` + `phosphobot/endpoints/active_demo.py` — extension PR.
4. `sim/libero_active/` — the main-results codebase (LIBERO active-query pipeline).
5. Dataset release on HuggingFace: LIBERO active-query buffers + real Piper transfer demos.
6. Paper: **Path A** — sim-main, target ICRA / IROS / RA-L / CoRL workshop. Contribution =
   "sample-efficient active demonstration selection for VLAs", main results on LIBERO,
   cross-embodiment transfer evidence on a real Piper.

## Total Compute Budget

| Item | Est. |
|------|------|
| Phase S simulation (LIBERO main + ManiSkill secondary) | ~2.5 weeks (agentic-tool-driven, mostly unattended) |
| Phase 0 engineering (real arm) | 3-5 days |
| Phase 1 seed | 1 day |
| Phase 2 real-transfer sweep (reduced — transfer evidence only) | 4-6 days |
| Phase 3 stopping | 2 days |
| Phase 4 ablations | 2 days |
| Phase 5 writeup | 1 day |
| **Total** | **~5 weeks**, 1 person (Stage A ~2.5 wk + Stage B ~2.5 wk) |

> Stage B is now lighter than the original plan: the real Piper produces *transfer evidence*
> (the conformal-beats-random trend survives on hardware), not the main quantitative table.
> Phase 2's real-arm sweep can be scoped down to 1-2 tasks × 2 methods (conformal vs random) ×
> 2 seeds — enough to show transfer without a full real-hardware matrix.

## First Runs to Launch (sim-first order)

1. **PS.1–PS.3** (LIBERO via LeRobot + SmolVLA rollouts + corrective-expert + schema match — pure sim, autonomous-friendly).
2. **PS.4** (offline-pipeline sanity: collect→train→predict→≥80 % success on a LIBERO suite) — the make-or-break checkpoint.
3. **PS.5–PS.6** (full active loop + 5-method main-results sweep on LIBERO) → must clear the **Phase S exit gate**.
4. **PS.7** (generalization check on ManiSkill built-in tasks).
5. **P0.1–P0.3** (real-arm engineering prereqs) — only after Phase S is green.
6. **B1 / B4** (reduced real-Piper transfer sweep: conformal vs random, 1-2 tasks) — transfer evidence.

## Off-Ramps

- **PS.4 gate (offline sanity)**: if a SmolVLA fine-tuned on clean sim oracle demos can't reach
  ≥80 % success *in sim*, the whole approach is unproven — no active-query cleverness can rescue a
  collect→train→predict chain that doesn't work. Fix data format / training / inference here.
  This is the single cheapest, most important checkpoint in the project.
- **Phase S gate**: if the sim loop can't produce reproducible curves OR conformal fails to beat
  random/entropy/kNN *even with a perfect oracle*, the algorithm is broken — fix it in sim before
  spending any real-arm time. This is the cheapest place to fail.
- After B1+B4 if **B-Conf at N=20 fails to beat B-Rand at N=20**, pause and revisit. Likely culprits: bad real-data calibration, weak uncertainty signal, or fine-tune instability.
- After Phase 3 if **stopping-rule error > 10 demos on held-out**, downgrade C3 from claim to qualitative observation, keep C1+C2 as paper.
