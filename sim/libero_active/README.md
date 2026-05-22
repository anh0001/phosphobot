# Calibrated Demo Budgeting — LIBERO Active-Query Pipeline

Research code for **sample-efficient active demonstration selection for VLAs**.

- Proposal: `../../refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `../../refine-logs/EXPERIMENT_PLAN.md`
- Idea report: `../../idea-stage/IDEA_REPORT.md`

## What this is

During VLA rollout we compute **conformal action uncertainty** per timestep. When uncertainty
exceeds a calibrated threshold we query a corrective demonstration from that state, append it to a
LoRA fine-tuning buffer, and periodically retrain. The research question: does conformal-uncertainty
querying reach a target success rate with fewer demonstrations than random / entropy / kNN-coverage
/ human-gated selection?

**Path A**: main quantitative results in simulation on **LIBERO** (recognized benchmark);
generalization check on **ManiSkill3 built-in tasks**; real **Piper** arm = cross-embodiment
transfer evidence (separate, in `phosphobot/`).

## Layout

```
conformal_active/
  config.py          # experiment configs (dataclasses)
  uncertainty.py     # conformal action uncertainty: split-CQR + IQT (ConformalDAgger-style)
  query_methods.py   # 5 query strategies: random / entropy / knn / human-gated / conformal
  rollout.py         # LIBERO + SmolVLA rollout harness            (PS.1)
  oracle.py          # corrective-demo source: planner + demo pool  (PS.2)
  train.py           # SmolVLA LoRA fine-tune wrapper
  active_loop.py     # rollout -> trigger -> correct -> retrain     (PS.5)
offline_sanity.py    # PS.4 gate: collect -> train -> predict -> >=80%
run_sweep.py         # PS.6 main-results sweep (5 methods x suites x seeds)
scripts/setup_env.sh # one-shot environment install
configs/             # YAML run configs
results/             # metrics, curves, logs (gitignored)
```

## Setup

```bash
cd sim/libero_active
bash scripts/setup_env.sh        # creates .venv, installs lerobot + LIBERO + deps
source .venv/bin/activate
python -c "import lerobot, libero; print('ok')"
```

Environment: 2x RTX 6000 Ada (48 GB), Python 3.10 venv (LIBERO/robosuite pin to 3.10).

## Phase S milestones (see EXPERIMENT_PLAN.md)

| Step | Gate |
|------|------|
| PS.1 | SmolVLA produces rollouts on a LIBERO suite |
| PS.2 | Corrective oracle delivers a valid segment from any queried state |
| PS.4 | **Offline sanity**: fine-tune on ~40 demos -> >=80% success / 20 rollouts |
| PS.6 | **Main table**: conformal beats baselines on >=2 suites x >=3 seeds |
| PS.7 | Trend holds on ManiSkill built-in tasks |
