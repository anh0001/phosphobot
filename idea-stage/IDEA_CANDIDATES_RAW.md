# Raw Idea Candidates (Phase 2 brainstorm)

**Anchor**: Data-efficient VLA fine-tuning on a single AgileX Piper using phosphobot
**Brainstorm rule**: every idea must be doable by one user, one Piper, one consumer GPU (24-48 GB), in ≤2 weeks of wall time.

## Idea Pool (12)

### I1. UncertaintyQuery — Active demo collection via conformal action uncertainty
- **Pitch**: Use ReconVLA-style conformal action uncertainty during VLA rollout on Piper. When uncertainty spikes, **pause execution → request a teleop demo at exactly that state** via phosphobot's leader-arm. Train next round on those uncertainty-targeted demos.
- **Hypothesis**: 10 uncertainty-queried demos beat 30 random demos in success rate.
- **Phosphobot fit**: Native `/auto/stop` + leader-arm teleop + `/recording/start` already wired.
- **Risk**: Conformal calibration on small data is fragile; reproducing CQR head needs work.

### I2. PyBulletFreePlay-to-Piper — Cheap-sim pretraining for SmolVLA
- **Pitch**: Pretrain SmolVLA on procedurally generated PyBullet free-play (randomized objects, lighting, textures using the shipped Piper URDF), then fine-tune on ≤20 real Piper demos. Benchmark vs. cold-start fine-tuning.
- **Hypothesis**: PyBullet pretrain → ≥+15 pts success at N=10 demos.
- **Phosphobot fit**: `simulation/pybullet/` + Piper URDF + ACT/SmolVLA paths already exist.
- **Risk**: PyBullet rendering gap may dwarf the data-efficiency win; existing Isaac results may already dominate.

### I3. HITL-Correction-as-Data — Closed-loop teleop interrupt + retrain
- **Pitch**: During VLA rollout, operator presses a button → leader-arm takes over for the failing segment → corrected segment is automatically appended as labeled data → micro-LoRA update after every N corrections. Measure how the success-rate curve climbs vs. wall-clock.
- **Hypothesis**: 30 min of interactive correction beats 200 cold demos.
- **Phosphobot fit**: All endpoints exist; just glue logic + dataset writer.
- **Risk**: Talk-and-Tweak (2509.13774) and ConRFT cover much of this; novelty hinges on (a) single inexpensive arm, (b) micro-LoRA at correction granularity, (c) benchmark vs. cold demos.

### I4. ModalLatency × ChunkHorizon — Empirical study of remote-inference latency
- **Pitch**: Run Pi0/SmolVLA on Modal with artificial latency injection (0–500 ms). Sweep action-chunk horizon (4, 8, 16, 32). Measure success on N tabletop tasks. Derive a simple law / lookup table.
- **Hypothesis**: There's a knee — optimal chunk scales sub-linearly with latency; below 100 ms you can use short chunks; above 300 ms you need ≥16-step chunks.
- **Phosphobot fit**: Modal integration exists; phosphobot's action server is the natural harness.
- **Risk**: Borderline incremental; more workshop / blog-grade unless paired with adaptive chunking.

### I5. ChunkAdapt — Latency-adaptive action chunking
- **Pitch**: Extension of I4: train a tiny head that predicts current network RTT and dynamically sets the chunk horizon. Measure success vs. fixed chunks under variable network conditions.
- **Hypothesis**: Adaptive chunk beats best fixed chunk by ≥5 pts on average across latency regimes.
- **Phosphobot fit**: Slot-in head on top of any VLA + Modal client.
- **Risk**: Engineering-heavy; impact depends on real-world latency variance.

### I6. TeleopModalityFingerprint — How input device affects VLA learning
- **Pitch**: Collect the same tabletop tasks via {keyboard, gamepad, leader-arm, Quest}, same operator. Train SmolVLA on each demo set separately. Measure success, smoothness, generalization. Identify the highest-yield modality per demo.
- **Hypothesis**: Leader-arm > Quest > gamepad > keyboard at low N; gap shrinks at large N.
- **Phosphobot fit**: Unique — only this framework supports all four natively with the same backend.
- **Risk**: Operator variability dwarfs modality differences; one-person study is weak.

### I7. SimPretrainPlusActiveQuery — Combine I1 + I2
- **Pitch**: PyBullet pretrain → cold deploy → uncertainty-queried real demos. Two-axis ablation table.
- **Hypothesis**: Compound effect at N=5 — sim-pretrain + uncertainty query > either alone.
- **Phosphobot fit**: Same as I1+I2.
- **Risk**: Two confounds, harder to get crisp claim.

### I8. PiperKinematicJitterAug — Calibration-noise augmentation
- **Pitch**: Piper has known DH-offset / firmware variants (the driver itself has a `LEGACY_URDF` path). At training time inject FK noise simulating real Piper calibration drift. Test policy on a robot whose calibration was deliberately mis-set.
- **Hypothesis**: Calibration-noise aug → policy survives 1–3 cm DH drift without retraining.
- **Phosphobot fit**: Tight — uses phosphobot's exact failure mode.
- **Risk**: Narrow; reads as engineering not science.

### I9. SmolVLAOnConsumerGPU — Honest sample-complexity curves
- **Pitch**: Run SmolVLA fine-tuning on a single RTX 4090 / 5090 for {5, 10, 20, 50, 100} demos × {3 tabletop tasks} × {3 seeds}. Report the curve nobody has published cleanly for the cheap-arm regime. Combine with quick ablations (LoRA rank, frozen vs. unfrozen encoder).
- **Hypothesis**: The curve is concave and saturates by ~50 demos with LoRA-rank-16 on SmolVLA-450M.
- **Phosphobot fit**: Phosphobot is the natural data-collection + eval harness.
- **Risk**: Reads as benchmarking rather than research — but is *the* missing artifact for the cheap-arm community.

### I10. AutoDemoBudget — Stopping criterion for demo collection
- **Pitch**: Train a small regressor that, given current dataset statistics and a small held-out probe set, predicts marginal success gain from N→N+1 demos. Tell the user when to stop collecting.
- **Hypothesis**: Predicted marginal gain correlates ≥0.7 with measured gain; stopping rule saves ≥30 % of collection time.
- **Phosphobot fit**: Cleanly slots into the recording UI.
- **Risk**: Hard to validate without many task families.

### I11. CrossEmbodimentSimWarmstart — Train on SO-100 sim, deploy on Piper
- **Pitch**: Pretrain in PyBullet on the SO-100 URDF (which is far better-studied in LeRobot data), then transfer to real Piper via X-VLA-style soft prompts. Phosphobot has both URDFs.
- **Hypothesis**: SO-100→Piper soft-prompt transfer halves real demo cost.
- **Phosphobot fit**: Both URDFs shipped.
- **Risk**: Concurrent: UniVLA, X-VLA already studied cross-embodiment on Piper. Differentiation needs to be tight (e.g. sim-only source side).

### I12. FailDetect-as-Reward — Use FAIL-Detect to bootstrap RL fine-tuning without reward engineering
- **Pitch**: FAIL-Detect gives a sequential OOD signal. Convert it to a dense negative reward during online RL on Piper. No task-specific reward design.
- **Hypothesis**: OOD-as-reward gives ≥80 % of hand-engineered-reward sample efficiency on 2 tasks.
- **Phosphobot fit**: Phosphobot's safety stops make on-policy RL on real hardware tractable.
- **Risk**: Online RL on real cheap hardware is dangerous; needs careful workspace limits.

---

## First-Round Filter

| ID | Feasibility (1 arm, 1 GPU, 2 wk) | Conceptual novelty | Phosphobot leverage | Notes |
|----|-----------------------------------|--------------------|---------------------|-------|
| I1 | High | **High** | **High** | Plug into existing rollout loop |
| I2 | High | Medium | High | Risk of being dominated by Isaac results |
| I3 | High | Medium | **High** | Overlaps Talk-and-Tweak; differentiation must be sharp |
| I4 | High | Low | Medium | Workshop-grade alone |
| I5 | Medium | Medium | Medium | Better as Phase-2 of I4 |
| I6 | Medium | Medium | **High (unique)** | Single-operator confound |
| I7 | Medium | High | High | More moving parts than I1 alone |
| I8 | High | Low | Medium | Engineering, narrow |
| I9 | High | Low | High | Benchmarking, but actually *missing* |
| I10 | Medium | Medium | Medium | Hard to validate broadly |
| I11 | Medium | Medium | High | Overlaps UniVLA / X-VLA |
| I12 | **Low (safety)** | Medium | Low | Real-arm online RL is risky |

## Top 3 to Carry into Phase 3 (deep novelty)

- **I1 — UncertaintyQuery** (active demo collection via conformal action uncertainty)
- **I3 — HITL-Correction-as-Data** (closed-loop teleop interrupt + micro-update)
- **I7 — SimPretrainPlusActiveQuery** (compound: PyBullet pretrain + I1) — *kept as enriched variant of I1*

I2 and I9 held as backup ideas.
