# Literature Landscape — Data-Efficient VLA Fine-Tuning for Single Piper

**Anchor**: Single AgileX Piper (6-DOF + gripper, ~$4k, supported by phosphobot via `piper_sdk`)
**Stack constraints**: phosphobot backend (FastAPI), PyBullet sim with shipped Piper URDF, ACT/Pi0/Gr00t inference, multi-modal teleop, Modal remote inference, consumer/single-GPU compute.
**Date**: 2026-05-22

## Themes (2024-2026)

### 1. Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA on VLA**: consumer-GPU (≤8 GB) deployment shown, e.g. *Towards Accessible Physical AI* (arXiv 2512.11921) — 200 demos, freezing vs. unfreezing vision encoder studied.
- **VLA-GSE**: generalized + specialized experts, only 2.51 % of params trainable, +6.3 pts vs. full FT.
- **OpenVLA-OFT**: parallel decoding + action chunking → 26× faster.
- **X-VLA**: soft prompts to encode embodiment, frozen backbone.

### 2. Few-Shot Adaptation
- **ControlVLA** (arXiv 2506.16211): ControlNet-style object-centric heads, 76.7 % success on diverse tasks with **10–20 demos**.
- **CO-RFT**: chunked offline RL transfers gripper→dexterous with **30–60 samples**.
- Most "data-efficient" papers still consume **100–200 real demos** on a single embodiment.

### 3. Sim-Real Co-training / RL Fine-Tuning
- **VLA-RFT** (arXiv 2510.00406): world-model simulator, ≤400 fine-tuning steps beats SFT.
- **ConRFT** (arXiv 2502.05450): BC+Q offline → consistency-policy online + HITL, 96.3 % in 45–90 min.
- **RL-Co** (Beyond Imitation, arXiv 2602.12628): warm-start SFT on mixed sim+real → RL in sim with auxiliary real loss.
- *Grounding Sim-to-Real* (arXiv 2603.22876): empirical study, mostly Isaac/Mujoco; PyBullet barely studied.

### 4. Cross-Embodiment Transfer
- **UniVLA** (HKU + OpenDriveLab) — validated on AgileX **Piper** as the new embodiment.
- **TrajBooster**: end-effector-trajectory transfer.
- **X-VLA**: soft prompts as embodiment tags.

### 5. Uncertainty / Failure Detection (runtime safety)
- **ReconVLA** (arXiv 2604.16677): CQR action-level uncertainty + SMD latent OOD detector.
- Conformal action prediction on token logits → calibrated confidence without retraining.
- **FAIL-Detect** (arXiv 2503.08558): sequential OOD detection, no failure data needed.
- These are framed as **runtime safety**, not as signals for *what demo to collect next*.

### 6. Human-in-the-Loop Refinement
- **Dual-Actor / Talk-and-Tweak** (arXiv 2509.13774): physical corrections → language commands → refinement actor.
- ConRFT HITL during online phase.
- Phosphobot ships leader-arm + Quest + gamepad teleop natively — none of these papers use a $4k arm + integrated multimodal teleop.

## Structural Gaps Identified

| # | Gap | Why phosphobot+Piper is well-positioned |
|---|-----|------------------------------------------|
| G1 | No clean **N-demos → success-rate** curve on a single low-cost 6-DOF arm with modern VLAs (Pi0 / SmolVLA / ACT / Gr00t) | Repo has trainable pipelines for all four, and one user can collect controlled demo budgets via the same teleop stack |
| G2 | **PyBullet sim → real Piper** pretraining is essentially untested in the VLA literature (everyone uses Isaac/Mujoco) | URDF + PyBullet integration already shipped; simulation/pybullet/ package exists |
| G3 | **Conformal / SMD uncertainty as an *active demo query signal*** (not just runtime fallback) is unexplored | Phosphobot can pause a rollout and trigger teleop takeover programmatically |
| G4 | **Teleop-correction-as-data** closed-loop (interrupt VLA → leader-arm takeover → write back as labeled correction → fine-tune) is alluded to but never benchmarked end-to-end with a single inexpensive arm | The exact endpoints (`/auto/start`, `/auto/stop`, `/recording/*`, leader-arm teleop) already exist; no paper has glued them |
| G5 | **Modal remote-inference latency** (150–300 ms) effect on policy success vs. action-chunk horizon — no systematic study | Phosphobot already supports Modal-based remote inference; an empirical study is low-risk and useful to the community |
| G6 | **Demo-quality estimation** when the same task is demonstrated via keyboard vs. gamepad vs. leader-arm vs. Quest — does input modality affect VLA learning? | All four modalities supported in one framework; no other repo offers this |

## Open Problems Worth Anchoring On

1. **Sample-complexity floor**: What is the *minimum* number of teleop demos a 6-DOF cheap arm needs to reach ≥80 % success on a class of tabletop tasks across {Pi0, SmolVLA, ACT, Gr00t}, with vs. without sim pretraining, with vs. without HITL correction?
2. **Active demo selection**: Can a conformal/SMD uncertainty signal *during VLA rollout* tell the operator *which next demonstration to collect* and outperform random demo collection?
3. **PyBullet free-play pretraining**: Can phosphobot-style PyBullet random play + procedural object generation provide a useful pretraining signal for SmolVLA/ACT that transfers to real Piper, with no Isaac-class compute?
4. **Modal-latency-aware action chunking**: How should chunk horizon adapt to network latency for remote VLA inference to maximize success?

> Auto-proceeding to Phase 2 with this landscape (AUTO_PROCEED=true).
