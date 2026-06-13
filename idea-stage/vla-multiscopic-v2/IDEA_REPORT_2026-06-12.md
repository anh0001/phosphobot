# Robotics Idea Discovery Report — vla-multiscopic-v2

**Direction**: Pivot the dead "VLA × multi-scopic preemption" idea into a new publishable, experiment-backed
idea — still genuinely inspired by Saputra et al. 2022's multi-scopic primitives (the unused ones), built on the
keystone mislocalization evidence, novel vs the Jan–Jun 2026 wave, feasible on SmolVLA + LIBERO + one A6000.
**Date**: 2026-06-12 · **Branch**: `idea/vla-multiscopic-cognition`
**Pipeline**: frame → research-lit (8-angle scite+web survey, 98 papers) → idea-creator (5 lenses × 3 judges →
top-3 synthesis) → novelty-check (5 adversarial hunts) → research-review (GPT-5.5 xhigh, hostile AC persona)
**Artifacts**: `_problem_frame.md` · `_feasibility_notes.md` · `_survey_digest.md` · `_candidates_v2.md` ·
`_novelty_check_v2.md` · this report.

---

## Robotics Problem Frame (condensed)

- **Embodiment / policy class**: 7-DoF Franka (LIBERO sim); chunked flow-matching VLA (SmolVLA, 50-step chunks,
  relative eef actions).
- **Task family**: tabletop pick/place (libero_spatial, libero_object) under controlled object displacement,
  episode-start and mid-rollout.
- **Assets**: kubotaserver2 (A6000); custom displacement harness `sim/libero_active/preempt/` (manual chunk
  control, qpos nudge with known d, mislocalization probe, RNG hygiene); LoRA fine-tunes feasible.
- **Constraints**: pilots ≤1 GPU-day; full study ≤1 GPU-week; sim-only; no resurrecting the falsified
  preemption/replan-timing branch; portfolio-orthogonal to the "Unreliable by Default" paper.
- **Keystone (ours, unpublished)**: after a 5cm displacement, the arm reaches the CANONICAL object location, not
  the displaced one (93% closer-to-canonical, failures stop ~5.4cm short ≈ the displacement vector) — a
  structured, low-dimensional, exploitable bias. Qualitative priority is LOST (AFI/LIBERO-PRO/Text-Latent);
  quantitative attribution is fully open.

---

## ✅ RECOMMENDED — Idea #1 (FLAGSHIP, collapsed per external review)

### "The Failure Is a Vector: Displacement-Relative Reach Fields and Causal Re-Anchoring for VLA Policies"

**One-liner**: A frozen VLA's object-displacement failure collapses to one low-dimensional quantity — the
canonical-mismatch vector ≈ the injected displacement — measured as the first *displacement-relative* continuous
reach-endpoint error field (episode-start AND mid-rollout), and proven causally sufficient by a sham-controlled
oracle virtual-frame re-anchoring probe whose action-side exactness is guaranteed by a known equivariance fact
(Wang et al. 2505.13431, Props 1–2) deployed test-time on a frozen third-person VLA for the first time.

**Headline claim (reworded per novelty check)**: Under controlled object displacement (2–10cm × direction ×
timing × task × checkpoint), chunked flow-matching VLAs exhibit a canonical-prior reach whose endpoint-error
field quantitatively recovers the displacement vector (DVR statistic: median cos(e,−d) and magnitude-transfer
slope, cluster-bootstrapped); an oracle re-anchoring that neutralizes only object position (agentview translate
+ proprio offset; identity action map; sham d=0 / random-direction / wrong-object controls) restores the
majority of the displacement-induced endpoint error and conditioned-success gap → mislocalization, not execution
or timing, is the dominant causal bottleneck. The warp is a probe, never a method.

**Contributions (final wording discipline — "first DISPLACEMENT-RELATIVE characterization", never "first to
show"; phenomenon credit to AFI 2512.07472; lemma credit to 2505.13431)**:
1. The reach-field probe + DVR statistic — released as a metric LAYER running unchanged on LIBERO-PRO and HELM.
2. Sham-controlled causal-sufficiency probe via oracle virtual-frame re-anchoring (restoration, where prior
   intervention work ablates/masks = necessity-only; combination unclaimed per novelty check C3: CLEAR_WITH_CITATION).
3. Detection→ESTIMATION evidence layer (E7-lite, log-only): FK-endpoint-of-chunk vs object as a calibrated
   VECTOR estimate of d + wasted-action lead time; internal-detector blindness comparison
   (C4: CLEAR_WITH_CITATION vs ProbeAct 2606.09740 / ReconVLA 2604.16677).

### Core evidence package (reviewer-collapsed; main paper = E0 + E1r + E2r + E7-lite)

| Exp | Content | GPU-h |
|---|---|---|
| E0 (GATE) | Strong checkpoint (public standard SmolVLA-LIBERO FT or train to ≥65% clean) + keystone replication: closer-to-canonical ≥60%, DVR cos ≥0.4 — else STOP (weak-checkpoint artifact) | 24–30 |
| E1r | Reach-field, strong ckpt, libero_spatial: 4 mag × 4 cardinal dir + 5cm diagonal spot-check + DEPTH-AXIS arm; episode-start full grid, mid-rollout @5cm; pre-registered endpoint definitions (pre-contact / closest-approach / final); task×seed cluster bootstrap | ~35 |
| E2r | Causal probe @5cm × 4 dir: {no-warp, full oracle, image-only, proprio-only, sham d=0, random-direction, wrong-object, clean-scene-warp, border-fill ablation}; PRIMARY = endpoint-error reduction; success restoration secondary, paired, strong-ckpt only | ~30 |
| E7-lite | Log-only re-analysis: FK-mismatch vector vs {flow dispersion, STAC, embedding-density, one action-stat monitor}; calibration curves cos(m,d), |m| vs |d|; framed as diagnosis evidence, zero "reliability" vocabulary | ~5 |

Deferred (appendix / rebuttal / follow-up): E3 INT-ACT adjudication; E4 visual-token patching bridge (novelty-
crowded, foreground 2603.19233 if kept); E5 pose-LoRA removal (instrument, not contribution); E6 portability
(one HELM or LIBERO-PRO smoke test only if claiming "metric layer"); AFI head-to-head (only if any
method-superiority claim is made — the probe framing avoids it).

**Core total ≈ 95–100 GPU-h** (within 1 GPU-week with margin; reviewer distrusted the original 150–165h plan).

### Decisive pilot (≈23 GPU-h, weak ckpt OK for signal discovery only)
- (a) **Field grid first** (~13 GPU-h): δ∈{2.5,5,7.5,10}cm × 4 dir, episode-start, libero_spatial 10×4–5 seeds,
  ~800 rollouts → ~200 endpoint vectors; log everything once (latents, flow samples, STAC, dispersion, kinematics).
  **K1 (continue)**: median cos ≥0.55 with bootstrap CI clearly above chance, slope 0.5–1.4, R² ≥0.25,
  closer-to-canonical ≥70% — cluster bootstrap by task×seed, NOT pooled-IID. Kill cost: 13 GPU-h.
- (b) **Mid-rollout + warp arms** (~10 GPU-h): mid-rollout @5cm×4dir; warp arms {full, image-only, proprio-only,
  sham, random-direction}. **K2 (pilot)**: PRIMARY = endpoint-error reduction (≥50% green); binary restoration
  only directional on weak ckpt; true-warp must beat sham+random by pre-registered margin. Red verdict requires
  strong ckpt, n≈80–100 paired cases, CI excluding useful restoration.
- **No arXiv drop after the weak-ckpt pilot** (reviewer override of the synthesis): preprint only after
  strong-checkpoint E0 + E1r/E2r with clean controls — a weak pilot preprint hands the scoop-watch groups the
  missing measurement.

### Kill criteria (revised)
- K1 (pilot-a, above) → kill paper at 13 GPU-h.
- K-A (E0): strong-ckpt keystone fails → stop before atlas spend; workshop salvage.
- K2 (paper claim): endpoint-correction primary; success-restoration McNemar at n≥80 strong-ckpt;
  paper-grade DVR bar: median cos ≥0.65 (lower CI ≥0.5), slope 0.7–1.3, R² ≥0.4, consistent across most tasks.
- K3: DVR fails on libero_object on strong ckpt → workshop downgrade.
- K6 (scoop watch, weekly): continuous endpoint-error-vs-displacement appears → re-pivot headline to
  adjudication/bridge legs (E3/E4 move from appendix to spine).

### Multi-scopic story (genuine, placement per reviewer: discussion paragraph at ICLR; one-paragraph lens at CoRL — never title/abstract/contributions)
- Intero/exteroceptive coordination (Saputra P-spine) = the diagnosis itself: canonical-prior reach is a measured
  failure of exteroception to override an interiorized prior; DVR is the unreconciled residual AS A VECTOR.
- Meso-level LER (P6/P7) = the oracle re-anchoring probe: a meso-scale world-state estimate re-anchors the micro
  level's egocentric inputs (image translate + proprio offset), action map = identity by the relative-action
  equivariance fact.
- Affordance-effectivity fit (P4) = the E7 estimation layer: FK endpoint of the predicted chunk (effectivity) vs
  perceived object pose (affordance), as a SIGNAL, not an interrupt.
- P3 (density-follows-attention) maps to the deferred E4 token bridge.

### Novelty status (post-hunt, 6.5/10 overall)
C1 PARTIAL_COLLISION (AFI owns protocol+phenomenon; vector field + DVR survive) · C2 theory KILLED → reworded as
known-fact-novel-deployment (2505.13431) · C3 CLEAR_WITH_CITATION (VLA-Trace masks=necessity; we restore=
sufficiency+controls) · C4 CLEAR_WITH_CITATION (ProbeAct belief-vector ≠ chunk-FK; "calibrated VECTOR" wording) ·
C5 PARTIAL_COLLISION (2603.19233 foregrounded; deferred anyway). Tier-1 mandatory citations: 2505.13431,
2307.03659, 2409.12894, 2605.30117, 2606.09740, 2605.28726, 2604.16677, 2507.01723, 2407.01812, 2603.05487,
2509.00328, 2603.19183 (+Tier-2 list in `_novelty_check_v2.md`). Scoop-watch: AFI authors (highest), ProbeAct,
Not-All-Features, VLA-Trace, Northeastern equivariance line, AHEAD, Robust-Skills, HELM, Stanford interp,
proprio-encoding line — standing queries listed in `_novelty_check_v2.md`.

### External review (GPT-5.5 xhigh, hostile AC persona)
Novelty 6 · Significance 6 · Soundness 6.5 · Feasibility 4.5 (pre-collapse) · Clarity 5.5.
ICLR 2027: MAJOR-REVISION trajectory, ~25% as-scoped → **35–40% collapsed** with strong results + disciplined
wording. CoRL 2027: ~20% diagnosis-only → 30–35% if #2 activates legitimately. Single most valuable change
(APPLIED above): collapse to reach-field + sham-controlled oracle re-anchoring on a strong checkpoint.

### Risks
(1) Scoop velocity — mitigated by 13 GPU-h K1, weekly watch, no-early-preprint discipline, strong-ckpt-first
ordering. (2) Warp artifacts — full control battery in E2r; if sham moves success → inpainting engineering
(+1 wk). (3) Strong ckpt may shrink the bias — either direction is a finding (emergence axis), E0 gates spend.
(4) Single-model scope — explicit scoping to chunked flow-matching VLAs; thin second-model arm only if budget.
(5) Sim-only — positioned as science-of-failure with controlled sim displacement as the instrument.

---

## Ranked alternates

### #2 (UPGRADE PATH) — "Measuring, Cancelling, and Exploiting the Canonical-Prior Reach Bias"
Activation: pilot oracle restoration ≥50% AND tracker spot-check ‖d̂−d‖ ≤2cm AND willingness to make the AFI
fairness fight load-bearing. Adds: tracked deployable re-anchoring + grounded preemptive gate + head-to-head vs
AFI/VLS/Eq.Bot. Ladder-of-claims wording (each rung independently falsifiable). ~150 GPU-h + 2 wks eng.

### #3 (CONTINGENT) — "The Missing Meso Level"
Activation: #1 pilot green AND E2r component arms separate well above noise; else it is one discussion paragraph
in #1. Highest multi-scopic fidelity (the full Saputra transplant as a training-free meso loop), but brushes the
survey-KILLED S3 cell (AFI+HELM) and carries the system-paper reject pattern. CoRL framing only.

## Eliminated
- **Method-first "Move the World, Not the Weights"** — scoop_resistance 3/3/4; 70% bar undecidable at n=14;
  fate hangs on a contestable self-ported AFI head-to-head. Lemma + validity map grafted into #1.
- **Detector-first "Failure Has a Direction"** — pilot cannot fail (rubber-stamp gates); "privileged perception
  trivially wins" dismissal; heaviest reimplementation load; bleeds into "Unreliable by Default" identity.
  Estimation framing + lead time + complementarity matrix grafted into #1's E7-lite.
- **S3 standalone (persistent memory re-anchoring)** — killed at survey by AFI 2512.07472 + HELM 2604.18791.
- **S5 standalone (pose-token method)** — killed at survey by Pose-VLA/ST4VLA/SG-VLA; survives only as the
  deferred E5 instrument.
- **Anything in the falsified timing/preemption branch** — remains dead; no retrieved paper resurrects it.

## Evidence package summary (for downstream /experiment-plan)
- Required baselines: sham/random-direction/wrong-object/clean-scene warp controls (E2r);
  internal-detector set on logs (E7-lite); AFI only if method claims appear.
- Required metrics: DVR cosine + slope (cluster bootstrap), closer-to-canonical fraction, endpoint-error
  reduction (primary causal metric), paired conditioned success (secondary), calibration curves, wasted-action
  lead time; per-chunk wall-clock.
- Required failure analyses: warp-validity boundary (depth-axis, relational tasks, border artifacts, staleness),
  per-task DVR heterogeneity, endpoint-definition sensitivity.
- Real-robot evidence: NOT required (science-of-failure framing); sim displacement is the instrument.

## Next steps
- [ ] Implement `oracle_warp` + `reach_field` variants in `sim/libero_active/preempt/` (camera-matrix pixel
      projection; endpoint logging at 3 pre-registered definitions; log-once-analyze-many).
- [ ] Run pilot (a) field grid (13 GPU-h) → decide K1 overnight.
- [ ] If K1 passes: pilot (b) warp arms (10 GPU-h) → K2 bands select #1 vs #2 framing.
- [ ] E0 strong checkpoint in parallel with pilot analysis (public FT first, else train).
- [ ] Weekly K6 scoop sweep (standing queries in `_novelty_check_v2.md`).
- [ ] After strong-ckpt E0+E1r+E2r: arXiv preprint, then full paper → /experiment-plan → /auto-review-loop.
