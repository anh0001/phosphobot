# Robotics Problem Frame — vla-multiscopic-v2 (2026-06-11)

**Goal**: Pivot the dead "preemption/override" idea into a NEW publishable, experiment-backed idea that
(a) is genuinely inspired by Saputra et al. 2022's multi-scopic primitives (the UNUSED ones),
(b) builds on our keystone mechanistic evidence, and (c) is novel vs the Jan–Feb 2026 wave.

## Frame
- **Embodiment**: single 7-DoF arm (Franka in LIBERO sim); policy class = chunked flow-matching VLA (SmolVLA).
- **Task family**: tabletop pick/place/rearrangement (LIBERO-Spatial, LIBERO-Object; optionally LIBERO-Long).
- **Environment**: tabletop sim (MuJoCo via LIBERO), quasi-static + our mid-rollout object-displacement protocol.
- **Observation**: RGB (2 cams, 360→256), proprioception; language task string.
- **Action interface**: 7-D relative end-effector deltas, 50-step chunks, flow-matching decoder (num_steps=10).
- **Learning regime**: pretrained SmolVLA + LoRA fine-tune feasible; training-free test-time methods preferred but
  NOT required this round (relaxed vs v1: multi-day A6000 budget acceptable, no backbone retrain from scratch).
- **Available assets**:
  - kubotaserver2 (single A6000, ~48GB); LeRobot 0.5.x; SmolVLA checkpoints incl. `fixedbuf_random_N20/seed0`.
  - Custom rollout+perturbation harness `sim/libero_active/preempt/` (manual chunk control,
    object-qpos nudge, mislocalization probe, oracle-handover variants, RNG-confound fixed).
  - HuggingFaceVLA/libero dataset (NO object poses → privileged pose supervision requires sim-side relabeling
    or MuJoCo ground truth at eval; LIBERO sim gives free ground-truth object poses at runtime).
- **Compute budget**: pilots ≤1 GPU-day; full study ≤1 GPU-week. No real robot.
- **Safety**: sim-only. No hardware risk.
- **Desired contribution type**: mechanistic diagnosis + intervention method (causal proof), or
  diagnosis + benchmark protocol. NOT another adaptive-compute / replan-timing method (falsified branch).

## Hard constraints carried from v1 (falsified / occupied — do NOT regenerate)
1. DEAD: fast-loop preemption/override/replan-timing (P0: n=1 hurts −10pp; E1: open_loop == replan_only;
   recovery probe: best-case ≈ none; all timing claims confounded by pose-OOD fragility).
2. DEAD: "recovery manifold" / "replanning isn't recovery" diagnostic at δ=5cm (crux: perturbed_start 1/14 —
   displaced pose is OOD-by-construction; fragility claim already owned by LIBERO-Plus 2510.13626).
3. OCCUPIED (Jan–Feb 2026): object-centric fixed-K slots (STORM 2601.20381), attention spotlighting
   (Spotlighting 2601.21416), spatial-prior training (ST4VLA 2602.10109), affordance conditioning
   (RT-Affordance), additive chunk correction (A2C2), pre-execution resampling (Pre-VLA 2605.22446),
   tactile reflex (Fast Safety Reflex 2601.14628), inpainting-under-delay (RTC), start-of-episode
   perturbation benchmarks (LIBERO-Plus/PRO/V).
4. WEAK SIGNAL: flow-head dispersion is a poor spatial allocator (jury, v1); dispersion-gated replan possibly
   vacuous on static LIBERO (Direction B, unresolved but low-value).

## The exploitable keystone (v1's product — this round's raw material)
After a 5cm object displacement at episode start: arm reaches the CANONICAL/expected location (mean 3.9cm)
not the displaced object (mean 5.2cm); 93% (13/14) closer-to-canonical; failures stop ~5.4cm short ≈ exactly
the displacement vector. → SmolVLA executes a **canonical-prior reach**, i.e., the failure is a structured,
predictable, low-dimensional BIAS — not noise. A structured bias can be (i) measured as a vector field,
(ii) predicted, and (iii) potentially inverted/exploited at test time.

## Multi-scopic primitives still unused (the indigenous inspiration pool)
- P3 attention = adaptive representation density (DD-GNG δ-feedback: density follows attention/information).
- P4 affordance-effectivity FIT: embodiment-relative consistency check between perceived affordance and
  intended action (the non-timing half of AEF — as a SIGNAL/measurement, not an interrupt).
- P6/P7 meso-level LER: persistent localization + environmental reconstruction state that RE-ANCHORS the
  micro level's egocentric processing (memory that survives the visual stream).
- Intero/exteroceptive coordination: reconcile where-the-body-is (proprioception) with where-the-world-is
  (vision) — the canonical-prior reach is exactly a failure to let exteroception override a learned prior.

## Search/novelty backend
Primary: Scite MCP (mcp__claude_ai_scite__search_literature — load via ToolSearch in subagents).
Secondary: WebSearch/WebFetch for arXiv (Jan–Jun 2026 preprints scite may lag on).

## Namespacing
All artifacts → idea-stage/vla-multiscopic-v2/ and refine-logs/vla-multiscopic-v2/. Prior v1 runs untouched.
