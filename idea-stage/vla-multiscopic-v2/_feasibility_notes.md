# Feasibility Notes — vla-multiscopic-v2 (pre-ideation, 2026-06-11)

Verified against `sim/libero_active/preempt/harness_lib.py` + `perturb.py` (the existing E1 harness).

## What the harness already gives us
- Manual chunk control via `predict_action_chunk` (1,50,7 normalized), canonical pre/post-processing chain.
- Disturbance with a KNOWN world-frame vector: `perturb_nearest_object` / `perturb_object_by_name`
  (`sim.data.qpos[adr] += delta; sim.forward()`) → the ORACLE displacement d is free.
- Raw obs intercept point: `raw_obs["pixels"]` (360×360, agentview + eye-in-hand) and
  `raw_obs["robot_state"]` BEFORE `build_policy_batch` → clean injection point for obs transforms.
- `_refresh_obs(env)` re-reads obs without stepping; RNG-safe measurement pattern already established.
- MuJoCo model access (`rs.sim`) → camera matrices available (robosuite `camera_utils.get_camera_transform_matrix`)
  for projecting world displacement → pixel shift. Fallback: empirical calibration (apply known d, diff images).

## Key mathematical observation (load-bearing for ideation)
SmolVLA on LIBERO uses RELATIVE end-effector actions (`control_mode="relative"`, 6 pose deltas + gripper).
Define virtual frame V = real frame R translated by −d (d = object displacement). In V:
- the object sits at its CANONICAL location → the warped image is back IN-DISTRIBUTION;
- proprio: eef_V = eef_R − d (subtract d from reported eef pos; quat unchanged for pure translation);
- the fixed third-person camera view in V ≈ real image translated by the image-projection of −d
  (homography/planar approx; exact enough at 5cm);
- the EYE-IN-HAND camera is INVARIANT: object-to-wrist-cam relative pose in V equals the real one
  (both camera and "world" shift together) → wrist image needs NO warp;
- relative deltas are frame-invariant → executing the V-frame policy output in R lands the eef at
  canonical+d = the TRUE object. **No action counter-warp needed.** The "counter-warp" is the identity,
  by construction of the relative-action interface.

So a test-time re-anchoring intervention = (image translate on agentview) + (eef-pos offset) only.
This simultaneously (a) re-anchors the reach target and (b) restores the visual input distribution —
which directly addresses the crux-control confound (perturbed_start collapse = scene-level OOD).

## Two-sided informativeness (why the oracle pilot is decisive either way)
- If ORACLE re-anchoring (known d) restores conditioned success toward clean levels → causal proof that
  canonical-prior mislocalization is THE failure mechanism + a working training-free intervention class;
  explains the perturbed_start collapse (shifted scene unparseable) as the same mechanism.
- If it fails → the OOD-ness is deeper than object position (relational/contextual), itself a sharp,
  publishable refinement of the LIBERO-Plus fragility story (and kills the intervention family honestly).

## Pilot cost estimate
- Oracle-warp variant in harness: ~1 day eng. Pixel projection via camera matrix or empirical calibration.
- Eval: reuse E1 protocol (10 tasks × 5 seeds × few variants, `--no-stale-harm`) ≈ 2–4 GPU-h.
- Tracker version (drop oracle): template/SAM2-lite tracking of the displaced object from agentview at
  meso rate (every chunk boundary) — the displacement estimate replaces oracle d. +1–2 days eng.
- Power-up to n≥60 conditioning pairs: more seeds + libero_object suite (harness param change).

## Constraints to respect
- Checkpoint `fixedbuf_random_N20/seed0` is weak (~30% clean). For a paper run, consider the stronger
  standard SmolVLA libero fine-tune or train longer; conditioning-pair count depends on clean success.
- HuggingFaceVLA/libero dataset has NO object poses → train-time localization supervision needs sim-side
  relabeling (feasible: replay episodes in LIBERO to extract poses, or use detector pseudo-labels).
