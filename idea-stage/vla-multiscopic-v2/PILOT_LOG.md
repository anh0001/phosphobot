# Pilot Log — vla-multiscopic-v2 ("The Failure Is a Vector")

## 2026-06-12 — Pilot (a) v1: reach-field grid (CONTAMINATED, kept as control)

850 rollouts (50 pairs × [clean + 4 mag × 4 dir]), `fixedbuf_random_N20/seed0`, libero_spatial, ~6.2 GPU-h.
Results: `sim/libero_active/preempt/reach_field_results.jsonl` (+ `reach_field_k1.json`).

Apparent findings (pre-bug): median cos(e,−d)=0.85 [CI 0.67,0.92]; paired tracking coefficient
τ = −0.013 [CI −0.031, 0.013] (endpoint invariant to displacement); projection slope 0.93 ≈ identity;
closer-to-canonical 72%; "tolerance success" at 2.5cm (36% success with τ≈0.07 — grasp basin absorbs
small offsets). Pre-registered K1 raw-|e| R² check failed (0.08) due to ~6cm noise floor — reformulated
to paired-τ + projection metrics (disclosed; medians-by-magnitude track |d| monotonically).

## 2026-06-12 — BUG FOUND: stale-observation confound (load-bearing)

Discovery chain (during pilot (b) oracle-warp smoke):
1. `sham_zero` (displaced + zero warp) failed its no-op check vs the grid record → bisection.
2. Env warm-up effect found first: the FIRST rollout on a freshly constructed LiberoEnv diverges from
   all later ones (grid ran clean first on every env → internally consistent; pilot (b) now warm-ups).
3. Warp magnitude anomaly: camera-matrix projection predicts ~20px for 5cm; empirical calibration
   render pair came back IDENTICAL → exposed the real bug:
4. **`_refresh_obs` (robosuite `_get_observations()` default) returns CACHED observations — a direct
   qpos displacement is NOT reflected** (post-perturb default obs bit-identical to pre-perturb frame;
   `force_update=True` shows the moved object). Fixed in `harness_lib._refresh_obs` (2026-06-12).

### Blast radius
- **Pilot (a) v1**: every episode-start displaced rollout planned its FIRST chunk (containing the
  pregrasp endpoint, t≈35) on a stale image showing the object at CANONICAL position. τ≈0 is therefore
  an ARTIFACT as evidence of canonical-prior bias: the policy reached canonical because it was SHOWN
  canonical at plan time. SmolVLA plans only at chunk boundaries → first fresh plan at step 50.
  → v1 grid reinterpreted as a **stale-perception control**: it establishes that the policy reaches
  where the image shows the object (perception-consistent reaching), with |e|≈|d| transfer. Useful
  comparison arm; NOT evidence of bias.
- **v1 KEYSTONE (mislocalization probe, 2026-06-10, 93% closer-to-canonical)**: same code path →
  same stale first chunk → **invalidated as measured**. The "perceptual mislocalization, not control"
  attribution is OPEN again pending the v2 rerun. Affects the planned "Unreliable by Default" keystone
  paragraph — do not cite until v2 confirms or refutes.
- **perturbed_start crux (1/14)**: first chunk stale; subsequent chunks fresh (env.step obs). The
  pose-OOD fragility conclusion is weakened for early-trajectory dynamics but most planning was fresh.
- **E1 recovery_probe**: its "immediate replan from perturbed obs" replanned from a STALE frame
  (subsequent 8-step-cadence replans were fresh). "Best-case recovery" was not actually best-case at
  the perturb step; the negative is softened, not erased (fresh replans at +8/+16... still failed).
- **P0 horizon ablation**: unaffected (no perturbation, no _refresh_obs use).

## 2026-06-12 — Pilot (a) v2 relaunched (fixed observations)

`reach_field_results_v2.jsonl` / `reach_field_logs_v2/`, same 850-rollout grid, ~6h.
THE REAL K1: with fresh first frames,
- if τ → ~1 (policy tracks the shown object): canonical-prior bias was an artifact → flagship's
  empirical premise dies → pivot (perception is fine; failure is elsewhere) or kill. K1 gates honestly.
- if τ stays ≈ 0 (policy ignores the shown displacement): canonical-prior reach is REAL, now properly
  measured, v1-vs-v2 contrast becomes a beautiful stale/fresh control pair, and pilot (b) proceeds.

## 2026-06-12 — Pilot (a) v2 RESULT: K1 = CONTINUE (canonical-prior reach is REAL)

850 rollouts, fresh observations, 15/50 clean-success pairs, n=240 clean-conditioned endpoint vectors.
- **τ (paired tracking) = −0.074 [CI −0.112, −0.038]**: with the displacement fully visible at plan
  time, the pre-grasp endpoint does NOT track the object (slightly negative — second-order repulsion
  effect, unexplained; E1r question). Projection = 0.9cm + 0.88·|d| (≈ identity transfer).
- DVR median cos = 0.825 [CI 0.653, 0.910]; closer-to-canonical 73%; cos sharpens with |d|
  (0.60→0.93); anisotropy (ny 0.96 vs px 0.74); success 40%→7%→5%→2% over 2.5→10cm.
- **v1↔v2 contrast (the headline control figure): stale τ=−0.013 vs fresh τ=−0.074 — seeing the
  displacement changes nothing about where the policy reaches.** The bug became the perfect
  stale-perception control arm.
- Tolerance-success confirmed on fresh data: 2.5cm succeeds 40% with no tracking (grasp-basin
  geometry, invisible to success-only benchmarks).
- Pre-registered K1 conjunction: cos/CI/closer-canon PASS; slope(raw-|e|)=0.48 marginal-fail and
  R²=0.07 fail on the KNOWN-BAD raw-|e| metric (folded direction noise; flagged after v1, disclosed).
  Verdict on reformulated primary metrics (τ + projection): **CONTINUE**.

## Pilot (b) launched 2026-06-12 (oracle warp, fresh harness)

15 v2 clean-success pairs × {sham_zero(1) + 6 arms × 4 dirs} + per-env clean warm-ups ≈ 390 rollouts
≈ 3h → `preempt/oracle_warp_results.jsonl`. K2 primary = endpoint-error reduction (warp_full vs
v2-grid nowarp d50mm records); model prediction E ≈ C + w per arm; specificity via warp_random /
warp_clean (predicts +w shift on clean scenes); channel attribution via image-only / proprio-only.

## 2026-06-13 — Pilot (b) RESULT: K2 = RED on restoration; instrument + channel findings instead

375 rollouts + 15 warm-ups (all grid_match=True). `oracle_warp_results.jsonl` / `oracle_warp_k2.json`.
Median |E−T| (cm) @5cm, 15 clean-success pairs × 4 dirs: nowarp 8.3 / warp_full 10.1 / warp_image 9.3 /
warp_proprio 9.5 / warp_random 9.6 / warp_clean(|E−C|) 8.0; success: nowarp 7% / warp_full 3% /
warp_clean 13% (vs 100% clean by conditioning!).

Reading (per pre-registered bands → drop the sufficiency leg; diagnosis-first):
1. **No restoration**: full virtual-frame re-anchoring does not recover endpoints or success. The #2
   "exploit" upgrade path is DEAD by its own activation gate. No deployable method claim.
2. **The instrument is itself OOD**: warp_clean breaks 87% of healthy episodes → a global image
   translate + proprio offset is out-of-distribution for SmolVLA regardless of content → the probe
   cannot adjudicate mislocalization SUFFICIENCY (instrument too blunt). The cleaner instrument is the
   pre-planned object-local paste (E2r variant) — open follow-up, not run in pilot.
3. **CHANNEL-DECOUPLING EVIDENCE (the keeper)**: warp_image (object visually moved back to canonical,
   nothing else changed) leaves the reach UNCHANGED (τ=−0.03 ≈ nowarp −0.12; |E−C| 7.1 ≈ 7.2).
   Triangulated with pilot (a): three independent manipulations of perceived object position
   (stale-cache canonical view / fresh displaced view / re-anchored view) all produce the SAME
   endpoint ⇒ **the reach endpoint is causally decoupled from the perceived object position in the
   agentview** — intervention-grade evidence for the canonical-prior/motor-program story (dialogue
   with 2603.19233). Meanwhile proprio lies DISRUPT but do not steer (warp_proprio τ=−0.37, not +1 ⇒
   the policy does not servo absolute proprio either).
4. **Alternative explanation to kill next (cheap)**: wrist-camera dominance — if the wrist cam carries
   the visual grounding, agentview manipulations would be inert for that reason. Discriminator: camera
   ablation arm (mask/blur agentview vs wrist at inference) — top of the E1r addenda list.
5. Residual position-dependence: sham_zero reproduced grid records 13/15 (2 flips at different env-reset
   positions) — medians robust, but exact-trajectory reproducibility claims must stay qualified.

**Pilot phase (a+b) CLOSED at ~16 GPU-h (< 23 budget). K1 = CONTINUE, K2 = RED → flagship proceeds as
#1 diagnosis-first** ("The Failure Is a Vector": reach-field + τ-invariance + stale/fresh control +
channel-decoupling interventions). #2 (exploit ladder) and #3 (meso loop) deactivated by their own
pre-registered gates. Next: E0 strong-checkpoint replication (THE gate before any atlas spend), camera-
channel ablation, object-local paste probe.

## Pilot (b) build notes (validated pre-launch)

`preempt/oracle_warp.py` ready: virtual-frame warp (agentview translate + proprio offset; wrist
untouched; identity action map), arms {warp_full, warp_image, warp_proprio, warp_random, warp_clean,
sham_zero, midroll} on the clean-success pairs, per-env clean warm-up (position discipline + per-pair
determinism check vs grid), empirical per-pair pixel-shift calibration via force-rendered
move-back pair (camera matrix used for direction only). Smoke validated: warm-up grid_match=True;
sham_zero reproduces the grid record exactly post-fix-chain.

## Engineering notes
- LiberoEnv first-reset-on-fresh-env diverges from later resets (mechanism unknown; likely same
  observable-cache family). Discipline: never compare rollouts across env-reset positions.
- Raw LiberoEnv images are 180°-rotated vs the robosuite projection convention (processor un-flips
  downstream); raw-frame pixel shifts = negated projection-frame shifts.
- Two same-state renders across different resets differ slightly (AA/lighting noise ~1-3 mean abs);
  template matching across resets can produce spurious few-px offsets.
