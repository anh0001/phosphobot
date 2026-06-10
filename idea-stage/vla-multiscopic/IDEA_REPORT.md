# Idea Discovery Report — VLA × Multi-Scopic Neuro-Cognitive Fusion

**Direction**: Combine modern VLA (vision-language-action) manipulation policies with the *indigenous*
primitives of Saputra et al. 2022 ("Multi-scopic neuro-cognitive adaptation for legged locomotion robots"),
change a **fundamental** VLA method, and verify in simulation (LIBERO / SmolVLA).
**Date**: 2026-06-08
**Branch**: `idea/vla-multiscopic-cognition`
**Pipeline**: research-lit → idea-creator (6 lenses, 30 candidates) → cross-model jury (GPT-5.5 xhigh) →
✅ checkpoint (Direction A selected) → novelty-check → refine round 2 (GPT-5.5) → experiment-plan → pilot P0 (running)
**Substrate**: SmolVLA (`lerobot/smolvla_base`, LeRobot 0.5.x) on LIBERO; flow-matching action expert (chunk n=50,
num_steps=10), async RobotClient/PolicyServer; harness already computes per-frame flow-head **dispersion**.

---

## ⛔ FINAL DECISION (2026-06-10): STOPPED — negative result (cross-model confirmed)

The VLA × multi-scopic **preemption/override** direction is **closed**. The pre-registered kill chain fired:
method killed at E1; recovery-manifold framing killed at the `perturbed_start` crux control (displaced pose
unsolvable from scratch → confound is object-pose OOD brittleness, already known via LIBERO-Plus). GPT-5.5 jury
confirmed STOP (no novel + likely-positive salvage on SmolVLA/LIBERO within ~1 day GPU). Total spend ~8–10
GPU-hours — fail-fast worked as designed.

**Single keepable takeaway** (→ candidate paragraph for the *Unreliable by Default* paper): *In SmolVLA/
LIBERO-Spatial, apparent mid-rollout recovery failures under object nudges are dominated by object-pose
distribution shift — the same 5 cm displacement at episode start collapses conditioned success to 1/14 — so
faster replanning or fast-loop override solves the wrong bottleneck.* Also keep: the perturbation harness
(`sim/libero_active/preempt/`), the RNG-confound lesson, and P0 (`n_action_steps=1` hurts vs `50`). Discard as
claims: "preemption helps", "VLAs lack a recovery manifold", "LIBERO mid-rollout recovery benchmark".

---

## Executive Summary

The reference paper's durable ideas are an *architecture of time and compute*, not a locomotion algorithm:
three independently-clocked adaptation loops (micro/meso/macro), attention-as-compute-allocation, embodiment-
relative affordance, and — its deepest idea — a **fast loop that can PREEMPT/OVERRIDE the slow loop**. Modern
VLAs have the opposite strengths (web-scale semantics, language, learned generalist control) but are
architecturally single-clock and execute action chunks ~open-loop. The fusion is therefore natural: graft the
multi-scopic temporal architecture onto a VLA.

We generated 30 grounded fusion candidates (each tied to exact SmolVLA code) and ran an independent cross-model
jury. **Key finding: nearly every "obvious" mechanism is already published (2025–2026).** The honest, still-open
contribution is one of two things, and they correspond to two paper types:

1. **Method paper — "Override, not replan."** A true AEF-style *preemption* controller where the fast micro
   loop can veto/hold/override the slow VLA's already-queued action *before* a replan returns. No existing
   dual-system VLA (GR00T N1, Helix, π0.5, Hi Robot) does this — they only execute what the slow loop
   authorized. **Requires a perturbed/dynamic LIBERO protocol** (static LIBERO gives reactivity nothing to bite on).
2. **Diagnostic paper — "Is dispersion-gated reactivity vacuous?"** A calibration audit that asks whether the
   flow-head dispersion signal (and the whole 2025-26 dispersion-gated-replan literature) actually carries
   affordance-mismatch information on LIBERO, or is just stochastic flow noise. A negative result is publishable
   and kills most of the field's recent "adaptive replan" claims on static sim.

**Bottom line (jury):** *On stock, static LIBERO there is no strong publishable reactivity method here — the
benchmark is too static and SmolVLA too open-loop for preemption claims to bite. A publishable path requires
either (a) building a perturbed LIBERO protocol for true fast-loop preemption, or (b) writing the diagnostic
paper. The only purely-static method worth running is the dispersion-shaped training loss, and even there expect
a modest/null result.*

---

## ✅ Decision & Refinement (post-checkpoint)

**Selected: Direction A — Preemption VLA**, full path (novelty-check → refine → plan → pilot).

**Confirmatory novelty (tightened the gap, honestly)**: found very close 2025–26 neighbors — **A2C2 /
"Leave No Observation Behind"** (real-time *additive* chunk correction every step), **Pre-VLA** (2605.22446,
*preemptive resampling before* execution), **Fast Safety Reflex** (2601.14628, tactile withdrawal bypass), and
**LIBERO-Plus / PRO / Libero-V** (perturb at *episode start*, not mid-rollout). The defensible gap narrowed to a
precise axis and was **reframed** by cross-model refine round 2 into the headline contribution:

> **When should a chunked VLA cede actuation authority (override) instead of rescheduling (replan) or blending
> a correction (residual)?** — delivered as a **phase diagram** over (disturbance magnitude × replan latency),
> stratified by manipulation phase. "Preemption VLA" is the mechanism; the phase diagram is the contribution.
> This turns the A2C2 collision into the central scientific question instead of a novelty threat.

**Two independent contributions**: (1) the **mid-rollout dynamic-disturbance LIBERO protocol** (genuinely new vs
start-of-episode LIBERO-Plus/PRO/V); (2) the **authority-handover-vs-correction-vs-replan phase diagram**.

**Artifacts** (this direction, namespaced):
- Proposal: [`refine-logs/vla-multiscopic/FINAL_PROPOSAL.md`](../../refine-logs/vla-multiscopic/FINAL_PROPOSAL.md)
- Experiment plan: [`refine-logs/vla-multiscopic/EXPERIMENT_PLAN.md`](../../refine-logs/vla-multiscopic/EXPERIMENT_PLAN.md)
- Tracker: [`refine-logs/vla-multiscopic/EXPERIMENT_TRACKER.md`](../../refine-logs/vla-multiscopic/EXPERIMENT_TRACKER.md)
- Jury + refine traces: `.aris/traces/idea-discovery/2026-06-08_run01/`

**Decisive next experiment (E1, dispersion-free oracle test)**: open-loop n=50 / oracle replan-only (execute
stale during L=8 latency) / oracle preempt-hold (same trigger+latency+macro count) / n=1. **Kill** if preempt-hold
doesn't beat replan-only by ≥+10 pp conditional perturbed success *or* cut stale-action harm ≥25% at equal macro
calls.

**Pilot P0 result (clean-LIBERO horizon ablation · libero_spatial · SmolVLA `fixedbuf_random_N20/seed0`):**

| `n_action_steps` | pc_success |
|---|---|
| 50 (open-loop chunk) | 30% (50 ep) · **31%** (200-ep anchor) |
| 8 | 32% (50 ep) |
| 1 (re-infer every step) | **20%** (50 ep) |

**C0 verdict: clean headroom = `succ(n=1) − succ(n=50)` = −10 pp (NEGATIVE).** Re-inferring every step *hurts*
— fresh flow-head re-sampling each step breaks chunk coherence (the smoothness-vs-reactivity tradeoff RTC
targets). Three implications, all *sharpening* Direction A (it does not kill it):
1. **The perturbation protocol P1 is mandatory** — static LIBERO has *no* reactivity headroom (confirms the
   jury's central doubt empirically).
2. **The contribution must be SELECTIVE intervention** (override on a trigger), because blanket "replan faster"
   (`n=1`) is empirically counter-productive — so *full replan is a weak baseline, not a strong one*, which
   strengthens the override-vs-replan framing.
3. **The override action must stay coherent** — `hold/retract` (mode 1) sidesteps the `n=1` incoherence, so it is
   validated as the first mechanism to try in E1.

**E1 result (ORACLE handover, dispersion-free · object-nudge δ-sweep · libero_spatial · 50 pairs, 14 clean-success):**

| arm | cond. success @5cm | cond. success @10cm |
|---|---|---|
| clean (reference) | 100% | 100% |
| open_loop (no recovery) | 21.4% | 21.4% |
| replan_only (run stale chunk during L=8 delay) | 21.4% | 21.4% |
| **preempt_hold** (freeze during L=8 delay) | **28.6%** | **14.3%** |

**Kill criterion FAILS at both magnitudes** → **Direction A's preemption mechanism is killed on LIBERO by its
own pre-registered gate.**
- `cond_gain (hold − replan)` = **+7.1 pp** @5cm, **−7.1 pp** @10cm — below the +10 pp bar and *sign-flips with
  magnitude*; at n=14 conditioning pairs ±7 pp = ±1 pair, i.e. **no detectable effect**.
- `stale-action harm`: HOLD diverges from the recovery action **−35% MORE** than the stale chunk does (freezing
  is *further* from what the policy wants than continuing the stale plan).
- **Decisive sub-finding**: `open_loop == replan_only` (21.4%) at *both* magnitudes → re-planning 8 steps earlier
  buys **zero**. The bottleneck is the policy's **recovery capability** (it cannot re-perceive + re-approach a
  displaced object in time), **not** stale-action execution timing — so preemption optimizes a non-bottleneck.

This confirms P0 (blanket reactivity hurts) and the GPT-5.5 jury's prediction (static manipulation sim lacks
closed-loop headroom for preemption). **It is a valid negative / diagnostic result, found in ~5 GPU-hours, not
weeks** — exactly what the kill-gated plan was for. **Pivot** (per plan): Direction B (is the dispersion trigger
even valid?) or reframe as a diagnostic — *"why fast-loop override does not improve VLA manipulation recovery on
standard benchmarks."*

**Recovery-capability probe (the keystone test, δ=5cm, same 14 clean-success pairs):** give the policy the
*best possible* recovery — at the disturbance, discard the stale plan and re-plan IMMEDIATELY from the perturbed
observation (zero latency) + track with an 8-step cadence.

| arm | conditional success | per-14 |
|---|---|---|
| clean (ref) | 100% | 14/14 |
| open_loop (no recovery) | 21.4% | 3/14 |
| replan_only | 21.4% | 3/14 |
| preempt_hold | 35.7% | 5/14 |
| **recovery_probe (best-case)** | **28.6%** | **4/14** |

**Thesis CONFIRMED: best-case recovery (4/14) ≈ no-recovery (3/14)** — within ±1 pair. The disturbance roughly
halves success and *no* strategy reliably restores it; the per-pair success pattern is essentially **uncorrelated
with strategy** (e.g. some pairs only the probe recovers, others only `hold`, others the probe *fails* where
`open_loop` succeeds — the signature of noise, not a mechanism). **→ the bottleneck is the policy's recovery
capability, not execution timing.**

**Two honest caveats (must fix before any claim):**
1. **Underpowered**: n=14 conditioning pairs; ±1–2 pairs = noise. Needs n≥60 (more seeds / stronger checkpoint).
2. **Measurement-RNG confound found**: the stale-action-harm secondary metric calls `predict_action_chunk`,
   which advances the policy's flow-sampling RNG and perturbs the *primary* outcome (`preempt_hold` = 28.6% with
   harm-measurement ON in the sweep vs 35.7% with it OFF here — both within noise, but the measurement must be
   decoupled from the rollout RNG). Clean numbers come from `--no-stale-harm` runs.

**Status: Direction A method killed; the diagnostic (Option B) is now well-supported directionally.** To make it
publishable (Codex, 1-week cap): power up to n≥60, add `libero_object` (and ideally OpenVLA/π0), fix the RNG
confound. Results: `sim/libero_active/preempt/e1_recovery_d005.json`.

**Crux control — `perturbed_start` (Codex must-run; the framing gate): FAILED.** Displace the SAME object by the
same 5 cm but at **episode start**, solve from scratch:

| condition | conditional success (on the 14 clean-success episodes) |
|---|---|
| clean | 100% (14/14) |
| **perturbed_start (5 cm at reset)** | **7.1% (1/14)** |
| open_loop (5 cm mid-rollout) | 21.4% (3/14) |
| recovery_probe (best-case mid-rollout) | 28.6% (4/14) |

**The displaced object pose is NOT solvable from scratch** — actually *harder* than the mid-rollout disturbance
(a 5 cm shift pushes the object out of the calibrated init-state distribution → OOD). Per Codex's pre-registered
rule, *"if `perturbed_start` also collapses, the honest claim downgrades to 'SmolVLA lacks robustness to object-
pose shift,' not 'recovery manifold.'"* So the mid-rollout preemption negative is **confounded by object-pose
fragility**, and that fragility is **already established** (LIBERO-Plus: 95%→<30% under layout perturbation).

**Consequence: the novel "replanning isn't recovery" diagnostic does NOT hold at δ=5 cm.** The whole chain
(P0: reacting hurts · E1: no strategy beats open-loop · recovery-probe: best-case ≈ none · crux: can't solve the
displaced pose from scratch) converges on a single, *non-novel* explanation: **SmolVLA's failure under object
displacement is perceptual/distributional brittleness, not control-timing or recovery strategy.** Possible (low-
odds) rescue: a small-δ sweep to find a regime where the pose stays solvable-from-scratch but mid-rollout still
fails. Otherwise this direction is **done** — method dead, diagnostic confounded/low-novelty. Results:
`sim/libero_active/preempt/perturbed_start_control.json`.

---

## The Reference Paper's "Indigenous" Primitives (transferable)

See `REF_PAPER_SUMMARY.md` for detail. The seven primitives:
- **P1 Three-clock multi-rate** adaptation (micro ~20ms / meso ~500ms / macro on-intention-change).
- **P2 Bottom-up ↔ top-down**, reconciled at the meso level.
- **P3 Attention = adaptive compute allocation** (density where the scene is information-rich).
- **P4 Affordance-centric, embodiment-relative** perception.
- **P5 Affordance-Effectivity-Fit INTERRUPT** — fast loop **preempts** the motor plan. *(The deepest, least-matched.)*
- **P6 Embodiment-aware** memory/map.
- **P7 Topological** (graph) representation.

---

## Literature Landscape (from 7-angle survey, 89 papers)

- **VLA inference is converging on flow/diffusion action chunking** (π0, SmolVLA, Octo): one heavy backbone pass
  → action expert emits a 16–50 step chunk in K=4–10 flow steps → execute ~open-loop ~0.5–1s → re-infer.
- **Temporal decoupling** is handled by *async inference* (SmolVLA RobotClient/PolicyServer) and *RTC*
  (NeurIPS 2025, freeze-prefix/inpaint-suffix) — smooth, delay-robust, no retraining.
- **Hierarchy tops out at TWO coupled rates**: System-2 VLM (~7–10 Hz) + System-1 control (120–200 Hz) in
  GR00T N1 / Helix; reasoning-level hierarchy in π0.5 / Hi Robot. **Crucially, the fast loop never preempts the
  slow loop** — it only executes slow-authorized actions. This is the open structural gap vs P5.
- **Adaptive compute is crowded**: adaptive flow-steps (ProbeFlow, AdaFlow, D3P), token pruning (VLA-Pruner,
  LightVLA), adaptive horizon (Mixture-of-Horizons, AutoHorizon, Adaptive Action Chunking), cache reuse
  (TIDAL, AC²-VLA, VLA-Cache), joint routers (AC²-VLA/SCALE).
- **LIBERO is quasi-static**; no standard reactive/perturbation manipulation benchmark is widely adopted — a gap
  that the method paper would have to fill.

---

## 30 Candidates — Cross-Model Jury Triage (novelty risk that it is already done)

★ = survived to top-6. Full annotations in `_jury_input.md`; jury trace in `.aris/traces/idea-discovery/2026-06-08_run01/`.

| ID | Idea (abbrev) | Novelty risk | Note |
|---|---|---|---|
| 0 | Dispersion mid-chunk preempt vs empty-queue | HIGH | stale-prefix can't see changed scene; needs perturbation+RTC |
| 1/26 | Three-clock KV reuse | HIGH | StreamVLA/TIDAL/AC² ; little backbone headroom |
| 2/11/15/27 | Adaptive flow-step budget | HIGH | ProbeFlow/AdaFlow/D3P — direct kill |
| 3/8/12/28 | Affordance/saliency token routing | HIGH | VLA-Pruner/LightVLA; SmolVLA 64-token compressed path |
| 4 | Typed abort routing | MED | first mildly interesting variant; geometry may be unmeasurable |
| 5/10/16/23 | Adaptive execution horizon | HIGH | AutoHorizon/MoH/AAC |
| ★6 | Two-clock → reframed as **preemption** | MED | cache story done; **override story open** |
| 7 | Per-timestep flow freeze | MED | no FLOP saving in joint denoiser |
| ★9 | Typed RTC guidance | MED | narrow but concrete under delay/perturbation |
| 13/17 | Adaptive depth / cache | HIGH | DeeR-VLA/VLA-Cache |
| ★14 | **Dispersion-shaped flow loss** | MED | real training change; static-valid; expect modest |
| 18 | Joint compute router | HIGH | AC²-VLA/SCALE verbatim |
| 19 | Adaptive sample-K | HIGH | not fundamental; adaptive self-consistency |
| ★20 | Within-chunk failure-onset autopsy | MED | promising after fixing "point of no return" label |
| ★21 | **Dispersion calibration autopsy** | MED | can kill/justify the whole dispersion program |
| 22 | Language-channel ablation→gate | HIGH | LIBERO-Plus + Stable Language Guidance |
| ★24 | Intra- vs inter-chunk disagreement | MED | good as signal-selection, not new replan method |
| 25 | Dispersion AEF interrupt | HIGH | dup of 0/16 (method); perturbation eval might matter |
| 29 | Bottom-up suffix conditioning | MED | fundamental but loses to AsyncVLA/DiG-Flow unless cleaner |

---

## Recommended Directions (pick one at the checkpoint)

### 🏆✅ Direction A — "Override, not Replan": a true AEF preemption VLA  *(SELECTED · synthesis of [6]+[9]+[25])*
- **Fundamental method change**: replace SmolVLA's FIFO open-loop chunk execution
  (`_check_get_actions_condition` / `n_action_steps` queue-drain in `select_action`) with a **priority-arbitrated
  controller**: a fast micro loop can veto/hold/override the queued slow-loop action *immediately*, while the
  macro SmolVLA replan is still pending. Not "replan earlier" — "the fast loop temporarily owns actuation."
- **Contribution (matters either way)**: directly tests the reference paper's unmatched primitive P5 — *fast
  reactivity preempts slow cognition*. Positive ⇒ preemption is a missing VLA capability; negative ⇒ LIBERO-class
  manipulation lacks the closed-loop headroom for it (itself a useful, citable result).
- **Why still novel**: AAC/MoH/RTC/BID/StreamVLA/TIDAL and GR00T/Helix/π0.5 *shorten, verify, cache, or replan
  slow-authorized chunks*; none test an **unauthorized fast override before slow authorization returns**.
- **LIBERO protocol**: `libero_spatial` → `libero_object`; clean **and perturbed** runs (move target / inject
  end-effector displacement at a fixed step). Baselines: fixed-50 SmolVLA, `n_action_steps=1`, RTC, scalar-
  dispersion replan [25]. Metrics: post-perturb success, recovery latency, stale-action steps, clean-success
  drop, macro calls/episode. **Perturbation protocol REQUIRED** (must be built).
- Risk: MED–HIGH (needs perturbation harness). Effort: weeks. Payoff: highest; a method paper.

### Direction B — Diagnostic: "Is dispersion-gated reactivity vacuous?"  *(synthesis of [21]+[24]+[20])*
- **Fundamental change**: before using dispersion as a controller, compare candidate replan signals at **equal
  compute** — intra-sample flow dispersion vs inter-chunk inconsistency vs image/state delta vs ProbeFlow-style
  velocity geometry; the winner (if any) becomes the gate. Re-purposes `num_steps`/the K-sample loop as a *sensor*.
- **Contribution**: tells us whether the flow-head dispersion signal you already compute is a real affordance-
  mismatch signal or stochastic noise — and whether the 2025-26 "adaptive replan" literature is vacuous on static
  sim. A **negative result kills most candidates above** and is publishable as a diagnostic.
- **Why novel**: ProbeFlow/BID/RTC/Sentinel use related signals but none *audit SmolVLA K-sample dispersion on
  static vs perturbed LIBERO and expose the stale-prefix failure mode*.
- **LIBERO protocol**: `libero_spatial`+`libero_object`, ~40 eps each. Static arm: signal calibration vs episode
  failure (AUROC, ECE/Brier, lead time, compute cost, equal-cost gate success). Optional perturbed arm for
  reactivity claims. **Perturbation optional** for the core static contribution.
- Risk: LOW–MED. Effort: days–weeks. Payoff: solid diagnostic; reuses existing dispersion code; lowest risk.

### Direction C — Dispersion-shaped flow-matching training loss  *([14])*
- **Fundamental change**: modify `VLAFlowMatching.forward` so the flow loss preserves the chunk-step axis and
  applies per-step dispersion weights instead of uniform averaging (allocate gradient density by self-disagreement).
- **Contribution**: tests whether bottom-up self-disagreement should drive learning capacity during LoRA fine-tune.
- **Why novel**: vs FreqPolicy (frequency), RTC (fixed late-step), Min-SNR (noise-time) — this is per-chunk-step
  *self-dispersion* weighting.
- **LIBERO protocol**: `libero_long` primary + `libero_spatial` control; baselines uniform LoRA / late-step
  weighting; metric paired `pc_success`, ≥3 train seeds. **No perturbation.** Purely static.
- Risk: LOW (but jury expects modest/null). Effort: days. Payoff: safest, smallest.

---

## Strategic Reframe (the one-line thesis)
> The reference paper's "interrupt" is **not** "adaptive replan." Most of the field (and most of our candidates)
> weakened P5 into a thresholded queue-refill, which prior art already owns. The real, open axis is
> **OVERRIDE vs REPLAN** — letting a fast loop preempt the slow VLA's authorized plan. Frame the paper on that
> axis, not on "dispersion vs entropy."

## Eliminated (representative)
Adaptive flow-steps (2/11/15/27 — ProbeFlow/AdaFlow/D3P), token routing (3/8/12/28 — VLA-Pruner), cache/3-clock
reuse (1/13/17/26 — TIDAL/AC²/VLA-Cache), joint router (18 — AC²-VLA), adaptive sample-K (19 — not fundamental).

## Next Steps
- [x] Checkpoint: **Direction A selected**.
- [x] Confirmatory novelty-check (A2C2 / Pre-VLA / Fast Safety Reflex / LIBERO-Plus-PRO-V) → gap reframed.
- [x] Refine round 2 (GPT-5.5) → `FINAL_PROPOSAL.md` + `EXPERIMENT_PLAN.md` + `EXPERIMENT_TRACKER.md`.
- [🔄] Pilot P0 (horizon ablation, clean-LIBERO headroom) — running on A6000.
- [ ] Build custom rollout+perturbation harness (`sim/libero_active/preempt/`) → run **E1 oracle handover** (kill gate).
- [ ] E2 vs bounded-residual A2C2 → E3 dispersion trigger → E4 phase diagram → `/auto-review-loop`.
