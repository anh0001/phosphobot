# P2T Results — The Prior Is Sticky

**Study:** probe-to-train pilot (idea #4) + round-2 constraint elevation.
**Instruments:** reach field (steering geometry), paste dissociation, 432-demo
LIBERO-spatial SmolVLA. **Base:** retrained e0_full_expert_100k (74% clean —
digit-for-digit replication of the wiped original: anchoring cos
0.637/0.769/0.902/0.943 vs 0.63/0.76/0.90/0.94 at 25/50/75/100 mm).

## Headline

> Counterfactual fine-tuning at practical dose improves displaced success
> **without restoring the causal visuomotor gain**: every condition — including
> the winner — keeps median steering proj ≈ 1.0 (the reach still goes to the
> canonical training location). Success gains are basin-widening, not tracking.
> Success-only benchmarks would misread this as "fixed."

## Round 1 — acquisition comparison (pre-registered, matched budget 88 eps, 20k steps)

### Table 1: success and steering by condition

| Condition | Clean | Disp @50 mm | proj @50 | Disp @75 mm | proj @75 |
|---|---|---|---|---|---|
| A gain-targeted | 76% | 34% | 1.00 | 17% | 0.95 |
| B no-aug | 76% | 29% | 1.01 | 16% | 0.99 |
| C uniform | 78% | 32% | 0.99 | 15% | 0.98 |
| D failure-targeted | 70% | **38%** | 0.97 | 16% | 0.95 |
| E sham | 72% | 30% | 1.04 | 14% | 0.99 |
| base (retrained) | 74% | 31% | 1.00 | 18% | 1.00 |

proj = median dot(e,−d)/‖d‖² on the pre-grasp endpoint; 1 = fully anchored,
0 = tracks the object. n = 200 displaced rollouts per condition per magnitude.

### Table 1b: steering invariance with uncertainty (task-cluster bootstrap 90% CI)

| Condition | proj @50 mm | proj @75 mm |
|---|---|---|
| A gain-targeted | 1.00 [0.93, 1.09] | 0.95 [0.89, 1.00] |
| B no-aug | 1.01 [0.96, 1.03] | 0.99 [0.97, 1.02] |
| C uniform | 0.99 [0.91, 1.07] | 0.98 [0.92, 1.06] |
| D failure-targeted | 0.97 [0.91, 1.03] | 0.95 [0.92, 0.98] |
| E sham | 1.04 [1.03, 1.09] | 0.99 [0.97, 1.04] |
| base (retrained) | 1.01 [0.95, 1.06] | 1.00 [0.94, 1.01] |

Every lower CI bound ≥ 0.89 — no condition approaches the "steering acquired"
gate (proj ≤ 0.70). The invariance is established with uncertainty, not
eyeballed.

### Table 1c: success-conditioned steering (the reviewer's make-or-break check)

If the improved conditions' *successful* rollouts tracked the object
(proj ≈ 0), the pooled median would hide a bimodal distribution and the
sticky-prior claim would be wrong. They do not:

| Condition | n succ | proj (successes) | n fail | proj (failures) |
|---|---|---|---|---|
| A gain-targeted | 68 | 0.82 | 132 | 1.09 |
| B no-aug | 58 | 0.95 | 142 | 1.03 |
| C uniform | 63 | 0.95 | 137 | 1.00 |
| D failure-targeted | 77 | **0.91** | 123 | 0.98 |
| E sham | 59 | 0.84 | 141 | 1.07 |

Even the successful displaced rollouts — including D's +7 pp of new successes —
reach toward canonical (median proj 0.82–0.95, nowhere near 0). At 50 mm an
anchored reach can still contact the bowl (rim + gripper span); success is
basin geometry, not perception. Pre-grasp endpoint defined in 100% of rollouts
(no closest-approach fallback).

### Table 2: pre-registered contrasts (displaced success @50 mm, A − X, task-cluster bootstrap 90% CI)

| Contrast | Δ | 90% CI | Gate | Result |
|---|---|---|---|---|
| A − B (does synthesis help?) | +5.0 pp | [+1.0, +9.0] | CI > 0 | ✅ PASS |
| A − C (does placement matter?) | +2.5 pp | [−4.0, +9.0] | > 0 | ⚠️ ns |
| A − D (causal gain > failure signal?) | −4.5 pp | [−10.5, +1.5] | ≥ +5 pp, CI > 0 | ❌ FAIL |
| A − E (specificity) | +4.5 pp | [−1.0, +10.0] | > 0 | ⚠️ ns |
| clean(A) ≥ clean(B) − 5 pp | 76% vs 76% | — | — | ✅ PASS |

**Round-1 verdict (as pre-registered): negative for the core claim.** The
measured causal gain field does not beat a generic failure signal as an
acquisition function.

### Table 3: what actually changed — the mechanism decomposition

| Quantity | Base | Best condition (D) | Read |
|---|---|---|---|
| Displaced success @50 | 31% | 38% (+7 pp) | improved |
| Median steering proj @50 | 1.00 | 0.97 (−0.03) | unchanged |
| Median steering proj @75 | 1.00 | 0.95 (−0.05) | unchanged |
| Clean success | 74% | 70% (−4 pp) | regressed |

Success moved; steering did not. Fig 2 (mechanism scatter) and Fig 4
(displacement-normalized endpoint clouds: median u = 1.01 / 1.00 / 0.97 for
base / A / D) show every policy still reaches canonical.

### Ruled out on existing data (zero GPU)

- **Coverage**: A delivered 43/93 episodes at ≥62 mm (Fig 3) — ample large-|d|
  exposure, zero tracking. D's realized distribution is *larger*-|d|-shifted
  than A's and D still won @50 — placement distributions do not explain the
  ranking.
- **Eval-magnitude mismatch**: at 75 mm (where A's mass sat) A gets zero lift
  (17% ≈ base 18%).
- **Under-training**: fine-tune loss 1.27 → 0.037 (converged; the synthetic
  episodes were fit).

## The binding constraint (Theory-of-Constraints pass)

Round 1 tested the *acquisition* level. The proj ≈ 1.0 invariance shows the
constraint sits below it: fine-tuning takes the least-resistance path —
**widen the tolerance basin rather than wire a visuomotor gain** — even when
trained on data only a tracking policy explains. Note the training recipe
freezes the entire VLM (vision + LM; `train_expert_only=true`): the only
trainable path is the action expert reading frozen features.

## Round 2 — constraint elevation (pre-registered 2026-07-12, running)

| Arm | Data (20k steps from base) | Trainable | Discriminates |
|---|---|---|---|
| M1_ratio50 | 88 A-synthetic + 88 originals (50/50) | expert only | H1 weak (ratio at fixed count) |
| M2_pure | all 352 pooled synthetic, 0 originals | expert only | H1 strong (no anchoring signal left) |
| M3_plastic | same 352 synthetic | expert + full VLM + vision | H2 (plasticity locus) |

**Primary metric = median steering proj** (not success).
Gate "steering acquired": **proj @50 mm ≤ 0.70**.
- H1 (mixture ratio) confirmed if M1 or M2 crosses with expert-only.
- H2 (plasticity locus) confirmed if only M3 crosses.
- H3 (representation: chunked expert cannot express conditional retargeting)
  supported if none cross with converged loss.

### Round-2 RESULTS (2026-07-13) — VERDICT: H1-strong; H3 refuted; H2 marginal

| Arm | Clean | Succ @50 | proj @50 | Succ @75 | proj @75 | proj (successes) @50 |
|---|---|---|---|---|---|---|
| M1 50/50, expert-only | 72% | 37% | 0.92 | 20% | 0.83 | — |
| **M2 100% syn, expert-only** | 52% | 37% | **0.68 ✅** | 23% | 0.77 | 0.51 |
| **M3 100% syn, unfrozen VLM** | 60% | 36% | **0.64 ✅** | 24% | 0.73 | **0.39** |

**Dose-response (proj @50 vs counterfactual fraction): 17% → 1.00, 50% → 0.92,
100% → 0.68.** (Fig 5.)

1. **The binding constraint is gradient competition with the canonical
   demonstrations** (H1-strong): remove them and the expert-only pathway
   acquires partial steering. The prior is not stuck — it is *defended by the
   data that taught it*.
2. **H3 refuted**: the frozen VLM features carry usable object-position signal;
   the chunked expert CAN express conditional retargeting.
3. **H2 marginal, not binding**: unfreezing adds a real but secondary gain
   (proj 0.64 vs 0.68; success-conditioned steering 0.39 vs 0.51) and buys
   back clean competence (60% vs 52%).
4. **Mechanism transition visible in the readout**: in round 1 even successes
   were anchored (proj_succ 0.82–0.95); in M2/M3 successes increasingly track
   (0.51/0.39) while failures stay anchored (0.92/0.99) — steering emerges as
   a bimodal split, exactly what the instrument was built to see.
5. **Sharp Pareto tension**: steering acquisition is paid for in canonical
   competence (76% → 52% clean along the frozen curve; M3 recovers to 60%).
   At no point does any arm reach full tracking (proj 0) — dose/steps scaling
   is the open axis.

### Round-2 CORRECTION (external gpt-5.5 audit, 2026-07-15) — verdict downgraded to PARTIAL

The proj gate ("proj ≤ 0.70 = steering acquired") **overstated** the result.
Endpoint-concentration analysis in the displacement-normalized frame
(u = component along −d: 0 = object T, 1 = canonical C; ρ = ‖e‖/‖d‖; pregrasp
endpoint defined in 100% of rollouts — no closest-approach fallback inflation):

| Arm | u (successes) | u (failures) | ρ (successes) | successes within ρ ≤ 0.75 |
|---|---|---|---|---|
| base | 1.06 | 1.00 | 2.29 | 13% |
| A (round 1, 17%) | 0.82 | 1.09 | 1.44 | 13% |
| M1 (50%) | 0.66 | 0.98 | 1.28 | 20% |
| M2 (100%) | 0.51 | 0.92 | 1.27 | 24% |
| M3 (100%+unfrozen) | 0.39 | 0.99 | 1.32 | 14% |

**Two facts, both true:**
- **Not an artifact**: the u-shift is a genuine central-tendency move — successes
  march toward the object (1.06→0.82→0.66→0.51→0.39) monotonically with
  synthetic fraction, while *failures stay anchored* (0.92–1.09). A merely
  degraded/dispersed policy would not produce this success-specific directional
  shift. The "sticky prior" instrument and the dose-response are real.
- **Not full steering either**: ρ (successes) stays ≥ 1.27 (median successful
  endpoint still > 5 cm from the object in normalized error); only 14–24% of
  successes land within ρ ≤ 0.75. Steering is **partial / directional**, not
  endpoint-convergent.

**Corrected claim**: increasing counterfactual fraction produces a *partial,
directional* recovery of object-conditioned reaching (dissociable from
degradation), not settled steering. The verdict "H1 gradient competition" is
supported in DIRECTION but not proven as a full fix. Also flagged: (a) round-2
confounds mixture ratio with synthetic COUNT (M1=176 total, M2/M3=352) — needs
a 352-synthetic + 352-original factorial arm; (b) proj is weakly tied to
success (base successes have ρ = 2.29 — an anchored pregrasp can still grasp) —
add contact/lift-time endpoints; (c) `analyze_conditions.py` prints the base
row as all-mags under a "disp@50" header (reporting bug — figures use the
correctly mag-filtered base).

## Paper contribution (current framing)

1. **Anchored-not-steered, replicated**: independent retrain reproduces the
   anchoring curve digit-for-digit — the failure is a property of
   (data, recipe).
2. **The sticky prior**: five acquisition strategies × matched budget all
   improve success without touching steering — a mechanism-level negative that
   success-only evaluation cannot see, measurable only with an interventional
   steering readout (reach field).
3. **Constraint localization** (round 2): whether the stickiness is data-ratio,
   plasticity-locus, or representation — each outcome is a distinct,
   falsifiable, publishable claim.

Figures: `p2t/figs/fig1..fig4`. Raw: `p2t/eval*.jsonl`, `p2t/staging/*/meta.jsonl`.

## Adversarial review (3 lenses, 2026-07-12) — actions

Strongest objections and how they are being addressed:

1. **"Successes might be tracking (bimodality)"** (robot-expert) → tested,
   Table 1c: successes are anchored too (proj 0.82–0.95). Claim survives.
2. **"n=1 training seed per arm; ±5 pp is seed noise"** (statistician) → honest
   limitation; round-1 ranking claims (esp. D>A point estimate) demoted to
   "pre-registered gate failed" only. Seed replication queued for the
   claim-critical arms (B, D, and the round-2 winner) before submission.
3. **"Frozen VLM makes 'sticky' trivially expected (LP-FT/DFR regime)"**
   (AC + expert) → this is exactly what round-2 M3 arbitrates; paper holds
   until M3 lands. If M3 breaks the anchor, the story is *plasticity locus*
   (actionable recipe fix), not "mystery stickiness".
4. **"Servo labels may carry a proprio/copycat shortcut"** (expert) → real;
   pre-registered round-3 arm **M4 state-masked**: fine-tune on the same
   synthetic corpus with proprio dropped/noised. Plus **first-chunk
   intervention** probe (teacher-force k steps toward object; does the policy
   continue or snap back?).
5. **"Linear-probe the frozen features for object xy"** (expert + AC) → queued
   (cheap, decisive for H2 vs H3): if displacement is decodable at the
   expert's interface but never used, the locus is the expert; if not
   decodable, the VLM features are the wall.
6. **Statistics upgrades queued**: TOST-style equivalence bounds on Δproj
   (margin 0.15), wild-cluster bootstrap (10 clusters is undercovered),
   Holm correction across the contrast family, 95% CIs.
7. **Positioning** (AC): cite LP-FT (Kumar et al.), DFR (Kirichenko et al.),
   WiSE-FT, shortcut-learning (Geirhos et al.), causal confusion (de Haan et
   al.); frame the synthesis engine as MimicGen/DemoGen-lineage instrument,
   NOT a novel method; lead with the evaluation-blind-spot framing.

**Venue read** (consensus): as-is → workshop; with round-2 arbitration + probe
+ M4 + seeds + second suite → competitive CoRL / NeurIPS D&B submission built
around: *"success-only VLA benchmarks systematically misreport spatial
generalization; an interventional steering readout dissociates them, and the
stickiness localizes to ⟨round-2/3 outcome⟩."*
