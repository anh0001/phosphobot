# Final Top-3 — vla-multiscopic-v2 idea round (synthesized 2026-06-12)

Inputs: `_problem_frame.md`, `_feasibility_notes.md`, `_survey_digest.md`; 5 candidates; 3 judge reports
(hostile-reviewer, experimentalist, strategist). Ground rules enforced: no first-to-show claims, term =
"canonical-prior reach", AFI mandatory for interventions, detector set mandatory for detector claims,
Eq.Bot/VLS for warp/steering claims, ≤1 GPU-day pilot, ≤1 GPU-week full study, SmolVLA+LIBERO+A6000.

---

## Top-3 ranked ideas

### #1 (FLAGSHIP) — "The Failure Is a Vector: Reach Fields, Causal Re-Anchoring, and the Canonical-Prior Bias of Vision-Language-Action Policies"

**Lineage**: Candidate 0 chassis (diagnosis-first, intervention-free headline) + grafts mandated by all
three judges: C4's thesis sentence, log-reuse economy, and statistical hygiene; C1's identity-counter-warp
lemma and failure-boundary error model; C3's E0-first ordering and wrist-cam invariance prediction;
C2's estimation framing, wasted-action lead time, complementarity matrix, and tracker-noise curves.

**one_liner**: A science-of-VLA-failure paper showing that a frozen VLA's object-displacement failure
collapses to one low-dimensional quantity — the canonical-mismatch vector ≈ the injected displacement —
measured as the first continuous reach-endpoint error field (episode-start AND mid-rollout), proven
causally sufficient by an oracle virtual-frame re-anchoring probe whose action counter-warp is the
identity by a boxed lemma of the relative-action interface, localized internally by token patching,
shown removable by pose-supervised LoRA, shown invisible to every policy-internal failure detector while
being a calibrated ESTIMATE (not an anomaly score), and shipped as a metric layer that runs unchanged on
LIBERO-PRO and HELM.

**headline_claim**: Under controlled object displacement (2–10cm × direction × timing × task ×
checkpoint), chunked flow-matching VLAs exhibit a structured canonical-prior reach: the reach-endpoint
error field quantitatively recovers the displacement vector (pooled median cos(e, −d) ≥ 0.6, aspirational
0.7; magnitude-transfer slope ‖e‖/‖d‖ in [0.5, 1.3], R² ≥ 0.4), and an oracle virtual-frame re-anchoring
that neutralizes only object position (agentview translate + proprio offset; identity action counter-warp
by the relative-action lemma; sham- and random-direction-controlled) restores ≥ 50% (red line < 30%) of
the displacement-induced conditioned-success gap — establishing mislocalization, not execution or timing,
as the dominant causal bottleneck. The headline is intervention-free: the warp is a probe, never a method.

**contributions**:
1. The reach-field probe + displacement-vector-recovery (DVR) statistic — first continuous, quantitative
   endpoint-error vector field for VLAs under controlled displacement, incl. mid-rollout; released as a
   metric LAYER running unchanged on LIBERO-PRO (2510.03827) and HELM/LIBERO-Recovery (2604.18791),
   not a sixth fork.
2. Causal sufficiency by construction: the boxed identity-counter-warp lemma (relative-action policies
   commute with world-frame translations → obs-side re-anchoring needs zero action transform; wrist cam
   provably invariant — a pre-registered falsifiable prediction) + the oracle re-anchoring probe with
   sham (d=0) and random-direction controls, plus a contribution-grade failure-boundary characterization:
   a geometric error model predicting restoration from (|d|, depth-axis component, relational-ness,
   staleness, border-fill), incl. global-translation vs object-local-paste on relational tasks.
3. Causal removal + provenance: 3-arm LoRA on the FIXED SmolVLA (privileged pose head vs Don't-Blind
   2510.25616 generic alignment vs vanilla) scored at the ENDPOINT level — the experiment
   Pose-VLA/ST4VLA/SG-VLA never ran — plus visual-token patching bridging 2603.19233's internal
   "spatially grounded motor programs" to environment-side endpoint geometry on the same model.
4. Detection upgraded to ESTIMATION + controversy adjudication: the FK-endpoint-vs-object mismatch is a
   calibrated vector (cos(m,d), |m| vs |d| calibration curves, lead time = wasted-action count, can be 0),
   demonstrated against the full internal-detector set (Sentinel 2410.04640, SAFE 2506.09937, FAIL-Detect
   2503.08558, embedding-density 2603.05147, direction-reversal 2605.28726, flow dispersion) with a
   per-failure-class complementarity matrix + hybrid OR-gate (defusing the privileged-perception attack)
   and a perturbation-class × measurement-level matrix adjudicating INT-ACT 2506.09930
   (execution-bottleneck under semantic OOD) vs localization-bottleneck under spatial displacement,
   incl. the VLA-Trace 2605.30117 attention-IoU dissociation ("attention grounds, endpoint goes canonical").

**multi_scopic_story**: The spine is Saputra 2022's intero/exteroceptive coordination primitive: the
canonical-prior reach is a measured failure of exteroception (the visibly displaced object) to override an
interiorized prior, and the DVR statistic is the unreconciled residual of that coordination AS A VECTOR —
the diagnosis IS the primitive, quantified. P6/P7 meso-level LER is transplanted as the oracle
re-anchoring probe: a meso-scale world-state estimate (d) re-anchors the micro level's egocentric inputs
via image translation + proprio offset, with the relative-action lemma making the action counter-warp the
identity — memory correcting the visual stream, exactly LER's architectural role; one paragraph of C3's
"missing-meso-level predicts this pathology" argument is lifted into the discussion (framing only, never
a contribution). P4 affordance-effectivity fit enters as a SIGNAL, not an interrupt: FK endpoint of the
predicted chunk (effectivity) vs perceived object pose (affordance) yields the per-chunk mismatch series
powering mid-rollout attribution and the estimation layer. P3 adaptive representation density maps to the
token-patching bridge: where in the hierarchy does "density follows attention" break — attention tracks
the displaced object (VLA-Trace) while the motor program stays canonical.

**decisive pilot** (split into two sequenced halves, K1 decidable after half (a); total ≈ 23 GPU-h ≈ 1
A6000-day; existing weak checkpoint fixedbuf_random_N20/seed0 + existing harness `sim/libero_active/preempt/`):
- (a) FIELD GRID FIRST (~13 GPU-h, run while warp engineering proceeds in parallel): δ ∈ {2.5, 5, 7.5,
  10}cm × 4 cardinal directions, episode-start, libero_spatial 10 tasks × 4–5 seeds, paired-RNG clean
  conditioning (~800 rollouts; ~200+ pooled continuous endpoint vectors — statistically decidable on the
  weak checkpoint, unlike binary-success pilots). Hard logging requirement: every rollout logs latents,
  flow samples (k=5), STAC stats, dispersion, kinematics — no rollout is ever re-run for a detector.
  Decide K1 overnight: pooled median cos(e,−d) ≥ 0.6 AND slope ∈ [0.4,1.5] with R² ≥ 0.3 → continue;
  else KILL the paper at 13 GPU-h, not 23.
- (b) MID-ROLLOUT + WARP ARMS (~10 GPU-h): mid-rollout δ=5cm × 4 dir at chunk boundary 2 (~200 rollouts);
  oracle warp arms at δ=5cm × 4 dir: {full warp, image-only, proprio-only, sham d=0, random-direction}
  (~350 rollouts). Outcome bands (asymmetric, pre-registered): sham-corrected restoration ≥ 50% →
  strong green-light incl. estimation/correction legs; 30–50% → proceed with diagnosis-first framing
  (pre-drafted pivot outline), restoration gate RE-ADJUDICATED at n ≥ 40 pairs on the strong checkpoint
  (McNemar) before any kill; < 30% with clean sham/random controls → drop the sufficiency leg, pivot to
  "structured but insufficient — the OOD-ness is relational" co-headlined with E3+E4.
- Immediately after a green pilot: arXiv preprint of the field figure + lemma + pilot warp result, led by
  the "failure is a vector" sentence (priority staking, non-negotiable).

**full package** (E0-gated ordering; core ≈ 150–165 GPU-h within the 168 h cap):
- E0 — Strong checkpoint + keystone replication GATE (~24–30 GPU-h): public standard SmolVLA LIBERO
  fine-tune (else train, target ≥ 65% clean); replicate DVR + closer-to-canonical on it BEFORE any atlas
  or intervention spend (kill gate K-A below).
- E1 — Reach-field atlas (~45–55 GPU-h): strong checkpoint, libero_spatial 8 dir × 4 mag × 2 timings ×
  10 tasks × 4–5 seeds; reduced 4-dir grids on libero_object + weak checkpoint (emergence axis); one thin
  second-model arm (pi0 or OpenVLA-OFT via LeRobot, episode-start only) as the budget allows. All
  detector signals logged. Deliverable: the DVR field = Figure 1.
- E2 — Causal sufficiency + failure boundary (~40 GPU-h): arms {no-warp, full oracle, image-only,
  proprio-only, wrist-cam-warp-ADDED (theory predicts neutral-to-harmful), stale-warp, sham d=0,
  random-direction, global-translate vs object-local paste on relational tasks, border-fill ablation,
  Eq.Bot-style SE(2) canonicalization 2511.15194, AFI-style action repair 2512.07472 (scoped SmolVLA
  port + oracle-stall fairness variant), VLS 2602.03973 if budget else reported numbers}; magnitude sweep
  to 10cm; fit the geometric error model; report per-chunk wall-clock.
- E3 — INT-ACT adjudication (~5 GPU-h): {spatial × 2 mag} vs {paraphrase, distractor, texture} ×
  {binary intention rate 2506.09930, continuous endpoint error, attention IoU 2605.30117} on identical
  rollouts.
- E4 — Internal–behavioral bridge (~8 GPU-h): canonical↔displaced visual-token patching at 3 sites ×
  2 directions; endpoint-transfer rate; token-geometry→endpoint regression (gap #9).
- E5 — Causal removal (~30–40 GPU-h): 3 LoRA arms (privileged pose head / 2510.25616 alignment /
  vanilla) on the fixed SmolVLA, scored on reduced reach-field grid; trim lever: cut to 2 arms.
- E6 — Metric-layer portability (~12 GPU-h): probe UNCHANGED on LIBERO-PRO episode-start and HELM
  mid-rollout protocols (anti-fork positioning made falsifiable; gap #10).
- E7 — Estimation + blindness (~5 GPU-h, log re-analysis): full detector set vs the mismatch vector;
  AUROC / FPR@95TPR / lead-time-as-wasted-actions; calibration curves cos(m,d), |m| vs |d|;
  per-failure-class complementarity matrix + hybrid OR-gate; tracker-noise sensitivity σ ∈ {0.5,1,2,4}cm
  (appendix). Framed strictly as diagnosis evidence, not a detector method; zero "reliability" vocabulary
  (portfolio-orthogonal to "Unreliable by Default").

**baselines**: AFI 2512.07472 (E2 head-to-head, scoped port + oracle-stall variant + published numbers);
Eq.Bot 2511.15194 (E2 warp pattern); VLS 2602.03973 (E2, budget-conditional); DINOBot 2402.13181 (argued);
Sentinel 2410.04640 / SAFE 2506.09937 / FAIL-Detect 2503.08558 / embedding-density 2603.05147 /
direction-reversal 2605.28726 / flow dispersion (E7 subjects); INT-ACT 2506.09930 + VLA-Trace 2605.30117
(E3 measurement arms); Don't-Blind 2510.25616 (E5 arm); RoboEval 2507.00435 (metric-positioning);
LIBERO-PRO 2510.03827 / HELM 2604.18791 / LIBERO-Plus 2510.13626 / NEBULA 2510.16263 (protocol context;
first two = E6 substrates).

**metrics**: DVR cosine (bootstrap CIs); magnitude-transfer slope β + R²; fraction closer-to-canonical;
paired-RNG conditioned success; sham-corrected restoration ratio (McNemar at n ≥ 40); endpoint-transfer
rate + token-geometry R² (E4); AUROC / FPR@95TPR / wasted-action lead time (E7); calibration cos(m,d) and
|m|-|d| R²; attention-IoU dissociation index + intention rate (E3); Δ-DVR-slope and residual endpoint
error per LoRA arm (E5); per-chunk wall-clock (E2).

**kill criteria**:
- K1 (pilot a): pooled median cos < 0.5 OR slope ∉ [0.4,1.5] OR R² < 0.3 → kill the paper (cost: 13 GPU-h).
- K2 (pilot b / E2): sham-corrected restoration < 30% at n ≥ 40 on the strong checkpoint (McNemar),
  controls clean → drop the sufficiency leg; survive only if E3+E4 carry "structured-but-insufficient".
- K-A (E0 gate): strong-checkpoint closer-to-canonical < 60% or DVR cosine < 0.4 → bias is a
  weak-checkpoint artifact → stop before E1 spend; workshop-only salvage.
- K3 (E1): DVR fails on libero_object (median cos < 0.4) AND strong checkpoint → downgrade to workshop.
- K4 (E5): pose-LoRA achieves neither ≥ 30% relative DVR-slope reduction nor significant residual-error
  reduction vs vanilla → drop "removable"; paper narrows to measure+cause+adjudicate.
- K5 (E7): any training-free internal baseline reaches AUROC ≥ 0.8 pre-contact on canonical reaches →
  drop the blindness claim (not load-bearing for the headline).
- K6 (scoop watch, continuous): continuous endpoint-error-vs-displacement measurement appears on arXiv →
  re-pivot headline to E3 adjudication + E4 bridge; mitigation = post-pilot arXiv drop.

**novelty positioning**: First QUANTITATIVE, never first-to-show. Vs AFI 2512.07472: they name "memory
trap", repair reactively in action space, qualitative episode-start evidence; we provide the continuous
field across magnitude×direction×timing×task×checkpoint incl. mid-rollout, input-side causal sufficiency
with an identity counter-warp AFI cannot have, and head-to-head comparison; we adopt "canonical-prior
reach". Vs LIBERO-PRO 2510.03827 / Text-Latent 2505.03500: qualitative sightings, success-only; we claim
quantification. Vs 2603.19233 (ICLR 2026): activation-internal only; we supply environment-side endpoint
geometry on the same model + the patching bridge — complement, not collision. Vs HELM 2604.18791 /
NEBULA 2510.16263: they own protocols, score success only; our value is what is MEASURED, proven by E6
portability. Vs INT-ACT 2506.09930: adjudicated by perturbation class, not contradicted. Vs Eq.Bot
2511.15194: global symmetry action with learned counter-map vs object-relative bias-targeted warp with
identity counter-map (a property their framework neither exploits nor notes) used as a probe. Vs
Pose-VLA 2602.19710 / ST4VLA 2602.10109 / SG-VLA 2603.22760: they own pose supervision as method; we run
the endpoint-level bias-removal experiment none reports, as instrument. Vs Sentinel/SAFE/FAIL-Detect/
Pre-VLA 2605.22446: we don't ship a detector; we show the structural blind spot and upgrade detection to
estimation as evidence. Vs RoboEval 2507.00435: target-agnostic vs target-relative.

**risks**: (1) Crowding velocity — HELM/LIBERO-PRO groups are natural authors of an endpoint field;
mitigation: 13 GPU-h K1 decision + immediate arXiv with the "failure is a vector" lead. (2) Warp artifact
confounds — sham + random-direction controls mandatory in pilot; if sham alone moves success, inpainting
engineering (+1 week). (3) Strong-checkpoint behavior change — either direction is a finding (emergence,
gap #11); if no displacement failures remain, larger δ + libero_object. (4) AFI port fidelity —
conservative "AFI-style" framing + oracle-stall variant + published numbers; E2's headline is a PROBE
result. (5) Single-model concentration — weak/strong axis + one thin second-model arm + explicit scoping
to chunked flow-matching VLAs. (6) Engineering breadth (patching + 3 LoRA arms + 2 external integrations
+ 6 detectors) is realistically 4–6 person-weeks — sequence E6/E4 after the arXiv drop; E7 is log-only.
(7) Mixed E3 outcome blunts the controversy narrative — both framings pre-written. (8) Sim-only —
positioned as science-of-failure where controlled sim displacement is the instrument.

**venue**: ICLR 2027 primary (analysis taste; direct dialogue with 2603.19233 on the same model);
fallbacks CoRL 2027, RSS 2027; pilot-validated diagnosis core to arXiv immediately after P1.

**gpu_budget**: Pilot ≈ 23 GPU-h split 13 + 10 with overnight K1. Full ≈ 150–165 GPU-h: E0 24–30,
E1 45–55, E2 40, E3 5, E4 8, E5 30–40, E6 12, E7 ~5 (log reuse). Pre-registered trim levers: drop
second-model arm (−13), halve E1 directions (−15), E5 to 2 arms (−12), VLS to reported numbers (−8).

---

### #2 (UPGRADE PATH) — "Measuring, Cancelling, and Exploiting the Canonical-Prior Reach Bias of Frozen VLAs" (Candidate 4, repaired)

**Activation condition**: choose this framing over #1 ONLY if the pilot's oracle restoration lands ≥ 50%
AND an early tracker spot-check shows ‖d̂−d‖ ≤ 2cm — i.e., the exploit arms are validated, so the
deployable gate + tracked correction can enter the headline without making AFI-port fidelity load-bearing.

**one_liner**: The failure collapses to one low-dimensional quantity — the canonical-mismatch vector ≈
the displacement vector — measured as a continuous reach-error field, cancelled via training-free
virtual-frame re-anchoring (identity action counter-warp) to prove causation, thresholded as a preemptive
grounded gate internal detectors are structurally blind to, and inverted as a tracked correction that goes
head-to-head with action-space repair.

**headline_claim**: Under 2–10cm displacement, SmolVLA's reach-endpoint error is governed by the
canonical-mismatch vector (median cos ≥ 0.7, magnitude ratio ≈ 1); cancelling it at the input restores
≥ 50% of lost conditioned success; thresholding its tracked estimate predicts failure pre-execution with
≥ 0.10 AUROC margin over all internal detectors; inverting it corrects at least as well as AFI. Judge-
mandated repairs: (a) ring-fence the diagnosis core — write the headline as a ladder, each rung
independently falsifiable, so a weak tracker/AFI arm cannot take down the conjunction; (b) add the sham
d=0 and random-direction specificity controls C4 lacked; (c) budget wall-clock at 2× the original
"3 weeks" estimate for the AFI+VLS+tracker+SAFE/FAIL-Detect porting load.

**contributions / experiments / baselines / metrics**: as Candidate 4 (E0 checkpoint, E1 diagnosis with
log-once-analyze-many, E2 oracle probe + staleness, E3 INT-ACT adjudication, E4 detector shoot-out on
reused logs, E5 tracked closed-loop vs AFI/VLS/Eq.Bot, E6 LoRA removal, E7 token-patch bridge), plus the
pilot controls above and C2's wasted-action lead time + complementarity matrix.

**pilot**: shared with #1 (identical raw material) — the pilot outcome SELECTS between #1 and #2 framing.

**kill criteria**: C4's K1–K6 retained verbatim (keystone-replication CI bounds, McNemar n ≥ 40 causal
gate, +0.05 AUROC detector gate, AFI-parity intervention gate, 2cm tracker gate, LoRA negative-finding
gate), plus the new specificity-control gate: random-direction warp helping ≥ half as much as true warp →
the cancellation is artifact-driven → fall back to #1's framing.

**top risks**: conjunction headline attacked leg-by-leg ("packaging, not mechanism"); circularity jab
("your governing vector is the displacement you injected") must be pre-rebutted in the abstract via the
tracker arm + mid-rollout naturalness argument; tracker brittleness under arm occlusion exactly when
re-anchoring matters; AFI fairness fight is load-bearing here (unlike #1).

**venue**: ICLR 2027 primary; RSS 2027, ICRA 2027 fallbacks. **gpu_budget**: ~150 GPU-h + 2× wall-clock
engineering realism (~2 weeks eng).

---

### #3 (CONTINGENT SYSTEM ANGLE) — "The Missing Meso Level" (Candidate 3, hardened)

**Activation condition**: pursue ONLY if #1's pilot greenlights AND E2's component arms show early
separations well above noise; otherwise this collapses into #1's discussion section. It is the highest
multiscopic-fidelity framing (9/10 from two judges) but strategically inverted: its identity is contingent
on an ablation outcome (its own K6), and it brushes the survey-KILLED S3 cell.

**one_liner**: Frozen chunked VLAs fail under displacement because nothing between chunk-rate execution
and the task plan re-anchors the egocentric stream; a training-free meso loop (persistent tracker + P4
fit gate + egocentric re-anchoring with identity action counter-warp) restores the missing level and
reverses the bias — with component-necessity ablations as the load-bearing defense.

**headline_claim**: A training-free chunk-boundary meso loop restores ≥ 50% of the displacement-induced
conditioned-success gap on mid-rollout ±5cm displacement, beats an audited AFI 2512.07472 reimplementation
head-to-head, and every component is individually necessary. HARDENING (judge-mandated): (a) E2 budget
arithmetic was off 2–3× — reduce the grid to 4 mag × 4 dir × 2 timings and pool n ≥ 120 pairs across both
suites; (b) the 5pp component-necessity bar at n = 60 pairs is noise — either pool to n ≥ 120 with
bootstrap CIs or demote the necessity claim to "consistent directional effects"; (c) Saputra framing
lives in intro/discussion until E6 ablations pass (C3's own K6 discipline, applied up front);
(d) positioning must lead with the measurement + causal layers, meso loop as integrative demo, to avoid
the "AFI+HELM increment in a cognitive-architecture wrapper" reject pattern.

**pilot**: shared with #1 (the oracle warp + channel arms ARE the pilot's (b) half; image-only /
proprio-only arms pre-buy the coordination ablation). **experiments**: C3's E1–E7 with the budget fixes
above. **kill criteria**: C3's K1–K6 retained; K2 (keystone replication on strong checkpoint) runs first.

**top risks**: system-paper reject pattern (sim-only, single-model, composite loop); the only net novelty
over #1/#2 is the framing K6 may force it to abandon; component-necessity statistics underpowered;
S3-cell adjacency (AFI+HELM own the neighborhood).

**venue**: CoRL 2027 primary; RSS 2027 / ICLR 2027 fallback. **gpu_budget**: ~155 GPU-h after grid fixes.

---

## Judge aggregation (incl. disagreements)

### Rankings and rank-sums
| Judge | Ranking |
|---|---|
| J0 hostile-reviewer | 4 > 0 > 3 > 1 > 2 |
| J1 experimentalist | 0 > 4 > 3 > 2 > 1 |
| J2 strategist | 0 > 4 > 2 > 3 > 1 |

Rank sums (lower = better): **C0 = 4**, **C4 = 5**, C3 = 10, C2 = 12, C1 = 14.
No veto-grade feasibility scores (minimum = 5, from J1 on C1/C2/C3 — caution-level, not veto). Two
effective strategic vetoes: C1's scoop_resistance (3/3/4 across judges) for a method-first headline, and
J1's "the pilot cannot fail" finding on C2 (K1/K2 rubber-stamp rather than gate, deferring all risk to a
~100 GPU-h eng-heavy mid-game).

### Disagreement 1 — who is #1, C0 or C4?
J0 (hostile) puts C4 first: the "measure, cancel, threshold, invert one vector" packaging answers
"so what" on every axis and the four-leg bundle survives any single-leg scoop. J1 + J2 put C0 first for
complementary reasons: J1 because C0's pilot is the only one statistically decidable on the weak
checkpoint (~200+ continuous endpoint vectors vs 14 binary restoration pairs; only design with a sham
control; oracle-only headline deletes the tracker/AFI-fairness risk class), J2 because C0's headline is
intervention-free (AFI-port fidelity never load-bearing), maximally scoop-resistant (8/10), and
portfolio-orthogonal to "Unreliable by Default". **Resolution**: C0 chassis wins on evidence-feasibility
and review posture — this matches 2 of 3 synthesis advices verbatim — and C4's genuine advantages
(thesis sentence, log-reuse economy, statistical hygiene, headline-embedded falsification) are grafted in
rather than lost. C4 survives intact as the #2 upgrade path selected by the pilot outcome, which also
honors J0's preference conditionally.

### Disagreement 2 — who is #3, C3 or C2?
J0 + J1 rank C3 third (only structurally faithful Saputra transplant, multiscopic_fidelity 9; C2 suffers
the "privileged perception trivially beats blind detectors" one-line dismissal plus the heaviest
reimplementation load for the lowest-prestige genre). J2 ranks C2 third (fresh "estimation, not
detection" framing) and flags that C3 resurrects exactly the survey-killed S3 cell with an identity
contingent on its own K6 ablation. **Resolution**: C3 takes #3 by aggregate, but hardened (budget
arithmetic fixed, necessity statistics pooled, framing demoted until ablations pass) and explicitly
contingent; C2 is eliminated as a standalone with its two best instruments (calibrated-vector estimation
framing; wasted-action lead time; complementarity matrix; tracker-noise curves) grafted into #1's E7.

### Convergent synthesis directives applied to #1
All three judges independently demanded: (i) E0-first keystone-replication gating before atlas/intervention
spend (J1, J2 via C3/C4); (ii) log-once-analyze-many so detectors cost ~0 GPU-h (J1, J2 via C4);
(iii) the identity-counter-warp observation promoted to a named, boxed lemma (J0, J2 via C1);
(iv) sham + random-direction specificity controls in the pilot (J0, J1 via C0/C1); (v) wrist-cam
invariance as a free falsifiable prediction (J0, J2 via C3); (vi) post-pilot arXiv drop as non-negotiable
priority staking led by C4's sentence (all three); (vii) failure-boundary geometric error model as
contribution-grade, not appendix (J0, J2 via C1); (viii) pre-drafted pivot outline for the 30–50%
restoration mid-band (J0); (ix) write the lemma + calibration sections first — hardest layers to
replicate (J0 via C2's risk analysis); (x) split the pilot so a K1 kill costs 13 GPU-h not 23 (J1);
(xi) scrub reliability vocabulary from E7 to keep portfolio orthogonality (J2).

---

## Eliminated

- **Candidate 1 (method-first, "Move the World, Not the Weights")** — strategic veto: scoop_resistance
  3/3/4 (an AFI-v2 or Eq.Bot follow-up kills it outright with no fallback), the ≥70% recovery bar is
  undecidable at 14 conditioning pairs and probably false given border-artifact losses, and a sim-only
  single-model method paper hangs its fate on a contestable self-ported AFI head-to-head. Its two durable
  assets — the identity-counter-warp lemma and the failure-boundary validity map — are grafted into #1
  (boxed lemma + E2).
- **Candidate 2 (detector-first, "Failure Has a Direction")** — the pilot cannot fail (oracle FK-mismatch
  under a known injected displacement is discriminative nearly by construction, so the gates rubber-stamp),
  the headline invites the "privileged perception trivially wins" dismissal, it carries the heaviest
  reimplementation load (8 detectors + 3 interventions + tracker) for the lowest-prestige genre, and it
  bleeds into the adjacent "Unreliable by Default" identity. Its best instruments — estimation-not-
  detection framing, wasted-action lead time, per-failure-class complementarity matrix + OR-gate,
  tracker-noise sensitivity — are grafted into #1's E7.
- **Candidate 0 (original)** — not killed: absorbed as the chassis of #1 with the eleven convergent grafts
  listed above (title energy, statistical hygiene, lemma, controls, ordering, log economy).
