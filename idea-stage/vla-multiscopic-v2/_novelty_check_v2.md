# Novelty Check v2 — Final Verdict
## VLA Canonical-Prior Reach under Controlled Object Displacement (SmolVLA / LIBERO)

Date: 2026-06-12
Method: 5 adversarial prior-art hunts (scite + arXiv sweeps through Jun 2026, incl. deep full-text reads of the closest colliders). Known-adjacent set (AFI, LIBERO-PRO, Text-Latent, Not-All-Features, INT-ACT, HELM, NEBULA, RoboEval, Eq.Bot, AnyCamVLA/VistaBot/Mirage/BYOVLA, Sentinel/SAFE/FAIL-Detect/Pre-VLA/SCALE, Pose-VLA/ST4VLA/SG-VLA) treated as baseline; only harder collisions reported.

---

## Per-claim verdicts

| # | Claim | Verdict | Closest prior | Surviving delta | Required rewording |
|---|-------|---------|---------------|-----------------|--------------------|
| 1 | **Headline:** first continuous reach-endpoint error VECTOR FIELD under controlled displacement (mag × dir × timing × task × ckpt) + DVR statistic (median cos(endpoint-err, −d), magnitude-transfer slope) | **PARTIAL_COLLISION** | **AFI (arXiv:2512.07472)** — runs the magnitude×direction displacement sweep AND states the canonical-prior-reach phenomenon verbatim ("the VLA drives the end-effector toward the original location"), but reports only success/fail counts. Secondary: FactorWorld 2307.03659 (continuous success-vs-magnitude curves), VLATest 2409.12894 (EE3D endpoint distance vs ground truth), AHEAD 2606.02486 (timing axis, success-only). | The continuous endpoint-error **vector field** itself; the two displacement-**relative** statistics (DVR = median cos(endpoint-error, −displacement); magnitude-transfer slope of \|error\| on \|d\|); resolution across timing × task × checkpoint. No found work measures endpoint error *relative to a controlled displacement vector*. | Never claim "first controlled displacement protocol" or "first to document the phenomenon" — AFI pre-empts both. Scope "first" to: *"first DISPLACEMENT-RELATIVE characterization: a continuous endpoint-error vector field with displacement-vector-recovery statistics."* Cite AFI (qualitative phenomenon), FactorWorld (success-vs-magnitude), VLATest/INT-ACT (non-relative endpoint metrics) in the same sentence. |
| 2 | **Identity counter-warp lemma:** input-side virtual-frame translation needs identity action transform for delta-action policies; wrist view exactly invariant | **PARTIAL_COLLISION** (theory component **KILLED**) | **Wang et al., NeurIPS 2025 (arXiv:2505.13431)** — Props 1–2 state and PROVE the lemma in stronger form (full SE(3), eye-in-hand view "remains unchanged", g·aᵈ = aᵈ). Also SDP 2507.01723 (π(tS)=A, "T(3) acts trivially"), EquiDiff 2407.01812, State-free Policy 2509.18644, Hsu et al. 2203.12677. | The **zero-training, test-time, input-side intervention on a frozen third-person-camera VLA**: pixel-translate agent view + offset proprio by −d + touch nothing else; its use as a *causal diagnostic* of canonical-prior reach; the observation that translation is the unique case where canonicalization wrappers need no action back-mapping (vs Eq.Bot / 2505.11719 rotations). | Present the lemma as a **known equivariance fact with citations** (2505.13431 Props 1–2; 2507.01723; 2407.01812), novel only in its test-time deployment. Must also flag the approximation gap: 2D pixel translate of a perspective third-person view only approximates a true 3D scene translation — "exactly invariant" is claimable only for the wrist view and the action map. |
| 3 | **Oracle re-anchoring as sham-controlled causal-SUFFICIENCY probe** (warp displaced object back to canonical location; d=0 sham + random-direction controls) | **CLEAR_WITH_CITATION** | **VLA-Trace (arXiv:2605.30117)** — observation-level interventions (masking, semantic edits) for causal attribution on π0.5/OpenVLA, but verified "no spatial displacement occurs"; ablates (necessity) rather than restores (sufficiency). Also: ADS oracle-substitution 2401.10443, counterfactual-image lineage 2009.08856 / 2101.12446, Robust-Skills 2602.24143 (correlational only). | The full combination is unclaimed anywhere: counterfactual **restoration** of canonical appearance under controlled displacement, with **sham (d=0) and random-direction control arms**, quantifying causal sufficiency of mislocalization for the canonical-prior reach. | Claim the combination, not the ingredients. Don't claim first observation-space intervention for robot policies (BYOVLA, VLA-Trace, Olson et al.), first canonical-frame warp (Eq.Bot does as a method), or first oracle substitution (2401.10443 at module level). Emphasize necessity-vs-sufficiency (they mask, we restore) and probe-vs-method (Eq.Bot fixes globally, we diagnose per-object with controls). |
| 4 | **E7: detection → ESTIMATION** — FK endpoint of predicted chunk vs tracked object as calibrated VECTOR estimate of d (cos(m,d), \|m\| vs \|d\|, wasted-action lead time), vs internal detectors blind to confident canonical reaches | **CLEAR_WITH_CITATION** | **ProbeAct (arXiv:2606.09740, Jun 2026)** — training-free runtime failure handling on LIBERO with vector-valued spatial probe (hidden-state object-position regression) + EE-kinematics event detection; but never rolls the chunk through FK, no displacement estimation, no calibration, reactive/binary only. Also: 2605.28726 (scalar action-stat monitors, AUROC), ReconVLA 2604.16677 (scalar conformal "calibration"), DEFLECT 2605.19294, metamorphic testing 2602.22579, classical reach-target inference (Lyu et al. 2023, 10.1109/tcds.2022.3215093). | The estimation upgrade survives whole: FK-of-predicted-chunk endpoint vs externally tracked object as a **calibrated metric estimator of d** (direction + magnitude curves under controlled displacement), wasted-action lead time, and head-to-head blindness comparison vs internal detectors. No found work outputs a calibrated displacement vector from predicted-chunk kinematics vs perception. | Never claim: first kinematics/action-level runtime signal (ProbeAct, 2605.28726); first vector-valued spatial quantity from a VLA (ProbeAct's belief probe — contrast: theirs is the model's *belief*, ours the chunk's *kinematic implication*); bare "calibrated failure detection" (ReconVLA owns scalar calibration — always say "calibrated VECTOR/displacement estimate"); novelty of lead-time metrics per se (Sentinel/SAFE) — claim only the wasted-action operationalization; novelty of endpoint-vs-target as such (INT-ACT binary 5cm; classical human-reach inference). |
| 5 | **E4: visual-token swap canonical↔displaced with behavioral endpoint readout**, bridging activation-level "motor programs" to environment geometry on the same model | **PARTIAL_COLLISION (harder than mapped)** | **Not-All-Features (arXiv:2603.19233)** — collides HARDER than the known-adjacent mapping: its motor-program finding is itself behavioral (cross-episode activation injection with full rollout readout, 394k+ episodes, 99.8% source-trajectory override), run ON SMOLVLA ON LIBERO incl. pathway-isolated injection. The "activation intervention + behavioral readout on the same VLA" bridge per se is done. Also 2603.05487, 2509.00328, 2603.19183 (steering→behavior), 2310.08043 (maze goal-channel edits), 2410.07149 (visual-token causal localization in VLMs). | Three-part delta, verified absent by full-text read: (1) intervention unit = **visual tokens** (image-patch interchange), theirs is full-layer hidden states; (2) minimal pair = **same task, same object, canonical vs displaced** under controlled displacement, theirs is cross-task/cross-scene; (3) readout = **target-relative endpoint geometry in environment coordinates** (endpoint-shift vector ≈ displacement vector), theirs is trajectory-cosine + success. | Drop any "first to causally intervene on VLA internals with behavioral readout / first behavioral evidence of spatially grounded motor programs" — 2603.19233 does all of it on the same model+benchmark and must be FOREGROUNDED, not footnoted. Frame as: finer unit (visual tokens vs layers), controlled counterfactual (displacement minimal pairs vs cross-task scenes), geometric readout (endpoint-shift vector vs trajectory cosine). Cite 2509.00328, 2603.05487, 2603.19183; recommended 2310.08043, 2410.07149. |

---

## Overall novelty assessment

**Does the flagship survive? YES, re-scoped.** The headline claim survives as a *displacement-relative geometric characterization*, not as a discovery or a protocol claim. AFI (2512.07472) owns both the controlled magnitude×direction displacement protocol and the qualitative statement of the canonical-prior-reach phenomenon; the paper's identity must therefore be "first to *measure* the phenomenon as a continuous vector field and reduce it to two displacement-relative statistics (DVR cosine + magnitude-transfer slope), then causally dissect and exploit it" — a measurement + causal-toolkit paper, not a phenomenon-discovery paper.

**Claims needing rewording:** all five. Claims 1, 2, 5 carry hard pre-emption that makes current wording rejectable on sight; Claims 3 and 4 are clear but only under combination-scoped wording with the listed citations.

**Killed component and replacement:** the *theoretical-novelty* framing of Claim 2 is dead — the identity counter-warp lemma is Propositions 1–2 of Wang et al. 2025 (2505.13431), proved for all of SE(3). Replace with: "we exploit a known equivariance fact and turn it, for the first time, into a zero-training test-time input-side intervention on a frozen third-person VLA, used as a causal probe." This replacement is actually a *better* story: the prior theory guarantees our intervention is exact on the action side, which lends rigor to the probe in Claim 3.

**Overall novelty rating: 6.5 / 10.** No single claim is untouched, and the two most quotable assets (the displacement protocol + phenomenon statement, and the invariance lemma) are pre-empted by AFI and the equivariance literature respectively, capping the score. What keeps it solidly above the kill line: (i) the displacement-relative geometric core (vector field + DVR + slope) has zero collision across ~30 closest works including five Jan–Jun 2026 papers read at full text; (ii) two of five claims are CLEAR_WITH_CITATION, and both (sham-controlled sufficiency probe; calibrated displacement estimator) are methodological contributions other groups are demonstrably circling but have not landed; (iii) the five claims interlock — phenomenon geometry (C1) → causal mechanism (C3, C5) → exact invariance lever (C2) → deployable estimator (C4) — and no prior work holds more than one link of that chain. The risk profile is dominated by scoop velocity, not by existing prior art: ProbeAct, AHEAD, and VLA-Trace each sit one increment away from a partial collision.

---

## Mandatory citation additions

Beyond the already-mapped known-adjacent set, the paper MUST cite:

**Tier 1 — load-bearing (omission is a reviewer-findable flaw):**
- arXiv:2505.13431 — Wang, Hu, Walters et al., *A Practical Guide for Incorporating Symmetry in Diffusion Policy* (NeurIPS 2025). Props 1–2 = the counter-warp lemma; must be cited at the lemma.
- arXiv:2307.03659 — *FactorWorld / Decomposing the Generalization Gap*. Continuous performance-vs-displacement-magnitude precedent.
- arXiv:2409.12894 — *VLATest*. EE3D endpoint-distance metric precedent (non-relative).
- arXiv:2605.30117 — *VLA-Trace*. Closest observation-level causal-attribution work for C3.
- arXiv:2606.09740 — *ProbeAct*. Closest runtime vector-spatial failure-handling work for C4.
- arXiv:2605.28726 — *How VLAs Fail Differently*. Action-level black-box monitoring for C4.
- arXiv:2604.16677 — *ReconVLA*. Owns scalar "calibrated" VLA uncertainty; forces "calibrated VECTOR" wording.
- arXiv:2507.01723 — *Spherical Diffusion Policy*. T(3)-trivial action proof; lemma lineage.
- arXiv:2407.01812 — *Equivariant Diffusion Policy*. Root of relative-action canonicalization lineage.
- arXiv:2603.05487, arXiv:2509.00328, arXiv:2603.19183 — VLA activation steering with behavioral readout; required context for C5.

**Tier 2 — strongly recommended:**
- arXiv:2606.02486 — *AHEAD*. Jun-2026 timing-axis displacement work; cite to fence off the timing dimension.
- arXiv:2602.24143 — *Robust Skills, Brittle Grounding*. SmolVLA, same hypothesis family, categorical metrics.
- arXiv:2510.13626 — *LIBERO-Plus*. Large-scale object-layout perturbation, success-only.
- arXiv:2509.18644 — *State-free Policy*; arXiv:2605.13067 — *When Absolute State Fails*. Proprio-frame adjacency for C2.
- arXiv:2203.12677 — Hsu et al., wrist-view OOD folklore.
- arXiv:2505.11719 — test-time canonicalization of frozen policies (rotations; non-identity counter-warp contrast).
- arXiv:2401.10443 — ADS oracle-component substitution; precedent for oracle-substitution causal logic.
- arXiv:2009.08856, arXiv:2101.12446 — counterfactual-observation-editing lineage.
- arXiv:2602.22579 — metamorphic testing of VLAs; arXiv:2605.19294 — DEFLECT (stale-chunk mismatch).
- arXiv:2507.17049 — VLA trajectory-quality / TCP-to-object metrics.
- arXiv:2310.08043 — maze goal-channel editing; arXiv:2410.07149 — visual-token causal localization in VLMs.
- doi:10.1109/tcds.2022.3215093 — classical reach-target inference ancestor for C4.

---

## Scoop-watch list (K6)

Monitor weekly (arXiv cs.RO/cs.LG new listings + author pages) through submission:

1. **AFI authors (2512.07472)** — HIGHEST RISK. They own the displacement protocol and state our phenomenon verbatim. One follow-up adding endpoint geometry to their existing sweeps collides with C1 directly. Watch for v2/extensions.
2. **ProbeAct group (2606.09740)** — June 2026, same benchmark, training-free runtime spatial vectors. Swapping their hidden-state belief probe for chunk-FK is a one-line idea away from C4.
3. **Not-All-Features authors (2603.19233)** — already patch SmolVLA on LIBERO with behavioral readout. Adding same-task displacement minimal pairs or token-level granularity collides with C5. Watch their follow-up and citing papers.
4. **VLA-Trace authors (2605.30117)** — their framework is one "spatial relocation edit" away from C3. Watch for v2 adding object-position interventions.
5. **Northeastern/equivariance lineage (Wang, Walters, Platt: 2505.13431, 2407.01812, EquiBot orbit)** — if they ship a test-time canonicalization wrapper for pretrained VLAs, C2's deployment delta erodes. Also watch Eq.Bot (2511.15194) follow-ups.
6. **AHEAD group (2606.02486)** — dynamic-object timing axis; adding endpoint-error geometry to their conveyor sweeps would graze C1's timing dimension.
7. **Robust-Skills group (2602.24143)** — SmolVLA + placement randomization + grounding-vs-control attribution; adding continuous endpoint metrics collides with C1/C3.
8. **HELM (2604.18791)** — mid-rollout ±5cm protocol exists; success-only today. Watch for endpoint-resolved extension.
9. **Stanford VLA-interp group (2603.05487 / 2603.19183)** — feature steering with behavioral validation; token-level spatial patching is within their established toolchain (C5 risk).
10. **Proprio-encoding line (2509.18644, 2605.13067)** — if anyone tests *test-time* proprio offsetting of a frozen policy (vs retraining), C2's intervention delta shrinks.

Suggested standing queries: "displacement vector" + VLA; "endpoint error" + manipulation policy; "canonical position" / "memorized location" + vision-language-action; "counterfactual observation" + VLA; "action chunk" + forward kinematics + failure; SmolVLA + LIBERO + perturbation.
