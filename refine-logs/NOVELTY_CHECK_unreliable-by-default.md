# Novelty Check Report — "Unreliable by Default" (2026-06-06)

ARIS `/novelty-check`: Phase B 6-cluster web sweep (6 agents, 117 web tool-uses) +
Phase C cross-model verification (Codex `gpt-5.5`, xhigh).

## Proposed Method
A reliability/variance study of low-demo (5–20) SmolVLA-450M LoRA fine-tuning on
LIBERO: the claim that train-seed + per-task **interference** variance dominate
single-run active-demo-selection "wins," instrumented by **fixed-buffer LoRA-retrain
variance decomposition** and **signed per-task transfer cartography** per demo-budget
increment, packaged on an affordable open stack (phosphobot).

## Core Claims & Novelty
1. Low-demo SmolVLA+LoRA is high-variance (per-task swings 30–50pp) — **MEDIUM** — closest: Dodge'20, Henderson'18 (general), LIBERO-PRO/PhAIL (eval-side, not training).
2. Negative transfer when adding demos in multi-task low-demo VLA LoRA — **MEDIUM** — closest: LoRI (COLM'25), CORAL, M2Distill (fixes/forgetting, not a budget-indexed audit).
3. Active demo selection doesn't robustly beat random here — **MEDIUM** — closest: Munjal CVPR'22, "Navigating AL Pitfalls" NeurIPS'23, DataMIL (in-domain, concedes "small edge"); CUPID/Demo-SCORE/DemInf all claim wins.
4. Signed per-task transfer cartography per demo-budget increment — **MEDIUM** — closest: Standley ICML'20, TAG NeurIPS'21 (CV, compute-budget, single-run, no reliability framing).
5. Reliability protocol + LoRA-ensemble/soup mitigation — **MEDIUM** — closest: Bouthillier MLSys'21, Model Soups, LoRA-ensembles (off-the-shelf tools, not robot/low-demo).

## Closest Prior Work (verified arXiv IDs; 2026 IDs re-confirm at submission)
| Paper | Year | Venue | Overlap | Key difference |
|-------|------|-------|---------|----------------|
| Munjal, Robust & Reproducible Active Learning | 2022 | CVPR | AL wins often not significant vs random | Image classification; no IL/VLA/demos/interference |
| Navigating Pitfalls of AL Evaluation (2301.10625) | 2023 | NeurIPS | AL gains fragile to seeds/design; reporting protocol | Image classification; no robot/VLA/demo-budget |
| Deep RL that Matters (1709.06560) | 2018 | AAAI | Seed variance dominates; multi-seed protocol | RL MuJoCo; no IL/VLA/LoRA/interference |
| Accounting for Variance in ML Benchmarks (2103.03098) | 2021 | MLSys | Variance decomposition + efficient comparison | CV/NLP; not robot/low-demo/LoRA |
| LIBERO-PRO / LIBERO-Plus / LIBERO-X / PhAIL | '25–'26 | arXiv | VLA LIBERO results unreliable | **Eval-side** (perturbation/memorization), not training-seed/interference |
| CUPID (2506.19121) / Demo-SCORE (2503.03707) / DemInf (2502.08623) / DataMIL (2505.09603) | '25 | CoRL/RSS | Robot demo curation **beats random** | Claim positive wins; no seed/interference variance audit; not SmolVLA/LIBERO low-demo |
| TAG (2109.04617) / Standley (1905.07553) | '20–'21 | NeurIPS/ICML | Signed task cooperation/competition | CV MTL, compute-budget, single-run; not demo-budget/closed-loop VLA |
| LoRI (2504.07448) / CORAL (2603.09298) / Ortho-LoRA (2601.09684) | '25–'26 | COLM/arXiv | Cross-task LoRA interference | **Fix** methods, not a variance audit; not demo-budget-indexed |
| ConformalDAgger (2410.08852) / CRSAIL (2512.00453) / When-to-Act (2602.22474) | '25–'26 | ICLR/RSS | Conformal active demo query | Deployment-time; assume querying helps; no reliability audit |

## Overall Assessment
- **Score: 6.5/10** (ARIS Phase B: all clusters MEDIUM; Codex Phase C: 6.5/10).
- **Recommendation: PROCEED WITH CAUTION.** Publishable at ICRA/IROS/RA-L or strong CoRL workshop — *not* on "variance matters" (old), but on the robotics-specific wedge: **low-demo VLA/LoRA demo-selection claims are statistically underdetermined because retrain variance + task interference ≈ the alleged sample-efficiency gains** — a timely counterweight to CUPID/Demo-SCORE/DemInf/DataMIL.
- **Key differentiator:** closed-loop multi-task VLA interference measured **along actual demo-budget acquisition trajectories** + fixed-buffer LoRA-retrain decomposition (not gradient-affinity/static grouping, not eval-side robustness).
- **Biggest reviewer risk:** "Munjal/AL-reproducibility ported to VLAs." Defused only by the fixed-buffer retrain decomposition + signed transfer maps + a second suite.

## Load-bearing vs droppable (Codex)
- **Load-bearing:** (1) fixed-buffer LoRA-retrain variance decomposition [**the paper**]; (2) reliability protocol; (3) signed transfer cartography (only if operationalized w/ CIs + sign-stability).
- **Droppable/demote:** mitigation (only if it beats random under the retrain protocol); affordable-stack/Piper check (nice, not core).

## Positioning
Keep "SmolVLA LoRA" + "low-demo LIBERO" in the title; don't overclaim general VLA. Frame as the first empirical demonstration of this failure mode in the small-VLA low-demo regime, and as a reliability counterweight to the hot robot-demo-curation subfield.

## Elevation to a full venue (single highest-value addition)
Show the cartography **predicts/reduces** interference on a held-out setting: build the transfer map on libero_spatial → use it to choose replay/adapter-partitioning on libero_object → show reduced negative transfer + lower retrain variance under the same fixed-buffer protocol. "Measured, predicted, and mitigated a VLA fine-tuning failure mode."
