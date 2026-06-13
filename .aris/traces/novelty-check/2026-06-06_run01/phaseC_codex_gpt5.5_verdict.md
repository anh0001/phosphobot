# novelty-check Phase C trace — cross-model verification

- Skill: novelty-check, run 2026-06-06_run01
- Reviewer: Codex `gpt-5.5`, `model_reasoning_effort=xhigh`, sandbox=read-only
- Thread: 019e9b9e-b998-7a50-8e6e-8a4280ace057
- Idea: "Unreliable by Default" — low-demo SmolVLA+LoRA instability / interference / active-selection reliability.

## Verdict
- **Novelty 6.5/10; PROCEED-WITH-CAUTION** (ICRA/IROS/RA-L or strong CoRL workshop).
- Publishable wedge: low-demo VLA/LoRA demo-selection claims are statistically underdetermined because retrain variance + task interference ≈ alleged sample-efficiency gains. Timely vs CUPID/Demo-SCORE/DemInf/DataMIL.
- Without fixed-buffer retrain study → workshop-level. With it + 2nd suite → credible RA-L/IROS.

## Strongest rejection + rebuttal
- "Munjal 2022 / AL-reproducibility ported to VLAs."
- Rebuttal: closed-loop multi-task VLA IL is a different failure mode; fixed-buffer retrain isolates optimizer/seed variance from acquisition; signed task-transfer maps show which demos help/harm which tasks; falsifiable decomposition (method effect vs retrain/interference variance).

## Contributions
- Load-bearing: (1) fixed-buffer LoRA-retrain variance decomposition [the paper]; (2) reliability protocol; (3) signed transfer cartography (operationalized w/ CIs + sign-stability).
- Droppable/demote: mitigation (only if it beats random under retrain protocol); affordable-stack/Piper (nice not core).
- Cartography not new vs TAG/Standley; differentiator = closed-loop VLA interference along actual demo-budget acquisition trajectories.

## Must-have experiments
1. Fixed-buffer retrain N=20, k≥5 (ideally 8-10) per buffer; compare retrain spread to method delta vs random.
2. Budget sweep N=5/10/15/20, paired eval seeds, report AUC not endpoint.
3. Multiple random buffers (random-buffer variance, not only train-seed).
4. Variance decomposition (mixed-effects/hierarchical binomial: method, budget, buffer/acq seed, train seed, task, eval seed).
5. Signed transfer matrices per increment, Wilson/bootstrap intervals, sign-stability.
6. 2nd suite (libero_object).
Nice-to-have: real-Piper, machine A/B reproducibility appendix, LoRA rank sensitivity, mitigation if it beats random.

## Elevation to full venue
Build transfer map on libero_spatial → use it to choose replay/adapter-partitioning on libero_object → show reduced negative transfer + lower retrain variance. "Measured, predicted, mitigated a VLA fine-tuning failure mode."

(Phase B literature sweep saved in task output wyj4iurir; report: refine-logs/NOVELTY_CHECK_unreliable-by-default.md)
