# Round 1 Review (gpt-5.5, xhigh)

**Scores**: Problem Fidelity 9, Method Specificity 7, Contribution Quality 8,
Frontier Leverage 8, Feasibility 8, Validation Focus 6, Venue Readiness 7.
**Overall: 7.8/10. Verdict: REVISE.**

**Central weakness (CRITICAL, Validation Focus)**: the degradation-vs-grounding
defense is named but underspecified. Success-conditioned u-split is strong but not
bulletproof because M2/M3 also dropped clean success and ρ stays large — survivor/
selection concern (successful trajectories may be a shifted subset of a degraded
policy). Fix: a **clean-success-matched degradation null** (label/action noise,
early stop, or corrupted fine-tune matched to M2/M3 competence drop), evaluated with
the SAME success-conditioned endpoint stats, reported as a cluster-bootstrap
difference-in-differences:
`([u_succ − u_fail]_intervention) − ([u_succ − u_fail]_degradation-null)`
plus ρ / orthogonal dispersion + contact/lift-time endpoints. Intervention must show
a larger success-specific directional u-shift than the null WITHOUT merely inflating
spread. This is the single most important missing control.

**Simplification**: (1) move the 5-arm acquisition study to motivating negative
evidence, not a contribution; (2) collapse cause-localization into ONE factorial
(counterfactual fraction × canonical-presence-at-fixed-count × VLM plasticity);
(3) treat state-masking / first-chunk intervention / feature probe as validity
checks, not claims.

**Modernization**: (1) frozen-interface object-position decodability probe
("features lack position" vs "expert ignores position"); (2) paired counterfactual
rollouts (identical seed/task/magnitude) for tighter attribution; (3) frame
synthesis strictly as MimicGen/DemoGen-lineage instrument.

**Drift**: NONE (would drift only if it starts selling a fine-tuning recipe).

<details><summary>Raw response</summary>

Overall weighted 7.8/10. Degradation-vs-grounding defense is the crux; success-
conditioned u-split argues against pure degradation (successes move toward object,
failures stay anchored) but M2/M3 clean-success drop + large ρ leaves a survivor/
selection concern. CRITICAL fix = matched-degradation null + DiD on
[u_succ−u_fail]. Simplifications: acquisition→motivation; one factorial; probes as
checks. Modernization: decodability probe; paired rollouts; synthesis-as-instrument.
Verdict REVISE — strong core, not yet READY until the null control is explicit.

</details>
