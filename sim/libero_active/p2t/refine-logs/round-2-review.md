# Round 2 Review (gpt-5.5, xhigh, same thread)

**Scores**: Problem Fidelity 9, Method Specificity 8.5, Contribution Quality 8.5,
Frontier Leverage 8, Feasibility 7, Validation Focus 8, Venue Readiness 8.
**Overall: 8.3/10. Verdict: REVISE.** Anchor preserved; contribution sharper;
degradation null is necessary, not bloat.

**Remaining action items:**
1. **DiD SIGN ERROR (blocking)**: with u=0 at object, grounding makes u_succ − u_fail
   *more negative*, so my stated "positive DiD CI" is inconsistent. Redefine the
   directional grounding gain as
   **G = ([u_fail − u_succ]_intervention) − ([u_fail − u_succ]_degradation-null)**,
   require **G > 0** with cluster-bootstrap CI excluding 0.
2. **Pre-register the degradation-null matching**: tolerance windows for clean AND
   displaced success, matched by magnitude/task where possible; report ALL null
   families (or the STRONGEST/most-conservative null), never the easiest comparator.
   One pre-registered primary null + robustness variants.
3. **Pre-register TOST**: declare equivalence margins for u, ρ, |v|; the contrast
   family; the correction (Holm); the clustering unit (task; wild-cluster bootstrap,
   10 clusters).
4. **Name 2nd policy + 2nd suite** with a minimum replication criterion and a rule to
   scope the claim if one fails.

**Single most important remaining gap**: a formally correct, pre-registered
degradation-null estimand (right sign, no cherry-picking, beat the STRONGEST matched
null without larger orthogonal dispersion).

<details><summary>Raw response</summary>
Overall 8.3. Anchor preserved, contribution sharper, method simpler-but-validation-
heavy (justified). Biggest issue = DiD sign/estimand: grounding → u_succ−u_fail more
negative, so require G=([u_fail−u_succ]_int)−([u_fail−u_succ]_null)>0. Pre-register
null matching (tolerance windows, report strongest null), TOST margins + clustering
unit, and name 2nd policy/suite with min replication criterion. Simplifications:
state-mask/first-chunk → appendix checks; acquisition → motivating negative;
degradation-null → one primary + robustness. Modernization: NONE material. Verdict
REVISE (not READY until the null estimand is formally correct + pre-registered).
</details>
