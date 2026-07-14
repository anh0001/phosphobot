[Round 3 re-evaluation]

I applied all four round-2 action items. Check the Problem Anchor is preserved, then
judge whether the method is now top-venue robust.

Key changes:
1. Fixed the estimand sign. Directional success-specific grounding gain is now
   G = ([u_fail − u_succ]_intervention) − ([u_fail − u_succ]_degradation-null),
   claimed iff G > 0 with wild-cluster-bootstrap CI excluding 0 AND no larger
   orthogonal |v| dispersion than the matched null.
2. Pre-registered degradation null: primary = action-noise FT matched to the
   intervention's clean AND displaced success (±3 pp), per (task, magnitude);
   robustness = label-noise FT + early-stopped FT; anti-cherry-pick rule = must beat
   the STRONGEST matched null, report G vs all three.
3. Pre-registered TOST margins (|Δu|<0.15, |Δρ|<0.25, |Δ|v||<0.20), Holm-corrected
   over the enumerated contrast family, clustering unit = task via wild-cluster
   bootstrap (10 clusters).
4. Named 2nd suite = LIBERO-object (base already validated), 2nd policy = π0 LoRA
   (fallback Octo-Small / ACT), with a minimum replication criterion and an explicit
   scoping rule if a policy shows no anchoring.

Revised proposal path (read yourself):
/srv/storage/roboserver1/home/anhar/codes/phosphobot/sim/libero_active/p2t/refine-logs/round-2-refinement.md
(round-1-refinement.md holds the unchanged sections it builds on.)

Please re-score the same 7 dimensions + overall (same weighting). Say whether the
anchor holds, whether any BLOCKING issue remains, and give the verdict (READY only
if overall >= 9 and no blocking issue). If anything still blocks READY, name the
single most important remaining item precisely and concretely — do not invent new
scope. Same output format.
