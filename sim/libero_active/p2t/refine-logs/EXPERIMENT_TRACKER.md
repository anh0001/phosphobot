# Experiment Tracker — Basin, Not Grounding (MPM)

Status: TODO / RUNNING / DONE / REUSE (already have data) / BLOCKED.
Reuse column marks runs whose data already exists from round-1/round-2.

| Run ID | Milestone | Purpose | System / Variant | Suite | Seeds | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|-------|---------|----------|--------|-------|
| R001 | M0 | contact/lift endpoints offline | `endpoints.py` (3D disp + z-rise) | spatial | — | contact<=lift 7/8; lift-defined succ 65% vs fail 21% | MUST | **DONE** | CONTACT_M=1cm, LIFT_M=3cm |
| R002 | M0 | stats module | `stats.py`: wild-cluster boot + u_gap/G + TOST | — | — | reproduces base/A/M2/M3 u_succ exactly | MUST | **DONE** | u_gap monotone −.06/.27/.40/.60 |
| R003 | M0 | u/ρ/\|v\| success-cond @ 3 endpoints | base + A–E + M1–M3 (reuse) | spatial | reuse | M2 succ u: pregrasp .51 → contact .22 → lift .32 | MUST | **DONE** | anchored plan, corrected contact |
| R010 | M1 | degradation null: action-noise scale search | base + inference action-noise | spatial | 1 | clean 66/56/46% @ σ .10/.15/.20; null u_gap 0.10/0.24/−.06 | MUST | **DONE** | σ=0.15 primary (clean 56≈52); G(M2 vs null)=+.16 CI[−.22,+.40] → underpowered, need 3 seeds |
| R011 | M1 | robustness null: label-noise FT | base + label-noise FT | spatial | 1 | clean, disp@50 | MUST | TODO | match M2 competence |
| R012 | M1 | robustness null: early-stopped FT | base, early stop | spatial | 1 | clean, disp@50 | MUST | TODO | match M2 competence |
| R013 | M1 | Octo-Small integration smoke | Octo-Small | spatial | 1 | trains + reach field runs | MUST | TODO | biggest new eng; ACT fallback |
| R014 | M1 | N1 dataset build (352 syn + 352 orig) | — | spatial | — | dataset loads | MUST | TODO | reuse `make_mixed_dataset.py` |
| R020 | M2 | primary null reach field + G | action-noise null | spatial | 3 | **G vs M2**, |v| guard | MUST | TODO | decisive |
| R021 | M2 | robustness nulls reach field + G | label-noise, early-stop | spatial | 1 | G vs M2 | MUST | TODO | must beat strongest |
| R022 | M2 | critical cell N1 (fraction-vs-count) | 100%-syn + 352 canonical | spatial | 3 | u_gap 0.23±.03; **M2−N1 +0.20 [+.16,+.24] seed** | MUST | **DONE** | canonical PRESENCE re-anchors → gradient competition |
| R020 | M2 | primary null σ=0.15 × 3 noise-seed + G | base+action-noise | spatial | 3 | null u_gap 0.20±.10; G(M2) seed +.24[+.15,+.32] excl0, **cluster +.26[−.10,+.59] incl0** | MUST | **DONE** | grounding seed-robust, task-cluster underpowered |
| R024 | M2 | M2 3-seed | 100%-syn | spatial | 3 | **u_gap 0.43±0.02** (seed-robust) | MUST | **DONE** | kills n=1-seed concern |
| R030 | M2 | TOST equivalence on round-1 A–E | A,B,C,D,E (reuse) | spatial | reuse | TOST u/ρ/|v|, Holm | MUST | REUSE | stats-only |
| R040 | M3 | C1 replication — SmolVLA × object | base | object | 1 | u_succ dissociation | MUST | TODO | base object 68% already validated |
| R041 | M3 | C1 replication — Octo × spatial | Octo-Small base | spatial | 1 | u_succ dissociation | MUST | TODO | VLM-specificity test |
| R042 | M3 | C2 on chosen 2nd setting | base/100%-syn/N1/null | object OR octo | 3 | G, u | MUST | TODO | ≥1 setting; scope if only one |
| R050 | M3 | decodability probe | frozen SmolVLM feats | spatial | 1 | probe R²/MAE for obj xy vs u | MUST | TODO | CPU-cheap; feature-dump hook |
| R060 | M4 | state-masked arm | 100%-syn, proprio masked | spatial | 1 | Δu vs M2 | NICE | TODO | proprio-shortcut check |
| R061 | M4 | first-chunk intervention | teacher-force k toward object | spatial | 1 | continue vs snap-back | NICE | TODO | appendix |
| R070 | M4 | extended factorial / π0 | full 6-cell / π0 LoRA | both | 3 | u, G | NICE | TODO | extended, not claim-critical |

## First three to launch
1. **R001 + R002 + R003** (M0, ~0 GPU): contact/lift endpoints + stats module + recompute
   success-conditioned u/ρ/|v| on ALL existing data. This alone upgrades the current figures
   and validates the pipeline before spending GPU.
2. **R010–R012** (null noise-scale search): get a competence-matched null — the decisive control.
3. **R013 + R014** (Octo smoke + N1 dataset): unblock generality + the fraction-vs-count cell.
