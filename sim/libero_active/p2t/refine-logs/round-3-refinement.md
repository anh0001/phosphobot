# Round 3 Refinement

## Problem Anchor (verbatim)
- **Bottom-line problem**: Reported VLA "spatial generalization" is measured by
  success rate, which cannot distinguish grounding+reaching the displaced object from
  a canonical motor program whose grasp basin is wide enough to occasionally succeed.
- **Must-solve bottleneck**: No cheap, model-agnostic, interventional readout
  dissociates "reach steered by perceived object position" from "reach anchored to
  training location, succeeds via basin tolerance."
- **Non-goals**: not a new SOTA data-generation/acquisition method; not solving spatial
  generalization; not a new benchmark.
- **Constraints**: 1× RTX 6000 Ada, sim-first (LIBERO), SmolVLA primary + one 2nd
  policy on same GPU; CoRL / NeurIPS D&B bar; honesty over hype.
- **Success condition**: reviewer agrees readout is valid+cheap+interventional; on ≥2
  policies/≥2 suites gains are predominantly basin-widening with partial directional
  grounding; cause localized to gradient competition.

## Anchor Check
- Unchanged. The only round-3 fix is pre-registering scope/feasibility — no method or
  problem change. No drift.

## Simplicity Check
- No new machinery. Round-3 LOCKS the feasible primary 2nd policy (Octo-Small) and
  pre-declares the minimum publishable matrix, which if anything *reduces* scope risk.

## Changes Made

### 1. Locked feasible 2nd policy (removes the feasibility risk)
- **Reviewer said**: use the declared fallback immediately if π0 doesn't fit; lock it
  before experiments.
- **Action**: **primary 2nd policy = Octo-Small (~27 M)** — fits 48 GB comfortably,
  trains fast, and (crucially) has NO language-pretrained VLM, so it directly tests
  whether anchoring is VLM-specific or a general narrow-placement-BC phenomenon.
  **π0 (LoRA) is an optional STRETCH arm**, run only if time remains; it is not on the
  critical path and its absence does not scope the paper.
- **Impact**: the 2nd-policy result is now cheap and certain.

### 2. Pre-registered MINIMUM PUBLISHABLE MATRIX (the last gap)
- **Action**: declare, before running, the exact set whose completion constitutes a
  submittable paper — everything else is explicitly "extended":

  **MPM (submittable):**
  - **Readout validity (Claim 1)**: SmolVLA × LIBERO-spatial base — displacement grid,
    success-conditioned u/ρ, at pregrasp + contact + lift. REPLICATED on
    SmolVLA × LIBERO-object AND Octo-Small × LIBERO-spatial (dissociation must hold on
    both, per the min-replication criterion).
  - **Mechanism (Claim 2)** on SmolVLA × LIBERO-spatial: the 3 claim-critical factorial
    cells — {base, 100%-synthetic, 100%-synthetic + equal-count-canonical} × {frozen} —
    plus the **frozen-VLM row** for those cells, at **3 seeds each**; the remaining
    factorial cells (17%, 50%, unfrozen) at **1 seed**.
  - **Degradation null**: the PRIMARY action-noise null matched to the 100%-synthetic
    arm, at 3 seeds, + the 2 robustness nulls at 1 seed; statistic G with wild-cluster
    bootstrap; |v|-dispersion guard.
  - **Decodability probe**: SmolVLA frozen features, LIBERO-spatial (1 pass).
  - Claim 2 must reproduce on **≥1** of {LIBERO-object, Octo-Small}; if only one,
    scope the cause claim to that setting explicitly.

  **Extended (not required for submission):** full 6-cell factorial × 3 seeds × both
  policies × both suites; π0 LoRA; state-mask + first-chunk appendix checks.
- **Reasoning**: fixes the exact "selectively scoped after results" risk — the paper's
  claim scope is fixed before any number is seen.
- **Impact**: feasibility now bounded (~2–3 weeks on 1 GPU for the MPM), scope is
  pre-committed.

### 3. Conservative cause-localization scoping (restated as a rule)
- If only one of {LIBERO-object, Octo-Small} reproduces Claim 2, the cause-localization
  claim is reported as holding for that setting and flagged as not-yet-general — never
  silently generalized.

## Revised Proposal
All method/thesis/contribution/validation content is as in round-2-refinement.md,
with (a) Octo-Small locked as the primary 2nd policy, (b) the Minimum Publishable
Matrix above governing scope, (c) π0 and the appendix probes marked extended/optional.
Nothing conceptual changed; this round only bounds execution and pre-commits scope.

### Final one-line thesis
An interventional, displacement-normalized, success-conditioned reach readout
dissociates object-grounding from grasp-basin tolerance in VLA policies; validated
against a competence-matched degradation null (directional gain
G = [u_fail−u_succ]_int − [u_fail−u_succ]_null > 0, no excess dispersion), it shows
counterfactual-fine-tuning success gains are predominantly basin-widening, with only
partial directional grounding that emerges solely as synthetic data displaces the
canonical demonstrations — localizing the cause to gradient competition, not a
frozen-feature representational limit.
