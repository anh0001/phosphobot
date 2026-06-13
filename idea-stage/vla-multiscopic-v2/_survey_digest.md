# Survey Digest — VLA Canonical-Prior Reach Bias (S1–S5 threat assessment)

Synthesized from 8 area reports (2026-06-11). Stack context: SmolVLA (chunked flow-matching, 50-step chunks) on LIBERO; custom mid-rollout object-displacement harness + mislocalization probe. Keystone (ours, unpublished): 93% (13/14) of failed reaches after 5cm displacement land closer-to-canonical, stopping ~5.4cm short ≈ the displacement vector.

## Landscape matrix

Deduped across all 8 areas; every kills/adjacent paper included; threat level reflects the worst threat any area assigned.

| Paper | ID | Mechanism | What it does (1 line) | Threat to | Level |
|---|---|---|---|---|---|
| Affordance Field Intervention (AFI) | arXiv 2512.07472 | test-time | Names the "memory trap" (VLA drives EE to original object location); proprioceptive-stall detection + rollback + 3D affordance-field waypoints + FK re-ranking of frozen pi0/pi0.5 chunks; +20.2% LIBERO-Pro, +23.5% real OOD; canonical-reach evidence qualitative only, episode-start shifts only | S3 (kills), S4 (wounds), S1 (mandatory baseline), keystone naming | **kills** |
| Pose-VLA / Universal Pose Pretraining | arXiv 2602.19710 | train-time | Discrete camera-centric pose tokens as pretraining-scale spatial priors (96.0% LIBERO, RSS 2026); no displacement or bias-removal evaluation | S5 (method novelty) | **kills** |
| Eq.Bot | arXiv 2511.15194 | test-time | SE(2) group-equivariant canonicalization wrapper: warp obs → frozen policy (CLIPort, OpenVLA-OFT) → counter-map actions; symmetry-driven, no measured bias, no object displacement, no mid-rollout | S1 (wrapper skeleton) | adjacent-high |
| LIBERO-PRO | arXiv 2510.03827 | benchmark | Episode-start perturbations collapse VLAs to ~0%; states qualitatively "EE moves toward the original, memorized location"; success-only metrics | S2, keystone qualitative priority | adjacent-high |
| HELM / LIBERO-Recovery | arXiv 2604.18791 | test-time | Defines mid-rollout ±5cm silent object displacement at subgoal boundaries; frozen-VLA harness (CLIP keyframe memory + learned MLP verifier, 50K rollouts); recovery-success metric only | S3, S4, harness protocol novelty | adjacent-high |
| Not All Features Are Created Equal | arXiv 2603.19233 | diagnosis | ICLR 2026: activation injection/SAEs on 6 VLAs incl. SmolVLA reveal "spatially grounded motor programs" bound to scene coordinates; internal-only causal evidence, no physical displacement, no endpoint geometry | S2, keystone conceptual headline | adjacent-high |
| ST4VLA | arXiv 2602.10109 | train-time | Two-stage spatial-grounding pretraining + spatially guided action post-training (ICLR 2026); no displacement/bias test | S5 | adjacent |
| SG-VLA | arXiv 2603.22760 | train-time | Auxiliary decoders incl. target-object relative pose as dense supervision for mobile manipulation | S5 | adjacent |
| GraspVLA | arXiv 2505.03233 | train-time | Foundation-scale CoT interleaving aux detection + grasp-pose estimation with actions | S5 | adjacent |
| Don't Blind Your VLA | arXiv 2510.25616 | train-time | Action fine-tuning degrades VL representations; alignment-loss fix — the generic ablation S5 must beat | S5 | adjacent |
| BYOVLA | arXiv 2410.01971 | test-time | Runtime sensitivity-guided inpainting of task-IRRELEVANT regions for frozen VLAs; appearance-level, no geometric warp, no action transform | S1 | adjacent |
| AnyCamVLA | arXiv 2603.05868 | test-time | Training-free NVS re-rendering back to training camera config for frozen VLAs (viewpoint only; world unchanged so no action counter-warp) | S1 | adjacent |
| VistaBot | arXiv 2604.21914 | test-time | Depth/pose + reprojection + diffusion inpainting maps novel views to training view for frozen ACT/pi0 (viewpoint axis) | S1 | adjacent |
| Mirage | arXiv 2402.19249 | test-time | Cross-paints source robot into observation so frozen policy sees training embodiment (RSS 2024); edits embodiment, not object geometry | S1 | adjacent |
| Equivariant Adaptation (Mondal/Kaba) | arXiv 2310.01647 / 2211.06489 | test-time | Canonicalization net + frozen model + inverse output transform — exact group actions on static vision tasks, not manipulation MDPs | S1 | adjacent |
| VLS | arXiv 2602.03973 | test-time | VLM-derived differentiable rewards steer frozen diffusion/flow-policy sampling under spatial shift (+13% LIBERO-PRO) — strongest competing fix in action-sampling space | S1 (baseline), S4 | adjacent |
| AHEAD | arXiv 2606.02486 | test-time | Latent world model forecasts future patch tokens so a frozen VLA acts where moving objects WILL be — latency for continuous motion, not displacement bias | S1, S3 | adjacent |
| DINOBot | arXiv 2402.13181 | test-time | Servo the EE until live view matches demo bottleneck view, then replay — the physical "align-then-execute" alternative S1 must argue against | S1 | adjacent |
| CAG / LIBERO-CF | arXiv 2602.17659 | test-time | Diagnoses vision-over-language shortcut; training-free dual-branch counterfactual guidance — occupies "bias found → test-time correction" narrative template on the language axis | S1 narrative, keystone framing | adjacent |
| FTM/FLA (Libero-V) | arXiv 2512.02902 | test-time | 4K-param visual-token recalibration recovers viewpoint robustness 48.5→87.1%: spatial misalignment is the recoverable, input-side failure | keystone framing, S5 analog | adjacent |
| Robust Skills, Brittle Grounding | arXiv 2602.24143 | diagnosis | SmolVLA/pi0.5 multi-object picking: primitives robust, object-location-correlation grounding collapses; binary decomposed metrics, no endpoint geometry | S2 | adjacent |
| INT-ACT | arXiv 2506.09930 | diagnosis | Binary 5cm "intention correct rate": intention survives semantic OOD while success collapses → execution-bottleneck conclusion, in direct tension with our localization-bottleneck finding | S2 | adjacent |
| VLA-Trace | arXiv 2605.30117 | diagnosis | CKA + attention-knockout: attention grounds object regions even when behavior fails; no EE endpoint metrics | S2 | adjacent |
| How VLAs Fail Differently (SafeContract) | arXiv 2605.28726 | diagnosis | Black-box kinematic failure signatures (direction-reversal AUROC ≤0.93); continuous policies fail with "physically valid wrong actions" | S4 (baseline), S2 (support) | adjacent |
| Shortcut Learning in Generalist Policies | arXiv 2508.06426 | diagnosis | Dataset diversity/fragmentation origin of shortcuts; success/feature level only | S2 framing | adjacent |
| Sentinel (STAC) | arXiv 2410.04640 | test-time | Temporal action-distribution self-consistency + VLM progress monitor — structurally blind to consistent, confident mislocalization | S4 (primary baseline) | adjacent |
| SAFE | arXiv 2506.09937 | test-time | Learned scalar failure score from VLA internal latents + conformal thresholds (NeurIPS 2025) | S4 | adjacent |
| FAIL-Detect | arXiv 2503.08558 | test-time | Success-data-only epistemic-uncertainty scores + conformal time-varying thresholds (RSS 2025) | S4 | adjacent |
| Pre-VLA | arXiv 2605.22446 | test-time | Learned dual-branch verifier gates candidate action chunks pre-execution + adaptive resampling — occupies preemptive gating with a black-box head | S4 | adjacent |
| Act, Think or Abstain | arXiv 2603.05147 | test-time | SmolVLA embedding-density OOD routing (act/think/abstain) on LIBERO/LIBERO-PRO — abstains, cannot localize the error or recover | S4 | adjacent |
| Code-as-Monitor | arXiv 2412.04455 | test-time | VLM-compiled geometric constraint monitors over tracked scene elements — checks the EXECUTED scene, never the predicted chunk | S4, S3 | adjacent |
| FOREWARN | arXiv 2502.01828 | test-time | Latent world model imagines plan outcomes, VLM evaluates — may inherit the policy's canonical prior | S4 | adjacent |
| FailSafe | arXiv 2510.01642 | test-time | Fine-tuned VLM detects failures post-hoc and proposes corrective movements (+22.6% ManiSkill) | S4 | adjacent |
| SCALE | arXiv 2602.04208 | test-time | Self-uncertainty-conditioned looking/execution modulation, training-free — intrinsic-uncertainty family S4 argues against | S4 | adjacent |
| ReViP | arXiv 2601.16667 | train-time | "False completion" via proprioceptive-progress bias; Task-Stage Observer rebalances vision; Relayout/Object-Drop suite (between-rollout) | S4 motivation, keystone sibling bias | adjacent |
| MAP-VLA | arXiv 2511.09516 | test-time | Demonstration-derived soft-prompt memory retrieved by trajectory similarity for a FROZEN VLA — semantic, not geometric, channel | S3 | adjacent |
| RoboStream | arXiv 2603.12939 | test-time | Training-free persistent 3D object grounding + causal state graph — anchors a VLM PLANNER, never a visuomotor policy | S3 | adjacent |
| P3-PO | arXiv 2412.06784 | train-time | Policy trained on tracker-maintained semantic point state; large spatial-generalization gains — tracker interface requires training | S3 | adjacent |
| RoboMME | arXiv 2603.04639 | benchmark | Memory benchmark incl. Permanence suite (occlusion tracking); binary success only | S3 eval framing | adjacent |
| Embodied-SlotSSM / LIBERO-Mem | arXiv 2511.11478 | train-time | Slot-SSM object-centric memory VLA + non-Markovian LIBERO suite (AAAI 2026) | S3 | adjacent |
| SAM2Act(+) / MemoryBench | arXiv 2501.18564 | train-time | SAM2-style memory bank in a manipulation transformer; first spatial-memory benchmark (near-solved) | S3 | adjacent |
| MemoryVLA | arXiv 2508.19236 | train-time | Perceptual-cognitive memory bank + memory-conditioned diffusion expert; strong LIBERO numbers, no displacement test | S3 | adjacent |
| Notes-to-Self | arXiv 2602.21013 | train-time | Trained-in language scratchpad storing object positions for memory-dependent tasks | S3 | adjacent |
| SD-VLA / LIBERO-memory | arXiv 2602.03983 | benchmark | Episodic "where"/position-reset tasks on LIBERO; success only | S3 eval framing | adjacent |
| AVA-VLA | arXiv 2511.18960 | train-time | Trained-in recurrent belief state reweights visual tokens (POMDP framing; CVPR 2026 Highlight); implicit, not external/explicit | S3 | adjacent |
| NEBULA | arXiv 2510.16263 | benchmark | Dual-axis capability/stress eval incl. mid-rollout "target suddenly moves" (ManiSkill3); success + action-smoothness only | S2, harness novelty | adjacent |
| RoboEval | arXiv 2507.00435 | benchmark | Target-AGNOSTIC trajectory-quality metrics (path, jerk, coordination) + stage flags; no endpoint-vs-target measurement | S2 | adjacent |
| Text Latent for pi0 | arXiv 2505.03500 | test-time | Text-latent interpolation + libero-ood; reports qualitative "spatial overfitting" — EE travels to original object location when target relocated | S2, keystone qualitative priority | adjacent |
| MolmoAct | arXiv 2508.07917 | train-time | Editable visual-trace intermediate makes intended reach path explicit — but only for its own retrained architecture | S2, S4, S5 | adjacent |
| STORM | arXiv 2601.20381 | train-time | Semantic slots on frozen vision backbones for manipulation; distractor robustness, no displacement | S3/S5 motivation | adjacent |

**Background papers (compressed):** LIBERO-Plus (2510.13626, layout collapse 95%→<30%, success-only — our seed context); LIBERO-X (2602.06556, success-only LIBERO fork); COLOSSEUM (2402.08191); GemBench (2410.01345); RoboArena (2506.18123); SimVLA (2602.18224 — self-reports position robustness as THE open axis, gap evidence); Causal Confusion (1905.11979) + Copycat (2010.14876) — temporal-shortcut framing backbone for S2; EquiBot (2407.01479) — equivariance-by-construction motivation sentence for S1; Spotlighting/SBOCR (2601.21416); SlotVLA (2511.06754 — its annotated LIBERO+ boxes/tracks are useful infrastructure for S2); RT-Affordance (2411.02704); ReMem-VLA (2603.12942); TraceVLA (2412.10345).

## Seed verdicts

### S1 — Bias-exploiting test-time re-anchoring (warp obs → frozen VLA → counter-warp actions): **WOUNDED, survives**
The wrapper PATTERN is published; the bias-exploiting object-level instantiation is not.
- **Eq.Bot (2511.15194)**: publishes warp-obs → frozen-policy → counter-map-actions on a VLA (OpenVLA-OFT) — but as GLOBAL SE(2) canonicalization from symmetry theory, with no measured bias, no object tracking, no displacement focus, no mid-rollout operation. S1's warp is object-relative and targeted at the policy's empirically measured canonical location — not a group action.
- **AnyCamVLA (2603.05868) / VistaBot (2604.21914) / Mirage (2402.19249) / BYOVLA (2410.01971)**: observation editing for viewpoint/embodiment/distractors — none moves the task-RELEVANT object, and none needs an action counter-warp (their edits don't move the action frame). S1's counter-warp of the 50-step chunk back to the true frame is the unsolved approximation problem (perspective, arm pixels, wrist cam, warp staleness) and the research content.
- **AFI (2512.07472)**: training-free fix for the same failure, but entirely in ACTION space (rollback + affordance waypoints + chunk re-ranking) — never touches the observation; mandatory head-to-head baseline.
- Surviving claim: first bias-EXPLOITING geometric obs-warp + action counter-warp under (mid-rollout) object displacement, doubling as the causal-sufficiency probe for the keystone (if warping to canonical restores success, mislocalization is proven the cause). Mandatory baselines: Eq.Bot, AFI, VLS; must argue vs DINOBot servo-then-execute.

### S2 — Reach-field probe benchmark (WHERE the policy reaches): **WOUNDED, strongest survivor**
Qualitative priority is lost; quantitative attribution is fully open.
- **LIBERO-PRO (2510.03827)**, **Text-Latent (2505.03500)**, **AFI (2512.07472)**: three independent QUALITATIVE sightings of "arm goes to the memorized location" (anecdote/figure level, episode-start only). The framing "nobody showed where the arm goes" is dead.
- **Not All Features (2603.19233, ICLR 2026)**: "spatially grounded motor programs tied to scene coordinates" on SmolVLA — preempts the conceptual headline via ACTIVATION injection; but no physical displacement, no endpoint-vs-displacement geometry, no displacement-vector recovery.
- **INT-ACT (2506.09930)**: closest existing where-measurement is a BINARY 5cm intention rate, and concludes the opposite (execution bottleneck) under semantic OOD — a publishable controversy S2 can adjudicate by perturbation class.
- Surviving claim: first CONTINUOUS, quantitative reach-endpoint error vector field vs displacement magnitude/direction/task/model, with the displacement-vector-matching statistic (93% closer-to-canonical, ~5.4cm shortfall), under mid-rollout displacement. Pitch as a mechanistic metric LAYER, not another LIBERO fork (Plus/PRO/X/V saturate that). Cite 2603.19233, LIBERO-PRO, 2505.03500, 2602.24143 prominently.

### S3 — Meso-level persistent localization memory re-anchoring a frozen VLA: **KILLED (basic form)**
The core mechanism — external grounded localization state correcting a frozen VLA's canonical-prior failures, training-free — is published.
- **AFI (2512.07472)**: external localization module (GPT-4o + Grounded-SAM + depth → 3D affordance field) re-anchors frozen pi0/pi0.5 out of memory traps, +20–23.5%, on LIBERO-Pro — exactly S3's value proposition, in action space.
- **HELM (2604.18791)**: test-time memory harness (episodic keyframe memory + verifier + controller) on a frozen VLA against MID-ROLLOUT ±5cm displacement — occupies the protocol slot too.
- **MAP-VLA / RoboStream / P3-PO / memory-VLA wave**: crowd every neighboring cell (external memory for frozen VLA; training-free persistent object anchoring for planners; tracker-state-conditioned policies).
- The residual sliver (persistent perception-side tracker state re-anchoring egocentric INPUT, predictively rather than post-stall) is real but reads as an increment on AFI+HELM. Do not pursue standalone; fold the tracker into S1 (it needs one anyway) and S4 (detection source).

### S4 — Predicted-chunk-endpoint vs detected-object consistency gate: **WOUNDED, best surviving intervention seed**
The exact (preemptive timing × geometric grounding) cell of the detector matrix is empty, but every neighbor is taken.
- **AFI (2512.07472)**: computes FK-trajectory-vs-detected-target affordance scoring — but only REACTIVELY, after a proprioceptive stall, as a candidate re-ranker; never evaluated as a failure PREDICTOR (no AUROC/lead-time).
- **Pre-VLA (2605.22446)**: preemptive chunk gating, but a LEARNED black-box verifier, no interpretable spatial signal, no detected-object comparison.
- **Sentinel/SAFE/FAIL-Detect/SCALE/Act-Think-Abstain**: all policy-internal (consistency, latents, density, self-uncertainty) — structurally blind to confident, smooth, in-distribution-looking canonical reaches; this blindness is demonstrable on our harness.
- Surviving claim: training-free, geometric, PRE-EXECUTION gate — FK endpoint of the predicted chunk vs tracked object pose — evaluated as a calibrated detector (AUROC, lead time, FPR) on displacement perturbations, with the bonus that the mismatch vector is STRUCTURED (≈ displacement vector), so it supports correction (feeds S1), not just gating. Mandatory baselines: Sentinel STAC, SAFE, FAIL-Detect, embedding-density OOD (2603.05147), direction-reversal rate (2605.28726), flow dispersion.

### S5 — Privileged pose-token / localization-head LoRA: **KILLED as method, survives as instrument**
- **Pose-VLA (2602.19710, RSS 2026)** owns "pose tokens for VLAs" at pretraining scale; **ST4VLA (2602.10109, ICLR 2026)** owns spatial-grounding supervision pipelines; **SG-VLA (2603.22760)** owns auxiliary target-object-pose decoder heads; **GraspVLA (2505.03233)** owns aux detection/pose in-the-loop. S5 has zero method novelty left.
- Surviving claim (narrow but clean): the controlled CAUSAL experiment nobody ran — does LoRA-scale localization supervision on a FIXED pretrained SmolVLA remove the canonical-prior reach bias at the ENDPOINT level (not aggregate success) under controlled displacement? Compare against Don't-Blind-Your-VLA-style generic representation alignment (2510.25616) as the ablation. Run it as one experiment inside the S2 paper, not as a contribution headline.

## Open gaps

Ranked by (uniqueness to us × paper-level value):

1. **Quantitative reach-endpoint attribution** — no published continuous endpoint-error vector field vs displacement magnitude/direction/task/model anywhere; three qualitative sightings, zero measurements. Our 93%/5.4cm statistic is the only quantitative attribution. Must be claimed as "first quantitative", never "first to show". (S2 core.)
2. **Bias-exploiting obs-warp + action counter-warp under object displacement** — the exact mechanism cell is empty (Eq.Bot = symmetry-global, no tracking; viewpoint canonicalizers = no action transform; AFI = action-space). Doubles as the causal-sufficiency probe for the keystone. Characterizing WHEN warp+counter-warp breaks (non-group action, chunk staleness, wrist cam, contact onset) is itself unstudied. (S1 core.)
3. **Preemptive + grounded failure detection** — the (preemptive timing × geometric grounding) cell is empty: AFI grounds reactively, Pre-VLA gates preemptively but ungrounded/learned, everything else is policy-internal. Plus: no head-to-head of internal vs perception-grounded detectors on displacement, where internal signals should be structurally blind. (S4 core.)
4. **Structured failure signal** — all existing detection treats failure as unstructured anomaly; nobody exploits that the mismatch vector ≈ displacement vector, giving calibrated magnitude/direction usable for CORRECTION, not just gating. Unique consequence of the keystone. (S4→S1 bridge.)
5. **Mid-rollout displacement endpoint attribution** — protocols now exist (HELM LIBERO-Recovery ±5cm; NEBULA target-moves) but both score success/recovery only; episode-start vs mid-rollout comparison of the bias, and endpoint attribution under mid-rollout displacement, are unmeasured. Our harness keeps protocol value only when paired with the reach-field metric.
6. **Causal bias-removal test for localization supervision** — Pose-VLA/ST4VLA/SG-VLA never test whether pose supervision removes the canonical-prior bias under displacement. (S5-as-instrument.)
7. **Internal↔behavioral bridge on one model** — connect 2603.19233's activation-level motor programs to environment-side displacement geometry on SmolVLA (e.g., patch canonical vs displaced visual tokens, predict reach endpoint); nobody links the two evidence levels.
8. **Adjudicating the INT-ACT controversy** — execution-bottleneck (semantic OOD) vs localization-bottleneck (spatial displacement), separable by perturbation class and measurement level (attention IoU grounds correctly per VLA-Trace, endpoints go canonical per us).
9. **Architecture provenance of the bias** — vision encoder vs VLM fusion vs flow action expert; chunk-length/architecture dependence (diagnosis work uses pi0/OpenVLA families; chunked flow-matching is underdiagnosed).
10. **Reach-field as a metric layer across fragmented LIBERO forks** (Plus/PRO/X/V/Recovery/Mem all incompatible, all success-only) — positioning, not a new fork.
11. **Endpoint-level emergence law** — at what training position-diversity/chunk-length does position memorization emerge (2508.06426/2602.24143 address it only at success level).

## Surprises

1. **AFI (2512.07472) is the single most dangerous paper.** It names our phenomenon ("memory trap"), fixes it training-free on frozen VLAs with big numbers on the same benchmark family, and killed S3 outright. Every intervention we propose must beat it head-to-head. The name "memory trap" is taken — we should adopt "canonical-prior reach" and lead with quantification.
2. **Qualitative priority on the keystone is lost three times over** (LIBERO-PRO §2.3, Text-Latent 2505.03500, AFI Fig 1). The keystone survives only as the first QUANTITATIVE, directional, mid-rollout attribution. Any draft claiming "we discovered the arm goes to the canonical location" gets desk-rejected by a reviewer who knows these.
3. **ICLR 2026 (2603.19233) preempted the conceptual headline internally** — "spatially grounded motor programs bound to scene coordinates", measured on SmolVLA itself. Threat AND opportunity: their activation-side evidence + our environment-side displacement geometry on the same model is an unclaimed bridge experiment that makes S2 stronger, not weaker.
4. **Our mid-rollout harness is no longer unique as a protocol** — HELM's LIBERO-Recovery is essentially the same perturbation (±5cm, silent, mid-episode). The harness's value now lives entirely in what we MEASURE on it (reach fields), not the perturbation itself.
5. **The field hands us a controversy to adjudicate** — INT-ACT says perception is fine and execution is broken; LIBERO-PRO/AFI/our keystone say localization is broken; VLA-Trace says attention grounds correctly even when behavior fails. A perturbation-class × measurement-level resolution is a paper-grade contribution on its own.
6. **2512.02902 is an unexpected ally**: tiny input-side representation fixes (4K params) recover viewpoint robustness — independent evidence that competence is latent in frozen VLAs and input-side re-alignment suffices, which is exactly S1's premise transplanted from viewpoint to object displacement.
7. **Crowding velocity is extreme** (Jan–Jun 2026 produced direct neighbors for every seed; 5 incompatible LIBERO robustness forks already exist). Implication: ship the diagnosis (S2) + causal probe (S1) + predictive gate (S4) as ONE coherent paper with S5 as an internal ablation, rather than slicing seeds into separate submissions that each get scooped.
8. **Replan-timing remains falsified and nothing retrieved resurrects it** — no retrieved paper attributes displacement failure to timing; SafeContract independently shows continuous policies fail with kinematically valid trajectories, corroborating mislocalization-not-control.
