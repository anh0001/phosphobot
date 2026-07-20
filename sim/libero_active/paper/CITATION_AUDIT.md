# Citation Audit Report

**Date**: 2026-07-20 · **Bib**: `references.bib` · **Entries**: 17 (all cited) ·
**Sources**: scite MCP + arXiv abstract pages.

## Summary
| Verdict | Count |
|---------|------|
| KEEP (verified, correct) | 14 |
| FIX (metadata filled/corrected) | 3 |
| REPLACE / REMOVE | 0 |

No hallucinated entries, no wrong-context citations. All 17 exist at the claimed
arXiv IDs; the three near-collision papers were context-checked against how the draft
uses them.

## Verified entries (existence + metadata + context)

| Key | Real authors (verified) | Venue / year | arXiv | Context |
|---|---|---|---|---|
| dehaan2019causal | de Haan, Jayaraman, Levine | NeurIPS 2019 | 1905.11979 | phenomenon (anchoring) — SUPPORTS |
| afi2025memory | Xu, Wang, Wang, Xia, Huang, Xu | 2025 | 2512.07472 | memory-trap phenomenon + fix (not a readout) — SUPPORTS |
| liberox2026 | Wang, Zhang, Liu, Zhang, Cai, Liu, Liu | 2026 | 2602.06556 | success-only-eval critique benchmark — SUPPORTS |
| proprio2025 | Zhao, Lu, Zhang, ... Gao (13 auth) | 2025 | 2509.18644 | proprio shortcut hurts spatial gen — SUPPORTS |
| robustskills2026 | Emukpere, Deffayet, Renders | 2026 | 2602.24143 | decomposed success/reach metrics — SUPPORTS (Table 2 delta correct) |
| park2021object | Park, Seo, Liu, Zhao, Qin, Shin, Liu | NeurIPS 2021 | 2110.14118 | causal-confusion regularizer — SUPPORTS |
| adebayo2018sanity | Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim | NeurIPS 2018 | 1810.03292 | interventional attribution genealogy — SUPPORTS |
| mandlekar2023mimicgen | Mandlekar, Nasiriany, Wen, ... Fox | CoRL 2023 | 2310.17596 | counterfactual data lineage — SUPPORTS |
| demogen2025 | Xue, Deng, Chen, Wang, Xu | RSS 2025 | 2502.16932 | counterfactual data lineage — SUPPORTS |
| kumar2022finetuning | Kumar, Raghunathan, Jones, Ma, Liang | ICLR 2022 (Oral) | 2202.10054 | fine-tuning geometry — SUPPORTS |
| kirichenko2023last | Kirichenko, Izmailov, Wilson | ICLR 2023 | 2204.02937 | last-layer / spurious — SUPPORTS |
| wortsman2022wise | Wortsman, Ilharco, Kim, ... Schmidt (11) | CVPR 2022 | 2109.01903 | robust fine-tuning — SUPPORTS |
| smolvla2025 | Shukor, Aubakirova, Capuano, ... Cadene (14) | 2025 | 2506.01844 | the studied policy — SUPPORTS |
| liu2023libero | Liu, Zhu, Gao, Feng, Liu, Zhu, Stone | NeurIPS 2023 D&B | 2306.03310 | the benchmark — SUPPORTS |

## FIX (metadata filled; arXiv ID scite-confirmed, author list partial)
- `mobipi2025` (Mobi-$\pi$, 2505.23692) — first-author Yang + et al.; fill full author list at camera-ready.
- `n2m2025` (N2M, 2509.18671) — first-author Zhang + et al.; fill at camera-ready.
- `manibox2024` (ManiBox, 2411.01850) — first-author Tan + et al.; fill at camera-ready.

## Notes
- The near-collision papers (robustskills2026, liberox2026, afi2025memory) were
  context-verified: each is used in Related Work / Table 2 exactly for what it does
  (decomposed metrics / progressive-perturbation benchmark / memory-trap fix), and the
  draft's differentiation ("none conditions on successes / separates basin from
  grounding") is accurate.
- No entry is cited to support a claim the paper does not make.
