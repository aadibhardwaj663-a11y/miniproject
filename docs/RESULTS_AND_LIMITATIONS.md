# Auditable results and scientific limits

All results are **RMSE in LN_IC50 units**, calculated on real GDSC2 observed response values. The revised CID-grouped tests are **retrospective**. Full foldwise errors and per-pair predictions: `research/full_gdsc/chemistry_grouped/`.

| Setting | Combined Ridge | Annotation-only | Genomics-only | Global mean |
|---|---:|---:|---:|---:|
| 5-fold known-CID-grouped unseen drug test, all **160,512** responses | **2.4516** | **2.5498** | 2.6563 | 2.7470 |
| 5-fold known-CID-grouped simultaneous unseen cells/drugs, **32,149** responses | **2.5008** | **2.5539** | 2.6995 | 2.7470 |

Combined beats annotations-only by **3.9%** and **2.1%** RMSE respectively. Not evidence of a broadly generalizable chemical fingerprint model; no new fingerprint-based model was fitted. Compound aliases sharing known PubChem CID are grouped together before CV. Original primary benchmarks are retained: unseen cells 1.355 vs per-drug mean 1.462; unseen drug names 2.387 vs genomics-only 2.651 and annotation-only 2.488; original name-grouped paired double cold 2.418 vs annotations-only 2.476. The two sets have different fold membership and scaling protocols and cannot be used to infer a causal improvement.

**Why pooled Pearson can mislead:** pooled unseen-cell r ~0.870 vs median within-drug Pearson **0.449**, and 57/286 drugs do not beat their individual training-only mean. The old 200×20 fingerprint gain and MG-132 interpretation are exploratory, not proof of performance. The newer Nutlin-3a/TP53 held-out permutation relies on a **drug-specific model**, not the full-data deployment model; association, not causation.

**Chemical identity:** 236/286 PubChem CID + InChIKey pairs linked by original GDSC assay drug IDs. 50 have no source-linked chemical identifier. Two confirmed shared-CID clusters and two additional provisional parent-form alias pairs were grouped without imputing the missing assay forms. MK-2206 and oxaliplatin have inconsistent prior chemical forms; salt, stereoisomer and assay materials are **not independently confirmed**. Source linkages and flagged review records under `research/full_gdsc/identity_audit/`. The original name-only split allowed one known synonym cluster on opposing folds. New grouped splits fix that **specific known leakage** but cannot rule out unknown duplicates and chemically related compounds.

**Validation and responsible use:** DepMap 19Q4 omics vs GDSC2 fitted 2023 assay labels may introduce batch/year bias. No CTRP, PRISM, wet-lab, new-patient, or entirely new screened-pair external validation. Retrospective post-hoc regrouping weakens evidence for untouched holdout claims. IC50 is fitted assay response, not clinical treatment response. Do not deploy for patient decisions. The all-data 649×286 predictor and grid are exploratory, not test scores.
