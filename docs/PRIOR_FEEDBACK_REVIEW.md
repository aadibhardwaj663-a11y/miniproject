# Reviewer's earlier criticisms: checked against the current submission

This is a concise audit of whether criticisms of a prior **200-cell × 20-drug random-forest pilot** remain true of the current **649-cell × 286-name Ridge benchmark**. The pilot remains archived under `research/chemistry/` and `research/validation/`. It is not represented as the current full-catalog model.

| Earlier criticism | Current assessment, with verifiable evidence |
|---|---|
| Only 200 cells × 20 drugs, one split with five cold drugs | **Resolved for the principal benchmark**: 649 cells, 286 drug-name groups, 160,512 real responses; five-fold grouped holdouts. Identity-grouped 5-fold cold drug and paired double-cold added in `research/full_gdsc/chemistry_grouped/`. |
| Inflated pooled unseen-cell Pearson and poor MEK-inhibitor within-drug predictions | **Still relevant as an interpretation risk.** Pooled r ~0.870 vs median within-drug r **0.449**; 57/286 drug groups do not beat their mean. MEK errors are disclosed. Do not quote pooled r alone. |
| Trametinib / selumetinib / PD0325901 identical target/pathway tokens | **Partly fixed for familiar-drug inference** by drug-name one-hot identities. For unseen drug names, these IDs are unknown; targets/pathways remain an intrinsically weak chemical representation. No claim of validated fingerprint uplift. |
| Fragile 1.4% fingerprint gain in an old 5-drug pilot | **Not used as main evidence** and not claimed to have been repaired by fingerprint tuning. Current models use annotation/genomic ablations; mapped chemistry primarily corrects split identity. |
| MG-132 importance tiny and lineage associations are not model explanations | **Improved but not completely solved**: five separate drug-specific held-out Ridge fits with prespecified mutation/CN permutation analyses, including Nutlin-3a/TP53. They measure those fitted models' reliance, not causality or full predictor global importance. |
| Test-set reuse | **Remains.** Earlier evaluations were examined before this chemical regrouping, so it is retrospective, not a pristine blind holdout. Never call it independently prospective. |
| DepMap 19Q4 vs GDSC2 2023; no external test | **Remains**, recorded in `docs/DATA_PROVENANCE.md`. No CTRP/PRISM scores fabricated. |
| Duplicated reports, checkpoint directories, inconsistent evidence | **Improved**: one canonical report/deck in `deliverables/`; separate present all-data checkpoint and explicitly historical pilot models. `checkpoints/` has a tiny training-mean baseline artifact intentionally, which is not a learned genomic model. `model_code/` is the authoritative source directory. |
| Names not identical to chemical identities | **Partly resolved**: 236/286 GDSC-ID-linked PubChem CID/InChIKeys, known shared-CID and two additional potential active-parent groups stay together; 50 unresolved, salt/structure conflicts disclosed. |

**Evaluator priority:** correctness of known-identity holdouts, baselines, within-drug failure reporting, reproducible outputs, and nonclinical boundaries. Source-limited unresolved identities and retrospective evaluation appropriately cap scientific maturity.
