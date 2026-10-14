# Independent evaluator guide

**Suggested reading order:** `README.md` → `deliverables/scientific_report.pdf` → `deliverables/research_presentation.pdf` → `docs/RESULTS_AND_LIMITATIONS.md` → `docs/REPRODUCE.md`.

| Judge's rubric / requirement | Evidence to inspect |
|---|---|
| **40% Results:** unseen-cell / unseen-drug RMSE, MAE, Pearson and baselines | `research/full_gdsc/{fold_metrics.csv,fold_predictions.csv,summary.csv}`; `research/full_gdsc/chemistry_grouped/{fold_metrics.csv,test_predictions.csv,summary.csv}` |
| **35% Data/methodology:** original drug IDs, chemical aliases, paired holdout, omics features | `research/full_gdsc/identity_audit/`, `model_code/{full_gdsc_benchmark.py,chemical_grouped_benchmark.py,identity_resolution.py}`, `docs/CHEMICAL_IDENTITY.md` |
| **35% Biological interpretation and errors:** within-drug / lineage / gene permutation | `research/full_gdsc/{per_drug_skill_cell_holdout.csv,biological_interpretability/}`, `docs/RESULTS_AND_LIMITATIONS.md` |
| **10% Reproducibility:** dependencies, split manifests, recorded model outputs | `requirements.txt`, `splits/full_catalog/`, `research/full_gdsc/chemistry_grouped/splits/`, `tests/`, `checkpoints/`, `docs/REPRODUCE.md` |
| **15% Communication:** report, presentation, integrity | `deliverables/` (one report, one ten-slide deck, short speech and integrity statement) |
| **Predicted LN_IC50**: inference on known cells/drugs, honest OOF access | `model_code/predict_ln_ic50.py`, `checkpoints/full_gdsc_ln_ic50_ridge.joblib`, `predictions/full_catalog_predicted_LN_IC50.csv` |

Use `python model_code/audit_package.py` for a **read-only** consistency check and `python -m unittest discover -s tests -q` for the fuller suite. Only the latter may regenerate deterministic artifacts during testing. Verify `SHA256SUMS` beforehand if byte identity is important.

**Distinguish three outputs:** `predicted_LN_IC50` in the catalog is **fitted** (not a held-out result); the original `fold_predictions.csv` is genuine out-of-fold by cell or drug *name*; chemical-identity-grouped retrospective holdouts live in `chemistry_grouped/test_predictions.csv`. The both-cold test contains **only the held-out intersections**, not all potential cell–drug combinations.

**Scope:** current 649-cell/286-drug work is under `research/full_gdsc`. The small 200×20 pilot, four historical checkpoints, old plots and research extensions are retained for provenance and are **not independent replications** of the current results.
