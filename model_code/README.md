# Model code (current full dataset first)

| File | Action |
|---|---|
| `full_gdsc_benchmark.py` | Match/curate optional raw source ZIP and fit ten held-out-cell/drug folds using sparse Ridge (training-only genes and drug vocabulary) |
| `export_full_splits.py` | Write row-level outer-train/test and optional diagnostic inner-validation manifests for all three settings |
| `verify_full_gdsc.py` | Audit source response IDs, exact holdout membership and RMSE/MAE reconciliation |
| `drug_annotation_ablation.py` | Five-fold cold-drug annotation-only comparator |
| `strict_generalization.py` | Five-fold simultaneous new-cell AND new-drug study with three modality baselines |
| `compound_identity_audit.py` | Audit drug IDs, mapped provisional structures and unresolved chemical identity |
| `biological_model_interpretability.py` | Five predetermined drug-specific cell-cold genomic reliance tests |
| `predict_ln_ic50.py` | Train/load the full-data inference checkpoint; single pair, batch, out-of-fold and complete-grid predictions |
| `full_gdsc_figures.py` | Figures derived from actual primary fold predictions |
| `build_submission_documents.py` | Generate all report, presentation, and speech files; export reading PDFs |

The main benchmark's sparse refactor fits all ten folds using less RAM while preserving the originally defined entities, features and Ridge alpha. Solver-level changes are ~0.000020 in mean RMSE relative to the previous dense fit.

Other scripts are **historical 200 × 20 study and pilot utilities**. For judged experiment selection, use only the full-dataset entry points above. These are separate experiments. The final inference checkpoint uses sparse scaling and is fitted on all measured training responses, so it is not a source of out-of-fold metrics.

Start with `README.md` and `docs/REPRODUCE.md` from the project root.
