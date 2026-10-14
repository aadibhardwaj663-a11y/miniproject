# Checkpoints: primary vs historical

**Current usable inference model:** `full_gdsc_ln_ic50_ridge.joblib`, accompanied by `full_gdsc_ln_ic50_metadata.json`, fitted on all 160,512 observed pairs. Supports known cell and known drug group only; **not** the source of cross-validation accuracy claims. Train and use via `model_code/predict_ln_ic50.py`.

The other four `unseen_{cell,drug}_*.joblib` checkpoints are historical **200-cell × 20-drug pilot** baselines and are not comparable with the 649 × 286 full-dataset test scores. They remain present only for provenance and regression tests. Primary held-out models were refitted fold by fold and are represented by saved test predictions, metrics and code, rather than by a single all-data checkpoint. To rebuild, see `docs/REPRODUCE.md`.
