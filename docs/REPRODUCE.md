# Reproduce and independently evaluate this submission

Run from the directory containing `README.md` using Python **3.13**. Environment uses pinned versions in `requirements.txt`. BLAS may otherwise spawn too many threads on shared machines.

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Fast verification, without fitting models

```bash
python model_code/audit_package.py --checksums   # FIRST, before any scripts rewrite source-derived artifacts
python model_code/audit_package.py               # read-only metric/document validation
python -m unittest discover -s tests -q          # regression tests
```

Expect **71 automated tests** (including compound-identity, CID-fold and saved-manifest checks), if the environment supports all pinned dependencies and office/PDF tooling. The separate `verify_full_gdsc.py` command rewrites deterministic split-ID CSVs and `verification.json`; checksum verification is intended for an unchanged extracted submission **before** commands that regenerate files. Hashes only apply to the shipped artifact bytes. Test discovery should print `OK`.

## Refit experiments from packaged curated data

```bash
# Primary 5-fold unseen-cell and unseen-drug experiment (10 sparse Ridge fits)
python model_code/full_gdsc_benchmark.py --root .
python model_code/verify_full_gdsc.py .

# Separate cold-drug annotation-only ablation (5 folds)
python model_code/drug_annotation_ablation.py --root .

# Retrospective CID-grouped five-fold drug and joint-cold robustness (10 fits)
python model_code/identity_resolution.py
python model_code/chemical_grouped_benchmark.py --root .
python model_code/export_identity_splits.py --root .  # reference-only inner manifests, not used for tuning

# Simultaneous cold-cell + cold-drug experiment (5 paired folds)
python model_code/strict_generalization.py --root .

# Drug-specific feature reliance and data-derived charts
python model_code/biological_model_interpretability.py .
python model_code/full_gdsc_figures.py .

# Regenerate Word, PowerPoint, speech, and their PDF exports
python model_code/build_submission_documents.py  # writes deliverables/scientific_report, research_presentation and speaker_notes
```

Experiments use deterministic fold assignments, training-only feature selection, sparse non-centered scaling with an intercept and fixed Ridge alpha 100. This is mathematically equivalent to centered scaling for the linear predictor, up to solver precision; the complete optimized ten-fold benchmark was refitted. Mean RMSE differs by at most 0.000020 from the historical dense implementation. No held-out settings were changed to improve this number. No held-out model selection. Original principal metrics are stored in `research/full_gdsc/summary.csv`; additional experiments have independent summaries under `research/full_gdsc/{annotation_ablation,strict_generalization}/`. The auxiliary drug-specific analysis uses its own holdout seed for interpretation and is **not** the deployment model.

## Predict `LN_IC50` from the fitted full-data checkpoint

```bash
# Known matched cell and known source-normalized drug
python model_code/predict_ln_ic50.py --mode predict --cell-id ACH-000002 --drug-group trametinib

# Truly unseen-cell / unseen-drug estimates for measured pairs
python model_code/predict_ln_ic50.py --mode oof --oof-setting cell --cell-id ACH-000002 --drug-group trametinib
python model_code/predict_ln_ic50.py --mode oof --oof-setting drug --cell-id ACH-000002 --drug-group trametinib
# Retrospective known-CID-grouped holdouts, not a pristine blind test:
python model_code/predict_ln_ic50.py --mode oof --oof-setting identity-drug --cell-id ACH-000002 --drug-group trametinib
# --oof-setting identity-both works only for saved both-cold test intersections.

# Refit full-data checkpoint; fitted predictions must not count as test metrics
python model_code/predict_ln_ic50.py --mode train

# Regenerate 649 x 286 CSV, with observed-source-response flag
python model_code/predict_ln_ic50.py --mode grid --chunk-size 1500 --output-csv predictions/full_catalog_predicted_LN_IC50.csv
```

For a batch, pass `--batch-csv path.csv` with columns `DepMap_ID,drug_group`. Add `--output-csv path.csv` to write predictions. Unknown cells/drug groups are rejected (no fabricated omics). Full-grid `prediction_status` distinguishes fitted predictions from true OOF testing; `has_observed_measurement` specifies whether an experimental target exists. **25,102 rows are predictions without measured target**.

## Optional rebuilding from original six-file source ZIP

```bash
python model_code/full_gdsc_benchmark.py --root . --raw /path/to/CCLE_mutations\(1\).zip --prepare-only
```

The original ZIP is not included. Verify its hash against `docs/DATA_PROVENANCE.md` before curation; version changes can legitimately alter joins and final counts. Publisher access/redistribution restrictions apply. For a clean reconstruction, use a separate checkout so you do not overwrite the evidence files being inspected.

## Export split manifests for independent judges

```bash
python model_code/export_full_splits.py
```

The output at `splits/full_catalog/{cell,drug,both}/` contains outer training, outer testing, and nested inner-validation identifier files. The **inner** validation is an unused diagnostic split and is included again in the full outer fold training. It was not used for parameter tuning. Full split membership and the number of unused one-cold-entity pairs are recorded in the per-fold protocol JSON.

## Output and folder organization

- `deliverables/`: human-facing report, presentation and speech, each with PDF and editable formats.
- `model_code/`: preprocessing, training, splits, evaluation, strict generalization, prediction and document-builder source.
- `research/full_gdsc/`: primary validated experiment and supplemental current full-catalog ablations. The `split_ids` folder contains **entity-level** IDs, not row-level duplicate response labels.
- `checkpoints/`: only `full_gdsc_ln_ic50_ridge.joblib` is the **current full-catalog trained inference model**. Other checkpoints are older 200 × 20 pilot examples.
- `research/chemistry/` and top-level `metrics/`, `figures/`, `tables/`: **pilot artifacts**, preserved to avoid erasing past exploratory work and **not** used in the primary results.
- `logs/`: earlier run evidence; `logs/current_audit.txt` records the current package checks; `logs/historical/` holds earlier records.

Paper/text claims can be checked against data rather than accepted on trust. `requirements.txt` also pins `python-docx` and `PyMuPDF` because the report/PDF tests use them.

## Optional chemical source acquisition and unresolved-name search

```bash
# Requires outbound access to public GitHub; checks exact upstream Git blob SHA1 values
python model_code/fetch_and_verify_gdsc_structures.py

# Requires PubChem PUG REST and ChEMBL Internet access. Only candidate matches.
python model_code/lookup_unresolved_candidates.py
```

Current offline reproducibility needs neither remote service: the included 236 GDSC-ID/CID/InChIKey records are sufficient to reproduce the identity-grouped CV. The 50 unresolved drugs have NOT been imputed. In `docs/CHEMICAL_IDENTITY.md`, review the known duplicate-name compounds and two mismatched chemical forms.

**Note:** The CID-grouped inner-train/inner-validation manifests are archival reference partitions only; the reported fold models were trained on the full outer-training data. These inner partitions do not constitute independently tuned validation scores.
