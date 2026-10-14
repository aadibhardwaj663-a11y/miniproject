# Challenge 27 | Gene-expression-based drug sensitivity prediction

A reproducible research mini-project predicting **GDSC2 `LN_IC50`** from matched DepMap/CCLE expression, mutation and copy-number features, plus GDSC drug annotations. The target is natural-log fitted IC50, **not AUC or patient treatment response**.

**Data:** 649 matched cell lines, 286 screened drug-name groups, 160,512 measured responses. The linked chemical-identity audit records 236 GDSC-ID-associated PubChem CID/InChIKey pairs and flags 50 unresolved assay identities. Salt, mixture and stereochemical forms are not universally verified.

## Start here: evaluator deliverables

| What the challenge requests | Single authoritative location |
|---|---|
| Scientific report | [`deliverables/scientific_report.pdf`](deliverables/scientific_report.pdf) (`.docx` also supplied) |
| 10-slide presentation, including integrity slide | [`deliverables/research_presentation.pdf`](deliverables/research_presentation.pdf) (`.pptx` also supplied) |
| Brief presentation script | `deliverables/speaker_notes.pdf` (`.docx` also supplied) |
| Integrity declaration | [`deliverables/integrity_statement.md`](deliverables/integrity_statement.md) |
| Data processing, feature/split generation, training, evaluation and inference | [`model_code/`](model_code/) and [`docs/EVALUATOR_GUIDE.md`](docs/EVALUATOR_GUIDE.md) |
| Pinned environment | [`requirements.txt`](requirements.txt), Python 3.13 |
| Explicit train/validation/test splits | `splits/full_catalog/` and `research/full_gdsc/chemistry_grouped/splits/` |
| Fitted LN_IC50 inference checkpoint | `checkpoints/full_gdsc_ln_ic50_ridge.joblib` |
| Actual held-out predictions and metrics | `research/full_gdsc/{fold_predictions.csv,fold_metrics.csv}` and `research/full_gdsc/chemistry_grouped/` |
| Prediction grid, distinguishing measured from unmeasured | `predictions/full_catalog_predicted_LN_IC50.csv` (185,614 rows, including 25,102 unmeasured combinations) |
| Scientific provenance, identity decisions and limitations | [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md), [`docs/CHEMICAL_IDENTITY.md`](docs/CHEMICAL_IDENTITY.md), [`docs/RESULTS_AND_LIMITATIONS.md`](docs/RESULTS_AND_LIMITATIONS.md) |
| Exact reproduction commands | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) |

The current full-data **Ridge** inference checkpoint is distinct from the reported cross-validation fold fits. Four older RF/mean checkpoints and their 200 × 20 pilot evidence remain for traceability, **not** as primary performance claims. The helper inner-validation manifests are diagnostic only: they were **not** used to tune the reported outer-fold models.

## Key evaluation results

Five-fold mean RMSE, lower is better; different rows have different test populations and must **not** be interpreted as head-to-head model comparisons.

| Held-out setting | Combined genomics + annotations | Relevant baseline | Notes |
|---|---:|---:|---|
| Unseen cell lines, known drugs | **1.355** | Per-drug training mean: 1.462 | Pooled r 0.870, **median within-drug r 0.449** |
| Unseen drug **names** | **2.387** | Annotation-only: 2.488 | Old names-only holdout with known alias risk |
| Simultaneous unseen cell/drug names | **2.418** | Annotation-only: 2.476 | 31,998 test observations |
| Chemically **identity-grouped** unseen drugs | **2.452** | Annotation-only: 2.550 | 160,512 responses, known CID aliases grouped |
| Identity-grouped unseen cells **and** drugs | **2.501** | Annotation-only: 2.554 | 32,149 held-out intersection observations |

The newer CID-grouped evaluations are **retrospective, not blinded**. Identity grouping reduces *known* synonym leakage but does not prove chemically independent holdouts for all 286 names. No independent PRISM/CTRP or laboratory validation was performed; do not claim clinical or broad unknown-drug performance.

## Verify and predict

From the extracted project root:

```bash
python -m pip install -r requirements.txt
python model_code/audit_package.py             # read-only checks of packaged evidence
python -m unittest discover -s tests -q        # scientific and inference regression tests
python model_code/predict_ln_ic50.py --mode predict --cell-id ACH-000002 --drug-group trametinib
python model_code/predict_ln_ic50.py --mode oof --oof-setting identity-drug --cell-id ACH-000002 --drug-group trametinib
```

On an **unchanged extraction**, `python model_code/audit_package.py --checksums` also verifies every packaged SHA-256 digest. Scripts that regenerate split files or reports can alter their bytes, so checksum verification comes first. For full-fold refitting, rebuilding the catalog, batch input, and source-package reconstruction, see [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

**Research conduct:** original six-file publisher ZIP is not redistributed; its SHA-256 and version/provenance are supplied. Dataset redistribution rights remain those of the publishers. OpenAI ChatGPT assisted with implementation, auditing and writing. This project is for **educational and research use only**.
