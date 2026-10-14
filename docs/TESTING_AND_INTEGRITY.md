# Verification record and reproducibility boundary

The source archive was unpacked and checked using Python 3.13 with the exact pinned versions in `requirements.txt`, and single-threaded BLAS (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).

## Tests and integration checks

- **71/71 automated tests passed** on this audit revision, including five additional regression tests for exact annotation-encoding equivalence, identity-grouped cold-drug/cold-both OOF lookups and full-grid chunking.
- A separate read-only package audit checks input cardinalities, source curve IDs, metrics reconciled to saved held-out predictions, chemical-ID coverage, report/PPTX page counts, model and catalog output counts.
- Live known-pair inference and cell-held-out, drug-held-out and identity-drug-held-out OOF lookups succeeded; an identity-both request outside the actual paired test intersection raises an explicit error instead of inventing a prediction.
- Rebuilt the complete 185,614-row known-cell/drug grid with 13-drug chunks; all values matched the shipped grid within the documented floating-point tolerance, with **25,102** genuinely unmeasured pairs left labeled as predictions.
- **All ten primary CV folds were refitted successfully** with the optimized sparse Ridge implementation, in **28.9 seconds** (peak RSS ~1.32 GB) on this local run with single-threaded BLAS. Saved fold/test metrics were rewritten from these actual fits; they reconcile to the paired experimental labels. Against the earlier dense implementation, the largest shift in mean RMSE across the seven summary rows is **0.000020**, well below the report's three-decimal precision. Earlier dense retraining was interrupted after fold 1 because it exceeded the execution window; the new sparse run completed. These timing figures describe this environment, not a hardware-independent benchmark.
- Source-linked PubChem CID/InChIKey mapping: 236/286 assay name groups; 50 still lack exact assay identity. Two conservative active-parent groupings are provisional. No external/clinical experimental validation.

## Run the checks

```bash
# BEFORE regeneration, on pristine extraction:
python model_code/audit_package.py --checksums

# Read-only science/artifact verification:
python model_code/audit_package.py

# Automated unit, regression and integration checks:
python -m unittest discover -s tests -q

# Reproducible training (resource-intensive, writes new outputs):
python model_code/full_gdsc_benchmark.py --root .
```

The original raw publisher six-file ZIP is absent; fully reconstructing curation requires downloading it separately and verifying the recorded SHA-256 (`docs/DATA_PROVENANCE.md`). Exact source environment, nonclinical boundaries and retrospective test-set reuse are disclosed in the report and supporting documents. Historical `logs/` records are not proof that the current release was completely retrained.
