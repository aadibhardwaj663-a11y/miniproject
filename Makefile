PYTHON ?= python
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1

.PHONY: audit checksums test smoke all verify fold-retrain identity-fold-retrain documents predictor predictor-grid

# Non-mutating, fast evaluator check.
audit:
	$(PYTHON) model_code/audit_package.py

# Run this BEFORE commands that write generated files.
checksums:
	$(PYTHON) model_code/audit_package.py --checksums

test:
	$(PYTHON) -m unittest discover -s tests -q

smoke all: audit test

# Original verifier regenerates fold-ID CSV files and verification.json.
verify:
	$(PYTHON) model_code/verify_full_gdsc.py .

# Resource-intensive: writes back fold evidence, do not use for a smoke check.
fold-retrain:
	$(PYTHON) model_code/full_gdsc_benchmark.py --root .

identity-fold-retrain:
	$(PYTHON) model_code/identity_resolution.py
	$(PYTHON) model_code/chemical_grouped_benchmark.py --root .

documents:
	$(PYTHON) model_code/build_submission_documents.py

predictor:
	$(PYTHON) model_code/predict_ln_ic50.py --mode train

predictor-grid:
	$(PYTHON) model_code/predict_ln_ic50.py --mode grid --output-csv predictions/full_catalog_predicted_LN_IC50.csv
