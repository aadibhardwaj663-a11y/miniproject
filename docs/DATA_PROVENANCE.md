# Data provenance, curation and redistribution

**Raw source package:** originally uploaded `CCLE_mutations(1).zip`, SHA-256 `b07f5b2f1174082a0422ae1eaace944d5fb6b15be276d48d70f9eb041b2fddf2`. It is **not included in this submission ZIP**. Source contents: `sample_info.csv` (DepMap Public 19Q4), `CCLE_expression_full.csv`, `CCLE_mutations.csv`, `CCLE_gene_cn.csv`, `GDSC2_fitted_dose_response_27Oct23.xlsx`, and source README (`data/source_depmap_README.txt`). Publisher sites: https://depmap.org/portal/data_page/ and https://www.cancerrxgene.org/downloads/drug_data.

Source GDSC2 fitted response workbook had **242,036** records; conservative mapping yielded **160,512** measured pairs for **649** cells and **286** normalized drug names. Requirements: exact COSMIC/DepMap join, triple-omics availability, cell name/Sanger ID checks, conservative exclusions, finite response values, one source drug ID per normalized name group picked by coverage (never by target response). Outputs: `research/full_gdsc/{matched_responses.csv,matched_omics.npz,audit.json,canonical_drug_ids.csv}`. No unmeasured LN_IC50 target was filled.

**Raw reconstruction** requires obtaining the original six-file package yourself and passing its path with `--raw` to `model_code/full_gdsc_benchmark.py` (see `docs/REPRODUCE.md`). It will not reproduce from online sources automatically. All model experiments can be rerun **without** the raw ZIP from the packaged source-derived matched data. No fresh external dataset or prospectively measured response is included.

**Molecular labels:** expression 58,676 source columns, training-selected top 100 by variance among *unique training cells*; binary variant-record panel and numeric copy numbers for ten predefined genes; drug names, putative targets and pathways (18,807 of the 160,512 rows lack a putative-target entry). Source linked name groups alone do not establish chemical identity: see `research/full_gdsc/identity_audit/`.

**Redistribution:** The packaged matched matrices and source-derived response records may be subject to original DepMap, CCLE, GDSC and dependent-provider licenses. Review the publishers' current redistribution terms before uploading this repository publicly. Sharing this ZIP for evaluation does **not** grant new rights to republish any upstream datasets. This work is for educational/research purposes, not clinical use.

**AI disclosure:** OpenAI ChatGPT was used to assist with code, audits, report writing and presentation generation. Metric tables are derived from saved experimental response data; provenance limitations are not hidden.

## GDSC drug identity source linkage (2026-10 audit)

The 236 GDSC `DRUG_ID` -> PubChem CID -> InChIKey links were retrieved from public GCNPath2026 supplementary CSVs and recorded with Git object SHA1 values in `docs/CHEMICAL_IDENTITY.md` and `research/full_gdsc/identity_audit/audit.json`. GCNPath2026 is an annotation **secondary source**, not a newly measured assay dataset. Manual checking of the experimental salt, stereochemistry, mixture and concentration remains outstanding; 50 group names lack corresponding CID/Key. See `docs/CHEMICAL_IDENTITY.md` for source URLs, discrepancy records, online refresh script and limits.
