# Challenge 27 | Research extension: simultaneous cold generalization and biological evidence

**Status: Executed for all three holdout settings over five random seeds each. Chemical-structure augmentation is prepared but awaits authenticated drug structures.**

## Main methodological advance

The original college mini-project evaluated previously unseen cell lines and previously unseen drugs separately. This extension adds a **strict simultaneous cold split**: no test cell identity, test drug group or known drug alias appears in the fitted training data. Every seed creates 200 measured test pairs (40 held-out cell lines x 5 held-out drug groups). The 1,400 mixed cross-quadrant pairs are deliberately excluded from the joint-cold fit and its test; they are not recycled as training examples. Frozen split IDs and source curve IDs are in `research/validation/`.

For comparison, the cell-only and drug-only holdout tests each have 800 and 1,000 test pairs per seed. Seeds are 107, 207, 307, 407 and 507. **The same 200 lines and 20 compounds recur across the five partitions, so these are repeated random partitions of one dataset, not five independent external cohorts.**

Each model uses a predeclared training-only expression variance filter for the top 100 features, 10 mutation indicators, 10 copy-number values and source-assigned target/pathway drug annotations. All ablations in the new experiment use the **same regularized Ridge estimator (alpha=100)** and the same preprocessing family, so differing results primarily reflect feature blocks. The earlier 100-tree Random Forest experiments and their verified test metrics remain unchanged and should never be compared as if they used the same training partition or estimator.

## Repeated held-out results (mean +/- SD across 5 seeds, RMSE)

| Held-out mode | Global training mean | Ridge, full features | Ridge, cell only | Strong drug-specific baseline |
|---|---:|---:|---:|---:|
| Unseen cell lines | 3.158 +/- 0.063 | **1.781 +/- 0.113** | 3.066 +/- 0.052 | 1.865 +/- 0.145 |
| Unseen drugs | 3.270 +/- 0.208 | 3.358 +/- 0.289 | **3.132 +/- 0.203** | N/A |
| Jointly unseen cells and drugs | 3.387 +/- 0.179 | 3.512 +/- 0.291 | **3.295 +/- 0.202** | N/A |

The forest's earlier successful unseen-cell result does **not** translate to reliable joint-cold extrapolation. In the full-feature Ridge ablation, adding available drug target/pathway annotations does not improve mean joint-cold performance. This could be due to incomplete drug descriptors, limited drug diversity, high response heterogeneity and confounded assay ranges. It does not imply chemical structure or drug identity is generally unimportant.

## Feature ablation, strict joint holdout

| Fixed Ridge feature set | Mean RMSE, five independent splits |
|---|---:|
| Expression (100) + mutation/CN (20) + drug annotations | 3.512 |
| Expression (100) + drug annotations | 3.533 |
| Mutation/CN (20) + drug annotations | 3.566 |
| Drug annotations alone | 3.592 |
| Expression (100) + mutation/CN (20), without drug annotations | **3.295** |

These are **exploratory retrospective results**, not an independently registered prospective benchmark. All feature selection and encoding occurs within the training entity partition of each new split.

## Applicability-domain diagnostic

For each jointly unseen cell line, we calculated the Euclidean distance to its closest training cell in the 100 training-selected, standardized expression features. Mean absolute error across the held-out drug groups was positively correlated with this distance when treating **200 cell-line-by-seed groups** as units (Spearman rho about 0.225; nominal p=0.00135). This suggests farther molecular contexts are more difficult, but pseudo-replication remains because cell identities can appear in multiple seeds. No distance threshold for reliable prediction is established.

## Literature-guided candidate biology (NOT model-explanation causality)

Eight predetermined drug/gene pairs were screened in the 200-cell selected cohort using Mann-Whitney tests and Benjamini-Hochberg correction; within-lineage label permutation (999 shuffles) was additionally applied for exploratory adjustment. Mutation-positive means **a recorded nonsilent coding variant**, while mutation-negative means no such retained record, **not independently verified wild type**. These observational findings are neither held-out biomarkers nor wet-lab confirmed mechanisms.

| Source drug | Gene | Variant-record / no-record cell counts | Observed median LN_IC50 difference | Within-lineage permutation p (exploratory) |
|---|---|---:|---:|---:|
| Nutlin-3a (-) | TP53 | 139 / 61 | +2.689 | 0.001 |
| Selumetinib | KRAS | 35 / 165 | -1.008 | 0.046 |
| Selumetinib | BRAF | 21 / 179 | -1.005 | 0.036 |
| Taselisib | PIK3CA | 16 / 184 | -1.845 | 0.001 |
| Trametinib | KRAS | 35 / 165 | -0.638 | 0.171 |

Positive median difference corresponds to higher fitted LN_IC50 (apparent resistance); negative corresponds to lower fitted LN_IC50 (apparent sensitivity). Do not interpret the nominal p-values as independent clinical discoveries, and account for multiple comparisons and limited tissue-level counts.

**External literature comparisons:**

- Nutlin-3a antagonizes MDM2 and generally depends on functional p53. Ovarian and AML cell-line investigations observed sensitivity in wild-type TP53 backgrounds and resistance in mutated TP53 backgrounds, consistent with the direction of our cohort-level association. [Nutlin-3a ovarian study](https://doi.org/10.1371/journal.pone.0135101); [AML MDM2 antagonist study](https://pmc.ncbi.nlm.nih.gov/articles/PMC1895324/). This study does **not** verify whether all source-listed mutations abolish p53 function.
- Selumetinib inhibits MEK1/2. Independent preclinical cell-line screens linked RAS/RAF pathway alterations with sensitivity in certain cancer subtypes, but results are lineage- and background-dependent. [Molecular Cancer Therapeutics, PMID 20587667](https://pubmed.ncbi.nlm.nih.gov/20587667/); [transcriptional pathway signatures](https://pmc.ncbi.nlm.nih.gov/articles/PMC3166660/).
- Taselisib targets PI3K, with PIK3CA mutation-associated sensitivity in preclinical breast models. Our smaller, histologically mixed observational result is directionally consistent but **not** a prospective validation. [Molecular Cancer Therapeutics, DOI 10.1158/1535-7163.MCT-19-0284](https://aacrjournals.org/mct/article/doi/10.1158/1535-7163.MCT-19-0284/87983/am/Predictive-and-Pharmacodynamic-Biomarkers-of).

## Methodology context in prior research

- [DeepCCDS (2025)](https://doi.org/10.1002/advs.202416958) emphasizes separate cell-line, drug and joint split strategies and interpretable cancer driver signals.
- [TransCDR (2024)](https://doi.org/10.1186/s12915-024-02023-8) shows that cold scaffold and simultaneous cold evaluations are materially harder than easier warm-start settings.
- [DrugGene (2024)](https://doi.org/10.1186/s12859-024-05669-x) integrates expression, mutations, copy numbers, biological subsystems and chemical drug features.
- [2026 cross-dataset benchmark](https://pubmed.ncbi.nlm.nih.gov/41525315/) emphasizes evaluating cross-dataset generalization, which our small single-GDSC2 subset has not yet demonstrated.

## Research prototype

- `model_code/predict_cli.py` supports curated known cell/drug pairs using the original trained Random Forest, including a warning when the cell was present during training and global feature importance (not individual causal attribution).
- `model_code/predict_external_profile.py` accepts a one-row fully supplied 120-feature genomic profile and drug pathway/target annotation, after creating a blank column template. It requires all numeric values to be finite and does not supply or invent missing measurements.
- This is not a graphical application or a patient-facing clinical predictor.

## Blocked improvement: validated chemical structures

The original ZIP has **no drug SMILES/InChIKeys**. The container cannot access PubChem directly to automatically fetch records. `research/data/compound_structures_TEMPLATE.csv` lists exactly 20 selected groups and their original IDs and lookup hints. The `model_code/chemical_features.py` parser uses RDKit to confirm the real structures, derive canonical isomeric SMILES, verify InChIKeys, reject duplicate compounds, identify Murcko scaffolds and compute 1,024-bit radius-2 Morgan fingerprints.

**No chemical-structure model has been trained. No scaffold validation has been claimed.** An authenticated 20-drug chemical mapping needs to be supplied and reviewed first. Suggested sources and file format are in `DATA_NEEDED.md`.

## Reproduction

```bash
python model_code/predict_cli.py --cell-id ACH-000002 --drug-group mg-132
python model_code/predict_external_profile.py --template research/data/EXTERNAL_PROFILE_TEMPLATE.csv
python model_code/robust_repeated.py .  # may exceed per-execution time limit; five seeds in each setting
python model_code/new_research_figures.py
python -m unittest discover -s tests -v
```

The full five-seed metrics and output predictions are already saved. For reliable reproduction on constrained environments, run each mode and seed individually with `robust_repeated.run(root, modes=(mode,), seeds=[seed])`, collecting output tables after each run; the saved manifest files are independent of the outcome values.

## Remaining integrity concerns

No external GDSC cohort, chemistry-aware scaffold split, wet-lab intervention, protein expression assays, independently verified TP53 functional classification, or external patient outcomes are available. Source omics and drug responses come from different release periods. The 200 cells and 20 drugs were coverage-selected; statistical inferences do not generalize automatically to the full original screening universe. The three plot/evaluation modes are repeated retrospective tests, not a registered blinded benchmark.

### Chemistry update (completed, provisional)
20 structures now have source-linked PubChem candidates and independent matching RDKit InChIKeys. Fixed five-seed fingerprint-enhanced Ridge models were executed, with numerical outputs and new report at `research/chemistry/CHEMICAL_EXTENSION_REPORT.md`. **This supersedes prior claims that structures are unavailable.** Assay-specific reagent forms remain unverified; the 20-drug universe is too limited for strong independent external generalization claims.
