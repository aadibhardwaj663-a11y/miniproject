# Challenge 27: Provisional molecular-fingerprint extension

## Question and method
20/20 selected GDSC2 drug groups were associated with published PubChem compound CIDs and stereospecific SMILES candidates. 18 records were transcribed from the public [GCNPath2026 GDSC molecular-structure table](https://github.com/MinhoLee-DGU/GCNPath2026/blob/main/data/drug_data/SMILES_GDSC.csv) (Git blob `40628645d02943f0fe2cad6520713cbee2882a08`); the other 2 came from the earlier PubChem cross-check records. The mapped structures parse in RDKit and match reported InChIKeys. All 20 InChIKeys and Murcko scaffolds are unique. **The exact biological assay reagent formulation, salt, and stereochemistry have NOT independently been authenticated against GDSC batch metadata.**

Fixed Morgan radius=2, 1024-bit *non-chiral* fingerprints. Stereochemistry was retained in the validated source SMILES, but the default fingerprint encoder does not represent chirality (`includeChirality=False`). Platinum-complex oxaliplatin has a multicomponent molecular graph: organic fingerprints should not be assumed to handle that chemistry optimally.

Models were fitted with the same 5 seeds `107,207,307,407,507`, same strict entity disjoint training and heldout splits, training-only 100-expression-gene feature selection and fitting of preprocessing. Five candidate ablations plus a training-mean control were trained with a common predeclared `Ridge(alpha=100,solver='lsqr')` and no test-based hyperparameter tuning. This is an exploratory extension after the original baseline was defined and does not replace the original RF checkpoints.

## Measured performance, five-seed RMSE (mean ± sample SD)

| Experiment | Global mean | Cell-only Ridge | Cell + annotations | Cell + molecular fingerprint | Cell + both drug features |
|---|---:|---:|---:|---:|---:|
| cell | 3.158 ± 0.063 | 3.067 ± 0.053 | 1.781 ± 0.114 | 1.707 ± 0.121 | 1.706 ± 0.121 |
| drug | 3.270 ± 0.208 | 3.132 ± 0.203 | 3.358 ± 0.289 | 3.092 ± 0.149 | 3.103 ± 0.122 |
| both | 3.387 ± 0.179 | 3.295 ± 0.201 | 3.512 ± 0.291 | 3.249 ± 0.137 | 3.260 ± 0.110 |


The full chemistry-featured joint-cold model (cell + fingerprint) has mean RMSE **3.249**, versus the global mean **3.387**, the cell-only model **3.295**, and cell plus annotated targets/pathways **3.512**. Its relative RMSE change against global mean is **+4.1%** (positive means improvement). The joint cold result should not be interpreted as successful novel drug prediction unless both absolute and relative performance support that claim. No external pharmacology experiment or independent drug cohort was evaluated.

The maximum observed Tanimoto similarity between distinct selected drugs is **0.333**; all 20 Murcko scaffolds are distinct. Consequently, the ordinary leave-drug-out splits are also scaffold-disjoint for this *small selected set*, but this is not a broad scaffold-generalization benchmark on a diverse chemical library.

## Evidence and limitations
- `chemical_fingerprint_seed_metrics.csv` contains all 105 actual numeric evaluations (3 modes × 5 seeds × 7 prediction types).
- `chemical_fingerprint_test_predictions.csv` contains response/prediction rows for all 15 outer tests. Metric values are independently recalculated and checked.
- `morgan_1024_fingerprints.npz` preserves fingerprints and ordered drug names.
- `structure_mapping_20_provisional.csv` records source URL/CID/InChIKey and `ASSAY_FORM_UNVERIFIED` status per compound.
- `tanimoto_similarity.csv` stores pairwise chemical similarity; no two distinct drugs share a source InChIKey or Murcko scaffold.
- No chemical structure field was generated from drug targets, pathways, or response values. No fabricated structure or performance metric.
- Repeated seeds partition the **same** 200 cells and 20 drugs; per-seed summaries do not constitute independent external biological validation. No clinical inference or causal biomarker claim.

## References
- GCNPath2026 public GDSC compound structure table: https://github.com/MinhoLee-DGU/GCNPath2026/blob/main/data/drug_data/SMILES_GDSC.csv
- RDKit Morgan generation: https://www.rdkit.org/docs/GettingStartedInPython.html
- The specification game, Journal of Cheminformatics (2025): https://link.springer.com/article/10.1186/s13321-025-00972-y
- RDKit scaffold definitions: https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html
