"""Predict fitted LN_IC50 for a selected cell line and supplied drug SMILES.

WARNING: Research-only, NOT clinically validated. Model is fixed joint-cold
seed=107 Ridge trained on the curated subset, with non-chiral Morgan fingerprint.
This is an inference demonstration, not independent external validation.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from research_extension import read_project

def predict(root,cell_id,drug_group=None,smiles=None):
 root=Path(root);d=read_project(root)
 if cell_id not in d['lookup']:raise ValueError('Cell ID not present in curated expression panel: '+cell_id)
 if (drug_group is None)==(smiles is None):raise ValueError('Supply exactly one of --drug-group or --smiles')
 table=pd.read_csv(root/'research/chemistry/structure_mapping_20_provisional.csv')
 if drug_group is not None:
  rows=table[table.drug_group==drug_group]
  if len(rows)!=1:raise ValueError('Drug group not in provisional 20-structure list: '+drug_group)
  smiles=rows.iloc[0].canonical_smiles
 mol=Chem.MolFromSmiles(smiles)
 if mol is None:raise ValueError('Invalid chemical SMILES')
 model=joblib.load(root/'research/chemistry/joint_cold_seed107_fingerprint_model.joblib')
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
 fp=np.zeros((1024,),dtype=np.float32)
 DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),fp)
 i=d['lookup'][cell_id]
 vec=np.concatenate([d['expr'][i,model['selected_expression_indices']],d['mut'][i],d['cn'][i],fp]).reshape(1,-1)
 assert vec.shape[1]==model['fitted_matrix_columns']
 v=float(model['pipeline'].predict(vec)[0])
 # Plain Morgan similarity to compounds seen in training seed 107
 fingerprints=np.load(root/'research/data/morgan_1024_fingerprints.npz',allow_pickle=False)
 labels=fingerprints['drugs'].tolist();arr=fingerprints['bits'].astype(bool)
 seen={str(k) for k in model['training_drug_groups']}
 near=[]
 for name,row in zip(labels,arr):
  if name not in seen:continue
  near.append((name,float(np.logical_and(row,fp>0).sum()/max(1,np.logical_or(row,fp>0).sum()))))
 near.sort(key=lambda item:item[1],reverse=True)
 return {'cell_id':cell_id,'drug_group':drug_group or 'EXTERNAL_SMILES_UNVERIFIED',
         'predicted_LN_IC50':v,'max_training_drug_tanimoto':near[0][1],
         'closest_training_drug':near[0][0],
         'cell_seen_in_training_seed107':cell_id in json.loads((root/'research/validation/both_seed107_ids.json').read_text())['train_cells'],
         'drug_seen_in_training_seed107':drug_group in seen if drug_group else False,
         'interpretation':'In-vitro fitted LN_IC50 only; research demonstration. Not a clinical or patient treatment prediction. Compound-to-GDSC assay form unverified.'}
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
 parser.add_argument('--cell',required=True)
 grp=parser.add_mutually_exclusive_group(required=True)
 grp.add_argument('--drug-group');grp.add_argument('--smiles')
 args=parser.parse_args()
 print(json.dumps(predict(args.root,args.cell,args.drug_group,args.smiles),indent=2))
