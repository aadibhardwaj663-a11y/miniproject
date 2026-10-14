"""Chemical identity audit and real RDKit fingerprints from user-provided structures.
No placeholder structures. NOT executed as a model benchmark until all 20 structures are verified.
"""
from pathlib import Path
import argparse,sys
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

def validate(root,csv_path):
    root=Path(root)
    required=pd.read_csv(root/'data/drug_identity_annotations.csv').drug_group.astype(str).tolist()
    df=pd.read_csv(csv_path,dtype=str).fillna('')
    must={'drug_group','canonical_smiles','source_url','pubchem_cid','inchikey'}
    if not must.issubset(df.columns):raise ValueError(f'Required structure columns: {sorted(must)}')
    if len(df)!=len(required) or set(df.drug_group)!=set(required) or df.drug_group.duplicated().any():
        raise ValueError('All 20 selected drug groups must appear once, exactly.')
    if any(df[x].eq('').any() for x in ['canonical_smiles','source_url']):
        raise ValueError('Missing chemical structure or source. Do not fabricate SMILES.')
    mols=[];canon=[];keys=[];scaffolds=[]
    for r in df.itertuples():
        mol=Chem.MolFromSmiles(r.canonical_smiles)
        if mol is None:raise ValueError('Unparseable SMILES for '+r.drug_group)
        can=Chem.MolToSmiles(mol,isomericSmiles=True)
        key=Chem.MolToInchiKey(mol)
        if r.inchikey and r.inchikey!=key:raise ValueError('External InChIKey mismatch for '+r.drug_group)
        mols.append(mol);canon.append(can);keys.append(key)
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    if len(set(keys))!=len(keys):raise ValueError('Different drug names share a molecular InChIKey; canonicalize before splitting')
    df['canonical_smiles_rdkit']=canon;df['inchikey_rdkit']=keys;df['murcko_scaffold']=scaffolds
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    matrix=np.zeros((len(mols),1024),dtype=np.uint8)
    for i,m in enumerate(mols):DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),matrix[i])
    order=[df.index[df.drug_group.eq(group)][0] for group in required]
    matrix=matrix[order];df=df.iloc[order]
    output=root/'research/data';output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/'validated_drug_structures.csv',index=False)
    np.savez_compressed(output/'morgan_1024_fingerprints.npz',drugs=np.asarray(required),bits=matrix)
    print('Verified',len(df),'compound groups, RDKit parse and unique InChIKeys; structures and fingerprints saved')
    return df,matrix

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('structure_csv');parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=parser.parse_args()
    validate(a.root,a.structure_csv)
