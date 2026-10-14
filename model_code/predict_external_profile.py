"""Strict research inference from user-supplied molecular profile (no response label).

A profile CSV has one row with the exact 100 selected expression column names,
and MUT_<gene> and CN_<gene> for each of the 10 predeclared panel genes.
Drug target and pathway provided as strings, not response labels. Unknown drug
identities are permitted for exploratory inference ONLY, without claims of validation.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import PANEL, annotation_dict

def template(root,model,output):
    b=joblib.load(Path(root)/'checkpoints'/f'{model}.joblib')
    header=list(b['expression_gene_names'])+['MUT_'+g for g in PANEL]+['CN_'+g for g in PANEL]
    pd.DataFrame(columns=header).to_csv(output,index=False)
    return len(header)

def infer(root,model,profile,drug_name,putative_target,pathway):
    b=joblib.load(Path(root)/'checkpoints'/f'{model}.joblib')
    df=pd.read_csv(profile)
    names=list(b['expression_gene_names'])+['MUT_'+g for g in PANEL]+['CN_'+g for g in PANEL]
    if len(df)!=1 or not set(names).issubset(df.columns):
        raise ValueError('Provide exactly 1 cell profile and all 120 exact required feature columns. Generate --template first.')
    arr=df[names].to_numpy(dtype=float)
    if not np.isfinite(arr).all():raise ValueError('Every supplied molecular value must be finite; no invented or imputed values accepted.')
    for gene in PANEL:
        val=float(df.iloc[0]['MUT_'+gene])
        if val not in (0.0,1.0):raise ValueError('Mutation flags must be 0/1 for '+gene)
    ann=pd.DataFrame([{'PUTATIVE_TARGET':putative_target,'PATHWAY_NAME':pathway}]); vec=b['drug_annotation_vectorizer']
    a=vec.transform([annotation_dict(ann.iloc[0])])
    full=np.column_stack([arr,a]); expected=len(b['feature_names'])
    if full.shape!=(1,expected):raise ValueError('Feature dimension mismatch')
    y=float(b['pipeline'].predict(full)[0]);assert np.isfinite(y)
    return {'predicted_LN_IC50':y,'model':model,'drug_name_label':drug_name,
            'drug_target':putative_target,'pathway':pathway,
            'warning':'Exploratory cell-line model inference only. Unseen drugs lack validated chemical structures and extrapolation is unreliable. No clinical use.'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--model',default='unseen_cell_random_forest')
    p.add_argument('--template');p.add_argument('--profile');p.add_argument('--drug-name');p.add_argument('--target');p.add_argument('--pathway')
    a=p.parse_args()
    if a.template:print('Empty feature template with',template(a.root,a.model,a.template),'columns written to',a.template)
    else:
        if not all([a.profile,a.drug_name,a.target,a.pathway]):p.error('Need --profile --drug-name --target --pathway')
        print(json.dumps(infer(a.root,a.model,a.profile,a.drug_name,a.target,a.pathway),indent=2))
