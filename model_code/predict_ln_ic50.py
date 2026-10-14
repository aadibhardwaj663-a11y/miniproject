"""LN_IC50 inference using the full 649-cell / 286-drug matched benchmark.

Modes:
  train: fit and save a FULL-DATA deployment Ridge model with bounded-memory sparse input.
         Inference models fitted on all pairs are NOT held-out test models.
  predict: infer LN_IC50 for known DepMap cell IDs and known GDSC drug-name groups.
  oof: read honest out-of-fold LN_IC50 predictions from the saved grouped-fold benchmark.
  grid: score the 649 x 286 known cell-drug grid, explicitly flagging any response without a measurement.

No clinical inference or missing features invented. Drug features are target/pathway/name
annotations, NOT chemical structure for this full-catalog model. Predictions are ln(IC50),
not IC50 in molar units, not cell survival, and not clinical doses.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_gdsc_benchmark import tokens

MODEL_NAME='full_gdsc_ln_ic50_ridge.joblib'
SEED=27013
PANEL=['TP53','KRAS','BRAF','EGFR','PIK3CA','PTEN','APC','RB1','MYC','CDKN2A']


def load_data(root:Path):
    out=root/'research'/'full_gdsc'
    obs=pd.read_csv(out/'matched_responses.csv')
    with np.load(out/'matched_omics.npz',allow_pickle=False) as z:
        cells=z['cell_ids'].astype(str).tolist()
        E=z['expression'].astype(np.float32)
        M=z['mut'].astype(np.float32)
        CN=z['cn'].astype(np.float32)
        headers=z['expression_headers'].astype(str).tolist()
    if obs.duplicated(['DepMap_ID','drug_group']).any():
        raise ValueError('Repeated matched drug/cell pair')
    if not np.isfinite(obs['LN_IC50'].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite response present')
    if E.shape!=(len(cells),len(headers)) or M.shape!=(len(cells),10) or CN.shape!=(len(cells),10):
        raise ValueError('omics matrix dimensions differ from manifest')
    if not all(np.isfinite(a).all() for a in (E,M,CN)):
        raise ValueError('Nonfinite omics values present')
    meta=obs.drop_duplicates('drug_group').set_index('drug_group').sort_index()
    if obs.groupby('drug_group')['PUTATIVE_TARGET'].nunique(dropna=False).max()>1 or obs.groupby('drug_group')['PATHWAY_NAME'].nunique(dropna=False).max()>1:
        raise ValueError('Ambiguous within-drug annotations; must resolve before inference')
    return obs,cells,E,M,CN,headers,meta


def train_full(root:Path):
    obs,cells,E,M,CN,headers,drug_meta=load_data(root)
    cell_lut={c:i for i,c in enumerate(cells)}
    uniq=np.arange(len(cells))
    var=E[uniq].var(axis=0,dtype=np.float64)
    gene_idx=np.lexsort((np.arange(len(var)),-var))[:100]
    if len(gene_idx)!=100 or not np.isfinite(var).all():
        raise ValueError('Invalid training-selected expression genes')
    cell_matrix=np.column_stack((E[:,gene_idx], M, CN)).astype(np.float32)
    drug_names=drug_meta.index.astype(str).tolist()
    # Build drug descriptors once per distinct drug, then index to pair rows.
    vec=DictVectorizer(sparse=True,sort=True)
    tuples=[tokens(row) for row in drug_meta.reset_index().itertuples(index=False)]
    drug_sparse=vec.fit_transform(tuples).astype(np.float32)
    i_cells=np.fromiter((cell_lut[x] for x in obs['DepMap_ID']),dtype=np.int32,count=len(obs))
    lut_drug={v:i for i,v in enumerate(drug_names)}
    i_drugs=np.fromiter((lut_drug[x] for x in obs['drug_group']),dtype=np.int32,count=len(obs))
    # Select genes BEFORE response-row expansion (avoids a >25GB allocation).
    # CSR format avoids materializing the full dense [cell | onehot drug] array.
    X=sparse.hstack((sparse.csr_matrix(cell_matrix[i_cells]),drug_sparse[i_drugs]),format='csr',dtype=np.float32)
    y=obs['LN_IC50'].to_numpy(dtype=np.float64)
    assert X.shape[0]==len(y) and np.isfinite(X.data).all()
    # The sparse fit has its own preprocessing; it must not be misrepresented
    # as exactly the dense fold model from the main benchmark.
    model=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(with_mean=False),Ridge(alpha=100,solver='lsqr',tol=1e-4))
    model.fit(X,y)
    test_pred=model.predict(X[:12])
    if test_pred.shape!=(12,) or not np.isfinite(test_pred).all():
        raise ValueError('Inference smoke test failed')
    saved={
        'pipeline':model,'cell_ids':cells,'drug_names':drug_names,
        'selected_expression_idx':gene_idx,'selected_expression_genes':[headers[j] for j in gene_idx],
        'panel':PANEL,'drug_vectorizer':vec,'drug_metadata':drug_meta[['PUTATIVE_TARGET','PATHWAY_NAME']].reset_index(),
        'training_observations':int(len(y)),
        'data_source':'research/full_gdsc/matched_responses.csv',
        'model_role':'all-data inference, NO held-out performance claim',
        'feature_matrix':{'num_cell':120,'num_drug':int(drug_sparse.shape[1]),'total':int(X.shape[1])},
        'response':'LN_IC50', 'training_seed':SEED,
        'feature_generation':'100 largest expression variances on all 649 training cells; fixed 10 mutation and 10 CN; training-fit drug identity/target/pathway',
        'model_family':'Ridge alpha 100; sparse StandardScaler(with_mean=False), unlike dense CV training',
    }
    ckpt=root/'checkpoints'/MODEL_NAME
    ckpt.parent.mkdir(exist_ok=True,parents=True)
    joblib.dump(saved,ckpt,compress=3)
    back=joblib.load(ckpt)
    np.testing.assert_allclose(back['pipeline'].predict(X[:12]),test_pred,atol=1e-9,rtol=1e-9)
    note={k:v for k,v in saved.items() if k not in ('pipeline','cell_ids','drug_names','selected_expression_idx','selected_expression_genes','drug_vectorizer','drug_metadata')}
    note['unique_cells']=len(cells); note['unique_drugs']=len(drug_names); note['checkpoint_bytes']=ckpt.stat().st_size
    (root/'checkpoints'/'full_gdsc_ln_ic50_metadata.json').write_text(json.dumps(note,indent=2))
    print(json.dumps(note,indent=2))
    return ckpt


def prepare_batch(root:Path,model,batch:pd.DataFrame):
    needed={'DepMap_ID','drug_group'}
    if not needed.issubset(batch.columns):
        raise ValueError('Input CSV must contain DepMap_ID and drug_group')
    _,cells,E,M,CN,headers,drug_meta=load_data(root)
    if model['cell_ids']!=cells:raise ValueError('Checkpoint uses different cell ID order')
    name_set=set(model['drug_names']);cell_lut={c:i for i,c in enumerate(cells)}
    unknown_cell=sorted(set(batch.DepMap_ID.astype(str))-set(cells))
    unknown_drug=sorted(set(batch.drug_group.astype(str))-name_set)
    if unknown_cell or unknown_drug:
        raise ValueError('Unknown cell IDs '+str(unknown_cell[:5])+' or drug groups '+str(unknown_drug[:5])+'. No fallback/imputed entities.')
    indices=np.fromiter((cell_lut[c] for c in batch.DepMap_ID.astype(str)),dtype=int,count=len(batch))
    gidx=np.asarray(model['selected_expression_idx'],dtype=int)
    numeric=np.column_stack((E[indices][:,gidx],M[indices],CN[indices])).astype(np.float32)
    md=model['drug_metadata'].set_index('drug_group')
    info=md.loc[batch.drug_group.astype(str)].reset_index()
    ann=model['drug_vectorizer'].transform([tokens(row) for row in info.itertuples(index=False)]).astype(np.float32)
    X=sparse.hstack((sparse.csr_matrix(numeric),ann),format='csr',dtype=np.float32)
    assert X.shape[1]==model['feature_matrix']['total'] and np.isfinite(X.data).all()
    return X


def infer(root:Path,batch:pd.DataFrame):
    saved=root/'checkpoints'/MODEL_NAME
    if not saved.exists():raise FileNotFoundError('Run --mode train first; full-catalog checkpoint is missing.')
    model=joblib.load(saved)
    X=prepare_batch(root,model,batch)
    pred=np.asarray(model['pipeline'].predict(X),dtype=float)
    if not np.isfinite(pred).all():raise ValueError('Invalid predicted LN_IC50')
    result=batch[['DepMap_ID','drug_group']].copy()
    result['predicted_LN_IC50']=pred
    result['prediction_status']='inference_from_all_data_fit_NOT_heldout'
    return result


def out_of_fold(root:Path,batch:pd.DataFrame,mode:str):
    allowed=('cell','drug','identity-drug','identity-both')
    if mode not in allowed:
        raise ValueError(f'OOF setting must be one of: {", ".join(allowed)}')
    if mode.startswith('identity-'):
        # These are retrospective, chemical-identity-*grouped* test predictions.
        # No arbitrary unseen-pair inference is possible for the both-cold test:
        # only the fold's held-out intersection has an honest OOF prediction.
        oof=pd.read_csv(root/'research/full_gdsc/chemistry_grouped/test_predictions.csv')
        selection=oof.loc[oof['setting'].eq(mode.split('-',1)[1])].copy()
        fields=['DepMap_ID','drug_group','fold','LN_IC50','pred_combined','pred_annotations_only','pred_global_mean']
        estimate='pred_combined'
    else:
        oof=pd.read_csv(root/'research/full_gdsc/fold_predictions.csv')
        selection=oof.loc[oof['mode'].eq(mode)].copy()
        fields=['DepMap_ID','drug_group','fold','LN_IC50','pred_cell_plus_annotations','pred_global_mean']+(['pred_drug_mean'] if mode=='cell' else [])
        estimate='pred_cell_plus_annotations'
    if selection.duplicated(['DepMap_ID','drug_group']).any():
        raise ValueError('OOF predictions are not unique per cell-drug pair')
    r=batch[['DepMap_ID','drug_group']].merge(selection[fields],on=['DepMap_ID','drug_group'],how='left',validate='many_to_one',sort=False)
    if r[estimate].isna().any():
        missing=r.loc[r[estimate].isna(),['DepMap_ID','drug_group']].head(3).to_dict('records')
        raise ValueError(f'{len(r)-r[estimate].notna().sum()} pairs lack a measured {mode} test prediction. Example: {missing}')
    r=r.rename(columns={estimate:'predicted_LN_IC50','LN_IC50':'observed_LN_IC50'})
    r['prediction_status']='retrospective_identity_grouped_'+mode if mode.startswith('identity-') else 'out_of_fold_'+mode
    return r


def predict_known_grid(root:Path,output:Path,chunk_size:int=1500):
    """Score every known cell and drug without inventing observed outcomes.

    Source matrices and drug annotations are loaded ONCE. The previous naive
    version repeatedly decompressed 49 MB of omics data for each cell and
    caused timeouts; this version keeps memory and runtime bounded.
    """
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    model=joblib.load(root/'checkpoints'/MODEL_NAME)
    obs,cells,E,M,CN,_,meta=load_data(root)
    if cells!=model['cell_ids']:
        raise ValueError('Checkpoint and input cell ID order mismatch')
    names=model['drug_names']
    drug_ann=model['drug_metadata'].set_index('drug_group').loc[names].reset_index()
    sparse_drug=model['drug_vectorizer'].transform([tokens(row) for row in drug_ann.itertuples(index=False)]).astype(np.float32)
    selected=np.asarray(model['selected_expression_idx'],int)
    measured=obs.set_index(['DepMap_ID','drug_group'])['LN_IC50']
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    first=True;n=0;n_meas=0
    for i,cell in enumerate(cells):
        row=np.concatenate([E[i,selected],M[i],CN[i]]).astype(np.float32)
        pieces=[]
        # Honor the chunk_size option. Bound the working prediction matrix
        # independently of the number of drugs in a future larger catalog.
        for start in range(0,len(names),chunk_size):
            end=min(start+chunk_size,len(names))
            numeric=np.broadcast_to(row,(end-start,len(row)))
            X=sparse.hstack((sparse.csr_matrix(numeric),sparse_drug[start:end]),format='csr',dtype=np.float32)
            pieces.append(np.asarray(model['pipeline'].predict(X),dtype=float))
        estimated=np.concatenate(pieces)
        if not np.isfinite(estimated).all():raise ValueError('Nonfinite predictions in grid')
        frame=pd.DataFrame({'DepMap_ID':cell,'drug_group':names,'predicted_LN_IC50':estimated})
        frame['observed_LN_IC50']=[measured.get((cell,d),np.nan) for d in names]
        frame['has_observed_measurement']=frame['observed_LN_IC50'].notna()
        frame['prediction_status']='all_data_fit_NOT_cross_validated'
        frame.to_csv(out,index=False,mode='w' if first else 'a',header=first)
        first=False;n+=len(names);n_meas+=int(frame.has_observed_measurement.sum())
    expect=len(cells)*len(names)
    if n!=expect or n_meas!=len(obs):
        raise AssertionError(f'Grid mismatch: predicted={n}/{expect}, measured={n_meas}/{len(obs)}')
    return {'output_csv':str(out),'predicted_pairs':n,'measured_pairs':n_meas,
            'unmeasured_pairs':n-n_meas,'warning':'Unmeasured drug responses are only predicted, never fabricated observations. No independent test performance claim.'}


def main():
    a=argparse.ArgumentParser(description='Gene-expression-based LN_IC50 prediction from matched GDSC2 cell-line/drug data')
    a.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    a.add_argument('--mode',choices=['train','predict','oof','grid'],required=True)
    a.add_argument('--cell-id',help='Known DepMap cell ID, e.g. ACH-000002')
    a.add_argument('--drug-group',help='Known GDSC drug group, e.g. trametinib')
    a.add_argument('--batch-csv',help='CSV with DepMap_ID, drug_group')
    a.add_argument('--oof-setting',choices=['cell','drug','identity-drug','identity-both'],default='cell')
    a.add_argument('--output-csv',help='Output predictions CSV')
    a.add_argument('--chunk-size',type=int,default=1500,help='Grid inference rows per batch (default 1500)')
    args=a.parse_args();root=Path(args.root)
    if args.mode=='train': train_full(root);return
    if args.mode=='grid':
        if not args.output_csv:a.error('--mode grid requires --output-csv')
        print(json.dumps(predict_known_grid(root,Path(args.output_csv),args.chunk_size),indent=2));return
    if args.batch_csv:
        if args.cell_id or args.drug_group:a.error('Use either --batch-csv OR --cell-id + --drug-group')
        batch=pd.read_csv(args.batch_csv,dtype={'DepMap_ID':str,'drug_group':str})
        if {'DepMap_ID','drug_group'}.issubset(batch.columns):
            batch['DepMap_ID']=batch['DepMap_ID'].astype('string').str.strip()
            batch['drug_group']=batch['drug_group'].astype('string').str.strip().str.casefold()
    else:
        if not args.cell_id or not args.drug_group:a.error('Supply --cell-id and --drug-group, or --batch-csv')
        batch=pd.DataFrame([{'DepMap_ID':args.cell_id,'drug_group':args.drug_group.strip().casefold()}])
    if batch.empty:raise ValueError('Empty batch')
    result=infer(root,batch) if args.mode=='predict' else out_of_fold(root,batch,args.oof_setting)
    if args.output_csv:
        dst=Path(args.output_csv);dst.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(dst,index=False);print('Saved:',dst)
    else:print(result.to_json(orient='records',indent=2))

if __name__=='__main__':main()
