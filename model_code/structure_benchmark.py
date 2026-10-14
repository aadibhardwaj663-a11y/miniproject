"""Provisional 20-compound Morgan fingerprint benchmark.
No test-set hyperparameter selection. Same five predeclared seeds and existing
cold-cell, cold-drug, jointly cold entity partitions. Alpha=100, ridge LSQR.
Absolute chemistry validity is limited by upstream reagent-form uncertainty.
"""
from pathlib import Path
import json,sys,time
import joblib
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from research_extension import read_project,split_pools,design,metrics,SEEDS
from chemical_features import validate

VARIANTS=['mean','annotation_only','fingerprint_only','cell_only','cell_plus_annotations',
          'cell_plus_fingerprint','cell_plus_both']


def run(root, modes=('cell','drug','both'),seeds=SEEDS):
 root=Path(root)
 chem=root/'research/chemistry/structure_mapping_20_provisional.csv'
 mapping,bits=validate(root,chem)
 names=mapping.drug_group.tolist();lookup={name:i for i,name in enumerate(names)}
 data=read_project(root)
 folder=root/'research/chemistry';folder.mkdir(parents=True,exist_ok=True)
 matrix=bits.astype(np.float32)
 from rdkit import DataStructs
 from rdkit.Chem import rdFingerprintGenerator
 # binary Tanimoto similarities are static structure descriptors independent of assay y
 similarity=np.zeros((20,20),dtype=float)
 for i in range(20):
  for j in range(20):
   a=matrix[i]>0;b=matrix[j]>0
   similarity[i,j]=np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum())
 pd.DataFrame(similarity,index=names,columns=names).to_csv(folder/'tanimoto_similarity.csv')
 result=[]; prediction=[];domain=[]
 for mode in modes:
  for seed in seeds:
   start=time.monotonic()
   _,_,train,test=split_pools(data,mode,seed,root)
   trainp,testp,selection,encoder=design(data,train,test)
   train_drug=train.drug_group.map(lookup).to_numpy(dtype=int)
   test_drug=test.drug_group.map(lookup).to_numpy(dtype=int)
   y=train.LN_IC50.to_numpy(float);yt=test.LN_IC50.to_numpy(float)
   tr_cell=np.column_stack([trainp['expression'],trainp['panel']]).astype(np.float32)
   te_cell=np.column_stack([testp['expression'],testp['panel']]).astype(np.float32)
   tr_a=trainp['drug'].astype(np.float32);te_a=testp['drug'].astype(np.float32)
   tr_f=matrix[train_drug];te_f=matrix[test_drug]
   pool={
    'annotation_only':(tr_a,te_a),
    'fingerprint_only':(tr_f,te_f),
    'cell_only':(tr_cell,te_cell),
    'cell_plus_annotations':(np.column_stack([tr_cell,tr_a]),np.column_stack([te_cell,te_a])),
    'cell_plus_fingerprint':(np.column_stack([tr_cell,tr_f]),np.column_stack([te_cell,te_f])),
    'cell_plus_both':(np.column_stack([tr_cell,tr_a,tr_f]),np.column_stack([te_cell,te_a,te_f])),
   }
   pframe=test[['NLME_CURVE_ID','DepMap_ID','drug_group','LN_IC50']].copy()
   pframe['mode']=mode;pframe['seed']=seed
   means=np.full(len(yt),float(np.mean(y)))
   result.append(dict(mode=mode,seed=seed,variant='mean',**metrics(yt,means)))
   pframe['pred_mean']=means
   for variant,(x,xt) in pool.items():
    assert np.isfinite(x).all() and np.isfinite(xt).all()
    reg=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100,solver='lsqr',tol=1e-4))
    reg.fit(x,y)
    pred=reg.predict(xt)
    assert np.isfinite(pred).all()
    result.append(dict(mode=mode,seed=seed,variant=variant,**metrics(yt,pred)))
    pframe['pred_'+variant]=pred
    if mode=='both' and seed==107 and variant=='cell_plus_fingerprint':
     joblib.dump({'pipeline':reg,'variant':variant,'train_seed':seed,'training_drug_groups':sorted(train.drug_group.unique()),
                  'selected_expression_indices':selection,'annotations_vectorizer':encoder,
                  'fingerprint_drug_names':names,'fitted_matrix_columns':x.shape[1]},folder/'joint_cold_seed107_fingerprint_model.joblib')
   if mode in ('drug','both'):
    seen=sorted({lookup[x] for x in train.drug_group})
    maxsim=np.max(similarity[test_drug][:,seen],axis=1)
    pframe['max_tanimoto_to_training_drugs']=maxsim
    pframe['abs_error_cell_plus_fingerprint']=np.abs(pframe['pred_cell_plus_fingerprint']-yt)
    domain.append(pframe[['mode','seed','NLME_CURVE_ID','drug_group','max_tanimoto_to_training_drugs','abs_error_cell_plus_fingerprint']])
   prediction.append(pframe)
   print('COMPLETE',mode,seed,'n',len(test),'time',round(time.monotonic()-start,2),flush=True)
 results=pd.DataFrame(result)
 summary=results.groupby(['mode','variant'],as_index=False).agg(n_seeds=('seed','nunique'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),r_mean=('pearson_r','mean'))
 results.to_csv(folder/'chemical_fingerprint_seed_metrics.csv',index=False)
 summary.to_csv(folder/'chemical_fingerprint_summary.csv',index=False)
 pd.concat(prediction,ignore_index=True).to_csv(folder/'chemical_fingerprint_test_predictions.csv',index=False)
 pd.concat(domain,ignore_index=True).to_csv(folder/'fingerprint_similarity_applicability.csv',index=False)
 print(summary.sort_values(['mode','rmse_mean']).round(4).to_string(index=False))
 return summary
if __name__=='__main__':
 run(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1],modes=tuple(sys.argv[2].split(',')) if len(sys.argv)>2 else ('cell','drug','both'),seeds=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else SEEDS)
