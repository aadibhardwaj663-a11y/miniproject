"""Verify all new 5-fold predictions against actual response IDs and metric tables."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from full_gdsc_benchmark import SEED

def run(root):
 root=Path(root);out=root/'research/full_gdsc';obs=pd.read_csv(out/'matched_responses.csv');m=pd.read_csv(out/'fold_metrics.csv');pr=pd.read_csv(out/'fold_predictions.csv')
 assert obs[['DepMap_ID','drug_group']].duplicated().sum()==0
 assert obs.NLME_CURVE_ID.is_unique
 assert obs.LN_IC50.notna().all() and np.isfinite(obs.LN_IC50).all()
 assert len(m)==(5*4+5*3)
 assert len(pr)==2*len(obs)
 byid=obs.set_index('NLME_CURVE_ID')[['DepMap_ID','drug_group','LN_IC50']]
 seen={}
 fold_dir=out/'split_ids';fold_dir.mkdir(exist_ok=True)
 for mode,field,seed in [('drug','drug_group',SEED),('cell','DepMap_ID',SEED+1)]:
  entities=np.array(sorted(obs[field].unique()))
  folds=KFold(n_splits=5,shuffle=True,random_state=seed)
  test_union=set()
  for fold,(tr,te) in enumerate(folds.split(entities),1):
   train=set(entities[tr]);test=set(entities[te]);assert not train&test
   pd.DataFrame({field:sorted(train)}).to_csv(fold_dir/f'{mode}_fold{fold}_train.csv',index=False)
   pd.DataFrame({field:sorted(test)}).to_csv(fold_dir/f'{mode}_fold{fold}_test.csv',index=False)
   rows=pr[(pr['mode']==mode)&pr['fold'].eq(fold)]
   assert len(rows)>0 and set(rows[field])==test
   assert rows.NLME_CURVE_ID.is_unique
   assert set(rows.NLME_CURVE_ID)==set(obs.loc[obs[field].isin(test),'NLME_CURVE_ID'])
   original=byid.loc[rows.NLME_CURVE_ID]
   np.testing.assert_allclose(original.LN_IC50,rows.LN_IC50,atol=0,rtol=0)
   models=['global_mean','cell_only','cell_plus_annotations']+(['drug_mean'] if mode=='cell' else [])
   for model in models:
    y=rows.LN_IC50.to_numpy();p=rows['pred_'+model].to_numpy()
    rmse=float(np.sqrt(np.mean((y-p)**2)));mae=float(np.mean(abs(y-p)))
    mrow=m[(m['mode']==mode)&(m['fold'].eq(fold))&(m['model']==model)].iloc[0]
    assert abs(rmse-mrow.rmse)<1e-7 and abs(mae-mrow.mae)<1e-7
   test_union|=test
  assert test_union==set(entities)
  print('CHECK_PASS',mode,'entity fold coverage',len(test_union),'predictions',len(pr[pr['mode']==mode]),flush=True)
 bio=pd.read_csv(out/'biological_interpretability/five_fold_metrics.csv')
 assert bio.drug.nunique()==5 and bio.fold.nunique()==5 and (bio.n>0).all()
 assert len(list((out/'figures').glob('*.png')))>=7
 (out/'verification.json').write_text(json.dumps({'actual_pairs':len(obs),'unique_cells':obs.DepMap_ID.nunique(),'unique_drug_groups':obs.drug_group.nunique(),'fitted_models_fold':35,'full_kfold_entity_disjoint':'PASS','measurements_and_metrics_match':'PASS','bio_drug_folds':int(bio.drug.nunique()),'figure_count':len(list((out/'figures').glob('*.png')))},indent=2))
 print('ALL_FULL_GDSC_CHECKS_PASS',flush=True)
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
