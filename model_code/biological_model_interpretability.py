"""Five-fold held-out-cell drug-specific Ridge + TP53/KRAS/BRAF/PIK3CA model reliance.
Predeclare drugs to avoid cherry-picking; no causal interpretation.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

DRUGS=['nutlin-3a (-)','trametinib','palbociclib','selumetinib','taselisib']
PANEL=['TP53','KRAS','BRAF','EGFR','PIK3CA','PTEN','APC','RB1','MYC','CDKN2A']
SEED=27014

def main(root):
 root=Path(root); folder=root/'research/full_gdsc';out=folder/'biological_interpretability';out.mkdir(exist_ok=True)
 obs=pd.read_csv(folder/'matched_responses.csv')
 with np.load(folder/'matched_omics.npz',allow_pickle=False) as f:
  cells=f['cell_ids'].astype(str).tolist();X=f['expression'];M=f['mut'];C=f['cn']
 lookup={c:i for i,c in enumerate(cells)}
 indexes=np.array(cells)
 kf=KFold(n_splits=5,shuffle=True,random_state=SEED)
 rows=[];imp=[];preds=[]
 for drug in DRUGS:
  src=obs[obs.drug_group.eq(drug)].copy()
  assert len(src)>=200
  for fold,(tri,tei) in enumerate(kf.split(indexes),1):
   trainids=set(indexes[tri]); testids=set(indexes[tei]);tr=src[src.DepMap_ID.isin(trainids)].copy();te=src[src.DepMap_ID.isin(testids)].copy()
   assert not set(tr.DepMap_ID)&set(te.DepMap_ID)
   train_idx=np.asarray(sorted({lookup[x] for x in tr.DepMap_ID}),dtype=int)
   v=X[train_idx].var(axis=0,dtype=np.float64)
   sel=np.lexsort((np.arange(len(v)),-v))[:100]
   def feat(frame):
    idx=np.array([lookup[x] for x in frame.DepMap_ID]);return np.column_stack([X[:,sel][idx],M[idx],C[idx]])
   trX=feat(tr);teX=feat(te);y=tr.LN_IC50.to_numpy(float);yy=te.LN_IC50.to_numpy(float)
   pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100,solver='lsqr',tol=1e-4))
   pipe.fit(trX,y);prediction=pipe.predict(teX);mean=np.full(len(yy),y.mean())
   for name,pr in [('per_drug_mean',mean),('gene_expression_panel_ridge',prediction)]:
    rows.append({'drug':drug,'fold':fold,'model':name,'n':len(yy),'rmse':float(np.sqrt(np.mean((pr-yy)**2))),'mae':float(np.mean(abs(pr-yy))),'within_drug_r':float(np.corrcoef(yy,pr)[0,1]) if np.std(pr)>1e-10 else np.nan})
   temp=te[['DepMap_ID','drug_group','lineage','LN_IC50']].copy();temp['fold']=fold;temp['model_prediction']=prediction;temp['baseline_prediction']=mean;preds.append(temp)
   # Disease-specific predetermined gene mutations, grouped with CN, tested on held-out cells.
   pids={'nutlin-3a (-)':['TP53'],'trametinib':['KRAS','BRAF'],'palbociclib':['RB1'],'selumetinib':['KRAS','BRAF'],'taselisib':['PIK3CA']}[drug]
   baseline=float(np.sqrt(np.mean((prediction-yy)**2)))
   for gene in pids:
    gi=PANEL.index(gene); rng=np.random.default_rng(61000+fold*13+gi)
    changes=[]
    for rep in range(40):
     perm=rng.permutation(len(yy));xp=teX.copy()
     xp[:,[100+gi,110+gi]]=teX[perm][:,[100+gi,110+gi]]
     pr=pipe.predict(xp)
     changes.append(float(np.sqrt(np.mean((pr-yy)**2))-baseline))
    imp.append({'drug':drug,'gene':gene,'fold':fold,'test_n':len(yy),'baseline_rmse':baseline,'delta_rmse_mean':float(np.mean(changes)),'delta_rmse_std':float(np.std(changes,ddof=1)),'n_permutations':len(changes),'standardized_mutation_coef':float(pipe[-1].coef_[100+gi]),'standardized_cn_coef':float(pipe[-1].coef_[110+gi])})
  print('BIO_DRUG_COMPLETE',drug,flush=True)
 results=pd.DataFrame(rows);results.to_csv(out/'five_fold_metrics.csv',index=False)
 pd.DataFrame(imp).to_csv(out/'heldout_gene_group_permutation.csv',index=False)
 pd.concat(preds,ignore_index=True).to_csv(out/'test_predictions.csv',index=False)
 print(results.groupby(['drug','model']).agg(rmse=('rmse','mean'),mae=('mae','mean'),within_r=('within_drug_r','mean')).round(4).to_string(),flush=True)
 print(pd.DataFrame(imp).groupby(['drug','gene']).delta_rmse_mean.mean().round(4).to_string(),flush=True)

if __name__=='__main__':
 import sys
 main(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
