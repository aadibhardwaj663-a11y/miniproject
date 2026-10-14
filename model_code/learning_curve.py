
"""Training-subset-only exploratory learning curve, not the final test."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import load,prepare_split,annotation_dict,design,evaluate,PANEL
def run(root):
 d=load(root)
 train,_=prepare_split(d,'unseen_cell')
 eligible=np.sort(train.DepMap_ID.unique())
 rng=np.random.default_rng(2701)
 perm=rng.permutation(eligible)
 inner_validation=set(perm[:32])
 training_order=list(perm[32:])
 validation=train[train.DepMap_ID.isin(inner_validation)].reset_index(drop=True)
 data=[]
 for n in [20,40,80,128]:
  fit=train[train.DepMap_ID.isin(training_order[:n])].reset_index(drop=True)
  unique_indices=np.array(sorted({d['cell_idx'][id_] for id_ in fit.DepMap_ID}))
  var=d['expr'][unique_indices].var(axis=0,dtype=np.float64)
  selected=np.lexsort((np.arange(len(var)),-var))[:100]
  dv=DictVectorizer(sparse=False)
  dv.fit([annotation_dict(row) for _,row in fit.iterrows()])
  Xtr=design(fit,d,selected,dv); Xval=design(validation,d,selected,dv)
  for model_name,reg in [('training_mean',DummyRegressor(strategy='mean')),
    ('random_forest',RandomForestRegressor(n_estimators=70,max_depth=12,min_samples_leaf=3,max_features=.7,random_state=27,n_jobs=1))]:
   pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('regressor',reg)])
   pipeline.fit(Xtr,fit.LN_IC50)
   pred=pipeline.predict(Xval)
   row={'train_unique_cells':n,'validation_unique_cells':32,
        'train_rows':len(fit),'validation_rows':len(validation),
        'model':model_name,**evaluate(validation.LN_IC50.to_numpy(),pred)}
   data.append(row)
 df=pd.DataFrame(data)
 df.to_csv(Path(root)/'tables'/'internal_training_learning_curve.csv',index=False)
 print(df[['train_unique_cells','model','rmse','mae']].to_string(index=False))
 return df
if __name__=='__main__':run(Path(__file__).resolve().parents[1])
