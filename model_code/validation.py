"""Independent inner-fold validation, using only outer-training entities.

The published outer-test results and pre-existing checkpoints are NOT replaced.
This provides actual validation metrics for each split setting.
"""
from pathlib import Path
import sys
import numpy as np,pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import load,annotation_dict,design,evaluate

def validate(root):
    root=Path(root);d=load(root);obs=d['obs'];allrows=[]
    for split,field in [('unseen_cell','DepMap_ID'),('unseen_drug','drug_group')]:
        folder=root/'splits'/split
        tr=set(pd.read_csv(folder/'train_entities.csv',dtype=str)[field])
        val=set(pd.read_csv(folder/'validation_entities.csv',dtype=str)[field])
        te=set(pd.read_csv(folder/'test_entities.csv',dtype=str)[field])
        assert tr.isdisjoint(val) and tr.isdisjoint(te) and val.isdisjoint(te)
        train=obs[obs[field].isin(tr)].sort_values(['DepMap_ID','drug_group']).reset_index(drop=True)
        valid=obs[obs[field].isin(val)].sort_values(['DepMap_ID','drug_group']).reset_index(drop=True)
        unique_cells=np.asarray(sorted({d['cell_idx'][x] for x in train.DepMap_ID}))
        var=d['expr'][unique_cells].var(axis=0,dtype=np.float64)
        selection=np.lexsort((np.arange(len(var)),-var))[:100]
        dv=DictVectorizer(sparse=False)
        dv.fit([annotation_dict(row) for _,row in train.iterrows()])
        Xtrain=design(train,d,selection,dv)
        Xvalid=design(valid,d,selection,dv)
        for model,reg in [('training_mean',DummyRegressor(strategy='mean')),
                          ('random_forest',RandomForestRegressor(n_estimators=100,max_depth=12,
                            min_samples_leaf=3,max_features=.7,random_state=27,n_jobs=1))]:
            pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('regressor',reg)])
            pipe.fit(Xtrain,train.LN_IC50.to_numpy(float))
            pred=pipe.predict(Xvalid)
            assert np.isfinite(pred).all()
            allrows.append({'split':split,'model':model,'train_core_pairs':len(train),
                            'validation_pairs':len(valid),**evaluate(valid.LN_IC50.to_numpy(float),pred)})
    table=pd.DataFrame(allrows)
    table.to_csv(root/'metrics'/'inner_validation_metrics.csv',index=False)
    print(table.to_string(index=False))
    return table

if __name__=='__main__':
    validate(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
