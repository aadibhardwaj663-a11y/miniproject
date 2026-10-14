
"""Challenge 27. Load curated real measurements, fit leak-safe baselines and forests."""
from pathlib import Path
import json, re, hashlib, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr

PANEL = ['TP53','KRAS','BRAF','EGFR','PIK3CA','PTEN','APC','RB1','MYC','CDKN2A']
SEED=27

def load(root):
    root=Path(root)
    expr=np.load(root/'data/matched_expression_full.npz',allow_pickle=False)
    cell_ids=list(expr['cell_ids'])
    genes=list(expr['gene_columns'])
    expr_arr=expr['values'].astype(np.float32)
    obs=pd.read_csv(root/'data/curated_responses.csv')
    mut=pd.read_csv(root/'data/mutation_panel_binary.csv',index_col=0).loc[cell_ids,PANEL]
    cn_cols=[next(c for c in pd.read_csv(root/'data/copy_number_panel.csv',nrows=0).columns if c.startswith(g+' (')) for g in PANEL]
    cn=pd.read_csv(root/'data/copy_number_panel.csv',index_col=0).loc[cell_ids,cn_cols]
    assert len(obs)==4000 and obs.DepMap_ID.nunique()==200 and obs.drug_group.nunique()==20
    assert not obs.duplicated(['DepMap_ID','drug_group']).any()
    assert np.isfinite(obs.LN_IC50.to_numpy()).all()
    assert np.isfinite(expr_arr).all()
    assert np.isfinite(mut.to_numpy()).all() and np.isfinite(cn.to_numpy()).all()
    assert set(mut.to_numpy().ravel()).issubset({0,1})
    return {'root':root,'obs':obs,'cells':cell_ids,'genes':genes,'expr':expr_arr,
            'mut':mut.to_numpy(dtype=np.float32),'cn':cn.to_numpy(dtype=np.float32),
            'cell_idx':{x:i for i,x in enumerate(cell_ids)}}

def annotation_dict(row):
    pathway=str(row['PATHWAY_NAME'])
    target=str(row['PUTATIVE_TARGET'])
    # Source-defined pathway category plus lexical target tokens.
    tokens={'pathway:'+re.sub(r'\s+','_',pathway.strip().casefold()):1.0}
    for token in re.findall(r'[A-Za-z][A-Za-z0-9-]*',target):
        tokens['target:'+token.casefold()]=1.0
    return tokens

def design(rows, d, sel, dv):
    rows=rows.reset_index(drop=True)
    indices=np.fromiter((d['cell_idx'][x] for x in rows.DepMap_ID),dtype=int,count=len(rows))
    E=d['expr'][indices][:,sel]
    M=d['mut'][indices]
    C=d['cn'][indices]
    A=dv.transform([annotation_dict(row) for _,row in rows.iterrows()])
    return np.column_stack([E,M,C,A]).astype(np.float32)

def pearson_safe(y,p):
    if len(y)<3 or np.std(y)<1e-12 or np.std(p)<1e-12: return np.nan
    return float(pearsonr(y,p).statistic)

def evaluate(y,p):
    assert np.isfinite(y).all() and np.isfinite(p).all() and len(y)==len(p)
    return {'n':len(y),'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'mae':float(mean_absolute_error(y,p)),'pearson_r':pearson_safe(y,p)}

def ids(root,stem):
    return pd.read_csv(Path(root)/'splits'/f'{stem}.csv',dtype=str).iloc[:,0].tolist()

def prepare_split(d,split):
    o=d['obs']
    if split=='unseen_cell':
        tr_ids=set(ids(d['root'],'unseen_cell_train'))
        te_ids=set(ids(d['root'],'unseen_cell_test'))
        assert tr_ids.isdisjoint(te_ids)
        assert tr_ids|te_ids==set(d['cells'])
        train=o[o.DepMap_ID.isin(tr_ids)].copy().sort_values(['DepMap_ID','drug_group']).reset_index(drop=True)
        test=o[o.DepMap_ID.isin(te_ids)].copy().sort_values(['DepMap_ID','drug_group']).reset_index(drop=True)
    elif split=='unseen_drug':
        tr_ids=set(ids(d['root'],'unseen_drug_train'))
        te_ids=set(ids(d['root'],'unseen_drug_test'))
        assert tr_ids.isdisjoint(te_ids)
        assert tr_ids|te_ids==set(o.drug_group)
        train=o[o.drug_group.isin(tr_ids)].copy().sort_values(['DepMap_ID','drug_group']).reset_index(drop=True)
        test=o[o.drug_group.isin(te_ids)].copy().sort_values(['DepMap_ID','drug_group']).reset_index(drop=True)
        assert set(train.DRUG_ID).isdisjoint(set(test.DRUG_ID))
        manifest = pd.read_csv(d['root']/'data'/'compound_alias_manifest.csv')
        normal = lambda s: re.sub(r'[^a-z0-9]','',str(s).casefold())
        train_aliases = {normal(a) for a in manifest[manifest.drug_group.isin(tr_ids)].alias}
        test_aliases = {normal(a) for a in manifest[manifest.drug_group.isin(te_ids)].alias}
        assert not train_aliases.intersection(test_aliases), 'Cross-split compound alias collision'
    else: raise ValueError(split)
    assert not train.empty and not test.empty
    assert len(train)+len(test)==len(o)
    assert not set(train.NLME_CURVE_ID)&set(test.NLME_CURVE_ID)
    return train,test

def fit_split(d,split):
    root=d['root']
    train,test=prepare_split(d,split)
    train_cell_indices=np.array(sorted({d['cell_idx'][x] for x in train.DepMap_ID}))
    # TRAINING-ONLY unsupervised variance, one count per unique training cell line.
    var=d['expr'][train_cell_indices].var(axis=0,dtype=np.float64)
    assert np.isfinite(var).all() and (var>0).sum()>=100
    indices=np.lexsort((np.arange(len(var)),-var))[:100]
    dv=DictVectorizer(sparse=False,sort=True)
    dv.fit([annotation_dict(row) for _,row in train.iterrows()])
    Xtr=design(train,d,indices,dv); Xte=design(test,d,indices,dv)
    ytr=train.LN_IC50.to_numpy(dtype=np.float64); yte=test.LN_IC50.to_numpy(dtype=np.float64)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
    feature_names=[str(d['genes'][i]) for i in indices]+['MUT_'+x for x in PANEL]+['CN_'+x for x in PANEL]+['DRUG_'+x for x in dv.get_feature_names_out()]
    assert len(feature_names)==Xtr.shape[1] and len(feature_names)==Xte.shape[1]
    models={
     'training_mean': Pipeline([('imputer',SimpleImputer(strategy='median')),('regressor',DummyRegressor(strategy='mean'))]),
     'random_forest':Pipeline([('imputer',SimpleImputer(strategy='median')),('regressor',RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=3,max_features=0.7,random_state=SEED,n_jobs=1))])
    }
    records=[]; predictions=test[['DepMap_ID','DRUG_ID','drug_group','DRUG_NAME','NLME_CURVE_ID','LN_IC50']].copy()
    for name,model in models.items():
        model.fit(Xtr,ytr)
        p=model.predict(Xte)
        assert len(p)==len(yte) and np.isfinite(p).all()
        if name=='training_mean':
            np.testing.assert_allclose(p,ytr.mean(),atol=1e-12,rtol=0)
        predictions[name+'_predicted']=p
        metric=evaluate(yte,p)
        records.append({'split':split,'model':name,**metric,
                        'training_observations':len(train),'test_observations':len(test),
                        'train_unique_cells':train.DepMap_ID.nunique(),
                        'test_unique_cells':test.DepMap_ID.nunique(),
                        'train_unique_drugs':train.drug_group.nunique(),
                        'test_unique_drugs':test.drug_group.nunique()})
        bundle={'pipeline':model,'expression_selected_indices':indices,'expression_gene_names':[str(d['genes'][i]) for i in indices],
                'drug_annotation_vectorizer':dv,'feature_names':feature_names,'panel':PANEL,
                'split':split,'model_name':name,'random_seed':SEED,'feature_selection':'100 largest train-unique-cell expression variances',
                'response':'LN_IC50'}
        path=root/'checkpoints'/f'{split}_{name}.joblib'
        joblib.dump(bundle,path)
        restored=joblib.load(path)
        np.testing.assert_array_equal(restored['pipeline'].predict(Xte),p)
    predictions['abs_error_forest']=abs(predictions.LN_IC50-predictions.random_forest_predicted)
    predictions.to_csv(root/'predictions'/f'{split}_predictions.csv',index=False)
    pd.DataFrame({'gene':bundle['expression_gene_names'],'train_variance':var[indices],
                  'expression_source_index':indices}).to_csv(root/'tables'/f'{split}_top100_expression.csv',index=False)
    meta={'split':split,'train_cell_ids':sorted(train.DepMap_ID.unique().tolist()),'test_cell_ids':sorted(test.DepMap_ID.unique().tolist()),
          'train_drug_groups':sorted(train.drug_group.unique().tolist()),'test_drug_groups':sorted(test.drug_group.unique().tolist()),
          'drug_annotation_feature_count':len(dv.get_feature_names_out()),'numeric_feature_count':120,'expression_feature_count':100,
          'feature_selection_cells':len(train_cell_indices),'selected_gene_names':bundle['expression_gene_names'],
          'checks':{'response_finite':'PASS','preprocessing_finite':'PASS','reloaded_predictions_exact':'PASS',
                    'held_out_entity_disjoint':'PASS','curve_id_disjoint':'PASS'}}
    (root/'logs'/f'{split}_checks.json').write_text(json.dumps(meta,indent=2))
    return pd.DataFrame(records),train,test,models,indices,dv,Xte,yte,feature_names

def permutation_for_one_drug(d,split_result):
    met,train,test,models,indices,dv,Xte,yte,fn=split_result
    # Predeclared before looking at model error: MG-132 is highest-coverage source drug.
    drug='mg-132'
    mask=(test.drug_group.to_numpy()==drug)
    assert np.sum(mask)>=10, 'Not enough held-out MG-132 cell lines'
    X=Xte[mask].copy(); y=yte[mask]
    model=models['random_forest']
    base=evaluate(y,model.predict(X))['rmse']
    fi=model.named_steps['regressor'].feature_importances_
    top_exp=np.argsort(-fi[:100],kind='stable')[:10]
    feature_groups=[('All 100 selected expression genes',list(range(100)))]
    feature_groups += [('EXP: '+str(fn[i]),[int(i)]) for i in top_exp]
    feature_groups += [('MUT+CN: '+gene,[100+j,110+j]) for j,gene in enumerate(PANEL)]
    seed=np.random.default_rng(SEED)
    out=[]
    for label,columns in feature_groups:
        diffs=[]
        for r in range(8):
            perm=seed.permutation(len(y))
            Xp=X.copy()
            Xp[:,columns]=X[perm][:,columns]
            rp=model.predict(Xp)
            assert np.isfinite(rp).all()
            diffs.append(evaluate(y,rp)['rmse']-base)
        out.append({'drug_group':drug,'test_n':len(y),'feature_group':label,
                    'rmse_baseline_unpermuted':base,'rmse_increase_mean':float(np.mean(diffs)),
                    'rmse_increase_std':float(np.std(diffs,ddof=1)), 'repeats':8,
                    'min_increase':float(np.min(diffs)),'max_increase':float(np.max(diffs))})
    imp=pd.DataFrame(out)
    imp.to_csv(d['root']/'tables'/'mg132_drug_specific_permutation.csv',index=False)
    return imp

def run(root):
    d=load(root)
    results={}
    for split in ['unseen_cell','unseen_drug']:
        results[split]=fit_split(d,split)
    metrics=pd.concat([v[0] for v in results.values()],ignore_index=True)
    metrics.to_csv(d['root']/'tables'/'model_metrics.csv',index=False)
    imp=permutation_for_one_drug(d,results['unseen_cell'])
    assert len(metrics)==4 and len(imp)==21
    print(metrics.to_string(index=False))
    print('Permutation results:',len(imp),'rows; top 6:')
    print(imp.sort_values('rmse_increase_mean',ascending=False).head(6)[['feature_group','rmse_increase_mean']].to_string(index=False))
    return metrics,imp

if __name__=='__main__':
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parents[1]))
