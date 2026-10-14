"""Secondary benchmark using ONLY inner-validation partitions for model selection.

Fixed outer held-out sets were already examined in the original project. All new
outer-test results are explicitly POST-HOC exploratory, not independent blind tests.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import load, prepare_split, annotation_dict, design, evaluate, SEED

CANDIDATES = ['mean','drug_mean','pathway_mean','ridge_100','ridge_1000']

def preprocess(d,fit,eval_rows):
    ind=np.array(sorted({d['cell_idx'][k] for k in fit.DepMap_ID}))
    var=d['expr'][ind].var(axis=0,dtype=np.float64)
    sel=np.lexsort((np.arange(len(var)),-var))[:100]
    dv=DictVectorizer(sparse=False,sort=True)
    dv.fit([annotation_dict(r) for _,r in fit.iterrows()])
    return sel,dv,design(fit,d,sel,dv),design(eval_rows,d,sel,dv)

def direct_predict(candidate,fit,eval_rows, x_fit,x_eval):
    y=fit.LN_IC50.to_numpy(float)
    if candidate=='mean':
        return np.full(len(eval_rows),y.mean(),dtype=float),None
    if candidate=='drug_mean':
        mapping=fit.groupby('drug_group').LN_IC50.mean().to_dict()
        return eval_rows.drug_group.map(mapping).fillna(y.mean()).to_numpy(float),None
    if candidate=='pathway_mean':
        # Assumes pathway is available for previously unseen drugs.
        # Mean-shrunk to global mean, fitted WITHOUT validation/test responses.
        grp=fit.groupby('PATHWAY_NAME').LN_IC50.agg(['mean','count'])
        shrink=100  # predeclared, not tuned against the outer test
        means=((grp['mean']*grp['count']+y.mean()*shrink)/(grp['count']+shrink)).to_dict()
        return eval_rows.PATHWAY_NAME.map(means).fillna(y.mean()).to_numpy(float),None
    common= [('imputer',SimpleImputer(strategy='median'))]
    if candidate=='ridge_100':
        reg=Ridge(alpha=100,solver='lsqr',tol=1e-4)
        pipe=Pipeline(common+[('scale',StandardScaler()),('regressor',reg)])
    elif candidate=='ridge_1000':
        reg=Ridge(alpha=1000,solver='lsqr',tol=1e-4)
        pipe=Pipeline(common+[('scale',StandardScaler()),('regressor',reg)])
    elif candidate=='original_rf':
        reg=RandomForestRegressor(n_estimators=100,max_depth=12,min_samples_leaf=3,max_features=.7,random_state=SEED,n_jobs=1)
        pipe=Pipeline(common+[('regressor',reg)])
    elif candidate=='extra_trees':
        reg=ExtraTreesRegressor(n_estimators=120,max_depth=12,min_samples_leaf=3,max_features=.7,random_state=SEED,n_jobs=1)
        pipe=Pipeline(common+[('regressor',reg)])
    else:raise ValueError(candidate)
    pipe.fit(x_fit,y)
    pred=pipe.predict(x_eval)
    return pred,pipe

def fit(root):
    root=Path(root)
    d=load(root)
    results=[];chosen=[]
    (root/'metrics').mkdir(exist_ok=True)
    (root/'predictions').mkdir(exist_ok=True)
    (root/'checkpoints').mkdir(exist_ok=True)
    for split,field in [('unseen_cell','DepMap_ID'),('unseen_drug','drug_group')]:
        o=d['obs'];fold=root/'splits'/split
        ids=lambda n:set(pd.read_csv(fold/f'{n}_entities.csv',dtype=str)[field])
        train_ids,val_ids,test_ids=ids('train'),ids('validation'),ids('test')
        assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
        def rows(allowed):return o[o[field].isin(allowed)].copy().sort_values(['DepMap_ID','drug_group']).reset_index(drop=True)
        core=rows(train_ids);val=rows(val_ids)
        sel,dv,Xfit,Xval=preprocess(d,core,val)
        val_result=[]
        # Evaluate candidate baselines and simple models only on INNER validation.
        for candidate in CANDIDATES:
            if candidate=='drug_mean' and split=='unseen_drug':continue
            if candidate=='pathway_mean' and split=='unseen_cell':continue
            print(split,'testing',candidate,flush=True)
            pred,_=direct_predict(candidate,core,val,Xfit,Xval)
            met=evaluate(val.LN_IC50.to_numpy(float),pred)
            item={'split':split,'candidate':candidate,'phase':'inner_validation',**met}
            assert np.isfinite(pred).all()
            results.append(item);val_result.append(item)
        # Include the PRE-EXISTING, separately measured original RF inner metric,
        # without refitting the expensive tree model during this secondary audit.
        historic=pd.read_csv(root/'metrics'/'inner_validation_metrics.csv')
        saved_rf=historic[(historic.split==split)&(historic.model=='random_forest')].iloc[0]
        rf_row={'split':split,'candidate':'original_rf','phase':'inner_validation',
                'n':int(saved_rf['n']),'rmse':float(saved_rf.rmse),
                'mae':float(saved_rf.mae),'pearson_r':float(saved_rf.pearson_r)}
        results.append(rf_row);val_result.append(rf_row)
        # Select by inner validation ONLY, never by the already inspected outer test.
        best=min(val_result,key=lambda item:(item['rmse'],item['candidate']))
        # Fit on final training entities after candidate selection; keep test untouched by fitting.
        final=rows(train_ids|val_ids);test=rows(test_ids)
        assert set(final.NLME_CURVE_ID).isdisjoint(set(test.NLME_CURVE_ID))
        sel,dv,Xfinal,Xtest=preprocess(d,final,test)
        if best['candidate']=='original_rf':
            original=joblib.load(root/'checkpoints'/f'{split}_random_forest.joblib')
            x=design(test,d,original['expression_selected_indices'],original['drug_annotation_vectorizer'])
            pred=original['pipeline'].predict(x)
            pipe=original['pipeline']
            # Saved original fit is the full train+validation partition.
            # It was generated before this supplemental experiment.
            joblib.dump(original,root/'checkpoints'/f'{split}_post_hoc_selected.joblib')
        else:
            pred,pipe=direct_predict(best['candidate'],final,test,Xfinal,Xtest)
        met=evaluate(test.LN_IC50.to_numpy(float),pred)
        chosen.append({'split':split,'inner_validation_selected':best['candidate'],'inner_validation_rmse':best['rmse'],
                       'outer_test_revisited_rmse':met['rmse'],'outer_test_revisited_mae':met['mae'],
                       'outer_test_revisited_pearson_r':met['pearson_r']})
        results.append({'split':split,'candidate':best['candidate'],'phase':'post_hoc_outer_test_revisit',**met})
        output=test[['NLME_CURVE_ID','DepMap_ID','drug_group','LN_IC50']].copy()
        output['post_hoc_selected_prediction']=pred
        output.to_csv(root/'predictions'/f'{split}_post_hoc_selected_predictions.csv',index=False)
        # save the model and necessary preprocessor mapping; no artifact for global means required if used.
        if pipe is not None and best['candidate']!='original_rf':
            joblib.dump({'pipeline':pipe,'selected_indices':sel,'annotation_vectorizer':dv,
                         'name':best['candidate'],'selection_basis':'inner_validation_only'},
                         root/'checkpoints'/f'{split}_post_hoc_selected.joblib')
        elif best['candidate']!='original_rf':
            record={'name':best['candidate'],'global_mean':float(final.LN_IC50.mean()),'selection_basis':'inner_validation_only'}
            if best['candidate']=='drug_mean':record['drug_means']=final.groupby('drug_group').LN_IC50.mean().to_dict()
            if best['candidate']=='pathway_mean':
                group=final.groupby('PATHWAY_NAME').LN_IC50.agg(['mean','count'])
                record['pathway_means']=((group['mean']*group['count']+record['global_mean']*100)/(group['count']+100)).to_dict()
            joblib.dump(record,root/'checkpoints'/f'{split}_post_hoc_selected.joblib')
        print(split,'selected by validation',best['candidate'],f'validation RMSE {best["rmse"]:.4f}',f'outer RMSE {met["rmse"]:.4f}',flush=True)
    df=pd.DataFrame(results)
    df.to_csv(root/'metrics'/'supplemental_candidate_comparison.csv',index=False)
    pd.DataFrame(chosen).to_csv(root/'metrics'/'supplemental_validation_selected_models.csv',index=False)
    return df,pd.DataFrame(chosen)
if __name__=='__main__':fit(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
