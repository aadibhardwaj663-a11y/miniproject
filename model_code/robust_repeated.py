"""Fast, repeatedly split, jointly cold baseline/ablation evaluation.
Fixed regularized Ridge alpha 100, train-only feature transformations, 5 seeds.
"""
from pathlib import Path
import sys
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from research_extension import read_project,split_pools,design,stack,ABLS,metrics,SEEDS

def run(root, modes=('cell','drug','both'), seeds=SEEDS):
    root=Path(root);d=read_project(root)
    results=[];prediction=[]
    for mode in modes:
        for seed in seeds:
            core,val,train,test=split_pools(d,mode,seed,root)
            fitparts,testparts,sel,dv=design(d,train,test)
            yy=train.LN_IC50.to_numpy(float);ytest=test.LN_IC50.to_numpy(float)
            baseline=np.full(len(ytest),yy.mean())
            basepred={'global_mean':baseline}
            if mode=='cell':basepred['training_drug_mean']=test.drug_group.map(train.groupby('drug_group').LN_IC50.mean()).to_numpy(float)
            for name,pr in basepred.items():
                results.append({'mode':mode,'seed':seed,'variant':name,**metrics(ytest,pr)})
            for variant in ['all','expression_drug','panel_drug','drug_only','cell_only']:
                model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100))
                model.fit(stack(fitparts,variant),yy)
                pr=model.predict(stack(testparts,variant))
                results.append({'mode':mode,'seed':seed,'variant':'ridge_'+variant,**metrics(ytest,pr)})
                if variant=='all':
                    temp=test[['NLME_CURVE_ID','DepMap_ID','drug_group','lineage','LN_IC50']].copy()
                    temp['seed']=seed;temp['mode']=mode;temp['prediction_ridge_all']=pr;temp['prediction_global_mean']=baseline
                    prediction.append(temp)
            print('Completed',mode,seed,'test_pairs',len(test),flush=True)
    df=pd.DataFrame(results);df.to_csv(root/'research/metrics/repeated_seed_metrics.csv',index=False)
    pd.concat(prediction,ignore_index=True).to_csv(root/'research/data/repeated_seed_predictions.csv',index=False)
    summ=df.groupby(['mode','variant'],as_index=False).agg(n_seeds=('seed','nunique'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),r_mean=('pearson_r','mean'))
    summ.to_csv(root/'research/metrics/repeated_seed_summary.csv',index=False)
    print(summ.sort_values(['mode','rmse_mean']).round(3).to_string(index=False),flush=True)
    return summ
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
