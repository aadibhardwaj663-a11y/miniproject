"""Post-hoc stronger baselines and entity-cluster bootstrap, no invented values.
This assesses uncertainty CONDITIONAL on the previously examined outer test entities.
It does NOT replace independent repeated cold splits or establish blind testing.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RNG_SEED=2727

def run(root):
    root=Path(root)
    obs=pd.read_csv(root/'data/curated_responses.csv')
    baselines=[];bo=[];predicted=[]
    for setting,entity in [('unseen_cell','DepMap_ID'),('unseen_drug','drug_group')]:
        outer_tr=set(pd.read_csv(root/'splits'/f'{setting}_train.csv').id.astype(str))
        core_ids=set(pd.read_csv(root/'splits'/setting/'train_entities.csv')[entity].astype(str))
        val_ids=set(pd.read_csv(root/'splits'/setting/'validation_entities.csv')[entity].astype(str))
        test_ids=set(pd.read_csv(root/'splits'/setting/'test_entities.csv')[entity].astype(str))
        assert core_ids|val_ids==outer_tr and not test_ids&outer_tr
        for phase,fit_ids,eval_ids in [('inner_validation',core_ids,val_ids),('outer_test_revisit',outer_tr,test_ids)]:
            train=obs[obs[entity].astype(str).isin(fit_ids)]
            valid=obs[obs[entity].astype(str).isin(eval_ids)]
            truth=valid.LN_IC50.to_numpy(float)
            fallback=train.LN_IC50.mean()
            if setting=='unseen_cell':
                known=train.groupby('drug_group').LN_IC50.mean()
                p=valid.drug_group.map(known).fillna(fallback).to_numpy(float)
                candidate='per_drug_training_mean'
            else:
                group=train.groupby('PATHWAY_NAME').LN_IC50.agg(['count','mean'])
                shrink=(group['count']*group['mean']+100*fallback)/(group['count']+100)
                p=valid.PATHWAY_NAME.map(shrink).fillna(fallback).to_numpy(float)
                candidate='pathway_shrunk_training_mean'
            rmse=float(np.sqrt(np.mean((truth-p)**2)))
            mae=float(np.mean(abs(truth-p)))
            baselines.append({'split':setting,'phase':phase,'candidate':candidate,'n':len(truth),'rmse':rmse,'mae':mae})
            if phase=='outer_test_revisit':
                d=valid[['NLME_CURVE_ID','DepMap_ID','drug_group','LN_IC50']].copy()
                d['stronger_baseline_prediction']=p
                predicted.append((setting,d))
        # Bootstrap 95% interval of paired test RMSE difference across held-out entities.
        path=root/'predictions'/f'{setting}_predictions.csv'
        df=pd.read_csv(path)
        strong=dict(predicted)[setting]
        df=df.merge(strong[['NLME_CURVE_ID','stronger_baseline_prediction']],on='NLME_CURVE_ID',validate='one_to_one')
        assert len(df)==(800 if setting=='unseen_cell' else 1000)
        rng=np.random.default_rng(RNG_SEED)
        entities=sorted(df[entity].astype(str).unique())
        # resample whole cells for cold cell; whole drugs for cold drug
        values={e:df[df[entity].astype(str).eq(e)] for e in entities}
        rmse=lambda y,p:np.sqrt(np.mean((y-p)**2))
        y=df.LN_IC50.to_numpy(float)
        pred_rf=df.random_forest_predicted.to_numpy(float)
        strong_pred=df.stronger_baseline_prediction.to_numpy(float)
        baseline_rf=float(rmse(y,pred_rf)-rmse(y,strong_pred))
        delta=[]
        for _ in range(2000):
            batch=[values[entities[i]] for i in rng.integers(0,len(entities),len(entities))]
            sample=pd.concat(batch,ignore_index=True)
            v=rmse(sample.LN_IC50.to_numpy(),sample.random_forest_predicted.to_numpy())-rmse(sample.LN_IC50.to_numpy(),sample.stronger_baseline_prediction.to_numpy())
            delta.append(float(v))
        ci=np.quantile(delta,[.025,.975])
        bo.append({'split':setting,'cluster':'held-out cell line' if setting=='unseen_cell' else 'held-out compound group',
                   'unique_test_clusters':len(entities),'bootstrap_repeats':len(delta),
                   'rf_minus_baseline_rmse':baseline_rf,'conditional_CI_lower':float(ci[0]),'conditional_CI_upper':float(ci[1]),
                   'interpretation':'Post-hoc, conditional on this fixed outer split; not across-split performance uncertainty'})
        df.to_csv(root/'predictions'/f'{setting}_supplemental_baseline_comparison.csv',index=False)
    baselines=pd.DataFrame(baselines);bo=pd.DataFrame(bo)
    baselines.to_csv(root/'metrics'/'supplemental_stronger_baselines.csv',index=False)
    bo.to_csv(root/'metrics'/'conditional_cluster_bootstrap.csv',index=False)
    print(baselines.to_string(index=False))
    print('\nCluster bootstrap\n'+bo.to_string(index=False))
    return baselines,bo

if __name__=='__main__':
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
