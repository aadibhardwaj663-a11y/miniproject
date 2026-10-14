"""Exact primary cold-drug fold annotation-only ablation.

Reuses published fold assignments, features and dense Ridge preprocessing, but trains
from scratch on drug target/pathway/identity annotations alone. No test tuning.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from full_gdsc_benchmark import SEED,tokens,rmse,corr

def run(root:Path):
    output=root/'research/full_gdsc'/'annotation_ablation'
    output.mkdir(exist_ok=True)
    obs=pd.read_csv(root/'research/full_gdsc/matched_responses.csv')
    drugs=np.array(sorted(obs.drug_group.unique()))
    all_pred=[]; all_metrics=[]
    for fold,(tri,tei) in enumerate(KFold(n_splits=5,shuffle=True,random_state=SEED).split(drugs),1):
        test_drugs=set(drugs[tei])
        tr=obs.loc[~obs.drug_group.isin(test_drugs)]
        te=obs.loc[obs.drug_group.isin(test_drugs)]
        assert not set(tr.drug_group)&set(te.drug_group)
        # Same exact weighted least-squares objective as repeated pairs because
        # annotations do not vary across observations of any given drug.
        drug_train=tr.drop_duplicates('drug_group').sort_values('drug_group')
        drug_test=te.drop_duplicates('drug_group').sort_values('drug_group')
        grouped=tr.groupby('drug_group').LN_IC50.agg(['size','mean'])
        vec=DictVectorizer(sparse=False,sort=True)
        train=vec.fit_transform([tokens(x) for x in drug_train.itertuples(index=False)]).astype(np.float64)
        test=vec.transform([tokens(x) for x in drug_test.itertuples(index=False)]).astype(np.float64)
        counts=grouped.loc[drug_train.drug_group,'size'].to_numpy(float)
        ys=grouped.loc[drug_train.drug_group,'mean'].to_numpy(float)
        scaler=StandardScaler().fit(train,sample_weight=counts)
        ridge=Ridge(alpha=100,solver='lsqr',tol=1e-4).fit(scaler.transform(train),ys,sample_weight=counts)
        pred_by_drug=dict(zip(drug_test.drug_group,ridge.predict(scaler.transform(test))))
        pr=te.drug_group.map(pred_by_drug).to_numpy(float)
        yy=te.LN_IC50.to_numpy(float)
        all_metrics.append({'fold':fold,'model':'annotations_only','n':len(te),'rmse':rmse(yy,pr),'mae':float(abs(yy-pr).mean()),'pearson_r':corr(yy,pr)})
        df=te[['NLME_CURVE_ID','DepMap_ID','drug_group','LN_IC50']].copy()
        df['fold']=fold; df['pred_annotations_only']=pr;all_pred.append(df)
        print('DRUG_ANNOT_ABLATION',fold,'n',len(te),'RMSE',rmse(yy,pr),flush=True)
    metrics=pd.DataFrame(all_metrics);metrics.to_csv(output/'fold_metrics.csv',index=False)
    pd.concat(all_pred,ignore_index=True).to_csv(output/'test_predictions.csv',index=False)
    original=pd.read_csv(root/'research/full_gdsc/fold_predictions.csv')
    merged=original[original['mode']=='drug'][['NLME_CURVE_ID','fold','pred_cell_plus_annotations','pred_cell_only','LN_IC50']].merge(pd.concat(all_pred),on=['NLME_CURVE_ID','fold'],validate='one_to_one')
    np.testing.assert_allclose(merged.LN_IC50_x,merged.LN_IC50_y,atol=0,rtol=0)
    assert len(merged)==len(obs)
    print('DRUG_ABLATION_AGG',metrics[['rmse','mae','pearson_r']].mean().to_dict(),flush=True)
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));run(Path(p.parse_args().root))
