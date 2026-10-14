"""Supplemental, pre-specified 5-fold simultaneous cell+drug cold test and modality ablations.

The primary 5-fold experiments remain unchanged. Here fold f holds out BOTH the cells
and drug groups assigned to f. Only the pairwise intersection of the withheld groups
is scored; other pairs involving just one held-out entity are excluded from both fit
and test. Therefore this evaluates 20% of measured pairs, not all of them.

Feature selection, vocabularies and scaling are fitted on training entities only.
No model/alpha tuning based on test performance. Sparse standardization differs from
the primary published dense preprocessing, so results must not be conflated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from full_gdsc_benchmark import SEED, tokens, rmse, corr


def fit_and_evaluate(root: Path):
    root=Path(root)
    base=root/'research/full_gdsc'
    dest=base/'strict_generalization'
    dest.mkdir(parents=True,exist_ok=True)
    obs=pd.read_csv(base/'matched_responses.csv')
    with np.load(base/'matched_omics.npz',allow_pickle=False) as f:
        cell_ids=f['cell_ids'].astype(str).tolist()
        expression=f['expression']
        mutations=f['mut']
        cn=f['cn']
    assert len(obs)==160512 and len(cell_ids)==649 and obs.drug_group.nunique()==286
    lut={c:i for i,c in enumerate(cell_ids)}
    all_drugs=sorted(obs.drug_group.unique())
    splits={}
    for mode,entities,seed in [('cell',np.array(sorted(cell_ids)),SEED+1),('drug',np.array(all_drugs),SEED)]:
        folds=KFold(n_splits=5,shuffle=True,random_state=seed)
        splits[mode]=[set(entities[test]) for _,test in folds.split(entities)]
        assert set.union(*splits[mode])==set(entities)
        assert sum(map(len,splits[mode]))==len(entities)
    output=[]
    predictions=[]
    memberships=[]
    for fold in range(1,6):
        cold_cells=splits['cell'][fold-1]
        cold_drugs=splits['drug'][fold-1]
        cell_held=obs.DepMap_ID.isin(cold_cells)
        drug_held=obs.drug_group.isin(cold_drugs)
        train=obs.loc[~cell_held & ~drug_held].copy()
        test=obs.loc[cell_held & drug_held].copy()
        ignored=obs.loc[cell_held ^ drug_held]
        assert len(train)+len(test)+len(ignored)==len(obs)
        assert len(test)>0 and len(train)>0
        assert not(set(train.DepMap_ID)&set(test.DepMap_ID))
        assert not(set(train.drug_group)&set(test.drug_group))
        assert not(set(train.NLME_CURVE_ID)&set(test.NLME_CURVE_ID))
        for kind,entities in [('cell',cold_cells),('drug',cold_drugs)]:
            pd.DataFrame({('DepMap_ID' if kind=='cell' else 'drug_group'):sorted(entities)}).to_csv(dest/f'fold{fold}_heldout_{kind}.csv',index=False)
        # Fit feature selection ONLY on unique training cells.
        unique_train_idx=np.array(sorted({lut[c] for c in train.DepMap_ID}))
        var=expression[unique_train_idx].var(axis=0,dtype=np.float64)
        gene_idx=np.lexsort((np.arange(len(var)),-var))[:100]
        assert len(gene_idx)==100 and np.isfinite(var).all()
        # Construct annotation vocabulary ONLY on training drugs. A held-out name
        # gets an unknown identity feature (0) as intended in strict cold testing.
        drug_train_meta=train.drop_duplicates('drug_group')
        vec=DictVectorizer(sparse=True,sort=True)
        vec.fit([tokens(x) for x in drug_train_meta.itertuples(index=False)])
        def features(frame):
            idx=np.fromiter((lut[c] for c in frame.DepMap_ID),count=len(frame),dtype=int)
            cells=np.column_stack((expression[:,gene_idx][idx],mutations[idx],cn[idx])).astype(np.float32)
            ann=vec.transform([tokens(x) for x in frame.itertuples(index=False)]).tocsr().astype(np.float32)
            return sparse.csr_matrix(cells),ann
        x_c_train,x_a_train=features(train)
        x_c_test,x_a_test=features(test)
        y=train.LN_IC50.to_numpy(float)
        truth=test.LN_IC50.to_numpy(float)
        def model(xtr,xte):
            # Sparse centered scaling would densify this matrix. With_mean=False
            # is applied to ALL ablations for a controlled within-study comparison.
            pipe=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),
                               StandardScaler(with_mean=False),Ridge(alpha=100,solver='lsqr',tol=1e-4))
            pipe.fit(xtr,y)
            ans=np.asarray(pipe.predict(xte),float)
            assert np.isfinite(ans).all()
            return ans
        preds={
            'global_mean':np.full(len(truth),y.mean()),
            'genomics_only':model(x_c_train,x_c_test),
            'annotations_only':model(x_a_train,x_a_test),
            'genomics_plus_annotations':model(sparse.hstack((x_c_train,x_a_train),format='csr'),
                                               sparse.hstack((x_c_test,x_a_test),format='csr')),
        }
        frame=test[['NLME_CURVE_ID','DepMap_ID','drug_group','lineage','LN_IC50']].copy()
        frame['fold']=fold
        for name,p in preds.items():
            frame['pred_'+name]=p
            output.append({'fold':fold,'model':name,'n':len(test),'train_pairs':len(train),
                           'ignored_pairs':len(ignored),'train_cells':train.DepMap_ID.nunique(),
                           'test_cells':test.DepMap_ID.nunique(),'train_drugs':train.drug_group.nunique(),
                           'test_drugs':test.drug_group.nunique(),
                           'rmse':rmse(truth,p),'mae':float(np.abs(truth-p).mean()),
                           'pearson_r':corr(truth,p)})
        predictions.append(frame)
        memberships.append({'fold':fold,'train_pairs':len(train),'test_pairs':len(test),
                            'ignored_one_cold_entity_pairs':len(ignored),
                            'cold_cell_count':len(cold_cells),'cold_drug_count':len(cold_drugs)})
        print('STRICT_FOLD',fold,'train',len(train),'test',len(test),
              'RMSE', {k:round(rmse(truth,p),4) for k,p in preds.items()},flush=True)
    m=pd.DataFrame(output)
    pr=pd.concat(predictions,ignore_index=True)
    m.to_csv(dest/'fold_metrics.csv',index=False)
    pr.to_csv(dest/'test_predictions.csv',index=False)
    pd.DataFrame(memberships).to_csv(dest/'fold_memberships.csv',index=False)
    summary=m.groupby('model',as_index=False).agg(folds=('fold','size'),mean_rmse=('rmse','mean'),
                    sd_rmse=('rmse','std'),mean_mae=('mae','mean'),mean_pooled_r=('pearson_r','mean'))
    summary.to_csv(dest/'summary.csv',index=False)
    meta={'scope':'paired 5fold BOTH held cell/drug; disjoint train and test by both',
          'source_rows':len(obs),'evaluated_test_rows':len(pr),
          'evaluated_fraction':len(pr)/len(obs),'fold_seed_cells':SEED+1,
          'fold_seed_drugs':SEED,'alpha_fixed':100,
          'full_catalog_chemical_structure_dedup':False,
          'preprocessing':'sparse StandardScaler(with_mean=False); selected 100 expression genes on unique training cells only',
          'partial_cover_warning':'Only same-index cross-products evaluated, all one-entity-only cold pairs are ignored in each fold.',
          'other_warning':'Internal matched-data screen, not independent external validation; name-based compound identity limits remain.'}
    (dest/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('STRICT_SUMMARY',summary.to_string(index=False),flush=True)
    return summary

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));arg=a.parse_args()
    fit_and_evaluate(Path(arg.root))
