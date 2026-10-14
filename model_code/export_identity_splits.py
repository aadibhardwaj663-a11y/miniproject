"""Export outer train/test and *reference-only* inner train/validation manifests.

The existing reported five-fold models were refit on the FULL outer-training pools.
These inner subsets are supplied to auditors, not used for hyperparameter search,
checkpoint fitting or test scoring. No target values are stored in split manifests.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def run(root):
    base=Path(root)/'research/full_gdsc'
    obs=pd.read_csv(base/'matched_responses.csv',usecols=['NLME_CURVE_ID','DepMap_ID','drug_group'])
    identities=pd.read_csv(base/'identity_audit/drug_identity_review.csv',usecols=['drug_group','identity_key'])
    obs=obs.merge(identities,validate='many_to_one',on='drug_group')
    fold_data=pd.read_csv(base/'chemistry_grouped/test_predictions.csv',usecols=['setting','fold','NLME_CURVE_ID','DepMap_ID','drug_group','identity_key'])
    columns=['NLME_CURVE_ID','DepMap_ID','drug_group','identity_key']
    for setting in ('drug','both'):
        for fold in range(1,6):
            te=fold_data[(fold_data.setting==setting)&(fold_data.fold==fold)][columns].copy()
            test_keys=set(te.identity_key)
            holdout_cells=set(te.DepMap_ID) if setting=='both' else set()
            outer_train=obs[~obs.identity_key.isin(test_keys)&~obs.DepMap_ID.isin(holdout_cells)][columns].copy()
            assert not set(outer_train.identity_key)&test_keys
            if holdout_cells: assert not set(outer_train.DepMap_ID)&holdout_cells
            assert not set(outer_train.NLME_CURVE_ID)&set(te.NLME_CURVE_ID)
            assert (len(outer_train)>0) and len(te)>0
            # Deterministic grouped reference partition, NOT used by reported folds.
            keys=np.array(sorted(outer_train.identity_key.unique()))
            rng=np.random.RandomState(9307+fold)
            rng.shuffle(keys)
            valkeys=set(keys[:max(1,int(round(len(keys)*.15)))])
            mask=outer_train.identity_key.isin(valkeys)
            inner_validation=outer_train[mask]
            inner_train=outer_train[~mask]
            assert not set(inner_validation.identity_key)&set(inner_train.identity_key)
            assert len(inner_validation)+len(inner_train)==len(outer_train)
            location=base/'chemistry_grouped/splits'/setting
            location.mkdir(parents=True,exist_ok=True)
            for name,frame in [('outer_train',outer_train),('outer_test',te),('inner_train',inner_train),('inner_validation',inner_validation)]:
                frame.sort_values('NLME_CURVE_ID').to_csv(location/f'fold{fold}_{name}.csv',index=False)
            print('SPLIT_EXPORT',setting,fold,'train',len(outer_train),'inner_validation',len(inner_validation),'test',len(te),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=ap.parse_args();run(a.root)
