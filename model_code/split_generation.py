"""Verify and, if needed, deterministically recreate train/validation/test IDs.
The final checkpoint training IDs always equal train_core UNION validation.
The five-drug held-out test is never used for fitting.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

def main(root):
    root=Path(root)
    obs=pd.read_csv(root/'data/curated_responses.csv')
    for kind,field,seed,n_validation in [
        ('unseen_cell','DepMap_ID',2701,32),
        ('unseen_drug','drug_group',2702,3),
    ]:
        parent=lambda side: pd.read_csv(root/'splits'/f'{kind}_{side}.csv',dtype=str).iloc[:,0].tolist()
        full_train=parent('train'); test=parent('test')
        if kind=='unseen_cell':
            ordered=np.random.default_rng(seed).permutation(np.sort(np.asarray(full_train)))
            valid=sorted(ordered[:n_validation].tolist())
        else:
            valid=sorted(np.random.default_rng(seed).choice(
                np.asarray(sorted(full_train)),size=n_validation,replace=False).tolist())
        training=sorted(set(full_train)-set(valid))
        assert set(training).isdisjoint(valid)
        assert set(full_train).isdisjoint(test)
        assert set(training)|set(valid)==set(full_train)
        folder=root/'splits'/kind
        folder.mkdir(exist_ok=True)
        for name,values in [('train',training),('validation',valid),('test',test),('final_train',full_train)]:
            pd.Series(sorted(values),name=field).to_csv(folder/f'{name}_entities.csv',index=False)
            df=obs[obs[field].isin(values)].copy().sort_values(['DepMap_ID','drug_group'])
            df[['NLME_CURVE_ID','DepMap_ID','DRUG_ID','drug_group','LN_IC50']].to_csv(folder/f'{name}.csv',index=False)
        assert len(pd.read_csv(folder/'train.csv'))+len(pd.read_csv(folder/'validation.csv'))+len(pd.read_csv(folder/'test.csv'))==len(obs)
        print(f'{kind}: training {len(training)} entities, validation {len(valid)}, test {len(test)}; '
              f'final model fitting used train+validation ({len(full_train)})')
    print('All fold manifests generated from frozen outer holdouts.')

if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
