"""Export evaluator-friendly row-ID train/validation/test manifests for the full study.

IMPORTANT: Training evidence comes from each *entire* outer training pool. The
inner partitions are deterministic, unused diagnostic partitions; they were not
used to tune the fixed Ridge alpha or to produce the published RMSEs. The
validation rows were refitted as part of outer training. Never claim otherwise.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from full_gdsc_benchmark import SEED

COLS=['NLME_CURVE_ID','DepMap_ID','drug_group']

def export(root:Path):
    root=Path(root)
    obs=pd.read_csv(root/'research/full_gdsc/matched_responses.csv')
    cells=np.array(sorted(obs.DepMap_ID.unique()))
    drugs=np.array(sorted(obs.drug_group.unique()))
    cell_folds=[set(cells[te]) for _,te in KFold(n_splits=5,shuffle=True,random_state=SEED+1).split(cells)]
    drug_folds=[set(drugs[te]) for _,te in KFold(n_splits=5,shuffle=True,random_state=SEED).split(drugs)]
    base=root/'splits/full_catalog'
    for setting in ['cell','drug','both']:
        dest=base/setting;dest.mkdir(parents=True,exist_ok=True)
        for fold in range(1,6):
            test_c=obs.DepMap_ID.isin(cell_folds[fold-1])
            test_d=obs.drug_group.isin(drug_folds[fold-1])
            if setting=='cell':
                outer_test=test_c; outer_train=~test_c
            elif setting=='drug':
                outer_test=test_d; outer_train=~test_d
            else:
                outer_test=test_c & test_d
                outer_train=~test_c & ~test_d
            tr=obs.loc[outer_train, COLS]
            te=obs.loc[outer_test, COLS]
            excluded=obs.loc[~outer_train & ~outer_test,COLS]
            assert not set(tr.NLME_CURVE_ID)&set(te.NLME_CURVE_ID)
            if setting in ('cell','both'):
                assert not set(tr.DepMap_ID)&set(te.DepMap_ID)
            if setting in ('drug','both'):
                assert not set(tr.drug_group)&set(te.drug_group)
            # Nested inner validation (optional diagnosis, NO original model tuning).
            val_seed=SEED + (101 if setting=='cell' else 201 if setting=='drug' else 301)+fold
            rng=np.random.default_rng(val_seed)
            cids=sorted(tr.DepMap_ID.unique()); dnames=sorted(tr.drug_group.unique())
            valcells=set(rng.choice(cids,size=max(1,len(cids)//10),replace=False))
            valdrugs=set(rng.choice(dnames,size=max(1,len(dnames)//10),replace=False))
            vc=tr.DepMap_ID.isin(valcells); vd=tr.drug_group.isin(valdrugs)
            if setting=='cell':va=vc;it=~vc
            elif setting=='drug':va=vd;it=~vd
            else:va=vc&vd;it=~vc&~vd
            tr_inner=tr.loc[it];va_inner=tr.loc[va]
            assert len(va_inner)>0 and not set(tr_inner.NLME_CURVE_ID)&set(va_inner.NLME_CURVE_ID)
            for basename,frame in [('outer_train',tr),('outer_test',te),('inner_train',tr_inner),('inner_validation',va_inner)]:
                frame.to_csv(dest/f'fold{fold}_{basename}.csv',index=False)
            # Write only an aggregate excluded count. These joint one-cold rows must
            # not be misrepresented as training or test examples.
            metadata={'setting':setting,'fold':fold,'outer_train_pairs':len(tr),'outer_test_pairs':len(te),
                      'excluded_from_this_outer_fold_pairs':len(excluded),'inner_train_pairs':len(tr_inner),
                      'inner_validation_pairs':len(va_inner),'inner_validation_seed':val_seed,
                      'inner_purpose':'Unused diagnostic manifest; no test/inner tuning. Outer training included all inner validation rows.',
                      'not_final_fit':'Outer fold models are fitted on the ENTIRE outer_train pool, including inner_validation; do not claim separately validated hyperparameters.'}
            (dest/f'fold{fold}_protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print('EXPORT',setting,fold,'train',len(tr),'test',len(te),'inner validation',len(va_inner),flush=True)
    (base/'README.md').write_text('''# Full-catalog train, validation and test manifests

Settings: `cell`, `drug` and `both`; files for folds 1-5 contain **actual experimental response curve IDs and matched cell/drug identifiers**.

- `fold*_outer_train.csv`: all rows used to fit each held-out fold model.
- `fold*_outer_test.csv`: real observations scored for final metrics.
- `fold*_inner_train.csv`, `fold*_inner_validation.csv`: deterministic disjoint *diagnostic* subpartitions of outer training, supplied for evaluator inspection. **No alpha tuning was performed using them. The deployed outer-fold model was fitted on all outer_train, including inner_validation.**
- `fold*_protocol.json`: counts and ignored one-cold-entity observations in the simultaneous experiment.

`both`: outer training contains neither a held-out cell nor a held-out drug; test contains **both**. Rows in only one cold group are deliberately excluded. These test rows cover ~19.9% of all measured pairs across five folds. The same-index intersection is a partial cross-product evaluation, not an exhaustive full-grid test.

All targets remain in the source dataset, not copied into these identifier-only files. Derivation: `model_code/export_full_splits.py`, source `research/full_gdsc/matched_responses.csv`.
''')

if __name__=='__main__':export(Path(__file__).resolve().parents[1])
