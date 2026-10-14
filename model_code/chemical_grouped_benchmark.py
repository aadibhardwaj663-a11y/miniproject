"""Retrospective chemical-identity leakage audit: grouped unseen drugs and paired joint-cold folds.

Fixed 5-fold KFold on 282 identity keys, 649 cells in independent folds.
Unresolved compounds have separate GDSC-ID keys (not proof of unique chemistry).
Same fixed Ridge alpha 100 and same 100 training-only expression features as the prior
sparse strict benchmark. Never train on, tune against, or select based on test labels.
"""
from __future__ import annotations
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from full_gdsc_benchmark import tokens,rmse,corr,SEED
from identity_resolution import entity_fold_lists


def run(root,settings=('drug','both')):
    root=Path(root);base=root/'research/full_gdsc';output=base/'chemistry_grouped'
    output.mkdir(parents=True,exist_ok=True)
    drugs=entity_fold_lists(root)
    obs=pd.read_csv(base/'matched_responses.csv')
    identity=pd.read_csv(base/'identity_audit/drug_identity_review.csv').set_index('drug_group')
    obs['identity_key']=obs.drug_group.map(identity.identity_key)
    with np.load(base/'matched_omics.npz',allow_pickle=False) as f:
        cells=f['cell_ids'].astype(str);exp=f['expression'];mut=f['mut'];cn=f['cn']
    lut={c:i for i,c in enumerate(cells)}
    cv=KFold(n_splits=5,random_state=SEED+1,shuffle=True)
    cell_folds=[set(cells[te]) for _,te in cv.split(cells)]
    output_metrics=[];saved_predictions=[]; split_rows=[]
    for setting in settings:
        for fold in range(1,6):
            is_cold_drug=obs.drug_group.isin(drugs[fold-1])
            is_cold_cell=obs.DepMap_ID.isin(cell_folds[fold-1])
            if setting=='drug':
                tr=obs.loc[~is_cold_drug].copy();te=obs.loc[is_cold_drug].copy()
            elif setting=='both':
                tr=obs.loc[~is_cold_drug & ~is_cold_cell].copy();te=obs.loc[is_cold_drug & is_cold_cell].copy()
            else:raise ValueError(setting)
            assert len(tr) and len(te)
            assert set(tr.identity_key).isdisjoint(te.identity_key)
            assert set(tr.drug_group).isdisjoint(te.drug_group)
            if setting=='both':assert set(tr.DepMap_ID).isdisjoint(te.DepMap_ID)
            assert set(tr.NLME_CURVE_ID).isdisjoint(te.NLME_CURVE_ID)
            dpath=output/'splits'/setting;dpath.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'identity_key':sorted(te.identity_key.unique())}).to_csv(dpath/f'fold{fold}_heldout_identity_keys.csv',index=False)
            pd.DataFrame({'drug_group':sorted(te.drug_group.unique())}).to_csv(dpath/f'fold{fold}_heldout_drug_names.csv',index=False)
            if setting=='both':pd.DataFrame({'DepMap_ID':sorted(te.DepMap_ID.unique())}).to_csv(dpath/f'fold{fold}_heldout_cells.csv',index=False)
            split_rows.append({'setting':setting,'fold':fold,'train_pairs':len(tr),'test_pairs':len(te),
                               'test_identity_keys':te.identity_key.nunique(),'test_drug_names':te.drug_group.nunique(),
                               'test_cells':te.DepMap_ID.nunique()})
            ix=np.array(sorted({lut[x] for x in tr.DepMap_ID}),dtype=int)
            var=exp[ix].var(axis=0,dtype=np.float64)
            selected=np.lexsort((np.arange(len(var)),-var))[:100]
            assert len(selected)==100 and np.isfinite(var).all()
            vocab=DictVectorizer(sparse=True,sort=True)
            vocab.fit([tokens(x) for x in tr.drop_duplicates('drug_group').itertuples(index=False)])
            def build(frame):
                ci=np.fromiter((lut[x] for x in frame.DepMap_ID),dtype=int,count=len(frame))
                gen=np.column_stack((exp[:,selected][ci],mut[ci],cn[ci])).astype(np.float32)
                assert np.isfinite(gen).all()
                # Compute drug descriptors only once per distinct drug (at most 286),
                # then reuse the encoded sparse row for all measured cell-drug pairs.
                unique=frame.drop_duplicates('drug_group').reset_index(drop=True)
                drug_matrix=vocab.transform([tokens(x) for x in unique.itertuples(index=False)]).tocsr().astype(np.float32)
                location=pd.Series(np.arange(len(unique)),index=unique.drug_group)
                inverse=frame.drug_group.map(location).to_numpy(dtype=np.int32)
                assert (inverse>=0).all()
                annotations=drug_matrix[inverse].tocsr()
                return sparse.csr_matrix(gen),annotations
            x_c_train,x_a_train=build(tr); x_c_test,x_a_test=build(te)
            y=tr.LN_IC50.to_numpy(float);truth=te.LN_IC50.to_numpy(float)
            def predict(train,test):
                pipe=make_pipeline(StandardScaler(with_mean=False),Ridge(alpha=100,solver='lsqr',tol=1e-4))
                pipe.fit(train,y)
                result=np.asarray(pipe.predict(test))
                assert np.isfinite(result).all()
                return result
            preds={'global_mean':np.full(len(truth),y.mean()),
                   'genomics_only':predict(x_c_train,x_c_test),
                   'annotations_only':predict(x_a_train,x_a_test),
                   'combined':predict(sparse.hstack([x_c_train,x_a_train],format='csr'),sparse.hstack([x_c_test,x_a_test],format='csr'))}
            pr=te[['NLME_CURVE_ID','DepMap_ID','drug_group','identity_key','lineage','LN_IC50']].copy();pr['setting']=setting;pr['fold']=fold
            for label,yp in preds.items():
                pr['pred_'+label]=yp
                output_metrics.append({'setting':setting,'fold':fold,'model':label,'rmse':rmse(truth,yp),'mae':float(np.mean(np.abs(truth-yp))),'pearson_r':corr(truth,yp),'test_n':len(te),'train_n':len(tr)})
            saved_predictions.append(pr)
            print('IDENTITY_FOLD',setting,fold,'n',len(te),{k:round(rmse(truth,v),4) for k,v in preds.items()},flush=True)
        pd.DataFrame(output_metrics).to_csv(output/'fold_metrics.csv',index=False)
        pd.concat(saved_predictions,ignore_index=True).to_csv(output/'test_predictions.csv',index=False)
        pd.DataFrame(split_rows).to_csv(output/'fold_membership_summary.csv',index=False)
    m=pd.DataFrame(output_metrics)
    s=m.groupby(['setting','model'],as_index=False).agg(folds=('fold','size'),rmse=('rmse','mean'),rmse_sd=('rmse','std'),mae=('mae','mean'),pooled_r=('pearson_r','mean'),n_test=('test_n','sum'))
    s.to_csv(output/'summary.csv',index=False)
    test=pd.concat(saved_predictions,ignore_index=True)
    # Check no duplicate response measured twice within each evaluation setting.
    assert not test.duplicated(['setting','NLME_CURVE_ID']).any()
    for setting in settings:
        ds=test[test.setting.eq(setting)]
        assert all(ds.groupby('identity_key').fold.nunique()==1)
    protocol={'seed_drugs':SEED,'seed_cells':SEED+1,'alpha_fixed_before_analysis':100,'folds':5,
              'posthoc_discovery':'Earlier distinct drug-name folds allowed known same-CID compounds to appear on opposite sides of train/test; improved identity fold protocol added after inspecting chemical collisions',
              'identity_form_status':'Provisional GDSC assay ID to PubChem CID and InChIKey mappings; salts stereoisomers and 50 missing codes remain',
              'status':'retrospective robustness check, not prospective untouched test or external validation',
              'preprocessing':'100 highest-variance transcript features selected on training cells, mutations and CN, sparse non-centered scale; training-only drug token vocabulary',
              'splits':'5-fold identity clusters, no known same-CID drug names across train/test; joint test is only paired cell/drug intersections, not entire grid',
              'test_data_tuning':False,'comments':'Do not compare RMSEs between settings as equal difficulty. Original cell benchmark unchanged.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('IDENTITY_SUMMARY\n'+s.to_string(index=False),flush=True)
    return s

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));
    ap.add_argument('--settings',default='drug,both');x=ap.parse_args();run(x.root,x.settings.split(','))
