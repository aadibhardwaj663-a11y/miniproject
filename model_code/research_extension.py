"""Exploratory repeated cold-cell, cold-drug and strict both-cold experiments.

All new experiments use a prior fixed design; the original previously-inspected
holdouts and original checkpoints are unchanged. No chemical structure is implied.
"""
from pathlib import Path
import json, re, argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr

PANEL=['TP53','KRAS','BRAF','EGFR','PIK3CA','PTEN','APC','RB1','MYC','CDKN2A']
SEEDS=[107,207,307,407,507]
ABLS={'all':('expression','panel','drug'),
      'expression_drug':('expression','drug'),
      'panel_drug':('panel','drug'),
      'drug_only':('drug',),
      'cell_only':('expression','panel')}

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    assert len(y)==len(p)>0 and np.isfinite(y).all() and np.isfinite(p).all()
    return {'n':int(len(y)),'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'mae':float(np.mean(abs(y-p))),
            'pearson_r':float(np.corrcoef(y,p)[0,1]) if np.std(p)>1e-10 and np.std(y)>1e-10 else None}

def token(row):
    ans={'pathway:'+str(row.PATHWAY_NAME).lower().replace(' ','_'):1.0}
    for word in re.findall('[A-Za-z][A-Za-z0-9-]*',str(row.PUTATIVE_TARGET)):
        ans['target:'+word.lower()]=1.0
    return ans

def read_project(root):
    with np.load(root/'data/matched_expression_full.npz',allow_pickle=False) as z:
        cells=z['cell_ids'].astype(str).tolist();genes=z['gene_columns'].astype(str).tolist();expr=z['values'].astype('float32')
    obs=pd.read_csv(root/'data/curated_responses.csv')
    mut=pd.read_csv(root/'data/mutation_panel_binary.csv',index_col=0).loc[cells,PANEL]
    copy=pd.read_csv(root/'data/copy_number_panel.csv',index_col=0).loc[cells]
    cncols=[next(col for col in copy.columns if col.startswith(g+' (')) for g in PANEL]
    cn=copy[cncols].to_numpy('float32')
    M=mut.to_numpy('float32')
    assert obs.shape[0]==4000 and obs.drug_group.nunique()==20 and obs.DepMap_ID.nunique()==200
    assert not obs.duplicated(['DepMap_ID','drug_group']).any()
    assert np.isfinite(expr).all() and np.isfinite(M).all() and np.isfinite(cn).all()
    return {'obs':obs,'cells':cells,'genes':genes,'expr':expr,'mut':M,'cn':cn,'lookup':{s:i for i,s in enumerate(cells)}}

def safe_aliases(root, drug_train,drug_val,drug_test):
    aliases=pd.read_csv(root/'data/compound_alias_manifest.csv')
    norm=lambda s:re.sub('[^a-z0-9]','',str(s).lower())
    arrays=[]
    for part in (drug_train,drug_val,drug_test):
        arrays.append({norm(x) for x in aliases[aliases.drug_group.isin(part)].alias})
    for i in range(3):
        for j in range(i+1,3):
            assert not (arrays[i]&arrays[j]),'Known alias crosses partitions'

def split_pools(d,mode,seed,root):
    rng=np.random.default_rng(seed)
    cells=np.asarray(sorted(d['cells'])); drugs=np.asarray(sorted(d['obs'].drug_group.unique()))
    cl=rng.permutation(cells);dr=rng.permutation(drugs)
    ct=set(cl[-40:]);cv=set(cl[-60:-40]);ctr=set(cl[:-60]);ctr_final=set(cl[:-40])
    dt=set(dr[-5:]);dv=set(dr[-8:-5]);dtr=set(dr[:-8]);dtr_final=set(dr[:-5])
    safe_aliases(root,dtr,dv,dt)
    o=d['obs']
    if mode=='cell':
        train=o[o.DepMap_ID.isin(ctr)];val=o[o.DepMap_ID.isin(cv)];test=o[o.DepMap_ID.isin(ct)]
        final=o[o.DepMap_ID.isin(ctr_final)]
    elif mode=='drug':
        train=o[o.drug_group.isin(dtr)];val=o[o.drug_group.isin(dv)];test=o[o.drug_group.isin(dt)]
        final=o[o.drug_group.isin(dtr_final)]
    elif mode=='both':
        # Mixed cross-quadrant samples are deliberately unused, not assigned to training.
        train=o[o.DepMap_ID.isin(ctr)&o.drug_group.isin(dtr)]
        val=o[o.DepMap_ID.isin(cv)&o.drug_group.isin(dv)]
        test=o[o.DepMap_ID.isin(ct)&o.drug_group.isin(dt)]
        final=o[o.DepMap_ID.isin(ctr_final)&o.drug_group.isin(dtr_final)]
    else:raise ValueError(mode)
    assert set(final.NLME_CURVE_ID).isdisjoint(test.NLME_CURVE_ID)
    assert set(train.NLME_CURVE_ID).isdisjoint(val.NLME_CURVE_ID)
    if mode in ('cell','both'):
        assert set(final.DepMap_ID).isdisjoint(test.DepMap_ID)
    if mode in ('drug','both'):
        assert set(final.drug_group).isdisjoint(test.drug_group)
    meta={'mode':mode,'seed':seed,'train_cells':sorted(train.DepMap_ID.unique().tolist()),
          'val_cells':sorted(val.DepMap_ID.unique().tolist()),'test_cells':sorted(test.DepMap_ID.unique().tolist()),
          'train_drugs':sorted(train.drug_group.unique().tolist()),'val_drugs':sorted(val.drug_group.unique().tolist()),
          'test_drugs':sorted(test.drug_group.unique().tolist()),
          'train_curve_ids':train.NLME_CURVE_ID.astype(str).tolist(),
          'validation_curve_ids':val.NLME_CURVE_ID.astype(str).tolist(),
          'test_curve_ids':test.NLME_CURVE_ID.astype(str).tolist(),
          'final_training_curve_ids':final.NLME_CURVE_ID.astype(str).tolist(),
          'excluded_cross_block_pairs':4000-len(final)-len(test) if mode=='both' else 0,
          'protocol':'inner train/val disjoint; final refit on union of non-test entity partitions; test never fits any transformation'}
    path=root/'research/validation'/f'{mode}_seed{seed}_ids.json'
    path.write_text(json.dumps(meta,indent=2))
    return train.reset_index(drop=True),val.reset_index(drop=True),final.reset_index(drop=True),test.reset_index(drop=True)

def design(d,fit,eval_set):
    indices=np.asarray(sorted({d['lookup'][x] for x in fit.DepMap_ID}),int)
    variance=d['expr'][indices].var(axis=0,dtype=np.float64)
    sel=np.lexsort((np.arange(len(variance)),-variance))[:100]
    assert len(sel)==100
    vec=DictVectorizer(sparse=False,sort=True)
    vec.fit([token(r) for r in fit.itertuples(index=False)])
    def enc(frame):
        ii=np.array([d['lookup'][x] for x in frame.DepMap_ID]); return {
            'expression':d['expr'][ii][:,sel],
            'panel':np.column_stack((d['mut'][ii],d['cn'][ii])),
            'drug':vec.transform([token(r) for r in frame.itertuples(index=False)]).astype('float32')}
    return enc(fit),enc(eval_set),sel,vec

def stack(parts,variant):
    out=np.column_stack([parts[key] for key in ABLS[variant]])
    assert np.isfinite(out).all()
    return out

def fit_predict(tr,te,ytrain,seed,variant):
    x=stack(tr,variant);xt=stack(te,variant)
    rf=RandomForestRegressor(n_estimators=10,max_depth=8,min_samples_leaf=3,
           max_features=.7,random_state=seed,n_jobs=1)
    rf.fit(x,ytrain)
    return rf.predict(xt),rf

def run(root,modes=('cell','drug','both'),seeds=SEEDS):
    root=Path(root)
    d=read_project(root)
    logs=[]; preds=[]; applicability=[]; lineages=[]; importance=[]
    for mode in modes:
      for seed in seeds:
        print('START',mode,seed,flush=True)
        tr,val,final,te=split_pools(d,mode,seed,root)
        print('SPLIT_DONE',mode,seed,flush=True)
        trainParts,validationParts,_,_=design(d,tr,val)
        print('VAL_DESIGN_DONE',mode,seed,flush=True)
        # validation used for a diagnostic, fixed configuration, NOT tuning the outer test
        vpred,_=fit_predict(trainParts,validationParts,tr.LN_IC50.to_numpy(float),seed,'all')
        valmetrics=metrics(val.LN_IC50,vpred)
        print('VALIDATION_DONE',mode,seed,flush=True)
        # Train-only final feature fit after validation; choose no hyperparameters on test.
        fitparts,testparts,sel,dv=design(d,final,te)
        print('FINAL_DESIGN_DONE',mode,seed,flush=True)
        y=final.LN_IC50.to_numpy(float);actual=te.LN_IC50.to_numpy(float)
        mean=np.full(len(te),np.mean(y))
        # For familiar drugs, per-drug mean is a meaningful strong baseline.
        # For cold drugs, this is not permitted and is not used.
        if mode=='cell':
          perdrug=te.drug_group.map(final.groupby('drug_group').LN_IC50.mean()).to_numpy(float)
          base={'global_mean':mean,'training_drug_mean':perdrug}
        else:
          base={'global_mean':mean}
        for key,pred in base.items():
          logs.append({'mode':mode,'seed':seed,'variant':key,'phase':'outer_test',**metrics(actual,pred),
                       'validation_rmse_full_forest':valmetrics['rmse']})
        # Ridge-all anchor permits direct feature ablation comparisons.
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        anchor=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100))
        anchor.fit(stack(fitparts,'all'),y)
        rpred=anchor.predict(stack(testparts,'all'))
        logs.append({'mode':mode,'seed':seed,'variant':'ridge_all','phase':'outer_test',**metrics(actual,rpred),
                     'validation_rmse_full_forest':valmetrics['rmse']})
        for variant in ABLS:
          print('BEGIN_VARIANT',variant,mode,seed,flush=True)
          if variant=='all':
            pred,rf=fit_predict(fitparts,testparts,y,seed,variant)
          else:
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            # All feature ablations use a common Ridge family (fixed alpha),
            # so removing a modality is not confounded with changing estimator.
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100))
            model.fit(stack(fitparts,variant),y)
            pred=model.predict(stack(testparts,variant))
            rf=model
          logs.append({'mode':mode,'seed':seed,'variant':variant,'phase':'outer_test',**metrics(actual,pred),
                       'validation_rmse_full_forest':valmetrics['rmse']})
          if variant=='all':
            df=te[['NLME_CURVE_ID','DepMap_ID','drug_group','lineage','LN_IC50']].copy()
            df['seed']=seed;df['mode']=mode;df['prediction_full_forest']=pred
            df['prediction_global_mean']=mean
            if mode=='cell':df['prediction_training_drug_mean']=perdrug
            preds.append(df)
            print('PRED_APP_START',mode,seed,flush=True)
            # Out-of-distribution diagnostics ONLY; no new features added to fit.
            training_idx=np.array(sorted({d['lookup'][x] for x in final.DepMap_ID}))
            training_profiles=d['expr'][training_idx][:,sel]
            standard=StandardScaler().fit(training_profiles)
            te_idx=np.array([d['lookup'][x] for x in te.DepMap_ID])
            sample=standard.transform(d['expr'][te_idx][:,sel]); ref=standard.transform(training_profiles)
            # nearest by Euclidean in standardized top100 feature space
            from sklearn.metrics import pairwise_distances
            closest=pairwise_distances(sample,ref,metric='euclidean').min(axis=1)
            for j,r in enumerate(te.itertuples()):
              applicability.append({'mode':mode,'seed':seed,'curve_id':str(r.NLME_CURVE_ID),
                                    'distance_to_nearest_training_cell':float(closest[j]),
                                    'abs_error':abs(float(actual[j]-pred[j])),'drug_group':r.drug_group})
            # Biological lineage errors: only descriptive, no independent unit claims.
            for lineage,g in df.groupby('lineage',dropna=False):
              if g.DepMap_ID.nunique()<3:continue
              lineages.append({'mode':mode,'seed':seed,'lineage':str(lineage),
                    'n_pairs':len(g),'n_cells':g.DepMap_ID.nunique(),
                    'rmse':metrics(g.LN_IC50,g.prediction_full_forest)['rmse']})
            print('PRED_APP_DONE',mode,seed,flush=True)
            if mode=='both' and seed==seeds[0]:
              # save exact preprocessing for the strict both-cold model
              joblib.dump({'pipeline':rf,'gene_indices':sel,'gene_names':list(np.asarray(d['genes'])[sel]),
                    'drug_encoder':dv,'panel':PANEL,'model':'joint_both_cold_research_extension',
                    'seed':seed,'training_cell_ids':sorted(final.DepMap_ID.unique().tolist()),
                    'training_drug_names':sorted(final.drug_group.unique().tolist())},
                    root/'research/models/both_cold_seed107.joblib')
        print(f'{mode} seed{seed} train={len(final)} test={len(te)} full={logs[-1]["rmse"]:.3f} val_full={valmetrics["rmse"]:.3f}',flush=True)
    score=pd.DataFrame(logs); p=pd.concat(preds,ignore_index=True)
    score.to_csv(root/'research/metrics/repeated_seed_metrics.csv',index=False)
    p.to_csv(root/'research/data/repeated_seed_predictions.csv',index=False)
    pd.DataFrame(applicability).to_csv(root/'research/metrics/applicability_errors.csv',index=False)
    pd.DataFrame(lineages).to_csv(root/'research/metrics/lineage_errors.csv',index=False)
    summaries=score.groupby(['mode','variant'],as_index=False).agg(
           n_seeds=('seed','nunique'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),
           mae_mean=('mae','mean'),pearson_mean=('pearson_r','mean'))
    summaries.to_csv(root/'research/metrics/repeated_seed_summary.csv',index=False)
    print('SUMMARY\n'+summaries.sort_values(['mode','rmse_mean']).round(3).to_string(index=False),flush=True)
    return summaries

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parents[1]));args=parser.parse_args()
  run(args.root)
