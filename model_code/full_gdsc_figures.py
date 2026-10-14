"""Reproducible graphics and rigorous subgroup metrics from held-out fold predictions."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['global_mean','drug_mean','cell_only','cell_plus_annotations']

def run(root):
 folder=Path(root)/'research/full_gdsc';fig=folder/'figures';fig.mkdir(exist_ok=True)
 metrics=pd.read_csv(folder/'fold_metrics.csv');pred=pd.read_csv(folder/'fold_predictions.csv');bio=pd.read_csv(folder/'biological_interpretability/five_fold_metrics.csv')
 def save(name):plt.savefig(fig/f'{name}.png',dpi=190,bbox_inches='tight');plt.savefig(fig/f'{name}.svg',bbox_inches='tight');plt.close()
 def r(y,p):return float(np.corrcoef(y,p)[0,1]) if len(y)>3 and np.std(y)>1e-12 and np.std(p)>1e-12 else np.nan
 bydrug=[]
 for mode,df in pred.groupby('mode'):
  for drug,g in df.groupby('drug_group'):
   y=g.LN_IC50.to_numpy(float)
   for mdl in (['global_mean','cell_only','cell_plus_annotations']+(['drug_mean'] if mode=='cell' else [])):
    pr=g['pred_'+mdl].to_numpy(float)
    bydrug.append({'mode':mode,'drug_group':drug,'model':mdl,'n':len(g),'rmse':float(np.sqrt(np.mean((pr-y)**2))),'mae':float(np.mean(abs(pr-y))),'within_drug_pearson_r':r(y,pr),'response_sd':float(np.std(y,ddof=1))})
 byd=pd.DataFrame(bydrug);byd.to_csv(folder/'pooled_per_drug_metrics.csv',index=False)
 bylin=[]
 for mode,df in pred.groupby('mode'):
  for lineage,g in df.groupby('lineage'):
   if len(g)<30:continue
   for mdl in ['global_mean','cell_only','cell_plus_annotations']+(['drug_mean'] if mode=='cell' else []):
    y=g.LN_IC50.to_numpy(float);pr=g['pred_'+mdl].to_numpy(float)
    bylin.append({'mode':mode,'lineage':lineage,'n':len(g),'model':mdl,'rmse':float(np.sqrt(np.mean((pr-y)**2))),'mae':float(np.mean(abs(pr-y)))})
 byl=pd.DataFrame(bylin);byl.to_csv(folder/'pooled_lineage_metrics.csv',index=False)
 s=metrics.groupby(['mode','model'],as_index=False).agg(mean_rmse=('rmse','mean'),sd_rmse=('rmse','std'),mean_mae=('mae','mean'),mean_pearson=('pearson_r','mean'))
 # bar comparison
 fig0,axs=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
 pal={'global_mean':'#6F8195','drug_mean':'#F59F45','cell_only':'#4B83AA','cell_plus_annotations':'#138E81'}
 for ax,mode in zip(axs,['cell','drug']):
  q=s[s['mode']==mode];vals=q.mean_rmse.to_numpy();errs=q.sd_rmse.to_numpy();names=q.model.tolist()
  x=np.arange(len(names));ax.bar(x,vals,yerr=errs,color=[pal[n] for n in names],capsize=3)
  ax.set_xticks(x,[n.replace('_','\n') for n in names]);ax.set_title('Held-out '+('cell lines' if mode=='cell' else 'drug groups'))
  ax.set_ylabel('Mean test RMSE across five folds (LN_IC50)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  for i,v in enumerate(vals): ax.text(i,float(v)+float(errs[i])+0.055,f'{v:.3f}',ha='center',fontsize=9)
  ax.set_ylim(0,max(s.mean_rmse+s.sd_rmse)*1.25)
 fig0.suptitle('Full matched GDSC2 benchmark: 649 cells, 286 drugs, 160,512 responses',weight='bold',fontsize=13);fig0.tight_layout();save('01_full_kfold_rmse')
 # per-drug Pearson histogram, exclude constant drug mean
 q=byd[(byd['mode']=='cell')&(byd['model']=='cell_plus_annotations')].dropna(subset=['within_drug_pearson_r'])
 fig0,ax=plt.subplots(figsize=(9,5));ax.hist(q.within_drug_pearson_r,bins=np.linspace(-1,1,28),color='#138E81',edgecolor='white')
 ax.axvline(q.within_drug_pearson_r.median(),color='#E17B36',ls='--',label=f'Median r = {q.within_drug_pearson_r.median():.3f}')
 ax.set(xlabel='Pooled out-of-fold Pearson correlation within each drug',ylabel='Number of drugs',title='Within-drug predictive correlation across five cell-line folds')
 ax.legend();ax.grid(axis='y',alpha=.2);save('02_within_drug_pearson')
 # skill against per-drug baseline
 q1=byd[(byd['mode']=='cell')&(byd['model'].isin(['drug_mean','cell_plus_annotations']))].pivot(index='drug_group',columns='model',values=['rmse','response_sd','within_drug_pearson_r'])
 sk=pd.DataFrame({'drug':q1.index,'baseline':q1['rmse','drug_mean'].values,'model':q1['rmse','cell_plus_annotations'].values,'r':q1['within_drug_pearson_r','cell_plus_annotations'].values,'response_sd':q1['response_sd','drug_mean'].values})
 sk['delta_rmse']=sk.baseline-sk.model;sk.to_csv(folder/'per_drug_skill_cell_holdout.csv',index=False)
 fig0,ax=plt.subplots(figsize=(9,5));ax.scatter(sk.baseline,sk.model,alpha=.5,s=17,color='#2A8B82');lo=min(sk[['baseline','model']].min());hi=max(sk[['baseline','model']].max());ax.plot([lo,hi],[lo,hi],'--',color='#C66050')
 ax.set(xlabel='Per-drug training-mean RMSE',ylabel='Model RMSE',title='Within-drug benchmark against strong drug-specific mean\n(each point is one of 286 drugs)');ax.grid(alpha=.15);save('03_per_drug_skill_scatter')
 # selected MEK drug subgroup
 targets=['selumetinib','trametinib','pd0325901','nutlin-3a (-)','palbociclib']
 q=sk[sk.drug.isin(targets)].set_index('drug').reindex(targets).dropna()
 fig0,ax=plt.subplots(figsize=(10,5));xx=np.arange(len(q));ax.bar(xx-.2,q.model,.38,label='Genomic+drug-identity Ridge',color='#138E81');ax.bar(xx+.2,q.baseline,.38,label='Per-drug training mean',color='#EAA15E')
 ax.plot(xx,q.response_sd,ls=':',color='#333333',marker='o',label='Observed SD (reference only)')
 ax.set_xticks(xx,list(q.index),rotation=15);ax.set_ylabel('RMSE / observed LN_IC50 SD');ax.set_title('Selected mechanistic examples: out-of-fold prediction errors');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);save('04_mek_and_biomarker_drugs')
 # Biological drug specific models
 b=bio.pivot_table(index='drug',columns='model',values='rmse',aggfunc='mean')
 xx=np.arange(len(b));fig0,ax=plt.subplots(figsize=(9,5));ax.bar(xx-.2,b['per_drug_mean'],.4,label='Drug-specific mean',color='#EAA15E');ax.bar(xx+.2,b['gene_expression_panel_ridge'],.4,label='Drug-specific genomic Ridge',color='#138E81');ax.set_xticks(xx,list(b.index),rotation=20,ha='right');ax.set_ylabel('Mean held-out-cell RMSE');ax.set_title('Predeclared biological drug case studies, 5-fold cell holdouts');ax.legend();ax.grid(axis='y',alpha=.2);save('05_biological_drug_models')
 # TP53/MAPK/PIK3CA mutation + CN grouped permutation in drug-specific fitted model
 imp=pd.read_csv(folder/'biological_interpretability/heldout_gene_group_permutation.csv')
 sg=imp.groupby(['drug','gene'],as_index=False).agg(mean=('delta_rmse_mean','mean'),sd=('delta_rmse_mean','std'))
 sg['label']=sg['drug']+' · '+sg['gene'];sg=sg.sort_values('mean')
 fig0,ax=plt.subplots(figsize=(10,5));ax.barh(sg.label,sg['mean'],xerr=sg['sd'],color='#1D897C',capsize=3);ax.axvline(0,color='#666',ls='--');ax.set_xlabel('Mean change in held-out RMSE after permuting gene mutation+CN (40 repeats/fold)');ax.set_title('Drug-specific model reliance, not a causal biomarker test');ax.grid(axis='x',alpha=.2);save('06_drug_specific_gene_permutation')
 # lineage pooled error, show >=500 measurements across cell folds
 q=byl[(byl['mode']=='cell')&(byl['model']=='cell_plus_annotations')&(byl['n']>=300)].sort_values('rmse',ascending=False).head(15)
 fig0,ax=plt.subplots(figsize=(10,6));ax.barh(q.lineage,q.rmse,color='#5271B4');ax.invert_yaxis();ax.set_xlabel('Pooled out-of-fold RMSE');ax.set_title('Cell-line holdout: errors by recorded lineage (largest 15)');ax.grid(axis='x',alpha=.2);save('07_lineage_stratified_rmse')
 print('FULL_PLOTS_DONE',len(list(fig.glob('*.png'))),'FIGURES','within drug median r',q.shape,sk.r.median(), 'drugs model better than drug mean',int((sk.delta_rmse>0).sum()),'/',len(sk),flush=True)
 return sk,byd,byl
if __name__=='__main__':
 import sys
 run(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
