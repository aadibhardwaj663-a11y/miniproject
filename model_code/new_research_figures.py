from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]; fig=R/'research/figures';fig.mkdir(exist_ok=True)
metrics=pd.read_csv(R/'research/metrics/repeated_seed_metrics.csv')
summary=pd.read_csv(R/'research/metrics/repeated_seed_summary.csv')
biol=pd.read_csv(R/'research/metrics/biological_candidate_associations.csv')
obs=pd.read_csv(R/'data/curated_responses.csv');mut=pd.read_csv(R/'data/mutation_panel_binary.csv',index_col=0)
pred=pd.read_csv(R/'research/data/repeated_seed_predictions.csv')
assert set(metrics['mode'])=={'both','cell','drug'} and set(pred.seed)=={107,207,307,407,507}

def save(name):
 plt.tight_layout();plt.savefig(fig/(name+'.png'),dpi=200,bbox_inches='tight');plt.savefig(fig/(name+'.svg'),bbox_inches='tight');plt.close()
# paired 3-seed joint heldout RMSE
plot=metrics[(metrics['mode']=='both')&metrics.variant.isin(['global_mean','ridge_all','ridge_cell_only'])]
f,ax=plt.subplots(figsize=(8.2,4.8))
for key in ['global_mean','ridge_all','ridge_cell_only']:
 z=plot[plot.variant.eq(key)].sort_values('seed')
 ax.plot(z.seed.astype(str),z.rmse,marker='o',label=key.replace('_',' '))
ax.set_ylabel('RMSE (LN_IC50)');ax.set_xlabel('Independent split seed');ax.set_title('Simultaneously unseen cells and drugs: repeated splits');ax.legend();ax.grid(alpha=.2)
save('joint_cold_repeated_rmse')
# ablation ridge family only
ab=summary[(summary['mode']=='both') & summary.variant.str.startswith('ridge_')].sort_values('rmse_mean')
f,ax=plt.subplots(figsize=(8.5,4.7))
ax.barh(ab.variant.str.replace('ridge_','',regex=False),ab.rmse_mean,xerr=ab.rmse_sd,capsize=3)
ax.invert_yaxis();ax.set_xlabel('Mean joint-holdout RMSE, 5 seeds (error bars: SD)');ax.set_title('Multi-omics ablation: fixed Ridge family')
save('joint_cold_ablation')
# scatter per seed
f,ax=plt.subplots(figsize=(6.6,5.7));ax.scatter(pred.LN_IC50,pred.prediction_ridge_all,s=15,alpha=.4)
lo=min(pred.LN_IC50.min(),pred.prediction_ridge_all.min());hi=max(pred.LN_IC50.max(),pred.prediction_ridge_all.max());ax.plot([lo,hi],[lo,hi],linestyle='--')
ax.set(xlabel='Observed LN_IC50',ylabel='Ridge prediction',title='Jointly unseen pairs, 5 split seeds (n=1,000)')
save('joint_cold_scatter')
# biology evidence
cases=[('nutlin-3a (-)','TP53'),('selumetinib','KRAS'),('taselisib','PIK3CA')]
f,axes=plt.subplots(1,3,figsize=(12,4.1))
for ax,(drug,gene) in zip(axes,cases):
 sub=obs[obs.drug_group.eq(drug)].copy();sub['mutation']=sub.DepMap_ID.map(mut[gene]);
 ax.boxplot([sub.loc[sub.mutation.eq(0),'LN_IC50'].values,sub.loc[sub.mutation.eq(1),'LN_IC50'].values],tick_labels=['No variant\nrecord','Variant\nrecord'],showfliers=False)
 ax.set_title(f'{drug}\n{gene}');ax.set_ylabel('Observed LN_IC50')
f.suptitle('Hypothesis-led associations (exploratory, not causal)',fontsize=12)
save('genomic_literature_associations')
# research applicability
x=pd.read_csv(R/'research/metrics/joint_applicability_distance.csv')
g=x.groupby(['seed','cell']).agg(distance=('nearest_cell_expression_distance','first'),mae=('abs_error','mean')).reset_index()
f,ax=plt.subplots(figsize=(7,4.9));ax.scatter(g.distance,g.mae,alpha=.45,s=18)
ax.set(xlabel='Nearest training-cell distance (standardized selected expression)',ylabel='Mean absolute error across new drugs',title='Joint-cold applicability-domain diagnostic, 200 cell/seed clusters')
save('joint_cold_applicability')
print('Saved',len(list(fig.glob('*.png'))),'research figures')
