
"""Report figures, generated exclusively from saved source and prediction tables."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def store(fig,root,name):
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(Path(root)/'figures'/f'{name}.{ext}',
                    dpi=250 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)

def axes(title,xlabel=None,ylabel=None,figsize=(9,5)):
    fig,ax=plt.subplots(figsize=figsize)
    ax.set_title(title,fontsize=14,pad=14)
    if xlabel:ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.2)
    return fig,ax

def plot_all(root):
 root=Path(root)
 obs=pd.read_csv(root/'data/curated_responses.csv')
 metrics=pd.read_csv(root/'tables/model_metrics.csv')
 preds={s:pd.read_csv(root/'predictions'/f'{s}_predictions.csv') for s in ['unseen_cell','unseen_drug']}
 imp=pd.read_csv(root/'tables/mg132_drug_specific_permutation.csv')
 learn=pd.read_csv(root/'tables/internal_training_learning_curve.csv')
 mut=pd.read_csv(root/'data/mutation_panel_binary.csv',index_col=0)
 perdrug=[]
 for s,p in preds.items():
  for drug,g in p.groupby('drug_group'):
   for model,col in [('training_mean','training_mean_predicted'),('random_forest','random_forest_predicted')]:
    perdrug.append({'split':s,'drug_group':drug,'model':model,'n':len(g),
                   'rmse':float(np.sqrt(np.mean((g.LN_IC50-g[col])**2))),
                   'mae':float(np.mean(abs(g.LN_IC50-g[col])))})
 drug_error=pd.DataFrame(perdrug)
 drug_error.to_csv(root/'tables/per_drug_holdout_errors.csv',index=False)

 fig,ax=axes('Figure 1. Observed drug-response distributions',
             'LN_IC50 (natural log of fitted IC50)','Number of cell–drug observations')
 ax.hist(obs.LN_IC50.to_numpy(),bins=38)
 ax.text(.97,.96,f'n = {len(obs):,} actual measurements',transform=ax.transAxes,ha='right',va='top')
 store(fig,root,'01_response_histogram')

 meds=obs.groupby('drug_group').LN_IC50.median().sort_values()
 labels=meds.index.tolist()
 arrays=[obs.loc[obs.drug_group.eq(x),'LN_IC50'].to_numpy() for x in labels]
 fig,ax=axes('Figure 2. Response variation by drug','LN_IC50',figsize=(11,9))
 ax.boxplot(arrays,orientation='horizontal',tick_labels=labels,showfliers=False)
 ax.set_yticks(range(1,len(labels)+1),labels=labels)
 ax.grid(axis='x',alpha=.2)
 ax.grid(axis='y',visible=False)
 store(fig,root,'02_drug_response_boxplot')

 for metric in ['rmse','mae']:
  groups=['unseen_cell','unseen_drug']
  x=np.arange(len(groups))
  w=.31
  fig,ax=axes(f'Figure 3. Holdout {metric.upper()}: mean versus Random Forest',
             'Holdout type',f'{metric.upper()} (LN_IC50 units)')
  for k,model in enumerate(['training_mean','random_forest']):
   vals=[float(metrics.query('split==@s and model==@model')[metric].iloc[0]) for s in groups]
   b=ax.bar(x+(k-.5)*w,vals,w,label=('Training mean' if k==0 else 'Random Forest'))
   ax.bar_label(b,fmt='%.2f',padding=3)
  ax.set_xticks(x,['Unseen cell lines','Unseen drugs'])
  ax.set_ylim(0,float(metrics[metric].max())*1.2)
  ax.legend()
  store(fig,root,f'03_holdout_{metric}')

 for s,number in [('unseen_cell',4),('unseen_drug',5)]:
  p=preds[s]
  fig,ax=axes(f'Figure {number}. Observed versus predicted: {s.replace("_"," ")}',
              'Observed LN_IC50','Random Forest predicted LN_IC50',figsize=(7,7))
  ax.scatter(p.LN_IC50,p.random_forest_predicted,s=15,alpha=.45)
  low=min(p.LN_IC50.min(),p.random_forest_predicted.min())
  high=max(p.LN_IC50.max(),p.random_forest_predicted.max())
  ax.plot([low,high],[low,high],linestyle='--',label='Perfect prediction')
  ax.set_xlim(low-.4,high+.4); ax.set_ylim(low-.4,high+.4)
  ax.set_aspect('equal',adjustable='box')
  ax.legend(loc='upper left')
  store(fig,root,f'0{number}_{s}_scatter')

 for s in preds:
  p=preds[s]
  preds[s]['residual']=p.random_forest_predicted-p.LN_IC50
 fig,ax=axes('Figure 6. Random Forest test residuals',
             'Predicted − observed LN_IC50','Number of held-out observations')
 for s,p in preds.items():
  ax.hist(p.residual,bins=np.linspace(-10,10,45),alpha=.48,label=s.replace('_',' '))
 ax.legend()
 store(fig,root,'06_holdout_residuals')

 held=drug_error.query('split=="unseen_drug"').pivot(index='drug_group',columns='model',values='mae')
 x=np.arange(len(held)); w=.34
 fig,ax=axes('Figure 7. MAE for each previously unseen drug',
              'Held-out compound','MAE (LN_IC50 units)',figsize=(11,6))
 for i,mdl in enumerate(['training_mean','random_forest']):
  b=ax.bar(x+(i-.5)*w,held[mdl].values,w,label=('Training mean' if i==0 else 'Random Forest'))
 ax.set_xticks(x,list(held.index),rotation=24,ha='right')
 ax.legend()
 store(fig,root,'07_unseen_drug_per_drug_mae')

 imp2=imp[imp.feature_group.ne('All 100 selected expression genes')].sort_values('rmse_increase_mean').tail(15)
 fig,ax=axes('Figure 8. MG-132 drug-specific permutation importance',
              'Mean increase in test RMSE after permutation (8 repeats)',figsize=(10,7))
 ax.barh(imp2.feature_group,imp2.rmse_increase_mean,xerr=imp2.rmse_increase_std,
         capsize=3)
 ax.axvline(0,linestyle='--',linewidth=1)
 ax.grid(axis='x',alpha=.2);ax.grid(axis='y',visible=False)
 store(fig,root,'08_mg132_permutation_importance')
 fig,ax=axes('Figure 9. Internal validation learning curve',
              'Training cell lines','RMSE on separate 32-cell internal validation set')
 for mdl,label in [('training_mean','Training mean'),('random_forest','Random Forest')]:
  p=learn[learn.model.eq(mdl)].sort_values('train_unique_cells')
  ax.plot(p.train_unique_cells,p.rmse,marker='o',label=label)
 ax.set_xticks([20,40,80,128])
 ax.legend()
 store(fig,root,'09_learning_curve')

 response_matrix=obs.pivot(index='DepMap_ID',columns='drug_group',values='LN_IC50')
 response_matrix=response_matrix.sort_index().reindex(sorted(response_matrix.columns),axis=1)
 response_matrix.to_csv(root/'tables/response_heatmap_matrix.csv')
 fig,ax=axes('Figure 10. Measured LN_IC50 matrix for 200 cells × 20 drugs',
             figsize=(13,8))
 im=ax.imshow(response_matrix.to_numpy(),aspect='auto',interpolation='nearest')
 ax.set_xticks(range(response_matrix.shape[1]),response_matrix.columns,rotation=70,ha='right',fontsize=8)
 tick=np.arange(0,len(response_matrix),20)
 ax.set_yticks(tick,[response_matrix.index[i] for i in tick],fontsize=8)
 ax.set_ylabel('Cell lines (sorted by DepMap ID)')
 fig.colorbar(im,ax=ax,label='LN_IC50')
 store(fig,root,'10_response_heatmap')

 count=mut.sum().sort_values()
 fig,ax=axes('Figure 11. Selected coding-mutation record frequency',
             'Number of the 200 cell lines with a panel mutation',figsize=(8,5))
 b=ax.barh(count.index,count.values)
 ax.bar_label(b,padding=3)
 ax.set_xlim(0,max(count)*1.15)
 ax.grid(axis='x',alpha=.2);ax.grid(axis='y',visible=False)
 store(fig,root,'11_mutation_panel_frequency')
 (root/'tables'/'mutation_panel_frequency.csv').write_text(count.rename('n_cell_lines').to_csv())
 print('Created',len(list((root/'figures').glob('*.png'))),'PNG and SVG pairs')

if __name__=='__main__':
 import sys
 plot_all(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
