"""Supplemental charts from actual measured CSVs only."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run(root):
 root=Path(root)
 out=root/'report_assets';out.mkdir(exist_ok=True)
 base=pd.read_csv(root/'metrics'/'supplemental_stronger_baselines.csv')
 m=pd.read_csv(root/'metrics'/'outer_test_metrics.csv')
 v=[]
 for split in ['unseen_cell','unseen_drug']:
  b=m[(m.split==split)&(m.model=='training_mean')].iloc[0].rmse
  f=m[(m.split==split)&(m.model=='random_forest')].iloc[0].rmse
  enhanced=base[(base.split==split)&(base.phase=='outer_test_revisit')].iloc[0].rmse
  v.append([b,enhanced,f])
 v=np.array(v)
 fig,ax=plt.subplots(figsize=(9.5,4.8))
 x=np.arange(2);w=.24
 for j,label in enumerate(['Global training mean','Entity-aware mean','Random Forest']):
  bars=ax.bar(x+(j-1)*w,v[:,j],w,label=label)
  ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
 ax.set_xticks(x,['New cell lines','New compounds'])
 ax.set_ylabel('Held-out RMSE (LN_IC50) | lower is better')
 ax.set_ylim(0,4.8);ax.set_title('Honest comparison against stronger simple baselines')
 ax.legend(loc='upper left');ax.grid(axis='y',alpha=.15)
 fig.tight_layout();fig.savefig(out/'stronger_baselines.png',dpi=190);plt.close(fig)
 ci=pd.read_csv(root/'metrics'/'conditional_cluster_bootstrap.csv')
 fig,ax=plt.subplots(figsize=(9.2,3.6))
 for y,row in enumerate(ci.itertuples()):
  m=row.rf_minus_baseline_rmse
  ax.errorbar(m,y,xerr=[[m-row.conditional_CI_lower],[row.conditional_CI_upper-m]],fmt='o',capsize=5)
 ax.axvline(0,linewidth=1,linestyle='--')
 ax.set_yticks([0,1],['New cell lines: RF minus drug-specific mean','New drugs: RF minus pathway mean'])
 ax.invert_yaxis();ax.set_xlabel('RMSE difference and conditional 95% bootstrap interval')
 ax.set_title('Held-out-entity cluster bootstrap | fixed test split')
 ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(out/'conditional_bootstrap_intervals.png',dpi=190,bbox_inches='tight');plt.close(fig)
 print('Supplemental figures created')
if __name__=='__main__':
 import sys
 run(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
