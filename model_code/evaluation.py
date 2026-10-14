"""Recompute per-task error metrics from actual saved predictions (no fitting)."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import evaluate

def evaluate_saved(root):
    root=Path(root); rows=[]
    for split in ('unseen_cell','unseen_drug'):
        df=pd.read_csv(root/'predictions'/f'{split}_predictions.csv')
        y=df.LN_IC50.to_numpy(dtype=float)
        for model in ('training_mean','random_forest'):
            pred=df[f'{model}_predicted'].to_numpy(dtype=float)
            rows.append({'split':split,'model':model,**evaluate(y,pred)})
    frame=pd.DataFrame(rows)
    saved=pd.read_csv(root/'tables/model_metrics.csv')
    for _,row in frame.iterrows():
        matched=saved[(saved.split==row['split'])&(saved.model==row['model'])].iloc[0]
        assert row['n']==matched['n']
        for col in ['rmse','mae']:
            assert np.isclose(row[col],matched[col],atol=1e-10)
        assert (np.isnan(row['pearson_r']) and np.isnan(matched['pearson_r'])) or np.isclose(row['pearson_r'],matched['pearson_r'],atol=1e-10)
    print(frame.to_string(index=False))
    return frame

if __name__=='__main__':
    evaluate_saved(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
