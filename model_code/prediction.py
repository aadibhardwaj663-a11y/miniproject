"""Predict using saved checkpoints, fitted preprocessing and frozen test IDs."""
from pathlib import Path
import sys
import joblib
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import load,prepare_split,design

def predict(root,split):
    root=Path(root);d=load(root);_,test=prepare_split(d,split)
    output=test[['DepMap_ID','DRUG_ID','drug_group','DRUG_NAME','NLME_CURVE_ID','LN_IC50']].copy()
    for model_name in ('training_mean','random_forest'):
        bundle=joblib.load(root/'checkpoints'/f'{split}_{model_name}.joblib')
        X=design(test,d,bundle['expression_selected_indices'],bundle['drug_annotation_vectorizer'])
        predictions=bundle['pipeline'].predict(X)
        assert np.isfinite(predictions).all()
        output[f'{model_name}_predicted']=predictions
    output['abs_error_forest']=(output.LN_IC50-output.random_forest_predicted).abs()
    return output

if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
    for split in ('unseen_cell','unseen_drug'):
        df=predict(root,split)
        path=root/'predictions'/f'{split}_predictions.csv'
        df.to_csv(path,index=False)
        print(split,len(df),'rows',path)
