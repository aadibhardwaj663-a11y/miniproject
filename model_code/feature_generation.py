"""Leakage-safe feature construction utilities for fitted checkpoint bundles.

For training: feature selection uses only unique training cell line IDs and
the annotation vectorizer fits only training drug annotations.
For inference: reuse the saved selection and encoder in the checkpoint.
"""
from pathlib import Path
import sys
import numpy as np,pandas as pd
from sklearn.feature_extraction import DictVectorizer

sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import load,prepare_split,annotation_dict,design

def fit_features(root,split):
    d=load(Path(root))
    train,test=prepare_split(d,split)
    idx=np.asarray(sorted({d['cell_idx'][c] for c in train.DepMap_ID}))
    variance=d['expr'][idx].var(axis=0,dtype=np.float64)
    selected=np.lexsort((np.arange(len(variance)),-variance))[:100]
    dv=DictVectorizer(sparse=False,sort=True)
    dv.fit([annotation_dict(r) for _,r in train.iterrows()])
    return d,train,test,selected,dv,design(train,d,selected,dv),design(test,d,selected,dv)

def inference_features(root,split,model_name,rows):
    import joblib
    root=Path(root)
    d=load(root)
    saved=joblib.load(root/'checkpoints'/f'{split}_{model_name}.joblib')
    return design(rows,d,saved['expression_selected_indices'],saved['drug_annotation_vectorizer'])

if __name__=='__main__':
    d,tr,te,sel,dv,a,b=fit_features(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1],
                                  sys.argv[2] if len(sys.argv)>2 else 'unseen_cell')
    print('Training feature matrix',a.shape,'test matrix',b.shape,'expression features',len(sel))
