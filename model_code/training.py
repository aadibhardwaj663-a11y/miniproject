"""Retrain and save the original 200-cell/20-drug baseline and RF checkpoints.
The full 649-cell five-fold benchmark is reproduced separately by full_gdsc_benchmark.py.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import run

def train_all(root):
 root=Path(root)
 metrics,_=run(root)
 expected=[root/'checkpoints'/f'{setting}_{model}.joblib' for setting in ['unseen_cell','unseen_drug'] for model in ['training_mean','random_forest']]
 assert all(path.exists() for path in expected), 'Missing fitted RF/mean checkpoint'
 print('Original twenty-drug checkpoints saved:',len(expected))
 return metrics
if __name__=='__main__':train_all(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
