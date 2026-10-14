"""CLI research prototype for 200 curated CCLE profiles and 20 annotated GDSC drugs.

No clinical-use interface. Prints whether the chosen cell is an outer-test entity.
No target response is passed to the prediction pipeline.
"""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from challenge27_pipeline import load,design

def predict(root,cell_id,drug_group,model='unseen_cell_random_forest'):
    root=Path(root);d=load(root)
    pair=d['obs'].loc[d['obs'].DepMap_ID.eq(cell_id)&d['obs'].drug_group.eq(drug_group)]
    if len(pair)!=1:raise ValueError('Unknown curated (cell, drug) pair. Use --list to see supported identities.')
    row=pair.iloc[:1].drop(columns=['LN_IC50'],errors='ignore').copy()
    bundle=joblib.load(root/'checkpoints'/f'{model}.joblib')
    assert 'pipeline' in bundle and 'drug_annotation_vectorizer' in bundle
    XX=design(row,d,bundle['expression_selected_indices'],bundle['drug_annotation_vectorizer'])
    pred=float(bundle['pipeline'].predict(XX)[0]); assert np.isfinite(pred)
    outer_test=set(pd.read_csv(root/'splits/unseen_cell_test.csv',dtype=str).id)
    rf=bundle['pipeline'].named_steps['regressor']
    importance=rf.feature_importances_
    features=bundle['feature_names']
    tops=[{'feature':str(features[i]),'impurity_importance':round(float(importance[i]),6)}
          for i in np.argsort(-importance)[:8]]
    return {'cell_id':cell_id,'drug_group':drug_group,'model':model,'predicted_LN_IC50':pred,
            'evaluation_status':'unseen_cell_outer_test' if cell_id in outer_test else 'cell_was_seen_during_outer_training',
            'global_model_feature_importances_not_local_attribution':tops,
            'warning':'Cell-line assay model. No chemical novelty, clinical efficacy, causal biomarker, or patient prediction.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--cell-id');parser.add_argument('--drug-group');parser.add_argument('--list',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    if args.list:
        o=pd.read_csv(root/'data/curated_responses.csv')
        print('Cells:',', '.join(sorted(o.DepMap_ID.unique())[:25]),'... (first 25)')
        print('Drugs:',', '.join(sorted(o.drug_group.unique())))
        return
    if not args.cell_id or not args.drug_group:parser.error('--cell-id and --drug-group required')
    print(json.dumps(predict(root,args.cell_id,args.drug_group),indent=2))
if __name__=='__main__':main()
