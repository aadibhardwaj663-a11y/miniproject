"""Submission-specific tests for reproducible LN_IC50 inference outputs."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_code'))
from predict_ln_ic50 import MODEL_NAME,infer,out_of_fold,load_data

class LN_IC50InferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=ROOT
        cls.grid=pd.read_csv(ROOT/'predictions'/'full_catalog_predicted_LN_IC50.csv')
        cls.obs=pd.read_csv(ROOT/'research'/'full_gdsc'/'matched_responses.csv')
        cls.saved=joblib.load(ROOT/'checkpoints'/MODEL_NAME)

    def test_checkpoint_is_complete_and_finite(self):
        m=self.saved
        self.assertEqual(len(m['cell_ids']),649)
        self.assertEqual(len(m['drug_names']),286)
        self.assertEqual(len(m['selected_expression_idx']),100)
        self.assertEqual(m['training_observations'],160512)
        self.assertEqual(m['feature_matrix']['total'],689)
        self.assertIn('NO held-out performance claim',m['model_role'])

    def test_inference_matches_saved_grid(self):
        test=self.grid.sample(18,random_state=27)[['DepMap_ID','drug_group']].reset_index(drop=True)
        calculated=infer(ROOT,test)
        stored=test.merge(self.grid,on=['DepMap_ID','drug_group'],validate='one_to_one',sort=False)
        np.testing.assert_allclose(calculated.predicted_LN_IC50.to_numpy(),stored.predicted_LN_IC50.to_numpy(),atol=3e-6,rtol=2e-6)
        self.assertTrue(np.isfinite(calculated.predicted_LN_IC50).all())

    def test_grid_never_confuses_predictions_with_observations(self):
        x=self.grid
        self.assertEqual(len(x),185614)
        self.assertEqual(len(x[['DepMap_ID','drug_group']].drop_duplicates()),len(x))
        self.assertTrue(np.isfinite(x.predicted_LN_IC50.to_numpy()).all())
        self.assertEqual(int(x.has_observed_measurement.sum()),160512)
        self.assertEqual(int((~x.has_observed_measurement).sum()),25102)
        self.assertTrue(x.loc[~x.has_observed_measurement,'observed_LN_IC50'].isna().all())
        self.assertTrue(x.prediction_status.eq('all_data_fit_NOT_cross_validated').all())
        observed=x[x.has_observed_measurement].merge(
            self.obs[['DepMap_ID','drug_group','LN_IC50']],on=['DepMap_ID','drug_group'],validate='one_to_one')
        self.assertEqual(len(observed),160512)
        np.testing.assert_allclose(observed.observed_LN_IC50,observed.LN_IC50,atol=1e-10,rtol=0)

    def test_out_of_fold_is_truly_from_existing_test_splits(self):
        case=self.obs.sample(7,random_state=27)[['DepMap_ID','drug_group']].reset_index(drop=True)
        for setting in ('cell','drug'):
            result=out_of_fold(ROOT,case,setting)
            self.assertEqual(len(result),7)
            self.assertTrue(result.prediction_status.eq('out_of_fold_'+setting).all())
            self.assertTrue(np.isfinite(result.predicted_LN_IC50.to_numpy()).all())

    def test_invalid_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            infer(ROOT,pd.DataFrame([{'DepMap_ID':'NONEXISTENT','drug_group':'trametinib'}]))
        with self.assertRaises(ValueError):
            infer(ROOT,pd.DataFrame([{'DepMap_ID':'ACH-000002','drug_group':'NONEXISTENT'}]))

    def test_original_fold_prediction_metrics_still_reconcile(self):
        rows=pd.read_csv(ROOT/'research'/'full_gdsc'/'fold_predictions.csv')
        metrics=pd.read_csv(ROOT/'research'/'full_gdsc'/'fold_metrics.csv')
        for setting in ('cell','drug'):
            for fold in range(1,6):
                x=rows[(rows['mode']==setting)&(rows['fold']==fold)]
                y=x.LN_IC50.to_numpy(dtype=float)
                pred=x.pred_cell_plus_annotations.to_numpy(dtype=float)
                actual=float(np.sqrt(np.mean((y-pred)**2)))
                saved=metrics[(metrics['mode']==setting)&(metrics['fold']==fold)&(metrics['model']=='cell_plus_annotations')]
                self.assertEqual(len(saved),1)
                self.assertAlmostEqual(actual,float(saved.iloc[0].rmse),places=5)

if __name__=='__main__': unittest.main(verbosity=2)
