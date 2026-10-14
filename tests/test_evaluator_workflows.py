"""Regression checks for optimized encoding, stricter OOF access and grid chunking."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_code'))
from full_gdsc_benchmark import tokens
from predict_ln_ic50 import out_of_fold,predict_known_grid


class EvaluatorWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs=pd.read_csv(ROOT/'research/full_gdsc/matched_responses.csv')
        cls.cold=pd.read_csv(ROOT/'research/full_gdsc/chemistry_grouped/test_predictions.csv')

    def test_unique_drug_encoding_equivalent_to_rowwise(self):
        """Optimization must not change annotation columns or per-pair values."""
        train=self.obs[self.obs.drug_group.isin(self.obs.drug_group.unique()[:120])].copy()
        trial=self.obs.sample(500,random_state=807)
        v=DictVectorizer(sparse=False,sort=True)
        v.fit([tokens(x) for x in train.itertuples(index=False)])
        expected=v.transform([tokens(x) for x in trial.itertuples(index=False)]).astype(np.float32)
        drugmeta=self.obs.drop_duplicates('drug_group').set_index('drug_group').sort_index()
        names=drugmeta.index
        encoded=v.transform([tokens(x) for x in drugmeta.reset_index().itertuples(index=False)]).astype(np.float32)
        indices=names.get_indexer(trial.drug_group)
        self.assertTrue((indices>=0).all())
        np.testing.assert_array_equal(encoded[indices],expected)

    def test_chemical_drug_oof_matches_recorded_test_predictions(self):
        sample=self.cold[self.cold.setting.eq('drug')].sample(9,random_state=19)
        rows=sample[['DepMap_ID','drug_group']].reset_index(drop=True)
        result=out_of_fold(ROOT,rows,'identity-drug')
        np.testing.assert_allclose(result.predicted_LN_IC50.to_numpy(),sample.pred_combined.to_numpy(),atol=0,rtol=0)
        self.assertTrue(result.prediction_status.str.startswith('retrospective_identity').all())

    def test_chemical_both_has_only_honest_intersection_pairs(self):
        sample=self.cold[self.cold.setting.eq('both')].sample(8,random_state=9)
        r=out_of_fold(ROOT,sample[['DepMap_ID','drug_group']],'identity-both')
        np.testing.assert_array_equal(r.predicted_LN_IC50.to_numpy(),sample.pred_combined.to_numpy())
        joined=pd.merge(self.obs[['DepMap_ID','drug_group']],self.cold[self.cold.setting.eq('both')][['DepMap_ID','drug_group']],on=['DepMap_ID','drug_group'],how='left',indicator=True)
        missing=joined.loc[joined._merge.eq('left_only'),['DepMap_ID','drug_group']].head(1)
        with self.assertRaisesRegex(ValueError,'lack a measured identity-both test prediction'):
            out_of_fold(ROOT,missing,'identity-both')

    def test_reject_invalid_grid_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'chunk_size must be positive'):
                predict_known_grid(ROOT,Path(d)/'out.csv',chunk_size=0)

    def test_chunked_grid_matches_shipped_predictions(self):
        """13-drug chunks produce the same finite 185,614 paired estimates."""
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'grid.csv'
            details=predict_known_grid(ROOT,out,chunk_size=13)
            self.assertEqual(details['predicted_pairs'],185614)
            self.assertEqual(details['unmeasured_pairs'],25102)
            actual=pd.read_csv(out,usecols=['DepMap_ID','drug_group','predicted_LN_IC50','has_observed_measurement'])
            reference=pd.read_csv(ROOT/'predictions/full_catalog_predicted_LN_IC50.csv',usecols=actual.columns)
            self.assertTrue(actual[['DepMap_ID','drug_group']].equals(reference[['DepMap_ID','drug_group']]))
            np.testing.assert_allclose(actual.predicted_LN_IC50,reference.predicted_LN_IC50,atol=3e-6,rtol=2e-6)
            self.assertTrue(actual.has_observed_measurement.equals(reference.has_observed_measurement))


if __name__=='__main__':
    unittest.main()
