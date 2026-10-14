"""Independent numeric and leakage audit of full matched source evaluation."""
from pathlib import Path
import unittest,json
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/full_gdsc'
class FullGDSCChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.obs=pd.read_csv(D/'matched_responses.csv')
  cls.p=pd.read_csv(D/'fold_predictions.csv')
  cls.m=pd.read_csv(D/'fold_metrics.csv')
 def test_real_data(self):
  a=self.obs
  self.assertEqual(a.DepMap_ID.nunique(),649);self.assertEqual(a.drug_group.nunique(),286)
  self.assertEqual(len(a),160512);self.assertTrue(a.NLME_CURVE_ID.is_unique)
  self.assertFalse(a.duplicated(['DepMap_ID','drug_group']).any())
  self.assertTrue(np.isfinite(a.LN_IC50).all())
 def test_no_entity_overlap(self):
  for mode,col in [('drug','drug_group'),('cell','DepMap_ID')]:
   seq=[]
   for fold in range(1,6):
    train=set(pd.read_csv(D/'split_ids'/f'{mode}_fold{fold}_train.csv')[col].astype(str))
    test=set(pd.read_csv(D/'split_ids'/f'{mode}_fold{fold}_test.csv')[col].astype(str))
    self.assertFalse(train&test)
    p=self.p[(self.p['mode']==mode)&self.p.fold.eq(fold)]
    self.assertEqual(set(p[col]),test)
    seq.extend(list(test))
   self.assertEqual(len(seq),len(set(seq)))
   self.assertEqual(set(seq),set(self.obs[col]))
 def test_metrics_from_original_observations(self):
  orig=self.obs.set_index('NLME_CURVE_ID')
  for (mode,fold),q in self.p.groupby(['mode','fold']):
   self.assertTrue(q.NLME_CURVE_ID.is_unique)
   a=orig.loc[q.NLME_CURVE_ID]
   np.testing.assert_allclose(q.LN_IC50,a.LN_IC50,rtol=0,atol=0)
   for model in ['global_mean','cell_only','cell_plus_annotations']+(['drug_mean'] if mode=='cell' else []):
    pp=q['pred_'+model];y=q.LN_IC50
    saved=self.m[(self.m['mode']==mode)&(self.m.fold==fold)&(self.m.model==model)].iloc[0]
    self.assertAlmostEqual(float(np.sqrt(np.mean((pp-y)**2))),float(saved.rmse),places=6)
 def test_per_drug_within_corr(self):
  sk=pd.read_csv(D/'per_drug_skill_cell_holdout.csv')
  self.assertEqual(len(sk),286)
  self.assertTrue(np.isfinite(sk[['baseline','model','response_sd']]).all().all())
  self.assertEqual(int((sk.delta_rmse>0).sum()),229)
  self.assertAlmostEqual(float(sk.r.median()),.4492,delta=0.001)
 def test_mechanistic_holdouts(self):
  x=pd.read_csv(D/'biological_interpretability/five_fold_metrics.csv')
  imp=pd.read_csv(D/'biological_interpretability/heldout_gene_group_permutation.csv')
  self.assertEqual(x.drug.nunique(),5)
  self.assertEqual(x.fold.nunique(),5)
  self.assertTrue(np.isfinite(imp.delta_rmse_mean).all())
  self.assertTrue((imp.n_permutations==40).all())
 def test_source_zip_audit(self):
  checks=json.loads((D/'audit.json').read_text())
  self.assertEqual(checks['source_response_rows'],242036)
  self.assertEqual(checks['matched_pairs'],160512)
  self.assertFalse(checks['external_chemical_alias_verification_full_catalog'])
if __name__=='__main__':unittest.main()
