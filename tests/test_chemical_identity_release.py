"""Identity data integrity and independently recomputed 5-fold metric checks."""
import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_code'))
from identity_resolution import build,entity_fold_lists

class ChemicalIdentityReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=build(ROOT)
        cls.out=ROOT/'research/full_gdsc/chemistry_grouped'
    def test_all_286_exclusively_assigned(self):
        self.assertEqual(len(self.d),286)
        self.assertEqual(self.d.canonical_drug_id.nunique(),286)
        self.assertEqual(self.d.source_linked.sum(),236)
        self.assertEqual(self.d.pubchem_cid.isna().sum(),50)
        self.assertEqual(self.d.identity_key.nunique(),282)
    def test_known_double_names_collapsed(self):
        m=self.d.set_index('drug_group').identity_key
        self.assertEqual(m['bleomycin'],m['bleomycin (50 um)'])
        self.assertEqual(m['bdoca000347a'],m['of-1'])
        self.assertEqual(m['gsk-lsd1'],m['gsk-lsd1-2hcl'])
        self.assertEqual(m['gsk2830371'],m['gsk2830371a'])
        self.assertNotEqual(m['trametinib'],m['selumetinib'])
        self.assertNotEqual(m['trametinib'],m['pd0325901'])
    def test_conflicting_forms_not_silently_accepted(self):
        x=self.d.set_index('drug_group')
        self.assertTrue(bool(x.loc['mk-2206','prior_structure_conflict']))
        self.assertTrue(bool(x.loc['oxaliplatin','prior_structure_conflict']))
    def test_every_drug_group_in_one_key_safe_fold(self):
        folds=entity_fold_lists(ROOT)
        groups=[v for s in folds for v in s]
        self.assertEqual(len(groups),286)
        self.assertEqual(len(set(groups)),286)
        for s in folds:
            keys=set(self.d.loc[self.d.drug_group.isin(s),'identity_key'])
            self.assertFalse(set(self.d.loc[~self.d.drug_group.isin(s),'identity_key'])&keys)
    def test_release_split_manifests(self):
        for setting in ('drug','both'):
            for fold in range(1,6):
                loc=self.out/'splits'/setting
                tr=pd.read_csv(loc/f'fold{fold}_outer_train.csv')
                te=pd.read_csv(loc/f'fold{fold}_outer_test.csv')
                inner=pd.read_csv(loc/f'fold{fold}_inner_train.csv')
                val=pd.read_csv(loc/f'fold{fold}_inner_validation.csv')
                self.assertFalse(set(tr.identity_key)&set(te.identity_key))
                self.assertFalse(set(tr.NLME_CURVE_ID)&set(te.NLME_CURVE_ID))
                if setting=='both':
                    self.assertFalse(set(tr.DepMap_ID)&set(te.DepMap_ID))
                self.assertFalse(set(inner.identity_key)&set(val.identity_key))
                self.assertEqual(set(tr.NLME_CURVE_ID),set(inner.NLME_CURVE_ID)|set(val.NLME_CURVE_ID))
                self.assertFalse(set(inner.NLME_CURVE_ID)&set(val.NLME_CURVE_ID))
                self.assertEqual(len(te),len(pd.read_csv(self.out/'test_predictions.csv',usecols=['setting','fold']).query('setting==@setting and fold==@fold')))

    def test_saved_predictions_recalculate_and_holdouts_disjoint(self):
        p=pd.read_csv(self.out/'test_predictions.csv')
        m=pd.read_csv(self.out/'fold_metrics.csv')
        self.assertEqual(len(m),40)
        for setting in ('drug','both'):
            q=p[p.setting.eq(setting)]
            self.assertFalse(q.duplicated(['NLME_CURVE_ID']).any())
            self.assertTrue((q.groupby('identity_key').fold.nunique()==1).all())
            if setting=='drug':self.assertEqual(len(q),160512)
            for fold in range(1,6):
                test=q[q.fold.eq(fold)]
                for model in ('global_mean','genomics_only','annotations_only','combined'):
                    expected=m[(m.setting==setting)&(m.fold==fold)&(m.model==model)].iloc[0]
                    score=np.sqrt(np.mean(np.square(test.LN_IC50.to_numpy()-test['pred_'+model].to_numpy())))
                    self.assertAlmostEqual(score,expected.rmse,delta=1e-7)
                    self.assertTrue(np.isfinite(test['pred_'+model]).all())

if __name__=='__main__':unittest.main()
