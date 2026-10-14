
"""Independently verify data, identity leakage, splits, metrics, models and figures."""
from pathlib import Path
import unittest,sys,re,json
import numpy as np,pandas as pd,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_code'))
from challenge27_pipeline import load,prepare_split,design,evaluate,PANEL

ROOT=Path(__file__).resolve().parents[1]
class Challenge27Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=load(ROOT)
        cls.metrics=pd.read_csv(ROOT/'tables/model_metrics.csv')
        cls.aliases=pd.read_csv(ROOT/'data/compound_alias_manifest.csv')
    def test_source_dataset_identity(self):
        log=json.loads((ROOT/'logs/input_validation.json').read_text())
        self.assertEqual(log['zip_integrity'],'PASS')
        self.assertEqual(log['selected_pairs'],4000)
        self.assertEqual(len(log['zip_member_sha256']),6)
    def test_shape_and_finite(self):
        d=self.d
        self.assertEqual(d['expr'].shape,(200,58676))
        self.assertEqual(d['mut'].shape,(200,10))
        self.assertEqual(d['cn'].shape,(200,10))
        self.assertTrue(np.isfinite(d['expr']).all())
        self.assertTrue(np.isfinite(d['mut']).all())
        self.assertTrue(np.isfinite(d['cn']).all())
        self.assertTrue(np.isfinite(d['obs'].LN_IC50).all())
    def test_unique_source_pairs(self):
        o=self.d['obs']
        self.assertEqual(len(o),4000)
        self.assertEqual(o.DepMap_ID.nunique(),200)
        self.assertEqual(o.drug_group.nunique(),20)
        self.assertFalse(o.duplicated(['DepMap_ID','drug_group']).any())
        self.assertFalse(o.NLME_CURVE_ID.duplicated().any())
    def test_fixed_panel_and_binary_mutations(self):
        m=self.d['mut']
        self.assertTrue(set(np.unique(m)).issubset({0,1}))
        self.assertEqual(len(PANEL),10)
    def test_entity_disjoint_holdouts(self):
        for split in ['unseen_cell','unseen_drug']:
            train,test=prepare_split(self.d,split)
            self.assertEqual(len(train)+len(test),4000)
            self.assertTrue(set(train.NLME_CURVE_ID).isdisjoint(set(test.NLME_CURVE_ID)))
            if split=='unseen_cell':
                self.assertTrue(set(train.DepMap_ID).isdisjoint(set(test.DepMap_ID)))
            else:
                self.assertTrue(set(train.drug_group).isdisjoint(set(test.drug_group)))
                self.assertTrue(set(train.DRUG_ID).isdisjoint(set(test.DRUG_ID)))
    def test_alias_guard(self):
        for split in ['unseen_drug']:
            tr,te=prepare_split(self.d,split)
            normalize=lambda v:re.sub('[^a-z0-9]','',str(v).lower())
            a={normalize(x) for x in self.aliases[self.aliases.drug_group.isin(tr.drug_group)].alias}
            b={normalize(x) for x in self.aliases[self.aliases.drug_group.isin(te.drug_group)].alias}
            self.assertFalse(a&b)
    def test_training_only_variance_selection(self):
        for split in ['unseen_cell','unseen_drug']:
            train,_=prepare_split(self.d,split)
            unique=np.array(sorted({self.d['cell_idx'][x] for x in train.DepMap_ID}))
            variances=self.d['expr'][unique].var(axis=0,dtype=np.float64)
            indices=np.lexsort((np.arange(len(variances)),-variances))[:100]
            bundle=joblib.load(ROOT/'checkpoints'/f'{split}_random_forest.joblib')
            np.testing.assert_array_equal(indices,bundle['expression_selected_indices'])
            self.assertEqual(len(bundle['expression_gene_names']),100)
    def test_prediction_regeneration_and_metrics(self):
        for split in ['unseen_cell','unseen_drug']:
            _,test=prepare_split(self.d,split)
            test=test.reset_index(drop=True)
            for model in ['training_mean','random_forest']:
                bundle=joblib.load(ROOT/'checkpoints'/f'{split}_{model}.joblib')
                x=design(test,self.d,bundle['expression_selected_indices'],bundle['drug_annotation_vectorizer'])
                pred=bundle['pipeline'].predict(x)
                self.assertTrue(np.isfinite(pred).all())
                saved=pd.read_csv(ROOT/'predictions'/f'{split}_predictions.csv')
                # Align rows explicitly to curated identifiers, not just row order.
                merged=test[['NLME_CURVE_ID']].merge(saved,on='NLME_CURVE_ID',validate='one_to_one',sort=False)
                np.testing.assert_allclose(pred,merged[model+'_predicted'],atol=1e-10,rtol=0)
                measured=evaluate(test.LN_IC50.to_numpy(),pred)
                metric=self.metrics.query('split==@split and model==@model').iloc[0]
                self.assertAlmostEqual(measured['rmse'],metric.rmse,places=10)
                self.assertAlmostEqual(measured['mae'],metric.mae,places=10)
                if model=='training_mean':
                    self.assertTrue(pd.isna(metric.pearson_r))
    def test_permutation_drug_specific(self):
        im=pd.read_csv(ROOT/'tables/mg132_drug_specific_permutation.csv')
        self.assertEqual(len(im),21)
        self.assertTrue(im.drug_group.eq('mg-132').all())
        self.assertTrue(im.repeats.eq(8).all())
        self.assertTrue(np.isfinite(im.rmse_increase_mean).all())
    def test_report_figures_generated(self):
        from PIL import Image
        figs=list((ROOT/'figures').glob('*.png'))
        self.assertEqual(len(figs),12)
        for p in figs:
            with Image.open(p) as img:
                self.assertTrue(img.width>=800 and img.height>=500)
                img.verify()
    def test_learning_curve_uses_only_training_cells(self):
        _, final_test=prepare_split(self.d,'unseen_cell')
        fulltrain,_=prepare_split(self.d,'unseen_cell')
        self.assertTrue(set(fulltrain.DepMap_ID).isdisjoint(set(final_test.DepMap_ID)))
        l=pd.read_csv(ROOT/'tables/internal_training_learning_curve.csv')
        self.assertEqual(sorted(l.train_unique_cells.unique().tolist()),[20,40,80,128])
        self.assertTrue(l.validation_unique_cells.eq(32).all())
if __name__=='__main__':
    unittest.main(verbosity=2)
