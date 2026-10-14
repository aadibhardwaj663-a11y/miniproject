"""Evidence-focused checks for new strictly cold research experiments."""
from pathlib import Path
import unittest,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class ResearchExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=pd.read_csv(ROOT/'research/metrics/repeated_seed_metrics.csv')
        cls.p=pd.read_csv(ROOT/'research/data/repeated_seed_predictions.csv')
    def test_five_repeated_seeds_three_modes(self):
        self.assertEqual(set(self.m['mode']),{'cell','drug','both'})
        for mode in ['cell','drug','both']:
            self.assertEqual(set(self.m[self.m['mode']==mode].seed),{107,207,307,407,507})
        self.assertTrue(np.isfinite(self.m[['rmse','mae']]).all().all())
    def test_joint_entity_isolation(self):
        for seed in [107,207,307,407,507]:
            record=json.loads((ROOT/'research/validation'/f'both_seed{seed}_ids.json').read_text())
            self.assertFalse(set(record['train_cells']) & set(record['test_cells']))
            self.assertFalse(set(record['train_drugs']) & set(record['test_drugs']))
            self.assertEqual(len(record['test_curve_ids']),200)
            self.assertEqual(record['excluded_cross_block_pairs'],1400)
            self.assertFalse(set(record['final_training_curve_ids']) & set(record['test_curve_ids']))
    def test_source_prediction_values_and_metric_reproduction(self):
        source=pd.read_csv(ROOT/'data/curated_responses.csv')[['NLME_CURVE_ID','LN_IC50']]
        for mode in ['cell','drug','both']:
            for seed in [107,207,307,407,507]:
                p=self.p[(self.p['mode']==mode)&self.p.seed.eq(seed)]
                self.assertEqual(len(p),{'cell':800,'drug':1000,'both':200}[mode])
                self.assertFalse(p.NLME_CURVE_ID.duplicated().any())
                s=p.merge(source,on='NLME_CURVE_ID',validate='one_to_one')
                np.testing.assert_allclose(s.LN_IC50_x,s.LN_IC50_y,rtol=0,atol=0)
                rmse=np.sqrt(np.mean((p.LN_IC50-p.prediction_ridge_all)**2))
                row=self.m[(self.m['mode']==mode)&(self.m.seed==seed)&self.m.variant.eq('ridge_all')]
                self.assertEqual(len(row),1)
                self.assertAlmostEqual(rmse,float(row.iloc[0].rmse),places=8)
    def test_research_ablations_cover_all(self):
        required={'ridge_all','ridge_expression_drug','ridge_panel_drug','ridge_drug_only','ridge_cell_only','global_mean'}
        for mode in ['cell','drug','both']:
            self.assertTrue(required.issubset(set(self.m[self.m['mode']==mode].variant)))
    def test_biology_explicitly_exploratory(self):
        d=pd.read_csv(ROOT/'research/metrics/lineage_adjusted_mutation_associations.csv')
        self.assertEqual(len(d),8)
        self.assertTrue(d.permutation_p_two_sided.between(0,1).all())
        self.assertTrue((ROOT/'deliverables/scientific_report.docx').exists())
    def test_chemical_structures_cannot_be_fabricated(self):
        df=pd.read_csv(ROOT/'research/data/compound_structures_TEMPLATE.csv',keep_default_na=False)
        self.assertEqual(len(df),20)
        self.assertTrue(df.canonical_smiles.eq('').all())
        self.assertTrue((ROOT/'research/data/morgan_1024_fingerprints.npz').exists())
    def test_working_inference_from_saved_checkpoint(self):
        import sys
        sys.path.insert(0,str(ROOT/'model_code'))
        from predict_cli import predict
        result=predict(ROOT,'ACH-000002','mg-132')
        self.assertTrue(np.isfinite(result['predicted_LN_IC50']))
        self.assertIn('warning',result)
if __name__=='__main__':unittest.main(verbosity=2)
