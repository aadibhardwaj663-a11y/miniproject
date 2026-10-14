"""Verify newly added cold-both and same-fold annotation-only ablations."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_code'))
from full_gdsc_benchmark import SEED
from compound_identity_audit import run as identity_audit

class SupplementalScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs=pd.read_csv(ROOT/'research/full_gdsc/matched_responses.csv')
        cls.base=ROOT/'research/full_gdsc/strict_generalization'
        cls.pr=pd.read_csv(cls.base/'test_predictions.csv')
        cls.met=pd.read_csv(cls.base/'fold_metrics.csv')

    def test_both_heldout_entities_never_trained(self):
        x=self.obs
        self.assertEqual(len(self.pr),31998)
        self.assertTrue(self.pr.NLME_CURVE_ID.is_unique)
        for fold in range(1,6):
            rows=self.pr[self.pr.fold.eq(fold)]
            cold_cells=set(pd.read_csv(self.base/f'fold{fold}_heldout_cell.csv').DepMap_ID)
            cold_drugs=set(pd.read_csv(self.base/f'fold{fold}_heldout_drug.csv').drug_group)
            self.assertEqual(set(rows.DepMap_ID),cold_cells.intersection(set(x.DepMap_ID)))
            self.assertEqual(set(rows.drug_group),cold_drugs.intersection(set(x.drug_group)))
            actual=x[x.DepMap_ID.isin(cold_cells)&x.drug_group.isin(cold_drugs)]
            train=x[~x.DepMap_ID.isin(cold_cells)&~x.drug_group.isin(cold_drugs)]
            self.assertFalse(set(actual.DepMap_ID)&set(train.DepMap_ID))
            self.assertFalse(set(actual.drug_group)&set(train.drug_group))
            self.assertEqual(set(rows.NLME_CURVE_ID),set(actual.NLME_CURVE_ID))
            self.assertEqual(len(rows),len(actual))
            np.testing.assert_array_equal(rows.set_index('NLME_CURVE_ID').sort_index().LN_IC50,
                                         actual.set_index('NLME_CURVE_ID').sort_index().LN_IC50)

    def test_metrics_match_predictions(self):
        for (fold,mdl),g in self.met.groupby(['fold','model']):
            y=self.pr.loc[self.pr.fold.eq(fold)]
            p=y['pred_'+mdl].to_numpy(float)
            self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(float(np.sqrt(np.mean((y.LN_IC50-p)**2))),float(g.iloc[0].rmse),places=8)
        summary=pd.read_csv(self.base/'summary.csv').set_index('model')
        self.assertLess(summary.loc['genomics_plus_annotations','mean_rmse'],summary.loc['annotations_only','mean_rmse'])

    def test_cold_drug_annotation_only_performance(self):
        abl=pd.read_csv(ROOT/'research/full_gdsc/annotation_ablation/test_predictions.csv')
        met=pd.read_csv(ROOT/'research/full_gdsc/annotation_ablation/fold_metrics.csv')
        self.assertEqual(len(abl),len(self.obs))
        self.assertEqual(abl.NLME_CURVE_ID.nunique(),len(self.obs))
        primary=pd.read_csv(ROOT/'research/full_gdsc/fold_predictions.csv')
        primary=primary[primary['mode']=='drug'][['NLME_CURVE_ID','fold','LN_IC50','pred_cell_plus_annotations']]
        z=abl.merge(primary,on=['NLME_CURVE_ID','fold'],validate='one_to_one')
        np.testing.assert_allclose(z.LN_IC50_x,z.LN_IC50_y,rtol=0,atol=0)
        self.assertLess(float(np.mean([((g.LN_IC50_x-g.pred_cell_plus_annotations)**2).mean()**.5 for _,g in z.groupby('fold')])),
                        float(met.rmse.mean()))
        for fold,g in abl.groupby('fold'):
            rmse=np.sqrt(np.mean((g.LN_IC50-g.pred_annotations_only)**2))
            self.assertAlmostEqual(rmse,float(met[met.fold.eq(fold)].iloc[0].rmse),places=8)

    def test_full_catalog_train_validation_test_manifests(self):
        base=ROOT/'splits/full_catalog'
        for setting in ('cell','drug','both'):
            visited=set()
            for fold in range(1,6):
                loc=base/setting
                tr=pd.read_csv(loc/f'fold{fold}_outer_train.csv')
                te=pd.read_csv(loc/f'fold{fold}_outer_test.csv')
                vi=pd.read_csv(loc/f'fold{fold}_inner_validation.csv')
                ti=pd.read_csv(loc/f'fold{fold}_inner_train.csv')
                self.assertFalse(set(tr.NLME_CURVE_ID)&set(te.NLME_CURVE_ID))
                self.assertFalse(set(vi.NLME_CURVE_ID)&set(ti.NLME_CURVE_ID))
                self.assertTrue(set(vi.NLME_CURVE_ID).issubset(set(tr.NLME_CURVE_ID)))
                self.assertTrue(set(ti.NLME_CURVE_ID).issubset(set(tr.NLME_CURVE_ID)))
                if setting in ('cell','both'):
                    self.assertFalse(set(tr.DepMap_ID)&set(te.DepMap_ID))
                if setting in ('drug','both'):
                    self.assertFalse(set(tr.drug_group)&set(te.drug_group))
                self.assertFalse(set(te.NLME_CURVE_ID)&visited)
                visited|=set(te.NLME_CURVE_ID)
            self.assertEqual(len(visited),len(self.pr) if setting=='both' else len(self.obs))

    def test_compound_evidence_is_not_overstated(self):
        audit=json.loads((ROOT/'research/full_gdsc/identity_audit/audit.json').read_text())
        self.assertEqual(audit['total_assay_name_groups'],286)
        self.assertEqual(audit['with_source_linked_cid_and_inchikey'],236)
        self.assertEqual(audit['unresolved_assay_name_groups'],50)
        self.assertFalse(audit['full_assay_form_verified'])
        self.assertFalse(audit['complete_chemistry_disjointness'])
        status=pd.read_csv(ROOT/'research/full_gdsc/identity_audit/drug_identity_review.csv')
        self.assertEqual(len(status),286)
        self.assertEqual(int(status.source_linked.sum()),236)
        self.assertEqual(int(status.prior_structure_conflict.sum()),2)

if __name__=='__main__':unittest.main()
