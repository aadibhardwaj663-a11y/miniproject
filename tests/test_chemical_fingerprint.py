"""Additional independent checks for the provisional molecular-structure experiment."""
from pathlib import Path
import unittest,json
import numpy as np,pandas as pd
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
CHEM=ROOT/'research'/'chemistry'
class ChemistryAudit(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.m=pd.read_csv(CHEM/'chemical_fingerprint_seed_metrics.csv')
  cls.p=pd.read_csv(CHEM/'chemical_fingerprint_test_predictions.csv')
  cls.s=pd.read_csv(CHEM/'structure_mapping_20_provisional.csv')
 def test_twenty_sourced_unique_structures(self):
  self.assertEqual(len(self.s),20)
  self.assertEqual(self.s.inchikey.nunique(),20)
  for r in self.s.itertuples(index=False):
   mol=Chem.MolFromSmiles(r.canonical_smiles)
   self.assertIsNotNone(mol)
   self.assertEqual(Chem.MolToInchiKey(mol),r.inchikey)
   self.assertTrue(r.source_url.startswith('https://'))
   self.assertIn('ASSAY_FORM_UNVERIFIED',r.review_status)
 def test_fingerprint_array(self):
  f=np.load(ROOT/'research/data/morgan_1024_fingerprints.npz',allow_pickle=False)
  self.assertEqual(f['bits'].shape,(20,1024))
  self.assertTrue(np.isin(f['bits'],[0,1]).all())
  self.assertEqual(len(set(f['drugs'].tolist())),20)
 def test_metric_partition_and_finiteness(self):
  self.assertEqual(len(self.m),105)
  self.assertEqual(len(self.p),10000)
  self.assertFalse(self.p[['mode','seed','NLME_CURVE_ID']].duplicated().any())
  for mode in ['cell','drug','both']:
   self.assertEqual(set(self.m[self.m['mode']==mode].seed),{107,207,307,407,507})
  self.assertTrue(np.isfinite(self.m[['rmse','mae']].to_numpy()).all())
 def test_regenerate_metrics_from_saved_predictions(self):
  for r in self.m.itertuples(index=False):
   a=self.p[(self.p['mode']==r.mode)&(self.p.seed==r.seed)]
   calc=np.sqrt(np.mean((a['LN_IC50']-a['pred_'+r.variant])**2))
   self.assertLess(abs(calc-r.rmse),1e-6)
 def test_source_structure_groups_disjoint_across_drug_splits(self):
  sc=pd.read_csv(ROOT/'research/data/validated_drug_structures.csv')
  self.assertEqual(sc.murcko_scaffold.nunique(),20)
  for mode in ['drug','both']:
   for seed in [107,207,307,407,507]:
    x=json.loads((ROOT/'research/validation'/f'{mode}_seed{seed}_ids.json').read_text())
    a=set(x['train_drugs']);b=set(x['test_drugs'])
    self.assertFalse(a&b)
    keyA=set(sc.loc[sc.drug_group.isin(a),'inchikey_rdkit'])
    keyB=set(sc.loc[sc.drug_group.isin(b),'inchikey_rdkit'])
    self.assertFalse(keyA&keyB)
    sfA=set(sc.loc[sc.drug_group.isin(a),'murcko_scaffold'])
    sfB=set(sc.loc[sc.drug_group.isin(b),'murcko_scaffold'])
    self.assertFalse(sfA&sfB)
 def test_result_not_claimed_clinical(self):
  report=(CHEM/'CHEMICAL_EXTENSION_REPORT.md').read_text().lower()
  self.assertIn('assay',report)
  self.assertIn('non-chiral',report)
  self.assertIn('independent',report)
if __name__=='__main__':unittest.main(verbosity=2)
