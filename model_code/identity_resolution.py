"""Auditable chemical identity map keyed by the *original* GDSC assay DRUG_ID.

The checked-in evidence has explicit GDSC2 drug-ID -> PubChem CID -> InChIKey
joins from GCNPath2026's Anno_Drugs.csv and SMILES_GDSC.csv (see provenance).
No CID is imputed by name or numerical similarity; missing compounds stay missing.
A CID match does NOT certify the salt/form/dose used in the assay.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import hashlib, json, re, csv
import numpy as np
import pandas as pd

SOURCE_ANNOT = 'https://github.com/MinhoLee-DGU/GCNPath2026/blob/main/data/drug_data/Anno_Drugs.csv'
SOURCE_STRUCTURE = 'https://github.com/MinhoLee-DGU/GCNPath2026/blob/main/data/drug_data/SMILES_GDSC.csv'
ANNOT_SHA = '386378bf486f4384af71195c2f278eda9ac326fa'
STRUCT_SHA = '40628645d02943f0fe2cad6520713cbee2882a08'


def build(root:Path):
    root=Path(root)
    dest=root/'research/full_gdsc/identity_audit';dest.mkdir(parents=True,exist_ok=True)
    canonical=pd.read_csv(root/'research/full_gdsc/canonical_drug_ids.csv')
    observed=pd.read_csv(root/'research/full_gdsc/matched_responses.csv',usecols=['drug_group','DRUG_ID','DepMap_ID'])
    raw=(dest/'external_gdsc_compound_mapping.tsv').read_text().strip().split(';')
    rows=[]
    for part in raw:
        x=part.split(':')
        if len(x)!=3 or not re.fullmatch(r'\d+',x[0]) or not re.fullmatch(r'\d+',x[1]) or not re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]',x[2]):
            raise ValueError('Malformed evidence record: '+part)
        rows.append((int(x[0]),int(x[1]),x[2]))
    ids=pd.DataFrame(rows,columns=['canonical_drug_id','pubchem_cid','inchikey'])
    assert len(ids)==236 and ids.canonical_drug_id.is_unique and ids.pubchem_cid.notna().all()
    assert canonical.drug_group.is_unique and len(canonical)==286
    assert canonical.canonical_drug_id.is_unique
    d=canonical.merge(ids,how='left',on='canonical_drug_id',validate='one_to_one')
    prev=pd.read_csv(root/'research/chemistry/structure_mapping_20_provisional.csv')
    prev=prev.rename(columns={'pubchem_cid':'previous_cid','inchikey':'previous_inchikey'})[['drug_group','previous_cid','previous_inchikey']]
    d=d.merge(prev,on='drug_group',how='left',validate='one_to_one')
    d['pubchem_cid']=d.pubchem_cid.astype('Int64')
    d['previous_cid']=d.previous_cid.astype('Int64')
    d['source_linked']=d.pubchem_cid.notna()
    d['identity_key']=np.where(d.source_linked,'pubchem:'+d.pubchem_cid.fillna(0).astype(int).astype(str),'gdsc_id:'+d.canonical_drug_id.astype(str))
    d['status']=np.where(d.source_linked,'SOURCE_LINKED_COMPOUND_FORM_UNVERIFIED','UNRESOLVED_ASSAY_CODE_OR_NAME')
    # Additional *conservative split-grouping only* of likely identical active
    # parents where the assayed form is not certified. Never insert a fake CID.
    # GSK-LSD1 2HCl is a salt of GSK-LSD1 (PubChem/FlyBase/Sigma).
    # GSK2830371A and GSK2830371 share a reported GSK active-parent name;
    # treat as a potential cross-form overlap, NOT confirmed same salt.
    parent_group_candidates={
        'gsk-lsd1-2hcl':('gsk-lsd1','https://flybase.org/reports/FBch0001151.html'),
        'gsk2830371a':('gsk2830371','https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=11103')
    }
    d['conservative_parent_of']=''
    d['parent_evidence_url']=''
    for possible,parent_data in parent_group_candidates.items():
        parent_name,evidence=parent_data
        candidate=d.loc[d.drug_group.eq(possible)]
        resolved_parent=d.loc[d.drug_group.eq(parent_name)]
        if len(candidate)!=1 or len(resolved_parent)!=1 or bool(candidate.iloc[0].source_linked) or not bool(resolved_parent.iloc[0].source_linked):
            raise ValueError(f'Parent alias assumptions require manual review: {possible}')
        d.loc[d.drug_group.eq(possible),'identity_key']=resolved_parent.iloc[0].identity_key
        d.loc[d.drug_group.eq(possible),'status']='POTENTIAL_SHARED_ACTIVE_PARENT_FORM_UNCONFIRMED'
        d.loc[d.drug_group.eq(possible),'conservative_parent_of']=parent_name
        d.loc[d.drug_group.eq(possible),'parent_evidence_url']=evidence
    conflict=d.previous_cid.notna() & d.pubchem_cid.notna() & ((d.previous_cid.astype('Int64')!=d.pubchem_cid.astype('Int64')) | (d.previous_inchikey!=d.inchikey))
    d['prior_structure_conflict']=conflict
    d.loc[conflict,'status']='CONFLICTING_SALT_STEREO_OR_FORM_REQUIRES_ASSAY_REVIEW'
    counts=d.groupby('identity_key').drug_group.transform('size')
    d['identity_shared_by_multiple_names']=counts.gt(1)
    d['source_annotation_url']=np.where(d.source_linked,SOURCE_ANNOT,'')
    d['source_structure_url']=np.where(d.source_linked,SOURCE_STRUCTURE,'')
    d['pubchem_record_url']=d.pubchem_cid.apply(lambda v: f'https://pubchem.ncbi.nlm.nih.gov/compound/{int(v)}' if pd.notna(v) else '')
    d['observed_pairs']=d.drug_group.map(observed.groupby('drug_group').size())
    assert d.drug_group.is_unique and len(d)==286 and d.observed_pairs.sum()==len(observed)
    assert not d.identity_key.isna().any()
    d=d.sort_values('drug_group').reset_index(drop=True)
    d.to_csv(dest/'drug_identity_review.csv',index=False)
    duplicates=d[d.identity_shared_by_multiple_names].sort_values('identity_key')
    duplicates.to_csv(dest/'shared_chemical_identifiers.csv',index=False)
    d[d.status!='SOURCE_LINKED_COMPOUND_FORM_UNVERIFIED'].to_csv(dest/'needs_manual_review.csv',index=False)
    n_pub=int(d.source_linked.sum())
    audit={'total_assay_name_groups':len(d),'with_source_linked_cid_and_inchikey':n_pub,
           'unresolved_assay_name_groups':int((~d.source_linked).sum()),
           'distinct_identity_keys':int(d.identity_key.nunique()),
           'known_shared_chemical_identifier_groups':int(d.identity_shared_by_multiple_names.sum()),
           'known_shared_identifier_clusters':int(duplicates.identity_key.nunique()),
           'source_confirmed_same_cid_clusters':2,
           'conservatively_grouped_active_parent_aliases':2,
           'conflicting_previous_structure_forms':int(conflict.sum()),
           'relevant_gdsc_compound_annotation_sha':ANNOT_SHA,
           'relevant_smiles_structure_table_sha':STRUCT_SHA,
           'full_assay_form_verified':False,
           'complete_chemistry_disjointness':False,
           'identity_grouping_rule':'use linked PubChem CID when available; unique source DRUG_ID otherwise; unknown forms are NOT chemically proven unique',
           'data_origin':'GCNPath2026 Anno_Drugs.csv keyed to original GDSC DRUG_ID, joined to SMILES_GDSC.csv by PubChem CID',
           'cold_split_warning':'remaining 50 exact-CID missing names (including two conservatively grouped by likely active parent) and salt/stereo variants prevent guaranteeing completely chemistry-disjoint folds'}
    (dest/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    return d


def entity_fold_lists(root,n_splits=5,seed=27013):
    from sklearn.model_selection import KFold
    d=pd.read_csv(Path(root)/'research/full_gdsc/identity_audit/drug_identity_review.csv')
    entities=np.array(sorted(d.identity_key.unique()))
    kf=KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    return [set(d.loc[d.identity_key.isin(entities[te]),'drug_group']) for _,te in kf.split(entities)]

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args();build(Path(a.root))
