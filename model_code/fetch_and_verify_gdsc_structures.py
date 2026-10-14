"""Optional online download and verification of GDSC-derived identity source CSVs.

NO internet access is required for the checked-in 236-ID identity map or CV.
This script only enriches local structure evidence when both public source URLs
are reachable. It never silently replaces a manually reviewed drug identity.
"""
from __future__ import annotations
import csv, hashlib, io, urllib.request
from pathlib import Path
import pandas as pd
from identity_resolution import ANNOT_SHA, STRUCT_SHA, build

SOURCE = 'https://raw.githubusercontent.com/MinhoLee-DGU/GCNPath2026/main/data/drug_data/'
FILES = [('Anno_Drugs.csv', ANNOT_SHA),('SMILES_GDSC.csv', STRUCT_SHA)]


def git_blob_sha(b):
    return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()


def run(root:Path):
    root=Path(root)
    records=[]
    for file,expected in FILES:
        request=urllib.request.Request(SOURCE+file,headers={'User-Agent':'Challenge27-Research-Identity-Audit/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            content=response.read()
        got=git_blob_sha(content)
        if got!=expected:
            raise ValueError(f'{file}: source changed: expected Git blob {expected}, got {got}. Stop; manually verify release identity.')
        records.append(pd.read_csv(io.BytesIO(content),dtype=str,keep_default_na=False))
    annotations,structures=records
    annotations=annotations[annotations.Datasets.str.contains('GDSC2',na=False)].copy()
    matched=build(root)
    m=matched[['drug_group','canonical_drug_id','pubchem_cid','inchikey','source_linked']].copy()
    m['canonical_drug_id']=m.canonical_drug_id.astype(str)
    m['pubchem_cid']=m.pubchem_cid.astype('Int64').astype(str).replace('<NA>','')
    joined=m.merge(annotations[['Drug_ID','Drug_CID','Name']],left_on='canonical_drug_id',right_on='Drug_ID',how='left',validate='one_to_one')
    joined=joined.merge(structures,left_on='pubchem_cid',right_on='Drug_CID',how='left',suffixes=('','_structure'),validate='many_to_one')
    assert (joined.loc[joined.source_linked,'Drug_CID']==joined.loc[joined.source_linked,'pubchem_cid']).all()
    assert (joined.loc[joined.source_linked,'INCHI_Key']==joined.loc[joined.source_linked,'inchikey']).all()
    # The source has different column names; always require a concrete identity key
    # and a source record before asserting that a structure is corroborated.
    assert int(joined.source_linked.sum())==236
    dst=root/'research/full_gdsc/identity_audit'
    joined.to_csv(dst/'verified_source_enriched_structures.csv',index=False)
    print(f'Saved {len(joined)} source-joined rows. Inspect any missing or variant forms manually before using as molecular input.')


if __name__=='__main__':
    run(Path(__file__).resolve().parents[1])
