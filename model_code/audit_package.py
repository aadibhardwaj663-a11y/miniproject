"""Read-only deliverable and scientific-artifact audit for Challenge 27.

Does not fit models, fetch network data, rewrite splits or mutate checkpoints.
Runs with the pinned dependencies, including PyMuPDF/python-pptx for documents.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from pptx import Presentation


def check(root:Path,checksums:bool=False)->dict:
    root=Path(root)
    essential=(
        'README.md','requirements.txt','Makefile',
        'deliverables/scientific_report.pdf','deliverables/scientific_report.docx',
        'deliverables/research_presentation.pdf','deliverables/research_presentation.pptx',
        'deliverables/integrity_statement.md',
        'docs/DATA_PROVENANCE.md','docs/REPRODUCE.md',
        'research/full_gdsc/matched_responses.csv','research/full_gdsc/matched_omics.npz',
        'research/full_gdsc/fold_predictions.csv','research/full_gdsc/fold_metrics.csv',
        'research/full_gdsc/chemistry_grouped/test_predictions.csv',
        'research/full_gdsc/chemistry_grouped/fold_metrics.csv',
        'research/full_gdsc/identity_audit/drug_identity_review.csv',
        'predictions/full_catalog_predicted_LN_IC50.csv',
        'checkpoints/full_gdsc_ln_ic50_ridge.joblib',
        'model_code/predict_ln_ic50.py','model_code/full_gdsc_benchmark.py',
    )
    missing=[v for v in essential if not (root/v).is_file()]
    if missing:raise ValueError('Missing required deliverables: '+repr(missing))
    with fitz.open(root/'deliverables/research_presentation.pdf') as doc:
        n_pdf=len(doc)
    n_slide=len(Presentation(root/'deliverables/research_presentation.pptx').slides)
    if (n_pdf,n_slide)!=(10,10):raise ValueError(f'Slide count {n_pdf} PDF, {n_slide} PPTX != 10')
    with fitz.open(root/'deliverables/scientific_report.pdf') as doc:
        report_pages=len(doc)
    obs=pd.read_csv(root/'research/full_gdsc/matched_responses.csv',usecols=['DepMap_ID','drug_group','NLME_CURVE_ID','LN_IC50'])
    n_pairs=len(obs);n_cells=obs.DepMap_ID.nunique();n_drugs=obs.drug_group.nunique()
    if (n_pairs,n_cells,n_drugs)!=(160512,649,286):raise ValueError('Unexpected sample counts')
    if obs.duplicated(['DepMap_ID','drug_group']).any() or not obs.NLME_CURVE_ID.is_unique:
        raise ValueError('Duplicate response identity')
    idmap=pd.read_csv(root/'research/full_gdsc/identity_audit/drug_identity_review.csv')
    if len(idmap)!=n_drugs or idmap.drug_group.nunique()!=n_drugs or int(idmap.source_linked.sum())!=236:
        raise ValueError('Chemical identity register discrepancy')
    result={'measured_pairs':n_pairs,'cells':n_cells,'drug_name_groups':n_drugs,
            'source_linked_compound_ids':int(idmap.source_linked.sum()),
            'unresolved_compound_ids':int((~idmap.source_linked).sum()),
            'report_pages':report_pages,'presentation_slides':n_slide}
    for prefix,stem,settingcol,settingvals,key,predprefix in [
        ('primary','', 'mode',('cell','drug'),'fold_predictions.csv','pred_cell_plus_annotations'),
        ('identity','chemistry_grouped/','setting',('drug','both'),'test_predictions.csv','pred_combined')]:
        p=pd.read_csv(root/'research/full_gdsc'/stem/key,
                      usecols=[settingcol,'fold','NLME_CURVE_ID','LN_IC50',predprefix])
        m=pd.read_csv(root/'research/full_gdsc'/stem/'fold_metrics.csv')
        for mode in settingvals:
            sub=p[p[settingcol].eq(mode)]
            if sub.NLME_CURVE_ID.duplicated().any():raise ValueError(f'Repeated {prefix}/{mode} test record')
            if prefix=='primary' and len(sub)!=n_pairs:raise ValueError('Incomplete primary OOF observations')
            if prefix=='identity' and mode=='drug' and len(sub)!=n_pairs:raise ValueError('Incomplete identity-drug test')
            for fold in range(1,6):
                q=sub[sub.fold.eq(fold)]
                if q.empty:raise ValueError(f'Empty fold: {prefix} {mode} {fold}')
                y=q.LN_IC50.to_numpy(float);estimate=q[predprefix].to_numpy(float)
                if not np.isfinite(estimate).all():raise ValueError('Nonfinite heldout values')
                rmse=float(np.sqrt(np.mean((y-estimate)**2)))
                row=m[(m[settingcol].eq(mode))&(m.fold.eq(fold))&(
                    m.model.eq('cell_plus_annotations' if prefix=='primary' else 'combined'))]
                if len(row)!=1 or abs(rmse-float(row.iloc[0].rmse))>1e-6:
                    raise ValueError(f'Metric mismatch: {prefix}/{mode}/{fold}')
            result[f'{prefix}_{mode}_heldout_records']=len(sub)
    catalog=pd.read_csv(root/'predictions/full_catalog_predicted_LN_IC50.csv')
    if len(catalog)!=n_cells*n_drugs or catalog.duplicated(['DepMap_ID','drug_group']).any():
        raise ValueError('Fitted catalog grid incomplete or duplicated')
    if int(catalog.has_observed_measurement.sum())!=n_pairs:
        raise ValueError('Fitted predictions mislabel observed outcomes')
    if not catalog.prediction_status.eq('all_data_fit_NOT_cross_validated').all():
        raise ValueError('Missing fitted-prediction integrity label')
    result['catalog_predictions']=len(catalog)
    result['catalog_without_response']=len(catalog)-n_pairs
    if checksums:
        manifest=root/'SHA256SUMS'
        checked=0
        for line in manifest.read_text().splitlines():
            if not line.strip():continue
            digest,relative=line.split('  ',1)
            file=root/relative
            if not file.is_file():raise ValueError(f'Manifest file missing: {relative}')
            with file.open('rb') as fh:
                h=hashlib.file_digest(fh,'sha256').hexdigest()
            if h!=digest:raise ValueError(f'Checksum mismatch: {relative}')
            checked+=1
        result['sha256_files_checked']=checked
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description='Read-only Challenge 27 scientific package audit')
    cli.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    cli.add_argument('--checksums',action='store_true',help='Verify shipped files against SHA256SUMS on an unchanged extraction')
    args=cli.parse_args()
    print(json.dumps(check(args.root,args.checksums),indent=2))
