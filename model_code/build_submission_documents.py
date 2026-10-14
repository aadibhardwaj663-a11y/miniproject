"""Build a fresh, self-contained Word report, 10-slide presentation and short speech.
All quantitative claims are read from saved source predictions/metrics, not typed results.
Execute: python model_code/build_submission_documents.py
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches as DIn, Pt as DPt, RGBColor as DColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / 'deliverables'
DELIVERABLES.mkdir(exist_ok=True)
D = ROOT / 'research' / 'full_gdsc'
S = pd.read_csv(D / 'summary.csv')
PRED = pd.read_csv(D / 'fold_predictions.csv')
OBS = pd.read_csv(D / 'matched_responses.csv')
SKILL = pd.read_csv(D / 'per_drug_skill_cell_holdout.csv')
BIO = pd.read_csv(D / 'biological_interpretability' / 'five_fold_metrics.csv')
IMP = pd.read_csv(D / 'biological_interpretability' / 'heldout_gene_group_permutation.csv')
JOINT = pd.read_csv(D / 'strict_generalization' / 'summary.csv').set_index('model')
ANN = pd.read_csv(D / 'annotation_ablation' / 'fold_metrics.csv')
AUDIT = __import__('json').loads((D / 'identity_audit' / 'audit.json').read_text())
assert AUDIT['with_source_linked_cid_and_inchikey']==236 and AUDIT['unresolved_assay_name_groups']==50
CHEM = pd.read_csv(D/'chemistry_grouped'/'summary.csv').set_index(['setting','model'])
chem_drug=float(CHEM.loc[('drug','combined'),'rmse']); chem_drug_ann=float(CHEM.loc[('drug','annotations_only'),'rmse'])
chem_both=float(CHEM.loc[('both','combined'),'rmse']); chem_both_ann=float(CHEM.loc[('both','annotations_only'),'rmse'])
chem_n=int(CHEM.loc[('both','combined'),'n_test'])
joint_rmse=float(JOINT.loc['genomics_plus_annotations','mean_rmse'])
joint_ann=float(JOINT.loc['annotations_only','mean_rmse'])
joint_cell=float(JOINT.loc['genomics_only','mean_rmse'])
annotation_drug=float(ANN.rmse.mean())
assert (len(OBS),OBS.DepMap_ID.nunique(),OBS.drug_group.nunique()) == (160512,649,286)
assert len(PRED)==321024 and not OBS.duplicated(['DepMap_ID','drug_group']).any()


def val(mode, model, key):
    x = S.query('mode==@mode and model==@model')
    assert len(x)==1
    return float(x.iloc[0][key])

newcell=val('cell','cell_plus_annotations','rmse_mean')
cellbase=val('cell','drug_mean','rmse_mean')
newdrug=val('drug','cell_plus_annotations','rmse_mean')
drugbase=val('drug','cell_only','rmse_mean')
within=float(SKILL.r.median())
beats=int((SKILL.delta_rmse>0).sum())
if not np.isfinite([newcell,cellbase,newdrug,drugbase,within,chem_drug,chem_both]).all():
    raise ValueError('Nonfinite reported model metric')
if not (0<newcell<cellbase and 0<newdrug<drugbase and -1<=within<=1):
    raise ValueError('Unexpected change in benchmark conclusions; review results')
assert beats==229 and len(SKILL)==286

REFS=[
 ('Barretina J, et al. The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature 483, 603-607 (2012).','https://doi.org/10.1038/nature11003'),
 ('Iorio F, et al. A Landscape of Pharmacogenomic Interactions in Cancer. Cell 166, 740-754 (2016).','https://doi.org/10.1016/j.cell.2016.06.017'),
 ('Ghandi M, et al. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature 569, 503-508 (2019).','https://doi.org/10.1038/s41586-019-1186-3'),
 ('Codicè F, et al. The specification game: rethinking the evaluation of drug response prediction for precision oncology. J Cheminform 17, 33 (2025).','https://doi.org/10.1186/s13321-025-00972-y'),
 ('TransCDR: a deep learning model for enhancing generalizability of drug activity prediction. BMC Biology (2024).','https://doi.org/10.1186/s12915-024-02023-8'),
 ('Small-molecule MDM2 antagonists reveal aberrant p53 signaling in cancer. Cancer Research (2006).','https://pmc.ncbi.nlm.nih.gov/articles/PMC1413632/'),
 ('Original user-supplied DepMap Public 19Q4 README and GDSC2 27 Oct 2023 response workbook.','data/source_depmap_README.txt; source archive hash in docs/DATA_PROVENANCE.md'),
 ('GCNPath2026 supplementary GDSC annotation and structure records, Drug_ID / Drug_CID / InChIKey, repository snapshot referenced in identity audit.','https://github.com/MinhoLee-DGU/GCNPath2026/tree/main/data/drug_data'),
 ('PubChem PUG REST documentation and compound records.','https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest')
]

INK=(25,43,63); TEAL=(10,120,112); GRAY=(92,107,124); PALE='EAF3F2'; HEADER='17394A'

def cell_fill(cell, color):
    tcPr=cell._tc.get_or_add_tcPr(); sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),color); tcPr.append(sh)


def add_table(doc, headers, rows):
    tab=doc.add_table(rows=1,cols=len(headers));tab.alignment=WD_TABLE_ALIGNMENT.CENTER
    tab.style='Table Grid'
    for j,s in enumerate(headers):
        c=tab.rows[0].cells[j];c.text=str(s);cell_fill(c,HEADER)
        for par in c.paragraphs:
            for run in par.runs: run.bold=True;run.font.size=DPt(8.5);run.font.color.rgb=DColor(255,255,255)
    for i,row in enumerate(rows):
        cs=tab.add_row().cells
        for j,item in enumerate(row):
            cs[j].text=str(item)
            cs[j].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i%2==0:cell_fill(cs[j],'F2F7F9')
            for par in cs[j].paragraphs:
                for run in par.runs:run.font.size=DPt(8.7)
    doc.add_paragraph('')
    return tab


def report():
    doc=Document();sec=doc.sections[0]
    sec.top_margin=DIn(.72);sec.bottom_margin=DIn(.65);sec.left_margin=sec.right_margin=DIn(.86)
    normal=doc.styles['Normal'];normal.font.name='Aptos';normal.font.size=DPt(10);normal.font.color.rgb=DColor(*INK)
    normal.paragraph_format.space_after=DPt(6)
    for sty,size in [('Title',26),('Heading 1',15),('Heading 2',11.5)]:
        q=doc.styles[sty];q.font.name='Aptos Display';q.font.size=DPt(size);q.font.color.rgb=DColor(*TEAL)
    doc.styles['Caption'].font.size=DPt(9);doc.styles['Caption'].font.italic=True
    def para(s, bold=False):
        q=doc.add_paragraph()
        r=q.add_run(s);r.bold=bold
        return q
    def heading(s):doc.add_heading(s,1)
    def figure(filename, caption, width=5.25):
        pp=doc.add_paragraph();pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
        pp.add_run().add_picture(str(D/'figures'/filename),width=DIn(width))
        cp=doc.add_paragraph(caption);cp.style='Caption';cp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    title=doc.add_paragraph(style='Title');title.add_run('Mini-project 27\nGene-expression-based drug sensitivity prediction')
    para('Reproducible oncology / pharmacogenomics mini-project')
    para('CCLE / DepMap molecular profiles and GDSC2 fitted drug-response measurements | October 2026')
    para('Research use only. Predictions concern cancer cell lines in laboratory assays, not patient treatment.',bold=True)

    heading('Abstract')
    para(f'The study predicts fitted natural-log IC50 (LN_IC50) using cancer-cell-line gene expression, mutation records, copy number and drug annotations. From the original experimental files, 649 cell lines and 286 drug-name groups were conservatively matched, producing 160,512 observed cell-drug responses. Two five-fold evaluations withheld complete cell-line identities or complete drug groups during training. A fixed Ridge regression pipeline using 100 training-selected expression features, a ten-gene mutation/copy-number panel and known drug descriptors was compared with strong baselines. For unseen cell lines, mean test RMSE was {newcell:.3f}, versus {cellbase:.3f} for a training-only per-drug mean. For unseen drug groups, RMSE was {newdrug:.3f} versus {drugbase:.3f} for genomics-only Ridge. However, median within-drug Pearson correlation was only {within:.3f}, and some mechanistically related drugs were poorly predicted. The original paired cell-plus-drug holdout achieved RMSE {joint_rmse:.3f} versus {joint_ann:.3f} for annotations alone. A retrospective GDSC-assay-ID-to-PubChem-CID regrouping of known aliases achieved RMSE {chem_drug:.3f} in cold drugs and {chem_both:.3f} in simultaneous cold cells and drugs ({chem_n:,} held-out pairs). Verified source-linked CIDs and InChIKeys cover 236 of 286 drug-name groups; the remaining 50 plus unverified assay salt forms prevent claiming perfect chemical separation or external validation.')

    heading('1. Challenge and scientific background')
    para('Cancer cell lines may respond differently to the same compound because gene activity, genomic variation and gene dosage differ across biological contexts. Gene expression provides an RNA-level view of cell state, while mutation and copy-number data add genomic information. The objective is to estimate LN_IC50 for a particular cell and drug. Lower fitted IC50 generally implies a lower inhibitory concentration in that assay; a negative logarithm is valid and does not indicate a bad measurement.')
    para('CCLE established links between molecular features and preclinical drug response [1], while GDSC work illustrated pharmacogenomic associations across larger screens [2]. Updated CCLE characterization provides detailed molecular reference data [3]. Modern evaluations distinguish unseen cells from unseen drugs because random observation-level splits can reuse familiar entities and overstate generalization [4,5]. This mini-project therefore compares both entity-level tasks and reports drug-specific error, rather than relying on one pooled correlation.')

    heading('2. Datasets and curation')
    add_table(doc,['Data item','Observed / verified'],[
        ('Original GDSC2 fitted response records','242,036 rows'),
        ('Conservatively cross-omics matched cell lines','649'),
        ('Distinct normalized drug-name groups','286'),
        ('Observed matched cell-drug measurements','160,512'),
        ('Available measurements in potential matched grid',f'{100*160512/(649*286):.1f}% of 185,614 possible pairs'),
        ('Omics reference','DepMap Public 19Q4'),
        ('Response reference','GDSC2 fitted dose response, 27 Oct 2023')])
    para('The original ZIP contains sample metadata, CCLE expression, mutations, gene copy-number profiles, a GDSC2 workbook and a DepMap README. Cell lines were joined by exact COSMIC/DepMap mappings and checked for name/Sanger-ID discrepancies. Only cells represented in all three molecular sources were kept. Repeated numeric drug IDs under one normalized drug name were consolidated using coverage, independent of response value. The subset represents all eligible matched responses, not the complete GDSC2 release. Missing outcomes were never fabricated.')
    para('Data-provenance checks include an archive SHA-256/CRC audit, schema validation, finite-value checks, unique response-curve IDs and saved fold identifiers. The updated source-ID-linked identity audit obtains PubChem CIDs and InChIKeys for 236 of 286 drug-name groups from public GDSC-derived annotations [8]; 50 remain unresolved, largely unpublished internal compound codes. Twenty stereochemical structures were present in the earlier local pilot; 236 linked identifiers are not 236 independently validated assay formulations. Two exact-CID name-pairs, plus two potential salt/active-parent pairs, are conservatively held together in the revised folds. MK-2206 and oxaliplatin show alternative salt/structure forms in prior annotations, flagged for manual review. No structure was guessed for missing internal codes.')

    heading('3. Processing and prediction pipeline')
    para('The model uses 100 expression columns selected by variance among unique training cell lines; the same genes need not be selected in every fold. A fixed genomic panel (TP53, KRAS, BRAF, EGFR, PIK3CA, PTEN, APC, RB1, MYC, CDKN2A) contributes ten binary nonsilent-variant-record indicators and ten numeric copy-number measurements. Drug descriptors comprise the supplied GDSC2 putative-target and pathway labels. For previously encountered drugs, a one-hot identity feature additionally distinguishes compounds with the same target/pathway annotations; an unseen-drug identity token cannot be learned in the training fold.')
    para('Training-only preprocessing applies median imputation, standardization and a fixed Ridge regression model (alpha=100, LSQR solver). Baselines include the global training response mean, a per-drug training-only mean for new-cell evaluation, and a genomics-only Ridge model for new-drug evaluation. An independent cold-drug annotation-only ablation isolates the additional contribution of genomics. These baselines prevent drug-average differences from being mistaken for cellular predictive performance.')
    para('End-to-end pipeline: source integrity -> conservative identity join -> fold generation over unique entities -> training-only feature selection and encoding -> fit baseline and Ridge -> infer on excluded entities -> recompute metrics -> subgroup and biological interpretation.')
    para('Prediction application: a separate full-data Ridge checkpoint for inference is saved in checkpoints/full_gdsc_ln_ic50_ridge.joblib and used by model_code/predict_ln_ic50.py. For any recognized cell/drug pair, the script retrieves its real expression, recorded variant/CN features and GDSC drug annotations to produce predicted LN_IC50; it also supports CSV batch inference and a complete 649 by 286 grid. The grid contains 185,614 predictions: 160,512 have measured LN_IC50 and 25,102 are model-only estimates, explicitly marked unmeasured. This model was fitted on all available observed pairs, so its predictions are not held-out performance evidence. All reported evaluation scores come strictly from the saved grouped cross-validation folds.')

    heading('4. Experimental design')
    add_table(doc,['Test','Held-out entity','Fold construction','Primary comparison'],[
        ('New cancer cell lines','DepMap cell-line ID','Five nonoverlapping cell-line folds; all known drugs remain represented during training','Per-drug training mean versus genomics + drug Ridge'),
        ('New drug groups','Normalized drug name','Five nonoverlapping drug-group folds; all known cells remain represented','Genomics-only and annotation-only versus combined model'),
        ('New cells AND new drugs','Paired cell-ID and drug-name folds','Five simultaneous double-holdout folds; other pairs excluded','Genomics only and annotations only versus combined model')])
    para('The two tests answer different questions. For cell holdouts, the same compound is familiar from other cell lines. For drug holdouts, the drug-name group is absent from training, but cell lines can be shared by design. Preprocessing is refitted inside each fold, and fold labels, measured responses and predictions are saved. All 286 compounds and 649 cells appear in one test fold within their respective evaluation. Fold standard deviation is descriptive, not an independent external confidence interval. The additional full-catalog simultaneous experiment uses the intersection of cell and drug test folds with the same index. Training excludes all test cells and all test drugs. This design evaluates 31,998 of 160,512 measured pairs (19.9%) across five disjoint test intersections; it does not establish performance on the untested 80.1%. It uses sparse, non-centered scaling with an intercept, as does the optimized primary model; these experiments still differ in split membership and feature combinations. The new PubChem-CID-grouped tests address known cross-name collisions, grouping both aliases of one CID together and leaving unknown structures as provisional unique GDSC IDs. They were designed after finding those collisions and are therefore retrospective checks, NOT an untouched holdout. Assay salt/stereo identity and missing compounds can still leak as yet-unknown chemical equivalents. All four model families are compared on the SAME identity-grouped folds.')

    doc.add_page_break()
    heading('5. Main predictive results')
    add_table(doc,['Test','Predictor','Mean RMSE ± fold SD','Mean MAE','Pooled Pearson r'],[
        ('New cells','Global training mean',f"{val('cell','global_mean','rmse_mean'):.3f} ± {val('cell','global_mean','rmse_sd'):.3f}",f"{val('cell','global_mean','mae_mean'):.3f}",'Undefined'),
        ('New cells','Per-drug training mean',f'{cellbase:.3f} ± {val("cell","drug_mean","rmse_sd"):.3f}',f'{val("cell","drug_mean","mae_mean"):.3f}',f'{val("cell","drug_mean","r_mean"):.3f}'),
        ('New cells','Genomics + drug features',f'{newcell:.3f} ± {val("cell","cell_plus_annotations","rmse_sd"):.3f}',f'{val("cell","cell_plus_annotations","mae_mean"):.3f}',f'{val("cell","cell_plus_annotations","r_mean"):.3f}'),
        ('New drugs','Global training mean',f'{val("drug","global_mean","rmse_mean"):.3f} ± {val("drug","global_mean","rmse_sd"):.3f}',f'{val("drug","global_mean","mae_mean"):.3f}','Undefined'),
        ('New drugs','Genomics-only Ridge',f'{drugbase:.3f} ± {val("drug","cell_only","rmse_sd"):.3f}',f'{val("drug","cell_only","mae_mean"):.3f}',f'{val("drug","cell_only","r_mean"):.3f}'),
        ('New drugs','Genomics + drug annotations',f'{newdrug:.3f} ± {val("drug","cell_plus_annotations","rmse_sd"):.3f}',f'{val("drug","cell_plus_annotations","mae_mean"):.3f}',f'{val("drug","cell_plus_annotations","r_mean"):.3f}')])
    para(f'Additional same-fold cold-drug ablation: annotation-only mean RMSE {annotation_drug:.3f}, compared with {newdrug:.3f} for combined genomic and drug annotations. This indicates approximately {100*(1-newdrug/annotation_drug):.1f}% improvement beyond drug annotations under the primary unseen-drug protocol. Drug identity is absent for previously unseen drug groups, so target/pathway labels supply the transferable drug signal.')
    heading('5.1 Simultaneous unseen cells and unseen drugs')
    add_table(doc,['Five-fold paired double-holdout model','Mean RMSE','Mean MAE'],[
        (name, f'{JOINT.loc[key,"mean_rmse"]:.3f}', f'{JOINT.loc[key,"mean_mae"]:.3f}') for name,key in [
            ('Global training mean','global_mean'),('Genomics only','genomics_only'),
            ('Drug annotations only','annotations_only'),('Genomics + drug annotations','genomics_plus_annotations')]])
    para(f'The combined model obtained RMSE {joint_rmse:.3f} versus {joint_ann:.3f} for drug annotations alone and {joint_cell:.3f} for genomics alone. The incremental advantage over annotation-only is approximately {100*(1-joint_rmse/joint_ann):.1f}%. This task excludes all test cell and drug identities from training, but it is still restricted to the same in-vitro screen and inherits unresolved chemical synonym risk. Full test rows, group IDs, metrics and protocol are saved under research/full_gdsc/strict_generalization/.')
    heading('5.2 Chemical-identifier-grouped robustness checks (retrospective)')
    add_table(doc,['Cold setting','Model','5-fold mean RMSE ± SD','Mean MAE'],[
        (lab,mod,f'{CHEM.loc[(setting,key),"rmse"]:.3f} ± {CHEM.loc[(setting,key),"rmse_sd"]:.3f}',f'{CHEM.loc[(setting,key),"mae"]:.3f}')
        for setting,lab in [('drug','New drug CIDs'),('both','New cells + drug CIDs')]
        for key,mod in [('annotations_only','Annotations only'),('combined','Genomics + annotations')]
    ])
    para(f'New-drug combined RMSE {chem_drug:.3f} versus annotation-only {chem_drug_ann:.3f} ({100*(1-chem_drug/chem_drug_ann):.1f}% reduction). Both unseen together RMSE {chem_both:.3f} versus {chem_both_ann:.3f} ({100*(1-chem_both/chem_both_ann):.1f}% reduction) on {chem_n:,} observed pairs. These are not directly comparable to the original drug-name split because folds were reassigned and sparse fold pipelines differ. The original model weights and LN_IC50 inference checkpoint remain unchanged; this is a leakage-aware internal sensitivity analysis. The full four-model comparison is available in the fold CSVs. Five-fold split files, predictions and foldwise metrics are saved in research/full_gdsc/chemistry_grouped/.')
    figure('01_full_kfold_rmse.png' ,'Figure 1. Original held-out performance over five normalized-name/entity folds; not PubChem-identifier-grouped.',width=5.7)
    para(f'Compared with meaningful baselines, genomic and drug features reduced mean RMSE by {100*(1-newcell/cellbase):.1f}% for new cell lines and {100*(1-newdrug/drugbase):.1f}% versus genomics-only Ridge for new drugs. These are numerical in-vitro screening results on one matched dataset. No model was tested on an independent PRISM or CTRP response cohort.')

    heading('6. Drug-specific and biological-context analysis')
    para(f'Pooled new-cell Pearson correlation was {val("cell","cell_plus_annotations","r_mean"):.3f}, but the median Pearson correlation computed separately for each drug was {within:.3f}. This drop is important: pooled correlation partly reflects differences in average response between compounds. The Ridge model reduced RMSE relative to the training-only per-drug mean for {beats}/286 compounds; for 57 it did not.')
    figure('02_within_drug_pearson.png','Figure 2. Distribution of within-drug correlation based on out-of-fold cell predictions.')
    para('MEK-pathway examples illustrate the failure modes. Despite distinct identity encodings for familiar drugs, selumetinib and trametinib did not beat their own per-drug training-mean baselines. Drug-specific error and tissue-lineage error tables are included to prevent a high pooled score from obscuring these cases.')
    figure('04_mek_and_biomarker_drugs.png','Figure 3. Selected drugs: model RMSE, per-drug mean RMSE and observed response SD.')

    heading('7. Biological interpretation')
    para('Five compounds were chosen before the drug-specific biological analysis: Nutlin-3a, trametinib, palbociclib, selumetinib and taselisib. For each, an independent expression-plus-panel Ridge model was evaluated on five cell-line folds. In held-out cells, the mutation-plus-copy-number values of mechanism-related genes were permuted 40 times per fold to measure changes in RMSE. The result measures how a fitted predictor relies on genomic variables; it does not establish a causal biomarker.')
    q=BIO.groupby(['drug','model']).rmse.mean().unstack()
    add_table(doc,['Drug','Per-drug mean RMSE','Drug-specific genomics Ridge'],[[drug,round(q.loc[drug,'per_drug_mean'],3),round(q.loc[drug,'gene_expression_panel_ridge'],3)] for drug in ['nutlin-3a (-)','palbociclib','selumetinib','taselisib','trametinib']])
    figure('06_drug_specific_gene_permutation.png','Figure 4. Held-out model reliance on predetermined mutation + copy-number gene groups.')
    nut=float(IMP[(IMP.drug=='nutlin-3a (-)')&(IMP.gene=='TP53')].delta_rmse_mean.mean())
    para(f'For Nutlin-3a, permuting the TP53 group increased held-out RMSE by approximately {nut:.3f} on average. The MDM2-p53 pathway has a known mechanistic link to Nutlin-3a response [6]. Nevertheless, sequence-variant record presence is not equivalent to functional TP53 status and is not an independently replicated clinical biomarker.')

    heading('8. Discussion and limitations')
    para('The model succeeds most clearly for familiar drugs tested on unseen cell lines. Its gains are meaningful but modest against each drug’s own training mean. Generalization to unseen drug-name groups is harder: the model learns transferable pathway and target information but has only provisional PubChem mappings for 236 of 286 names, not audited experimental salt/stereo forms. Testing within each drug and lineage makes this limitation visible. The older 200-cell/20-drug MEK fingerprint finding is exploratory: the claimed 1.4% average gain was not robust across seeds and should not be applied to the 649-cell benchmark. The older MG-132 permutation result was weak; current TP53/Nutlin-3a permutation is on a drug-specific held-out model rather than the deployed full model. A previous 20-compound fingerprint experiment gave inconsistent results and is kept as exploratory appendix data, not as evidence that fingerprints reliably solve novel-compound generalization.')
    para('Limitations: (1) DepMap 19Q4 genomics were matched to GDSC2 2023 responses; (2) 50 drug identities remain unresolved, 2 previously recorded chemical forms conflict, and no assay reagent form has been fully confirmed; (3) original paired cold-holdout covers 19.9% of pairs, and the new CID-grouped retrospective paired test covers a similar fraction, without independent external or pristine prospective validation; (4) test observations were revisited to improve split protocol and must not be interpreted as a fresh unbiased independent test; (5) mutation-record absence does not establish functional wild-type status, and IC50 does not predict patient outcomes. Fold test predictions remain the independent evaluation evidence. A separate all-data inference checkpoint was saved using sparse scaling, but it has not been independently externally validated.')

    heading('9. Conclusion')
    para('A reproducible genomics-plus-drug-annotation Ridge pipeline predicts held-out laboratory drug responses better than appropriate baselines on the matched CCLE/GDSC2 dataset, including a measured but modest improvement in the simultaneous cold-cell/cold-drug task. The strongest evidence is for new cell lines with known drugs. Drug-specific correlations, several challenging compounds and the lack of an independent validation set impose important boundaries on the claim. This is a transparent research mini-project, not a clinical precision-oncology system.')

    heading('AI assistance and integrity disclosure')
    para('OpenAI ChatGPT assisted with implementation, testing, drafting this report and presentation. All tabulated numerical outcomes are derived from saved experimental response records and executable evaluation code. This is educational, preclinical research, not a clinical decision system.')

    heading('References')
    for i,(name,url) in enumerate(REFS,1):
        par=doc.add_paragraph();par.paragraph_format.space_after=DPt(4)
        x=par.add_run(f'[{i}] {name}  {url}');x.font.size=DPt(8.8)
    path=DELIVERABLES/'scientific_report.docx';doc.save(path)
    return path

# Slide deck uses native editable text/shapes and real recorded figures.
DARK='122437';LIGHT='F5F8FC';C_TEAL='0A8C83';C_MUTED='546C80';C_AMBER='D38A44';C_WHITE='FFFFFF';C_INK='203346';C_FAINT='E6F4F1'
ppt=Presentation();ppt.slide_width=Inches(13.333);ppt.slide_height=Inches(7.5)

def C(x):return RGBColor.from_string(x)
def shape(sl,x,y,w,h,fill,rounded=False):
    sp=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sp.fill.solid();sp.fill.fore_color.rgb=C(fill);sp.line.fill.background();return sp

def text(sl,s,x,y,w,h,sz=18,bold=False,col=C_INK):
    z=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));f=z.text_frame;f.clear();f.word_wrap=True
    f.margin_left=Inches(.02);f.margin_top=Inches(.01)
    for i,line in enumerate(str(s).split('\n')):
        par=f.paragraphs[0] if i==0 else f.add_paragraph();par.text=line;par.font.name='Aptos';par.font.size=Pt(sz);par.font.bold=bold;par.font.color.rgb=C(col);par.space_after=Pt(7)
    return z

def slide(label,title,notes,dark=False):
    sl=ppt.slides.add_slide(ppt.slide_layouts[6]);shape(sl,0,0,13.333,7.5,DARK if dark else LIGHT)
    shape(sl,.59,.44,.08,.37,C_AMBER if dark else C_TEAL)
    text(sl,label.upper(),.83,.44,11,.27,11,True,'91DFD6' if dark else C_TEAL)
    text(sl,title,.60,.98,12.2,.93,32,True,C_WHITE if dark else C_INK)
    shape(sl,.65,7.04,12.02,.016,'4C6477' if dark else 'DDE5EE')
    text(sl,'CHALLENGE 27   /   GENE EXPRESSION & DRUG SENSITIVITY',.66,7.17,11.5,.22,9,False,'B4C5D2' if dark else C_MUTED)
    text(sl,f'{len(ppt.slides):02d}',12.2,7.15,.5,.22,11,True,'B4C5D2' if dark else C_MUTED)
    sl.notes_slide.notes_text_frame.text=notes
    return sl

def card(sl,x,y,w,h,head,body,col=C_TEAL):
    shape(sl,x,y,w,h,C_WHITE,True);shape(sl,x+.22,y+.26,.43,.05,col)
    text(sl,head,x+.23,y+.60,w-.46,.55,22,True)
    text(sl,body,x+.23,y+1.30,w-.48,h-1.52,17,False,C_MUTED)

def chart(sl,filename,x,y,w,h):
    shape(sl,x,y,w,h,C_WHITE,True)
    # native chart source image; no values are recreated in the deck
    pic=pptx_picture=sl.shapes.add_picture(str(D/'figures'/filename),Inches(x+.10),Inches(y+.10))
    w0,h0=pic.width,pic.height
    scale=min((w-.20)*914400/w0,(h-.20)*914400/h0)
    pic.width=int(w0*scale);pic.height=int(h0*scale)
    pic.left=Inches(x+(w-pic.width/914400)/2);pic.top=Inches(y+(h-pic.height/914400)/2)

# Slide 1
s=slide('Research mini-project','Can gene expression predict cancer-drug sensitivity?', 'Begin with the challenge: cell lines differ in genomic activity and can react differently to the same drug. We predict a laboratory inhibition response, not patient treatment.',True)
shape(s,.73,2.30,11.86,2.38,'244056',True)
for x,n,lbl in [(1.04,'649','matched cell lines'),(5.02,'286','drug-name groups'),(8.79,'160,512','observed response pairs')]:
    text(s,n,x,2.64,3.5,.88,42,True,C_WHITE);text(s,lbl,x,3.68,3.6,.54,16,False,'C9DDE8')
shape(s,.76,5.20,11.80,.86,'194A4D',True);text(s,'Question: What generalizes to new cells, and what generalizes to new drugs?',1.03,5.48,11.1,.35,20,True,C_WHITE)

# Slide 2
s=slide('Challenge & background','What is being predicted?', 'Gene expression measures activity of genes via RNA; mutation and copy number describe other genomic variation. The target is LN_IC50, the natural logarithm of fitted inhibitory concentration. A lower value generally means less compound is needed in the cell assay, not a recommended clinical dose.')
card(s,.70,2.06,3.83,3.54,'Cell profile','Gene expression\nMutation records\nGene copy number')
card(s,4.73,2.06,3.83,3.54,'Drug information','Compound identity if known\nPutative targets\nPathway labels')
card(s,8.76,2.06,3.83,3.54,'Response','Continuous LN_IC50\nFitted laboratory IC50\nLower = less concentration')
shape(s,.82,6.00,11.74,.62,C_FAINT,True);text(s,'A prediction is one cell-line and one drug combination.',1.05,6.20,10.94,.27,18,True,C_TEAL)

# Slide 3
s=slide('Dataset','How did we prepare the data?', 'CCLE/DepMap provides expression, mutation and copy-number files; GDSC2 provides fitted drug response. We matched exact cell identifiers, excluded name conflicts and combined source-normalized duplicated drug identifiers. This yields 649 matched cells, 286 drugs and 160,512 measured response pairs, about 86.5 percent coverage. The original source uses different years for genomics and response.')
for i,(headline,value,caption) in enumerate([('Molecular profiles','649','cross-omics matched cells'),('Drug groups','286','source-normalized names'),('Observed responses','160,512','86.5% of matched grid')]):
    x=.77+i*4.11;shape(s,x,2.17,3.84,2.58,C_WHITE,True);text(s,value,x+.22,2.54,3.4,.80,38,True,C_TEAL);text(s,headline,x+.22,3.52,3.37,.45,17,True);text(s,caption,x+.22,4.12,3.28,.42,13,False,C_MUTED)
shape(s,.80,5.16,11.72,1.03,'E7F2F0',True);text(s,'No missing response values were invented. Only verified, available measurements were modeled.',1.04,5.43,11.1,.49,17,True,C_INK)

# Slide 4
s=slide('Methodology','The prediction pipeline', 'The workflow starts with exact source joins, then generates folds over unique entities. Within each training fold it selects 100 expression features, includes the same ten-gene mutation/copy-number panel, encodes the appropriate drug information and fits Ridge. Then it predicts and audits previously excluded entities. No test response is used for feature selection.')
steps=[('01','Match sources','COSMIC / DepMap IDs'),('02','Hold out entities','Cells or drugs'),('03','Build features','100 expression + 20 genomic + drug'),('04','Fit model','Training-only preprocessing, Ridge'),('05','Evaluate','Held-out RMSE, MAE, per-drug r')]
for i,(num,head,desc) in enumerate(steps):
    x=.65+i*2.57;shape(s,x,2.45,2.42,3.34,C_WHITE,True);text(s,num,x+.19,2.73,1.20,.39,17,True,C_TEAL);text(s,head,x+.18,3.29,2.08,.62,17,True);text(s,desc,x+.18,4.25,2.11,1.30,14,False,C_MUTED)
text(s,'Predictor: scaled Ridge regression (alpha = 100); baseline models are fitted only to training responses.',.83,6.11,11.8,.55,16,True)

# Slide 5
s=slide('Evaluation','Three held-out tests', f'Original drug-name-grouped and cell-ID benchmarks are retained. Additional post-hoc CID-grouped five-fold experiments keep compounds sharing known PubChem identities together. New drug CID RMSE is {chem_drug:.3f} vs {chem_drug_ann:.3f} annotation-only; both new RMSE {chem_both:.3f} vs {chem_both_ann:.3f} annotation-only on {chem_n:,} pairs. The 50 unresolved compounds still prevent a fully chemistry-disjoint claim.')
card(s,.81,2.21,5.67,3.58,'New cell lines','Train: different cell lines\nTest: cell IDs never used in training\nFair baseline: mean for each familiar drug',C_TEAL)
card(s,6.78,2.21,5.67,3.58,'New drugs','Original: disjoint drug-name groups\nRevised: disjoint known PubChem CIDs\n50 drug codes remain unverified',C_AMBER)
text(s,f'Revised BOTH NEW ({chem_n:,} pairs): RMSE {chem_both:.3f} combined vs {chem_both_ann:.3f} annotations-only.',.90,6.19,11.9,.51,17,True,C_TEAL)

# Slide 6
s=slide('Results','What did the model achieve?', f'Original new-cell RMSE {newcell:.3f} versus {cellbase:.3f} per-drug mean. Original cold-drug RMSE {newdrug:.3f} versus {annotation_drug:.3f} annotations-only. Revised CID-grouped drug RMSE {chem_drug:.3f} versus {chem_drug_ann:.3f}, and joint cold RMSE {chem_both:.3f} versus {chem_both_ann:.3f}. All are internal results; the first two use the original normalized-name grouping.')
chart(s,'01_full_kfold_rmse.png',.74,2.01,8.12,4.63)
shape(s,9.10,2.02,3.53,2.12,C_WHITE,True)
text(s,f'{newcell:.3f}',9.34,2.31,3.01,.70,35,True,C_TEAL);text(s,f'Unseen-cell RMSE\nvs {cellbase:.3f} baseline',9.35,3.25,2.97,.72,17,False,C_MUTED)
shape(s,9.10,4.40,3.53,2.25,C_WHITE,True)
text(s,f'{chem_drug:.3f}',9.34,4.74,3.01,.68,35,True,C_TEAL);text(s,f'CID-grouped drug RMSE\nvs {chem_drug_ann:.3f} annotation-only',9.35,5.65,2.97,.79,16,False,C_MUTED)

text(s,'Figure: original normalized-name folds. CID-grouped result shown at right.',.90,6.83,11.80,.25,10,False,C_MUTED)

# Slide 7
s=slide('Meaning of accuracy','Pooled correlation is not the whole story', 'The pooled unseen-cell correlation is high, but it can reward predicting average drug effects. The median Pearson correlation calculated separately for each drug is only 0.449, a fairer measure of differences between cell lines for the same drug. The model beats the per-drug mean on 229 of 286 drug groups, but fails on 57.')
chart(s,'02_within_drug_pearson.png',.74,2.04,7.74,4.59)
shape(s,8.75,2.13,3.80,1.93,C_WHITE,True);text(s,f'{within:.3f}',9.02,2.51,3.22,.70,39,True,C_TEAL);text(s,'Median within-drug r',9.02,3.32,3.21,.37,16,False,C_MUTED)
shape(s,8.75,4.28,3.80,2.35,C_WHITE,True);text(s,f'{beats}/286',9.02,4.68,3.25,.74,33,True,C_TEAL);text(s,'Drugs with lower RMSE\nthan their training mean',9.03,5.57,3.26,.66,16,False,C_MUTED)

# Slide 8
s=slide('Error analysis','Which drugs remain difficult?', 'Some MEK inhibitors are still poorly predicted compared with their within-drug means, despite distinct input identities for familiar drugs. We show selumetinib, trametinib and PD0325901 next to the Nutlin-3a and palbociclib examples. A good global metric cannot guarantee good estimates for each compound.')
chart(s,'04_mek_and_biomarker_drugs.png',.72,2.00,8.95,4.84)
shape(s,9.91,2.20,2.71,4.29,C_WHITE,True);text(s,'Key observation',10.13,2.68,2.30,.63,22,True,C_INK);text(s,'A known drug identity does not ensure accurate variation between cell lines.\n\nMEK inhibitors remain challenging.',10.15,3.66,2.23,2.36,16,False,C_MUTED)

# Slide 9
s=slide('Biological discussion','Can genomic variables explain some responses?', 'We predeclared Nutlin-3a/TP53, selumetinib and trametinib/KRAS-BRAF, palbociclib/RB1 and taselisib/PIK3CA. Each drug had its own held-out-cell Ridge model. In Nutlin-3a the TP53 mutation-and-copy-number group showed predictive reliance, consistent with published MDM2-p53 pharmacology. Permutation importance is not proof of causality.')
chart(s,'06_drug_specific_gene_permutation.png',.75,2.00,8.69,4.81)
shape(s,9.71,2.18,2.88,4.45,C_WHITE,True)
text(s,'Nutlin-3a / TP53',9.94,2.62,2.34,.95,22,True,C_TEAL)
text(s,'Mechanism-linked model reliance on held-out cells.\n\nBiological association, not a validated treatment biomarker.',9.94,3.88,2.31,2.39,16,False,C_MUTED)

# Slide 10
s=slide('Integrity, discussion & limitations','What can we conclude safely?', 'State conclusions and limits explicitly. Genomic profiles and drug features improve regression within the matched preclinical cohort, but performance differs by compound. Assumptions: matched identities and source target/pathway labels are reliable. Limitations: mismatched data-release years, incomplete drug structure canonicalization, only single-dataset cross-validation, and no clinical/wet-lab validation. Final answer is a robust college-level proof of concept, not a clinical prediction tool.',True)
for i,(head,body) in enumerate([('Finding','Better than appropriate baselines on both grouped holdout tasks; within-drug performance varies.'),('Assumptions','Cross-source cell IDs and supplied drug target/pathway annotations are meaningful.'),('Limitations & uncertainty','Scientific risks: 50 missing CIDs; 2 chemical-form conflicts; retrospective tests; unverified salt forms; source-year mismatch; no external screen.')]):
    x=.72+i*4.15;shape(s,x,2.16,3.87,3.73,'244056',True);shape(s,x+.19,2.42,.42,.06,C_AMBER if i==2 else C_TEAL)
    text(s,head,x+.21,2.73,3.44,.69,21,True,C_WHITE);text(s,body,x+.21,3.74,3.41,1.94,16,False,'D6E5EF')
shape(s,.75,6.13,11.83,.56,'195050',True);text(s,'Nonclinical research; incomplete chemical aliases; AI-assisted development disclosed.',1.01,6.29,11.15,.30,16,True,C_WHITE)

assert len(ppt.slides)==10
PPT_PATH=DELIVERABLES/'research_presentation.pptx';ppt.save(PPT_PATH)

# Simple short, independent talk track: roughly 5 to 6 minutes, not the full report.
SPEECH=[
('1 | Research question','Cancer cells do not always respond the same way to a drug. This project tests whether we can use their gene-expression and genomic profiles to predict those differences. We analyzed 649 laboratory cancer cell lines, 286 drug groups and 160,512 measured responses. This is a preclinical machine-learning study, not a method for choosing treatment for patients.'),
('2 | What we predict','We describe each cell using RNA gene-expression measurements, recorded DNA variants and copy-number values. We combine those with information about each drug. The model predicts LN_IC50, a logarithm of the concentration associated with 50 percent inhibition in the assay. A lower fitted IC50 generally means a lower effective concentration in that laboratory test.'),
('3 | Data preparation','Our molecular files came from CCLE and DepMap, while drug responses came from GDSC2. We matched cell identities across the files, checked for naming conflicts and consolidated duplicate numeric drug identifiers under source-normalized drug names. The result contains 160,512 observed cell-and-drug pairs. We did not fill in unmeasured response values.'),
('4 | Model pipeline','The pipeline first separates training and test entities. Using training cells only, we select 100 expression features and include a fixed ten-gene mutation and copy-number panel. Drug identities help distinguish familiar compounds, while target and pathway labels allow some information transfer to new compounds. A Ridge model is fitted with training-only preprocessing.'),
('5 | Fair evaluation',f'Original five-fold tests hold out cells, drug names, or both. Our new independent fold assignment uses source-linked PubChem identities, keeping same-CID names plus two putative active-parent alias pairs together. The drug-identity grouped tests cover all 160,512 pairs, and the paired both-cold intersections test {chem_n:,} measured pairs. Fifty unverified identifiers remain provisional and the revised folds are post-hoc robustness checks.'),
('6 | Main results',f'Unseen-cell RMSE is {newcell:.3f} versus {cellbase:.3f} per-drug mean. The original new-drug RMSE is {newdrug:.3f} versus {annotation_drug:.3f} annotations-only. With known PubChem CIDs held out together, new-drug RMSE is {chem_drug:.3f} versus {chem_drug_ann:.3f}, and joint new-cell and new-drug RMSE is {chem_both:.3f} versus {chem_both_ann:.3f}. No external screen was used.'),
('7 | Within-drug results',f'One warning is that a high pooled correlation can mostly reflect differences among drugs. Calculating correlation separately within each drug gives a median of {within:.3f}. The model performs better than the per-drug average for {beats} of 286 compounds, but not for every drug. This is why our report does not rely on a single pooled score.'),
('8 | Errors','The model still struggles with specific compounds, including several MEK inhibitors. Their response differences across cell lines are not always captured better than by a simple training mean. Showing these failures is important: improving the overall score does not mean all drugs are predicted accurately.'),
('9 | Biology','We also examined selected gene-drug relationships. For Nutlin-3a, shuffling TP53 mutation and copy-number information made held-out predictions worse. This agrees in direction with established p53 and MDM2 pharmacology, but it only shows model reliance and an association. It does not prove that a gene mutation causes resistance in a patient.'),
('10 | Conclusion','In summary, gene expression and genomic features provide useful predictive information for laboratory drug-response data, especially for new cells treated with familiar drugs. The study has limits: the data releases differ in age, 50 drug codes have no reliable linked CID and tested chemical forms are unverified, joint cold coverage is 19.9 percent of measured pairs, and there is no independent validation. Our conclusion is therefore a reproducible mini-project result, not a treatment recommendation. Thank you.')]
sp=Document();s=sp.sections[0];s.top_margin=DIn(.72);s.bottom_margin=DIn(.65);s.left_margin=s.right_margin=DIn(.86)
sp.styles['Normal'].font.name='Aptos';sp.styles['Normal'].font.size=DPt(10.7);sp.styles['Normal'].paragraph_format.space_after=DPt(7)
sp.add_heading('Mini-project 27 | Presentation script',0)
sp.add_paragraph('Short slide-by-slide speech | Approximately 5-6 minutes at a measured pace.')
for label,body in SPEECH:
    sp.add_heading('Slide '+label,2)
    sp.add_paragraph(body)
sp.save(DELIVERABLES/'speaker_notes.docx')

if __name__=='__main__':
    report_path=report()
    print('REPORT_DOCX',report_path)
    print('PRESENTATION_PPTX',PPT_PATH,'slides',len(ppt.slides))
    print('SPEAKER_NOTES_DOCX',DELIVERABLES/'speaker_notes.docx','sections',len(SPEECH))

    # Export reading copies after generating all editable originals.
    import subprocess
    originals=[DELIVERABLES/'scientific_report.docx',PPT_PATH,DELIVERABLES/'speaker_notes.docx']
    cmd=['libreoffice','-env:UserInstallation=file:///tmp/lo_challenge27_documents',
         '--headless','--convert-to','pdf','--outdir',str(DELIVERABLES)]+[str(x) for x in originals]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
    if proc.returncode:
        raise RuntimeError('PDF export failed: '+proc.stderr[-900:])
    for orig in originals:
        assert orig.with_suffix('.pdf').is_file(),str(orig)
    print('PDF_EXPORT_OK',proc.stdout.strip())
