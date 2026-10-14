"""Create an independently readable scientific PDF using only checked project outputs."""
from pathlib import Path
import pandas as pd
import numpy as np
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether,HRFlowable)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'REPORT.pdf'
navy=colors.HexColor('#15273C'); teal=colors.HexColor('#147E75'); grey=colors.HexColor('#566779'); light=colors.HexColor('#EDF4F7')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CoverTitle2',fontName='Helvetica-Bold',fontSize=27,leading=31,textColor=navy,spaceAfter=13))
styles.add(ParagraphStyle(name='CoverSubtitle2',fontName='Helvetica',fontSize=13,leading=18,textColor=grey,spaceAfter=16))
styles.add(ParagraphStyle(name='Sect2',fontName='Helvetica-Bold',fontSize=17,leading=22,textColor=navy,spaceBefore=10,spaceAfter=9,keepWithNext=True))
styles.add(ParagraphStyle(name='Subsect2',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=teal,spaceBefore=10,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10.2,leading=15.2,spaceAfter=9,textColor=navy))
styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8.5,leading=12,spaceAfter=7,textColor=grey))
styles.add(ParagraphStyle(name='Legend2',fontName='Helvetica',fontSize=9,leading=13,spaceAfter=8,textColor=grey,alignment=TA_CENTER))
styles.add(ParagraphStyle(name='Bullet2',fontName='Helvetica',fontSize=9.8,leading=14,leftIndent=11,firstLineIndent=-9,spaceAfter=6,textColor=navy))

W,H=A4
PAGE_WIDTH=W-96
story=[]

def p(txt,style='Body2'):
 story.append(Paragraph(txt,styles[style]))
def head(txt):p(txt,'Sect2')
def sub(txt):p(txt,'Subsect2')
def bullet(txt):p('&#8226; '+txt,'Bullet2')
def fig(filename,width=PAGE_WIDTH,max_height=250,caption=''):
 from PIL import Image as PIL
 file=ROOT/filename
 with PIL.open(file) as im:tw,th=im.size
 ratio=min(width/tw,max_height/th)
 story.append(Image(str(file),width=tw*ratio,height=th*ratio))
 if caption:p(caption,'Legend2')
def table(rows,widths=None):
 from reportlab.lib import colors
 headerstyle=ParagraphStyle('HeaderCell',parent=styles['Small2'],textColor=colors.white,fontName='Helvetica-Bold')
 data=[[Paragraph(str(x),headerstyle if i==0 else styles['Small2']) for x in row] for i,row in enumerate(rows)]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([
 ('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),
 ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),
 ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CEDBE1')),
 ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
 ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),8)]))
 story.append(t);story.append(Spacer(1,12))
def page():story.append(PageBreak())

def decor(c,doc):
 c.saveState()
 c.setFillColor(navy);c.rect(0,H-28,W,28,fill=1,stroke=0)
 c.setFont('Helvetica-Bold',8.3);c.setFillColor(colors.white)
 c.drawString(48,H-18,'CHALLENGE 27  /  GENE EXPRESSION + DRUG SENSITIVITY')
 c.setStrokeColor(colors.HexColor('#D9E4EB'));c.line(48,41,W-48,41)
 c.setFillColor(grey);c.setFont('Helvetica',8)
 c.drawString(48,27,'Executed project | fixed random seed 27 | retrospective cell-line data')
 c.drawRightString(W-48,27,f'Page {doc.page}')
 c.restoreState()

metrics=pd.read_csv(ROOT/'metrics/model_metrics.csv') if (ROOT/'metrics/model_metrics.csv').exists() else pd.read_csv(ROOT/'metrics/outer_test_metrics.csv')
assert len(metrics)==4
p('Gene-Expression-Based<br/>Drug Sensitivity Prediction','CoverTitle2')
p('Challenge 27 | Scientific mini-project report | 13 October 2026','CoverSubtitle2')
p('<b>Question:</b> Can a small, leakage-controlled model estimate cancer cell-line drug sensitivity from transcriptomic and genomic features, and does that prediction generalize to a cell line or a compound not present during training?')
sub('Executive summary')
p('The study uses 4,000 measured GDSC2 LN_IC50 responses representing a coverage-selected, identity-matched matrix of 200 CCLE cell lines and 20 drug groups. The model uses 100 training-selected expression features, 10 mutation flags, 10 copy-number values, and source target/pathway annotations. Two separate entity-disjoint tests compare a global training mean with a 100-tree Random Forest.')
table([
 ['Held-out entity','Training mean RMSE','Random Forest RMSE','Pearson r (forest)'],
 ['Cell line (40, n=800)','3.0272','1.4934','0.8713'],
 ['Drug group (5, n=1,000)','3.2793','3.6344','0.3912'],
],[135,115,123,PAGE_WIDTH-373])
p('<b>Principal result:</b> the Random Forest improves unseen-cell RMSE by 50.7% compared with a global training mean, but performs 10.8% worse for unseen compounds. This is not evidence of successful new-drug generalization. A training-only per-drug mean is a substantially stronger unseen-cell baseline (RMSE 1.6392), narrowing the forest improvement to 8.9%.')
fig('report_assets/stronger_baselines.png',width=PAGE_WIDTH,max_height=235,caption='All bar heights were recalculated from saved, actual test responses. Lower RMSE is better.')
page()
head('1. Scientific background and data rationale')
p('A drug-sensitivity screen measures how cultured cancer cell lines respond to compounds. IC50 is the fitted half-maximal inhibitory concentration; the supplied target is its natural logarithm, LN_IC50. Lower observed values generally correspond to lower fitted inhibitory concentrations under the screening assay, not a clinical drug recommendation. Negative logarithms are valid and were preserved.')
p('Gene expression measures abundance of RNA transcripts. Copy number provides numeric gene-level dosage measurements. Mutation data provide recorded variants, which were reduced to binary coding-variant indicators. These features describe different aspects of a cell line but do not directly prove drug mechanism.')
sub('Source inventory and reconciliation')
table([
 ['Source','Records / dimensions','Role'],
 ['DepMap / CCLE Public 19Q4','1,249 expression lines × 58,676 genes','Cell line transcription'],
 ['CCLE mutation records','1,253,648 variant records','Ten-gene indicators'],
 ['CCLE copy number','1,682 lines × 27,639 genes','Ten-gene numeric CN'],
 ['GDSC2 fitted responses','242,036 fitted response records','LN_IC50, target/pathway'],
 ],[150,180,PAGE_WIDTH-330])
p('From the uploaded ZIP, six expected members passed CRC and SHA-256 integrity checks. The GDSC2 XLSX advertises a false A1 worksheet dimension; streaming actual rows recovered the 242,036 measurements. Exact COSMIC matching across omics produced 659 candidate cell lines. Conservative name/Sanger exclusions left 649 eligible cells before coverage-based selection.')
fig('figures/01_response_histogram.png',width=PAGE_WIDTH,max_height=218,caption='Exploratory distribution of the 4,000 observed source LN_IC50 values, not model predictions.')
page()
head('2. Curation, feature engineering and leakage control')
bullet('<b>Selection:</b> 20 high-coverage drug groups were chosen without using response magnitude; duplicate source drug IDs under the same normalized name were grouped. Then 200 eligible cell lines were chosen by coverage, giving 4,000 observed pairs with no response imputation.')
bullet('<b>Expression:</b> the 100 most variable columns were chosen separately using unique cells from each training partition, rather than using the full dataset or the labels.')
bullet('<b>Ten-gene panel:</b> TP53, KRAS, BRAF, EGFR, PIK3CA, PTEN, APC, RB1, MYC and CDKN2A. A zero mutation indicator means no retained variant call under the filtering rule; it does not establish wild type.')
bullet('<b>Drug descriptors:</b> known source PUTATIVE_TARGET tokens and PATHWAY_NAME categories were vectorized on training annotations only. These are presumed known for held-out compounds; no drug ID is included as a feature.')
sub('Explicit held-out-entity design')
table([
 ['Experiment','Inner train','Inner validation','Outer test','Final fitting'],
 ['New cell lines','128 cells','32 cells','40 cells, 800 pairs','160 non-test cells'],
 ['New drug groups','12 drugs','3 drugs','5 drugs, 1,000 pairs','15 non-test drugs'],
 ],[117,79,95,110,PAGE_WIDTH-401])
p('Final test entities were excluded from all fitting and final mean calculations. Inner validation used training-core-only fits. Final checkpoints were then refitted on train core + validation, excluding the outer test. Known source IDs and selected documented drug synonyms were checked for overlap. Comprehensive molecule/InChIKey alias reconciliation remains unfinished.')
fig('figures/10_response_heatmap.png',width=PAGE_WIDTH,max_height=219,caption='Complete 200 × 20 matrix of observed responses in the curated, coverage-selected subset.')
page()
head('3. Predictive models and original measured test results')
p('The original pipeline compared a global mean of training LN_IC50 observations with a scikit-learn Random Forest of 100 trees, maximum depth 12, minimum leaf size 3, max feature fraction 0.7 and fixed seed 27. Both retained fitted median preprocessing. The chosen expression columns and drug-annotation encoder were serialized alongside each model.')
table([
 ['Outer evaluation','Model','RMSE ↓','MAE ↓','Pearson r'],
 ['Unseen cells','Training mean','3.0272','2.4612','Undefined'],
 ['Unseen cells','Random Forest','1.4934','1.1395','0.8713'],
 ['Unseen drugs','Training mean','3.2793','2.6731','Undefined'],
 ['Unseen drugs','Random Forest','3.6344','2.7307','0.3912'],
 ],[108,115,72,72,PAGE_WIDTH-367])
p('The mean baseline has constant outputs, so Pearson correlation is mathematically undefined, not zero. The observed new-drug loss is dominated in part by staurosporine (RF MAE 6.757; global mean MAE 4.625 for that group), underscoring heterogeneous compound difficulty.')
fig('figures/03_holdout_rmse.png',width=PAGE_WIDTH,max_height=225,caption='Original baseline versus Random Forest on separate, fully held-out entity experiments.')
page()
head('4. Robust baselines, validation selection and uncertainty')
p('To address rubric concerns about weak comparators, we computed stronger task-aware baselines. For new cells, each familiar drug can use its own <b>training-only mean</b>. For new drugs, a 100-observation-shrunk training pathway mean is possible when source pathway annotation is supplied. These comparisons reuse previously examined outer test entities and are explicitly <b>post-hoc</b>.')
table([
 ['Outer test','Global mean','Stronger mean','Original RF'],
 ['New cells (RMSE)','3.0272','1.6392 (per drug)','1.4934'],
 ['New drugs (RMSE)','3.2793','3.8158 (pathway)','3.6344'],
 ],[122,96,134,PAGE_WIDTH-352])
p('A fixed small candidate comparison on <b>inner validation only</b> considered training mean, source-aware mean, Ridge at strengths 100 and 1000, and the original forest. Inner-validation selection retained the original forest for unseen cells (RMSE 1.6289), but preferred the global mean for new compounds (RMSE 3.7485 versus forest 4.8704). This policy changes which prediction is recommended, not the recorded original model performance.')
sub('Conditional cluster-bootstrap uncertainty')
p('Two thousand fixed-seed resamples of whole held-out entities compared RF RMSE minus the task-aware baseline. For unseen cells the observed difference is −0.1458, with 95% percentile interval [−0.2862, +0.0156]. For unseen drugs versus the pathway mean the difference is −0.1814, interval [−1.5449, +0.5099]. Both intervals include zero. With only five test-drug groups, this bootstrap cannot replace independent cold-drug test splits.')
fig('report_assets/conditional_bootstrap_intervals.png',width=PAGE_WIDTH,max_height=221,caption='Conditional RMSE differences on the existing, previously examined test. Negative values favour forest.')
page()
head('5. Model interpretation, error analysis and learning behaviour')
p('On 40 held-out MG-132 cell lines, eight-repeat genomic permutation importance measured the change in RMSE when selected features or groups were shuffled. Jointly perturbing all 100 expression features raised RMSE by mean 0.113 (repeat SD 0.047). Most individual mutation/copy-number effects were near zero; these are fitted-model reliance measures, not causal biomarkers.')
fig('figures/08_mg132_permutation_importance.png',width=PAGE_WIDTH,max_height=218,caption='MG-132-only drug-specific permutation importance. Bars show test RMSE change with variability across permutations.')
p('A separate within-training-pool validation curve fitted increasing numbers of cell lines, 20, 40, 80 and 128, against a fixed 32-cell inner validation group. Its forest used 70 trees, so it is a learning-size diagnostic rather than an exact refit of the final 100-tree model. Improvements were non-monotonic.')
fig('figures/09_learning_curve.png',width=PAGE_WIDTH,max_height=200,caption='Internal learning curve, not a repeated assessment of the final outer test set.')
page()
head('6. Integrity, limitations and recommendations')
sub('What was actually verified')
bullet('Uploaded archive CRC and hashes; selected dataset dimensions; cell/drug identity checks; exactly 4,000 measured pairs; complete selected numeric finiteness.')
bullet('Entity-disjoint splits, known aliases, training-only feature selection, four saved fitted model bundles, reproducible model predictions and recomputed test metrics. A prior submission passed 19 automated tests, and this supplement adds standalone repeatable code and measured CSV artifacts.')
bullet('An independent raw-source regeneration yielded byte-identical response, CN and mutation CSVs and exactly equal expression values, cell IDs and feature headers.')
sub('Scientific risks not eliminated')
bullet('Only one fixed outer test assignment and five held-out drug groups: no confidence intervals across independent compound choices. Conditional bootstrap intervals here are not a substitute.')
bullet('Synonym coverage is incomplete without full chemical structure/InChIKey harmonization. The genomic release is from 2019, while GDSC drug response is from 2023.')
bullet('Fitted IC50 may extrapolate beyond tested concentrations. A zero mutation flag is not confirmed wild type. Drug target annotations may be unavailable in a real zero-shot setting.')
bullet('The original outer test was previously viewed before the supplement: added baseline comparisons are transparent exploratory analyses and should not be reported as fresh blind validation.')
sub('Conclusions and highest-value next steps')
p('The Random Forest offers real error reduction for new cell lines receiving familiar drugs, including improvement over a drug-aware baseline. It does not beat a simple global mean for unseen compounds. A scientifically appropriate next benchmark would include multiple fully independent cold-drug splits, complete compound structural deduplication, stronger chemistry-based drug features, and a separately held final blind test. This report makes no patient-level prediction or clinical treatment recommendation.')
sub('Source and artifact references')
p('1. Source data: uploaded DepMap Public 19Q4 README, preserved as `data/source_depmap_README.txt`; GDSC2 fitted dose-response Oct 2023.\n2. GDSC drug data: https://www.cancerrxgene.org/downloads/drug_data\n3. DepMap data: https://depmap.org/portal/data_page/\n4. Project tables: `metrics/`, `predictions/`, `splits/`, `checkpoints/`, `PROVENANCE.md`, `LIMITATIONS.md`.', 'Small2')

if __name__=='__main__':
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=54,bottomMargin=56,
            title='Challenge 27: Gene-Expression-Based Drug Sensitivity Prediction',author='Challenge 27 project',allowSplitting=1)
    doc.build(story,onFirstPage=decor,onLaterPages=decor)
    print('PDF created:',OUT,'bytes:',OUT.stat().st_size)
