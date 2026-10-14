"""Optional ONLINE *candidate* search for the 50 unknown assay codes.

Never treat unverified name-search hits as experimental assay identity. A compound
can be a different salt, ester, racemate, analog, formulation, or mixture.
Candidates are written separately and are not consumed by training/splitting.
"""
from __future__ import annotations
import argparse, json, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

AGENT={'User-Agent':'Challenge27-Research-Identity-Audit/1.0'}

def get_json(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=AGENT),timeout=12) as response:
            return json.load(response),''
    except (TimeoutError,OSError,ValueError,urllib.error.HTTPError) as exc:
        return {},f'{type(exc).__name__}: {exc}'


def candidates(name):
    q=urllib.parse.quote(str(name),safe='')
    url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON'
    data,error=get_json(url)
    out=[]
    for cid in data.get('IdentifierList',{}).get('CID',[])[:10]:
        out.append({'query':name,'source':'PubChem','candidate_id':str(cid),'source_url':f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}','query_status':'candidate_only_name_lookup_no_assay_proof','error':''})
    if not out:
        url2='https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q='+q+'&limit=10'
        d,e=get_json(url2)
        for x in d.get('molecules',[]):
            ident=x.get('molecule_chembl_id','')
            if ident:out.append({'query':name,'source':'ChEMBL','candidate_id':ident,'source_url':f'https://www.ebi.ac.uk/chembl/compound_report_card/{ident}/','query_status':'candidate_only_name_lookup_no_assay_proof','error':''})
        error=error or e
    if not out:
        out.append({'query':name,'source':'none','candidate_id':'','source_url':'','query_status':'unresolved','error':error})
    return out


def main(root,delay=0.25):
    d=Path(root)/'research/full_gdsc/identity_audit'
    raw=pd.read_csv(d/'drug_identity_review.csv')
    names=raw.loc[~raw.source_linked,'drug_group'].tolist()
    rows=[]
    for name in names:
        rows.extend(candidates(name));time.sleep(delay)
    out=d/'online_name_lookup_CANDIDATES_UNVERIFIED.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    print('Saved',out,'for',len(names),'unresolved names. NO candidate is trusted or used by the benchmark.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=ap.parse_args();main(a.root)
