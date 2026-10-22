#!/usr/bin/env python3
"""Download legally available open-access papers used by MARE-FL.

The script NEVER bypasses paywalls. For each catalog entry it tries:
1) a curated direct OA PDF URL;
2) OpenAlex best_oa_location / primary_location;
3) arXiv DOI fallback when applicable.
If no legal OA PDF is found, it writes the DOI/official landing page to
closed_or_unresolved.csv so the paper can be accessed through a library.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
from urllib.parse import quote
import requests

UA='MARE-FL-literature-fetcher/1.0 (research reproducibility; respects OA only)'

def safe_name(s):
    s=re.sub(r'[^A-Za-z0-9._-]+','_',s).strip('_')
    return s[:160] or 'paper'

def looks_pdf(resp):
    ct=(resp.headers.get('content-type') or '').lower()
    return 'pdf' in ct or resp.content[:5]==b'%PDF-'

def download(url, out, timeout=45):
    r=requests.get(url,headers={'User-Agent':UA},timeout=timeout,allow_redirects=True)
    r.raise_for_status()
    if not looks_pdf(r):
        return False, f'not-pdf:{r.headers.get("content-type","")}'
    out.write_bytes(r.content)
    return True, f'{len(r.content)} bytes'

def openalex_pdf(doi):
    if not doi: return None
    url='https://api.openalex.org/works/https://doi.org/'+quote(doi,safe='')
    r=requests.get(url,headers={'User-Agent':UA},timeout=30)
    if r.status_code!=200: return None
    j=r.json()
    locs=[]
    for key in ('best_oa_location','primary_location'):
        v=j.get(key)
        if isinstance(v,dict): locs.append(v)
    locs += [x for x in (j.get('locations') or []) if isinstance(x,dict)]
    for loc in locs:
        if loc.get('is_oa') and loc.get('pdf_url'):
            return loc['pdf_url']
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--catalog',default='papers_catalog.csv')
    ap.add_argument('--out',default='pdfs')
    ap.add_argument('--all',action='store_true',help='download every catalog entry (default behavior)')
    ap.add_argument('--only',action='append',default=[],help='key to fetch; repeatable')
    ap.add_argument('--sleep',type=float,default=0.15)
    a=ap.parse_args()
    base=Path(__file__).resolve().parent
    cat=Path(a.catalog); cat=cat if cat.is_absolute() else base/cat
    out=Path(a.out); out=out if out.is_absolute() else base/out; out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(cat.open(encoding='utf-8')))
    if a.only: rows=[r for r in rows if r['key'] in set(a.only)]
    status=[]; unresolved=[]
    for n,r in enumerate(rows,1):
        key=r['key']; title=r['title']; doi=r.get('doi','').strip(); direct=r.get('direct_oa_pdf','').strip()
        target=out/(safe_name(key+'_'+title)+'.pdf')
        if target.exists() and target.stat().st_size>1000:
            status.append([key,'exists',target.name,'']); print(f'[{n}/{len(rows)}] {key}: exists'); continue
        candidates=[]
        if direct: candidates.append(('curated_oa',direct))
        try:
            oa=openalex_pdf(doi)
            if oa and oa not in [u for _,u in candidates]: candidates.append(('openalex_oa',oa))
        except Exception as e:
            pass
        ok=False; detail='no OA PDF located'
        for source,url in candidates:
            try:
                good,msg=download(url,target)
                if good:
                    status.append([key,'downloaded',target.name,source]); print(f'[{n}/{len(rows)}] {key}: downloaded ({source})'); ok=True; break
                detail=msg
            except Exception as e:
                detail=f'{type(e).__name__}: {e}'
        if not ok:
            landing=('https://doi.org/'+doi) if doi else ''
            status.append([key,'not_downloaded','',detail])
            unresolved.append([key,title,doi,landing,detail])
            print(f'[{n}/{len(rows)}] {key}: no legal OA PDF found; saved landing link')
        time.sleep(a.sleep)
    with (base/'download_status.csv').open('w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerows([['key','status','filename','detail'],*status])
    with (base/'closed_or_unresolved.csv').open('w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerows([['key','title','doi','official_landing','detail'],*unresolved])
    print(f'\nDone: {sum(x[1] in ("downloaded","exists") for x in status)}/{len(status)} PDFs available locally.')
    print('No paywall bypass is attempted. See closed_or_unresolved.csv for official links.')

if __name__=='__main__': main()
