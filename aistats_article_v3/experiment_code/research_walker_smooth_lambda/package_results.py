"""Compact source/data snapshot, SHA256 manifest, ZIP CRC verification."""
from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent
def main():
    selected={}
    for p in H.iterdir():
        if p.is_file() and p.suffix in ['.py','.md','.csv','.json','.tex','.pdf','.png','.txt']:
            if p.name=='MANIFEST.json' or p.name.startswith(('report_lambda_page','report_baselines_page','report_final_page')):continue
            selected[p.name]=p
    for p in H.glob('seed*/**/*'):
        if not p.is_file():continue
        if p.name in ['train.json','progress.jsonl','test.csv','validation.csv','metrics.json','action_MG_windows.csv','timecourse_MG_windows.csv']:
            selected[p.relative_to(H).as_posix()]=p
    selected['workspace_scripts/confirm_smooth_lambda.py']=H.parent/'confirm_smooth_lambda.py'
    entries=[dict(path=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(selected.items())]
    (H/'MANIFEST.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    dest=H.parent/'walker_smooth_lambda_results.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,p in selected.items():z.write(p,n)
        z.write(H/'MANIFEST.json','MANIFEST.json')
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for e in entries:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256'],e['path']
    print(f'{dest.name}: {len(entries)+1} files, {dest.stat().st_size} bytes, CRC and SHA256 verified')
if __name__=='__main__':main()
