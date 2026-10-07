from pathlib import Path
import hashlib,json,zipfile
from collect_evidence import BRANCHES
H=Path(__file__).resolve().parent;P=H.parent
def main():
    files={}
    for p in H.iterdir():
        if p.is_file() and p.suffix in ['.py','.md','.pdf','.tex','.csv','.json'] and p.name!='MANIFEST.json':
            files['overview/'+p.name]=p
    for branch in BRANCHES:
        root=P/f'research_walker_{branch}'
        for p in root.iterdir():
            if p.is_file() and p.suffix in ['.py','.md','.csv','.json','.txt'] and p.name!='MANIFEST.json':
                files[p.relative_to(P).as_posix()]=p
        for p in root.glob('*/train*.json'):files[p.relative_to(P).as_posix()]=p
    manifest=[dict(path=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(files.items())]
    (H/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    dest=P/'walker_complete_report.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,p in files.items():z.write(p,n)
        z.write(H/'MANIFEST.json','MANIFEST.json')
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for item in manifest:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    print(f'Verified {len(files)+1} files; {dest.stat().st_size} bytes; {dest.name}')
if __name__=='__main__':main()
