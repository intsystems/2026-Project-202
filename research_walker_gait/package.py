"""Create a complete local artifact bundle, excluding environments and caches."""
from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent

def main():
    entries=[];paths=[]
    for base in [H,H.parent/'code'/'actdim']:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.aux','.out','.synctex.gz'] or p.name=='MANIFEST.json':continue
            rel=p.relative_to(H.parent).as_posix()
            entries.append(dict(path=rel,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));paths.append(p)
    manifest=H/'MANIFEST.json';manifest.write_text(json.dumps(entries,indent=2),encoding='utf-8');paths.append(manifest)
    target=H.parent/'research_walker_gait_results.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,p.relative_to(H.parent))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for entry in entries:
            assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
    print(json.dumps(dict(archive=str(target),files=len(paths),bytes=target.stat().st_size,integrity='all hashes and ZIP CRC passed'),indent=2))

if __name__=='__main__':main()
