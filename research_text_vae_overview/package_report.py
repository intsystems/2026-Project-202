"""Package report and numerical evidence without checkpoints/corpus duplicates."""
from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent
ROOT=H.parent
S=ROOT/'research_text_vae'
files=[]
for p in H.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts:continue
    if p.name.startswith('preview-') or p.name in {'archive_manifest.json','report_extracted.txt'}:continue
    if p.suffix in {'.md','.py','.pdf','.tex','.csv','.json','.png'}:files.append(p)
for p in S.rglob('*'):
    if not p.is_file():continue
    parts=set(p.relative_to(S).parts)
    if parts & {'initial_three','before_protection','__pycache__'}:continue
    if p.suffix in {'.csv','.json','.md','.py'}:files.append(p)
files=sorted(set(files))
manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
               sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
mf=H/'archive_manifest.json'
mf.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
archive=H/'vae_complete_report.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files+[mf]:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for row in manifest:
        assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
print(f'{archive.name}: {len(files)+1} files, {archive.stat().st_size:,} bytes; ZIP and SHA256 checks passed.')
