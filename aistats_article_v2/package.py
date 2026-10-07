from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent
files=[]
for p in H.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts:continue
    rel=p.relative_to(H)
    # Internal legacy source contains author metadata and is not part of this independent edition.
    if rel.as_posix()=='evidence/aistats_article/aistats2027.tex':continue
    if p.name.startswith(('check-','preview-')) or p.name in ['package_manifest.json','review_layout.txt']:continue
    if p.suffix in {'.tex','.sty','.bib','.pdf','.py','.md','.csv','.json'}:files.append(p)
manifest=[dict(path=p.relative_to(H).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
m=H/'package_manifest.json';m.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
out=H/'aistats2027_v2_sources.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(files)+[m]:z.write(p,'aistats_article_v2/'+p.relative_to(H).as_posix())
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for row in manifest:assert hashlib.sha256(z.read('aistats_article_v2/'+row['path'])).hexdigest()==row['sha256']
print(f'{len(files)+1} files, {out.stat().st_size:,} bytes; ZIP and manifest verified.')
