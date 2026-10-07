from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
roots=[R/'EXPERIMENT_SEARCH_FINAL.md',R/'research_mg_real_forecast',R/'research_mg_deployment',R/'research_mg_certification',R/'research_mg_ucr',R/'research_mg_har',R/'research_mg_interventions']
files=[]
for root in roots:
 if root.is_file():files.append(root);continue
 for p in root.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts:continue
  if p.suffix in {'.raw','.zip','.npz','.pt','.pkl'} and p.stat().st_size>8_000_000:continue
  if p.suffix in {'.py','.md','.csv','.json','.pdf','.png','.pkl','.npz'}:files.append(p)
manifest=[dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))]
out=R/'mg_search_campaign_final.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(set(files)):z.write(p,p.relative_to(R).as_posix())
 z.writestr('mg_search_campaign_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for e in manifest:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
print(len(manifest),out.stat().st_size)
