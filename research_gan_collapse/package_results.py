from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
H=Path(__file__).resolve().parent
env=dict(python=sys.version,platform=platform.platform(),
    packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','torch','torchvision','scikit-learn','matplotlib','threadpoolctl']},
    gpu_used=False,training_threads_recorded=4,matched_benchmark_threads_recorded=1)
(H/'environment.json').write_text(json.dumps(env,indent=2))
files=[]
for f in H.rglob('*'):
    if not f.is_file() or any(part in ['data','__pycache__'] for part in f.relative_to(H).parts):continue
    if f.name.startswith('check_page') or f.name=='bundle_manifest.json':continue
    if f.suffix in {'.py','.md','.json','.csv','.npz','.pdf','.png'}:files.append(f)
    elif f.name=='classifier.pt':files.append(f)
    elif f.suffix=='.pt' and (f.name=='checkpoint_16384.pt' or f.name=='checkpoint_12288.pt' or f.name=='checkpoint_06144.pt'):files.append(f)
files.extend(f for f in (H.parent/'code/actdim').rglob('*.py') if '__pycache__' not in f.parts)
files.append(H/'report_ru.tex');files=sorted(set(files))
manifest={str(f.relative_to(H.parent)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(H/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=H.parent/'gan_collapse_pilot_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in files:z.write(f,f.relative_to(H.parent))
    z.write(H/'bundle_manifest.json','research_gan_collapse/bundle_manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
print(archive,round(archive.stat().st_size/1024**2,2),'MiB;',len(files),'files; integrity and SHA256 passed.')
