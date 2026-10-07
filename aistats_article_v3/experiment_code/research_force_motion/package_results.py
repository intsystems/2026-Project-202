"""Compact reproducibility bundle; preserve every raw experiment locally."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
import numpy as np

H=Path(__file__).resolve().parent
env=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
    packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl']},
    BLAS_threads=1)
(H/'environment.json').write_text(json.dumps(env,indent=2))
for f in (H/'results_chaotic'/'seed_7').glob('rollout_*.npz'):
    d=np.load(f)
    np.savez_compressed(f.with_name(f.stem.replace('rollout','scalar_observers')+'.npz'),
        y=np.tanh(d['states'][:,:3]),z=d['z'],t=d['t'])
allowed={'.py','.md','.csv','.json','.pdf','.png'}
paths=[]
for f in H.rglob('*'):
    if not f.is_file() or '__pycache__' in f.parts or f.name.startswith('check_page') or f.name=='bundle_manifest.json':continue
    if f.suffix in allowed:paths.append(f)
    elif f.name.startswith('lyapunov_') and f.suffix=='.npz':paths.append(f)
    elif f.parent.name=='seed_7' and f.name.startswith(('checkpoint_','scalar_observers_')):paths.append(f)
paths.extend(f for f in (H.parent/'code'/'actdim').rglob('*.py') if '__pycache__' not in f.parts)
paths.append(H/'report_ru.tex')
paths=sorted(set(paths))
manifest={str(f.relative_to(H.parent)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
(H/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=H.parent/'force_motion_report_and_sources.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in paths:z.write(f,f.relative_to(H.parent))
    z.write(H/'bundle_manifest.json','research_force_motion/bundle_manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(archive,round(archive.stat().st_size/1024**2,2),'MiB',len(paths),'files')
