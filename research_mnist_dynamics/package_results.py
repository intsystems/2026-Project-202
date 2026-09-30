"""Compact reproducibility package; never remove the large local trajectories."""
from pathlib import Path
import hashlib,json,platform,zipfile,importlib.metadata
import numpy as np
import torch
from run import data,H

if __name__=='__main__':
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib','scikit-learn','threadpoolctl','torch','torchvision']}
    env=dict(python=platform.python_version(),platform=platform.platform(),packages=versions,
        hardware='Intel Core i5-12500H CPU',cuda_available=torch.cuda.is_available())
    (H/'environment.json').write_text(json.dumps(env,indent=2))
    x,y,tx,ty,train_ids=data();rng=np.random.default_rng(10299)
    a=np.concatenate([rng.choice(np.flatnonzero(y.numpy()==c),10,False) for c in range(10)])
    b=np.concatenate([rng.choice(np.flatnonzero(ty.numpy()==c),10,False) for c in range(10)])
    np.savez(H/'selection_ids.npz',train_official_ids=train_ids,train_probe_subset_ids=a,
        validation_official_test_ids=b,other_test_ids=np.setdiff1d(np.arange(len(tx)),b),
        small_parameter_ids=np.random.default_rng(718).choice(15018,128,False))
    allowed={'.py','.md','.txt','.csv','.json','.npy','.npz','.pt','.png','.pdf','.tex'}
    files=[p for p in H.rglob('*') if p.is_file() and p.suffix in allowed
        and p.name!='trajectory.npy' and not p.name.startswith('report_page')
        and p.name!='report_rendered.txt' and '__pycache__' not in p.parts]
    files.extend((H.parent/'code/actdim').rglob('*.py'))
    files=sorted(set(files));manifest=[]
    target=H.parent/'mnist_dynamics_experiment_results.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            name=p.relative_to(H.parent).as_posix();content=p.read_bytes()
            z.writestr(name,content);manifest.append(hashlib.sha256(content).hexdigest()+'  '+name)
        z.writestr('MANIFEST.sha256','\n'.join(manifest)+'\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for line in z.read('MANIFEST.sha256').decode().splitlines():
            digest,name=line.split('  ',1)
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
    print(f'{target}: {target.stat().st_size/2**20:.2f} MiB, {len(files)} files; verified.')
