from pathlib import Path
import hashlib,importlib.metadata,json,platform,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parent.parent
if __name__=='__main__':
    environment=dict(python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib','scikit-learn','threadpoolctl','torch','pypdf']})
    (H/'environment.json').write_text(json.dumps(environment,indent=2))
    allowed={'.py','.md','.json','.csv','.pt','.pdf','.png','.tex','.txt'}
    files=[p for p in H.rglob('*') if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts
        and not p.name.startswith('report_page') and p.name!='report_rendered.txt']
    files += [H.parent/'run.py',H.parent/'requirements.txt',H.parent/'data/selection.json']
    files += list((ROOT/'code/actdim').rglob('*.py'))
    target=ROOT/'text_vae_cyclical_results.zip';manifest=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):
            data=p.read_bytes();name=p.relative_to(ROOT).as_posix();z.writestr(name,data)
            manifest.append(hashlib.sha256(data).hexdigest()+'  '+name)
        z.writestr('MANIFEST.sha256','\n'.join(manifest)+'\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for line in z.read('MANIFEST.sha256').decode().splitlines():
            digest,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==digest
    print(target,round(target.stat().st_size/2**20,2),'MiB;',len(files),'files verified')
