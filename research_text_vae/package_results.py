from pathlib import Path
import hashlib,importlib.metadata,json,platform,zipfile
H=Path(__file__).resolve().parent
if __name__=='__main__':
    packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib','scikit-learn','threadpoolctl','torch']}
    (H/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages=packages),indent=2))
    allowed={'.py','.md','.json','.csv','.pt','.pdf','.png','.tex','.txt'}
    files=[p for p in H.rglob('*') if p.is_file() and p.suffix in allowed
        and p.name not in ['ptb.train.txt','ptb.valid.txt','report_rendered.txt']
        and not p.name.startswith('report_page') and '__pycache__' not in p.parts]
    files+=list((H.parent/'code/actdim').rglob('*.py'))
    files=sorted(set(files));target=H.parent/'text_vae_nlp_results.zip';manifest=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            content=p.read_bytes();name=p.relative_to(H.parent).as_posix()
            z.writestr(name,content);manifest.append(hashlib.sha256(content).hexdigest()+'  '+name)
        z.writestr('MANIFEST.sha256','\n'.join(manifest)+'\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for line in z.read('MANIFEST.sha256').decode().splitlines():
            digest,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==digest
    print(target,round(target.stat().st_size/2**20,2),'MiB;',len(files),'files verified')
