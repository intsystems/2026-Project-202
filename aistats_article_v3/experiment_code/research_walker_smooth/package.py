from pathlib import Path
import hashlib,json,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parent;OLD=ROOT/'research_walker_repair'

def main():
    files=[]
    skip={'.pyc','.aux','.log','.out'}
    for p in H.rglob('*'):
        if p.is_file() and p.suffix not in skip and '__pycache__' not in p.parts and p.name not in ['MANIFEST.json','archive_check.json'] and not p.name.startswith('report_page-'):files.append(p)
    for p in (ROOT/'code/actdim').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:files.append(p)
    for name in ['features.py','motion.py','train.py','perturb.py','build_report.py','PROTOCOL.md','all_seeds.csv','requirements.txt']:
        files.append(OLD/name)
    files+=list((OLD/'anchor').glob('*'))
    for seed in range(211,216):
        root=OLD/f'seed{seed}';files.append(root/'pair.json')
        for step in [0,1048576]:
            cp=root/f'step{step:07d}'
            for name in ['policy.zip','normalize.pkl']:files.append(cp/name)
            for d in cp.glob('reset51*'):
                for name in ['metrics.json','trajectory.npz']:
                    if (d/name).exists():files.append(d/name)
    files=sorted(set(files));manifest=[]
    for p in files:manifest.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (H/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));archive=ROOT/'research_walker_smooth_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files+[H/'MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    result=dict(archive=str(archive),files=len(files)+1,bytes=archive.stat().st_size,all_sha256_and_crc_passed=True)
    (H/'archive_check.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
