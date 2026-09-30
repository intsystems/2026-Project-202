from pathlib import Path
import json,hashlib,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parent

def run():
    files=[]
    for name in ['research_walker_imitation','research_walker_phase','research_walker_phase_wide','code/actdim']:
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.aux','.log','.out','.lock'] and p.name not in ['MANIFEST.json','archive_check.json'] and not p.name.startswith('report_page-'):files.append(p)
    files+=list((ROOT/'research_walker_repair/anchor').glob('*'))
    for name in ['trajectory.npz','metrics.json']:files.append(ROOT/'research_walker_repair/seed210/step0000000/reset41001'/name)
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))]
    (H/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));archive=ROOT/'research_walker_reference_experiments.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(ROOT).as_posix())
        z.write(H/'MANIFEST.json',(H/'MANIFEST.json').relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    result=dict(files=len(manifest)+1,bytes=archive.stat().st_size,all_sha256_and_crc_passed=True)
    (H/'archive_check.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':run()
