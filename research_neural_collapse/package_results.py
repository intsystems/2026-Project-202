"""Bundle completed experiment and exact MG sources, without external image data."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'neural_collapse_experiment_results.zip'

def main():
    frozen=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (HERE/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
        cpu='Intel Core i5-12500H',pip_freeze=frozen.stdout),indent=2),encoding='utf-8')
    selected=[]
    for folder in [HERE,ROOT/'code'/'actdim']:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            if path.suffix in {'.aux','.out','.log'} or path.name.startswith('preview_'):continue
            selected.append(path)
    manifest=[]
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in selected:
            name=path.relative_to(ROOT).as_posix()
            z.write(path,name)
            manifest.append(dict(path=name,size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(OUT) as z:
        assert z.testzip() is None
    print(f'{OUT}\n{len(manifest)} files, {OUT.stat().st_size/1024**2:.2f} MiB; CRC verified')

if __name__=='__main__':main()
