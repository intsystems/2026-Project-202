"""Wait for authorized runs, then produce audited data and a reviewable report."""
from pathlib import Path
import json,subprocess,sys,time
H=Path(__file__).resolve().parent

def run(script):
    (H/'pipeline_status.json').write_text(json.dumps(dict(stage=script),indent=2));print('POSTPROCESS',script,flush=True)
    subprocess.run([sys.executable,str(H/script)],cwd=H.parent,check=True)

if __name__=='__main__':
    while True:
        status=H/'confirmation_status.json';pilot=H/'seed210/MG_summary.csv'
        if status.exists():
            try:rows=json.loads(status.read_text())
            except json.JSONDecodeError:rows=[]
            if len(rows)==5:
                assert all(r['status']=='complete' for r in rows),'A confirmation stage failed'
                if pilot.exists():break
        time.sleep(5)
    run('benchmark.py');run('benchmark_aligned.py');run('old_final_control.py');run('verify.py');run('summarize.py');run('build_report.py')
    (H/'pipeline_status.json').write_text(json.dumps(dict(stage='ready_for_visual_review'),indent=2))
