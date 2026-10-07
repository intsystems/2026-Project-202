"""Run the fixed seed expansion with bounded concurrency and per-stage logs."""
from pathlib import Path
import argparse,concurrent.futures,json,subprocess,sys,time
H=Path(__file__).resolve().parent

def complete(root):
    required=['summary.json','secondary_summary.json','audit.json']
    if not all((root/x).exists() for x in required):return False
    for arm in ['base','regularized']:
        p=root/arm/'meta.json'
        if not p.exists():return False
        if json.loads(p.read_text())['steps']!=3072:return False
    return True

def run_seed(seed):
    root=H/f'confirmation_seed{seed}';root.mkdir(exist_ok=True)
    if complete(root):return dict(seed=seed,status='existing_complete')
    started=time.perf_counter();print(f'START seed{seed}',flush=True)
    stages=[('train',['run.py','--seed',str(seed),'--out',str(root)]),
        ('analyze',['analyze.py','--root',str(root)]),('extras',['extras.py','--root',str(root)]),
        ('audit',['audit.py','--root',str(root)])]
    # Resume analysis after a completed training pair; do not overwrite existing runs.
    trained=all((root/arm/'meta.json').exists() and
        json.loads((root/arm/'meta.json').read_text()).get('steps')==3072
        for arm in ['base','regularized'])
    for stage,args in stages:
        if stage=='train' and trained:continue
        with (root/f'{stage}_process.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(H/args[0]),*args[1:]],cwd=H.parent,
                stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'FAILED seed{seed} stage={stage}',flush=True)
            return dict(seed=seed,status='failed',stage=stage,returncode=result.returncode)
    assert complete(root)
    s=json.loads((root/'summary.json').read_text())
    q=next(v for v in s['scalar'] if v['window']==512 and v['tau']==1)['paired']['MG']
    result=dict(seed=seed,status='complete',seconds=time.perf_counter()-started,
        independent_event=s['independent_event'],primary_MG_paired=q)
    print('DONE '+json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=list(range(3,10)))
    p.add_argument('--workers',type=int,default=3);a=p.parse_args();results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_seed,s) for s in a.seeds]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            (H/'expansion_status.json').write_text(json.dumps(sorted(results,key=lambda x:x['seed']),indent=2))
    if any(x['status']=='failed' for x in results):raise SystemExit('A stage failed; inspect its process log.')
    print('All planned expansion seeds completed.',flush=True)
