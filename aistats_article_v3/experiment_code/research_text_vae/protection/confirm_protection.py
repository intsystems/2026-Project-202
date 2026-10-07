"""Run all nine prespecified third branches, regardless of MG direction."""
from pathlib import Path
import concurrent.futures,json,subprocess,sys,time
H=Path(__file__).resolve().parent

def one(seed,mode):
    out=H/mode/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    if (out/'comparison.json').exists():return dict(seed=seed,status='existing_complete')
    stages=[]
    if not (out/'meta.json').exists():stages.append(('train','run_protection.py'))
    stages.append(('analysis','analyze_protection.py'))
    for stage,script in stages:
        print(f'START {mode} seed{seed} {stage}',flush=True)
        with (out/f'{stage}_process.log').open('w',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,str(H/script),'--mode',mode,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,cwd=H.parent.parent)
        if r.returncode:return dict(seed=seed,status='failed',stage=stage,code=r.returncode)
    d=json.loads((out/'comparison.json').read_text());p=d['scalar'][0]
    result=dict(seed=seed,status='complete',seconds=time.perf_counter()-start,
        protection_valid=d['retention']['protection_valid'],q_protected=p['q_protected'],R=p['protected_vs_regularized'])
    print('DONE '+json.dumps(result),flush=True);return result

if __name__=='__main__':
    selected=json.loads((H/'selection.json').read_text());mode=selected['mode']
    assert selected['reference_passed'] and selected['selection_did_not_use_MG']
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,seed,mode) for seed in range(1,10)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());(H/'confirmation_status.json').write_text(json.dumps(sorted(results,key=lambda x:x['seed']),indent=2))
    if any(r['status']=='failed' for r in results):raise SystemExit('A stage failed; inspect per-seed logs.')
