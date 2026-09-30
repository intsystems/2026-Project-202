"""Finish an already running pilot, without adapting scientific choices to MG."""
from pathlib import Path
import json,subprocess,sys,time
H=Path(__file__).resolve().parent

def run(script,*args):
    (H/'pipeline_status.json').write_text(json.dumps(dict(stage=script,args=list(args),updated=time.time()),indent=2))
    print('PIPELINE',script,*args,flush=True)
    subprocess.run([sys.executable,str(H/script),*map(str,args)],cwd=H.parent,check=True)

def main():
    horizon=4194304
    while not (H/f'seed200/train_{horizon}.json').exists():time.sleep(10)
    run('motion.py','--seed',200)
    run('motion.py','--seed',200,'--pair',horizon)
    pair=json.loads((H/'seed200/pair.json').read_text())
    if not pair['usable']:
        run('standard_eval_audit.py')
        run('feasibility_report.py')
        run('build_report.py')
        run('render_preview.py','--seed',200)
    else:
        run('perturb.py','--seed',200,'--step',pair['early'],'--reset',pair['common_resets'][0])
        run('select_pilot.py')
        # Seeds are launched regardless of the pilot MG result.
        with (H/'confirmation_process.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(H/'confirm.py')],cwd=H.parent,stdout=log,stderr=subprocess.STDOUT)
            run('perturb.py','--seed',200)
            run('features.py','--seed',200)
            (H/'pipeline_status.json').write_text(json.dumps(dict(stage='confirmation',pid=process.pid,updated=time.time()),indent=2))
            if process.wait():raise RuntimeError('Confirmation failed; see confirmation_process.log')
        run('benchmark.py');run('verify.py');run('summarize.py');run('build_report.py')
        example=json.loads((H/'example_selection.json').read_text());run('render_preview.py','--seed',example['seed'])
    (H/'pipeline_status.json').write_text(json.dumps(dict(stage='analysis_complete',updated=time.time()),indent=2))
    # Human/model inspection of PDF and final archive creation follow separately.

if __name__=='__main__':main()
