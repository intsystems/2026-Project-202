"""Fixed continuation: successful validation ->5 pairs; failed ->held-out pilots."""
import json,sys,subprocess,concurrent.futures
from pathlib import Path
from workflow import stage
H=Path(__file__).resolve().parent

def run():
    selected=json.loads((H/'selection.json').read_text())['selected_coef']
    if selected is None:
        def evaluate(c):
            label=f'seed240_lambda{c}';stage(label,'evaluate.py',['--label',label,'--split','test'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(evaluate,[0,1,3]))
        for c in [1,3]:stage(f'seed240_lambda{c}','measure.py',['--seed','240','--coef',str(c)])
    else:
        with (H/'confirmation_process.log').open('w') as f:r=subprocess.run([sys.executable,str(H/'workflow.py'),'confirm'],stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0
        def measure(s):stage(f'seed{s}_lambda{selected}','measure.py',['--seed',str(s),'--coef',str(selected)])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(measure,range(241,246)))
    print('ALL POSTPILOT WORK COMPLETE',flush=True)

if __name__=='__main__':run()
