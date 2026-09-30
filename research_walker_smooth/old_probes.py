from pathlib import Path
import json,shutil,torch
from threadpoolctl import threadpool_limits
from perturb_fixed import probe
H=Path(__file__).resolve().parent;OLD=H.parent/'research_walker_repair'

def run():
    rows=[]
    for seed in range(211,216):
        reset=json.loads((OLD/f'seed{seed}'/'pair.json').read_text())['common_resets'][0]
        for step in [0,1048576]:
            src=OLD/f'seed{seed}'/f'step{step:07d}';out=H/'diagnostic_probes'/('anchor' if step==0 else f'seed{seed}')
            out.mkdir(parents=True,exist_ok=True);(out/f'reset{reset}').mkdir(exist_ok=True)
            for name in ['policy.zip','normalize.pkl']:shutil.copy2(src/name,out/name)
            for name in ['trajectory.npz','metrics.json']:shutil.copy2(src/f'reset{reset}'/name,out/f'reset{reset}'/name)
            m=probe(out,reset,[0,1024]);rows.append(dict(seed=seed,step=step,**m))
        (H/'diagnostic_probes'/'summary.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run()
