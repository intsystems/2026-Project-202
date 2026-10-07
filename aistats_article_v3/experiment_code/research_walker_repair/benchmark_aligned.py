"""Supplement: equal five-window work for the inexpensive comparators."""
import json,time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from motion import H,state_reference,cheap

def main():
    benchmark=json.loads((H/'benchmark.json').read_text());assert benchmark['available']
    label=benchmark['seed'];reset=benchmark['reset'];rows=[]
    for stage,step in [('early',0),('late',1048576)]:
        data=np.load(H/label/f'step{step:07d}'/f'reset{reset}'/'trajectory.npz');qp=data['qpos'];qv=data['qvel']
        funcs={'full_state_five_windows':lambda:[state_reference(qp[e-2048:e],qv[e-2048:e]) for e in range(2048,4097,512)],
            'cheap_scalar_five_windows':lambda:[cheap(qp[e-2048:e,4]) for e in range(2048,4097,512)]}
        for f in funcs.values():f()
        for rep in range(9):
            for name in np.random.default_rng(rep+781).permutation(list(funcs)):
                start=time.perf_counter();funcs[name]();elapsed=time.perf_counter()-start
                rows.append(dict(stage=stage,operation=name,repeat=rep,seconds=elapsed))
    frame=pd.DataFrame(rows);frame.to_csv(H/'timings_aligned.csv',index=False)
    benchmark['operations']=[r for r in benchmark['operations'] if r['operation'] not in funcs]
    for (stage,op),p in frame.groupby(['stage','operation']):
        benchmark['operations'].append(dict(stage=stage,operation=op,median=float(p.seconds.median()),min=float(p.seconds.min()),max=float(p.seconds.max()),repeats=len(p)))
    benchmark['aligned_window_supplement']='Same five windows of length2048, stride512 for cheap features and full-state R/D. Original full-record timings retained. No training/selection change.'
    (H/'benchmark.json').write_text(json.dumps(benchmark,indent=2));print(frame.groupby(['stage','operation']).seconds.median().to_string())

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
