"""Descriptive matched-reset check of the previously failed final policy."""
import json,shutil
import torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,rollout

def main():
    out=H/'old_final_control';out.mkdir(exist_ok=True)
    source=H.parent/'research_walker_gait/seed200/step4194304'
    for name in ['policy.zip','normalize.pkl']:
        if not (out/name).exists():shutil.copy2(source/name,out/name)
    actor=Actor(out);rows=[]
    for reset in range(51001,51011):
        m=rollout(actor,out,reset);rows.append(dict(reset=reset,steps=m['steps'],walking=bool(m['complete'] and m['mean_speed']>=.5)))
    result=dict(walking=sum(r['walking'] for r in rows),total=10,rows=rows,
        purpose='Descriptive evaluation on the same held-out resets, not used for selection or a causal ablation')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
