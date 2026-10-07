from pathlib import Path
import json,pickle,hashlib
import numpy as np,pandas as pd,torch
from motion import Actor,state_reference,cheap
H=Path(__file__).resolve().parent

def run():
    selected=json.loads((H/'selection.json').read_text());assert selected['protocol_sha256']==hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()
    c=selected['selected_coef'];anchor=Actor(H/'anchor/step0000000');initial=anchor.model.policy.state_dict();final_hashes=[];counts=[]
    for s in range(221,226):
        for coef in [0,c]:
            root=H/f'seed{s}_lambda{coef:g}';start=Actor(root/'step0000000');final=Actor(root/'step1048576')
            for k,v in start.model.policy.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
            for norm in [start.norm,final.norm]:
                for kind in ['obs_rms','ret_rms']:
                    for field in ['mean','var','count']:np.testing.assert_array_equal(getattr(getattr(norm,kind),field),getattr(getattr(anchor.norm,kind),field))
            digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in final.model.policy.state_dict().values())).hexdigest();final_hashes.append(digest)
            frame=pd.read_csv(root/'test.csv');assert len(frame)==10
            assert len(pd.read_csv(root/'validation.csv'))==15
            for p in (root/'step1048576').glob('reset62*/metrics.json'):
                m=json.loads(p.read_text());d=np.load(p.parent/'trajectory.npz')
                np.testing.assert_allclose(np.sum(d['reward'])/4096,m['padded_reward'],rtol=1e-12)
                if len(d['actions'])>2:
                    for n in [1,2]:np.testing.assert_allclose(np.mean(np.diff(d['actions'],n=n,axis=0)**2),m[f'J{n}'],rtol=1e-7)
                if m['eligible']:
                    r,_=state_reference(d['qpos'],d['qvel'])
                    for key in ['recurrence','section_dispersion']:np.testing.assert_allclose(m[key],r[key],rtol=1e-10)
                    mg=pd.read_csv(p.parent/'MG_windows.csv');assert len(mg)==9
            counts.append(dict(label=root.name,heldout=len(frame),complete=int(frame.complete.sum())))
    assert len(set(final_hashes))==10
    probes=list(H.glob('seed22*/step1048576/reset62*/fixed600_2.json'))+list(H.glob('diagnostic_probes/*/reset*/fixed600_2.json'))
    for p in probes:
        m=json.loads(p.read_text());assert m['horizon']==600 and m['phase_radius']==75 and m['terminal_steps']==150 and m['zero_replay_max_error']<1e-8
        data=np.load(p.with_name('fixed600_2_curves.npz'))['distances'].reshape(68,601)
        f=pd.read_csv(p.with_suffix('.csv'));valid=~f.nearly_tangent&~f.fell
        recomputed=np.median(data[valid,-150:],axis=1)/data[valid,0]
        np.testing.assert_allclose(recomputed,f.loc[valid,'amplification'],rtol=1e-10)
    result=dict(passed=True,shared_initial_policy=True,normalization_frozen=True,distinct_final_policies=len(set(final_hashes)),confirmed_runs=counts,fixed_horizon_probe_records=len(probes),training_implementation=json.loads((H/'training_implementation_audit.json').read_text()))
    (H/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':torch.set_num_threads(1);run()
