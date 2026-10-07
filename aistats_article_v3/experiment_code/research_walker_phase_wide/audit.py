import json,hashlib,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch
from motion import H,Actor,reduced,state_reference
from evaluate import strobe
from measure import experiments
from whole_cycle import metric as cycle_metric

def run():
    rows=[]
    for name,seeds,resets in [('research_walker_phase',[260],'reset74*'),('research_walker_phase_wide',experiments(),'reset76*')]:
        folder=H.parent/name;sel=json.loads((folder/'selection.json').read_text());assert sel['protocol_sha256']==hashlib.sha256((folder/'PROTOCOL.md').read_bytes()).hexdigest()
        anchor=Actor(folder/'anchor');weights=anchor.model.policy.state_dict();hashes=[]
        for seed in seeds:
            for c in [0,3]:
                root=folder/f'seed{seed}_lambda{c}';start=Actor(root/'step0000000');final=Actor(root/'step1048576')
                for k,v in weights.items():torch.testing.assert_close(v,start.model.policy.state_dict()[k],rtol=0,atol=0)
                for norm in [start.norm,final.norm]:
                    for kind in ['obs_rms','ret_rms']:
                        for key in ['mean','var','count']:np.testing.assert_array_equal(getattr(getattr(norm,kind),key),getattr(getattr(anchor.norm,kind),key))
                hashes.append(hashlib.sha256(b''.join(v.numpy().tobytes() for v in final.model.policy.state_dict().values())).hexdigest())
                assert len(pd.read_csv(root/'test.csv'))==10
                for p in (root/'step1048576').glob(resets+'/metrics.json'):
                    m=json.loads(p.read_text());d=np.load(p.parent/'trajectory.npz');np.testing.assert_allclose(m['padded_reward'],np.sum(d['reward'])/4096,atol=1e-12)
                    if m['eligible']:
                        r,_=state_reference(d['qpos'],d['qvel']);s,_=strobe(d)
                        np.testing.assert_allclose(r['recurrence'],m['recurrence'],rtol=1e-10);np.testing.assert_allclose(s['D_strobe'],m['D_strobe'],rtol=1e-10)
                        if (p.parent/'whole_cycle.json').exists():
                            expected=json.loads((p.parent/'whole_cycle.json').read_text());actual=cycle_metric(d)
                            np.testing.assert_allclose(expected['C_cycle'],actual['C_cycle'],rtol=1e-10)
                        assert len(pd.read_csv(p.parent/'MG_windows.csv'))==15
                    for q in p.parent.glob('probe_eps*.json'):
                        z=json.loads(q.read_text());assert z['horizon']==600 and z['zero_replay_error']<1e-8
                        norms=np.load(q.with_suffix('.npz'))['norms'].reshape(-1,601);df=pd.read_csv(q.with_suffix('.csv'));valid=df.fall_step<0
                        np.testing.assert_allclose(np.median(norms[valid,-150:],axis=1)/z['epsilon'],df.loc[valid,'amplification'],rtol=1e-10)
                rows.append(root.name)
        assert len(set(hashes))==len(hashes)
    source=H.parent/'research_walker_imitation';spec=importlib.util.spec_from_file_location('imitation_reference',source/'reference.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);orbit=mod.Orbit()
    old=Actor(H.parent/'research_walker_repair/anchor');initial=old.model.policy.state_dict();ihashes=[]
    for coef in [0,1,3]:
        root=source/f'seed240_lambda{coef}';assert len(pd.read_csv(root/'test.csv'))==10
        start=Actor(root/'step0000000');final=Actor(root/'step1048576')
        for k,v in initial.items():torch.testing.assert_close(v,start.model.policy.state_dict()[k],rtol=0,atol=0)
        for kind in ['obs_rms','ret_rms']:
            for key in ['mean','var','count']:np.testing.assert_array_equal(getattr(getattr(final.norm,kind),key),getattr(getattr(old.norm,kind),key))
        ihashes.append(hashlib.sha256(b''.join(v.numpy().tobytes() for v in final.model.policy.state_dict().values())).hexdigest())
        for p in (root/'step1048576').glob('reset72*/metrics.json'):
            m=json.loads(p.read_text());d=np.load(p.parent/'trajectory.npz')
            if m['eligible']:
                r,_,_=orbit.metrics(reduced(d['qpos'],d['qvel']));np.testing.assert_allclose(r['D_section'],m['D_section'],rtol=1e-10)
        rows.append(root.name)
    assert len(set(ihashes))==3
    bandwidth=json.loads((H/'bandwidth.json').read_text());assert bandwidth['source_sha256']==hashlib.sha256((H.parent/bandwidth['source']).read_bytes()).hexdigest()
    result=dict(passed=True,runs=rows,total_final_tests=10*len(rows),shared_initialization_verified=True,frozen_norm_verified=True,raw_metrics_recomputed=True,zero_replay_verified=True)
    (H/'audit.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':torch.set_num_threads(1);run()
