import hashlib,json,pickle
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,state_reference
from perturb import zero_audit

def main():
    selection=json.loads((H/'selection.json').read_text());assert selection['selected_without_MG']
    assert hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()==selection['protocol_sha256']
    for f,digest in json.loads((H/'estimator_hashes.json').read_text()).items():assert hashlib.sha256((H.parent/f).read_bytes()).hexdigest()==digest
    labels=[selection['pilot_label']]+[f'seed{s}'+('_B' if selection['conservative'] else '') for s in selection['confirmation_seeds']]
    anchor=Actor(H/'anchor');original=anchor.model.policy.state_dict();last_hashes=set();result=[]
    for label in labels:
        out=H/label;meta=json.loads((out/'train.json').read_text());pair=json.loads((out/'pair.json').read_text())
        initial=Actor(out/'step0000000');last=Actor(out/'step1048576');current=last.model.policy.state_dict()
        for k in original:torch.testing.assert_close(initial.model.policy.state_dict()[k],original[k],rtol=0,atol=0)
        for attr in ['mean','var','count']:
            np.testing.assert_array_equal(getattr(last.norm.obs_rms,attr),getattr(anchor.norm.obs_rms,attr))
            np.testing.assert_array_equal(getattr(last.norm.ret_rms,attr),getattr(anchor.norm.ret_rms,attr))
        digest=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in current.values())).hexdigest()
        assert digest not in last_hashes;last_hashes.add(digest)
        actor_keys=[k for k in original if 'policy_net' in k or k.startswith('action_net') or k=='log_std']
        delta=float(np.sqrt(sum(float(((current[k]-original[k])**2).sum()) for k in actor_keys)))
        assert delta>1e-5 and meta['parameter_delta_norm']>1e-5
        val=pd.read_csv(out/'validation.csv');test=pd.read_csv(out/'test.csv')
        mg_summary=pd.read_csv(out/'MG_summary.csv')
        assert len(val)==45 and len(test)==20
        assert sorted(val.step.unique())==list(range(0,1048577,131072))
        assert sorted(test.reset.unique())==list(range(51001,51011))
        for step in [0,1048576]:
            for reset in pair['common_resets']:
                root=out/f'step{step:07d}'/f'reset{reset}';d=np.load(root/'trajectory.npz');m=json.loads((root/'metrics.json').read_text())
                reference,_=state_reference(d['qpos'],d['qvel'])
                for k in ['recurrence','section_dispersion','period']:np.testing.assert_allclose(m[k],reference[k],rtol=1e-12)
                raw=pd.read_csv(root/'MG_windows.csv');primary=raw[(raw.sensor=='right_knee')&(raw.window==2048)&(raw.tau==selection['tau'])]
                assert list(primary.end)==[2048,2560,3072,3584,4096]
                entry=mg_summary[(mg_summary.step==step)&(mg_summary.reset==reset)&(mg_summary.sensor=='right_knee')&(mg_summary.window==2048)&(mg_summary.tau==selection['tau'])].iloc[0]
                valid=np.isfinite(primary.MG)&~primary.degenerate
                assert bool(entry.all_valid)==bool(valid.all())
                if valid.any():np.testing.assert_allclose(entry.MG,primary.loc[valid,'MG'].median())
            if pair['common_resets']:
                reset=pair['common_resets'][0];root=out/f'step{step:07d}'/f'reset{reset}';m=json.loads((root/'metrics.json').read_text())
                d=np.load(root/'trajectory.npz');zero_audit(Actor(root.parent),d,0,m['period'])
                p=json.loads((root/'perturb_2.json').read_text());r=pd.read_csv(root/'perturb_2.csv')
                assert len(r)==p['probes']==68 and p['inference_mode']=='single_observation_as_nominal'
                assert p['zero_replay_max_error']==0 and r.fell.sum()==p['falls']
                curves=np.load(root/'perturb_2_curves.npz')['distances'];curves=curves.reshape(-1,curves.shape[-1])
                for i,row in enumerate(r.itertuples()):
                    if not row.fell and not row.nearly_tangent:np.testing.assert_allclose(np.median(curves[i,-p['period']:])/curves[i,0],row.amplification,rtol=1e-10)
        result.append(dict(label=label,passed=True,actor_delta=delta,final_sha256=digest,common_test_resets=len(pair['common_resets'])))
        print('AUDIT',label,'PASS',flush=True)
    (H/'audit.json').write_text(json.dumps(dict(passed=True,shared_initialization=True,seeds=result),indent=2))

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
