import hashlib,json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from motion import H,Actor,RESETS,SCALE,state_reference,cheap,reduced
from perturb import phase_dist,zero_audit

def geometry_tests():
    line=np.c_[np.arange(10),np.zeros((10,16))]
    point=np.zeros((2,17));point[0,0]=4.3;point[0,1]=.25;point[1,0]=4.8
    np.testing.assert_allclose(phase_dist(point,line,5,4),[.25,0.],atol=1e-14)
    # Removing forward translation does not remove forward speed.
    p=np.zeros((2,9));v=np.zeros((2,9));p[1,0]=10
    np.testing.assert_array_equal(reduced(p,v)[0],reduced(p,v)[1])
    v[1,0]=1;assert reduced(p,v)[1,8]!=reduced(p,v)[0,8]

def main():
    geometry_tests();selection=json.loads((H/'selection.json').read_text());assert selection['selected_without_MG']
    assert selection['protocol_sha256']==hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()
    assert selection['resource_amendment_sha256']==hashlib.sha256((H/'RESOURCE_AMENDMENT.md').read_bytes()).hexdigest()
    result=[];initial_hashes=set()
    for seed in [200]+selection['confirmation_seeds']:
        out=H/f'seed{seed}';train=json.loads((out/f"train_{selection['horizon']}.json").read_text())
        assert train['steps']==selection['horizon'] and train['seed']==seed
        initial_actor=Actor(out/'step0000000')
        digest=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in initial_actor.model.policy.parameters())).hexdigest()
        assert digest not in initial_hashes;initial_hashes.add(digest)
        table=pd.read_csv(out/'evaluation.csv');counts=table.groupby('step').eligible.sum()
        pair=json.loads((out/'pair.json').read_text());early=[int(t) for t in counts[counts>=2].index if t<pair['horizon']]
        assert pair['early']==(early[0] if early else None)
        assert set(table['reset'])==set(RESETS)
        assert sorted(table.step.unique())==list(range(0,selection['horizon']+1,131072))
        try:windows=pd.read_csv(out/'MG_summary.csv')
        except pd.errors.EmptyDataError:windows=pd.DataFrame()
        for step in [pair['early'],pair['late']] if pair['usable'] else []:
            actor=Actor(out/f'step{step:07d}')
            for reset in pair['common_resets']:
                root=out/f'step{step:07d}'/f'reset{reset}';m=json.loads((root/'metrics.json').read_text())
                assert m['eligible'] and m['complete'] and m['n_analysis']==4096
                data=np.load(root/'trajectory.npz');reference,_=state_reference(data['qpos'],data['qvel'])
                for k in ['recurrence','section_dispersion','period','state_variance']:
                    np.testing.assert_allclose(reference[k],m[k],rtol=1e-12,atol=1e-14)
                scalar=cheap(data['qpos'][:,4]);np.testing.assert_allclose(scalar['entropy'],m['cheap']['entropy'])
                zero_audit(actor,data,selection['anchors'][0],m['period'])
                name='perturb_'+str(len(selection['anchors']))
                probe=json.loads((root/(name+'.json')).read_text());rows=pd.read_csv(root/(name+'.csv'))
                assert probe['inference_mode']=='single_observation_as_nominal' and probe['zero_replay_max_error']<1e-8
                assert len(rows)==34*len(selection['anchors'])==probe['probes']
                assert rows.fell.sum()==probe['falls'] and rows.nearly_tangent.sum()==probe['nearly_tangent']
                vals=rows.amplification.dropna()
                if len(vals):np.testing.assert_allclose(vals.median(),probe['median_amplification'])
                curves=np.load(root/(name+'_curves.npz'))['distances']
                for i,r in enumerate(rows.itertuples()):
                    distances=curves.reshape(-1,curves.shape[-1])[i]
                    np.testing.assert_allclose(distances[0],r.initial_distance,rtol=1e-10)
                    if not r.nearly_tangent and not r.fell:
                        np.testing.assert_allclose(np.median(distances[-m['period']:])/distances[0],r.amplification,rtol=1e-10)
                raw=pd.read_csv(root/'MG_windows.csv');primary=raw[(raw.sensor=='right_knee')&(raw.window==2048)&(raw.tau==selection['tau'])]
                assert primary.end.tolist()==[2048,2560,3072,3584,4096]
                assert not raw.duplicated(['sensor','window','tau','end']).any()
                selected=windows[(windows.step==step)&(windows.reset==reset)&(windows.sensor=='right_knee')&(windows.window==2048)&(windows.tau==selection['tau'])].iloc[0]
                valid=np.isfinite(primary.MG)&~primary.degenerate
                assert bool(selected.all_valid)==bool(valid.all())
                if valid.any():np.testing.assert_allclose(selected.MG,primary.loc[valid,'MG'].median())
        result.append(dict(seed=seed,passed=True,usable=pair['usable']));print('AUDIT',seed,'PASS',flush=True)
    (H/'audit.json').write_text(json.dumps(dict(all_passed=True,geometry_tests=True,seeds=result),indent=2))

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):main()
