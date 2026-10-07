from pathlib import Path
import json,pickle,hashlib,time,platform
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from stable_baselines3 import PPO
from checkpoint_selection import H,R,eval_policy,MGCFG,estimate,entropy
from analyze import candidates,evaluate,METRICS

def main():
    checks={};manifest=[]
    for seed in range(231,236):
        d=pd.read_csv(H/f'checkpoint_selection_s{seed}.csv')
        assert len(d)==16*33 and not d.duplicated(['coef','step','split','reset','alpha','noise']).any()
        assert d[d.split=='target'].groupby(['coef','step']).size().eq(30).all()
        assert np.isfinite(d.reward).all()
        if seed>231:
            assert d.loc[(d.split=='nominal')&(~d.complete),METRICS].isna().all().all()
        original,_,_=evaluate(d,seed,['coef','step'],(.25,1048576))
        altered=d.copy();altered.loc[altered.split=='target','reward']=np.random.default_rng(seed).normal(size=480)
        permuted,_,_=evaluate(altered,seed,['coef','step'],(.25,1048576))
        for x,y in zip(original,permuted):
            if x['method'].startswith('oracle') or x['method']=='random_eligible':continue
            assert x['coef']==y['coef'] and x['step']==y['step']
        for coef in [0,.25,1,4]:
            for step in [262144,524288,786432,1048576]:
                p=R/'research_walker_smooth_lambda'/f'seed{seed}_lambda{coef:g}'/f'step{step:07d}'/'policy.zip'
                manifest.append(dict(seed=seed,coef=coef,step=step,path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    checks.update(full_candidate_coverage=True,selection_independent_of_test_rewards=True,
                  incomplete_confirmation_features_missing=True,all_falls_retained=True)
    root=R/'research_walker_smooth_lambda/seed232_lambda1/step0786432'
    model=PPO.load(root/'policy.zip',device='cpu');model.policy.set_training_mode(False)
    with (root/'normalize.pkl').open('rb') as f:norm=pickle.load(f)
    norm.training=False;norm.norm_reward=False
    replay=eval_policy(model,norm,range(84001,84006),.1,.02,False)
    stored=pd.read_csv(H/'checkpoint_selection_s232.csv').query('coef==1 and step==786432 and split=="target" and alpha==.1 and noise==.02').sort_values('reset')
    assert np.allclose([r['reward'] for r in replay],stored.reward.to_numpy(),rtol=1e-10,atol=1e-10)
    checks['saved_target_batch_replay']=True
    # Sequential timings after all parallel jobs finished. Same implementation and policy.
    times=[]
    for rep in range(3):
        tic=time.perf_counter();nom=eval_policy(model,norm,range(83001,83004),0,0,False)
        times.append(dict(name='three_nominal_rollouts',repeat=rep,seconds=time.perf_counter()-tic))
        tic=time.perf_counter()
        for a in [.05,.1,.2]:
            for noise in [0,.02]:eval_policy(model,norm,range(84001,84006),a,noise,False)
        times.append(dict(name='thirty_target_rollouts',repeat=rep,seconds=time.perf_counter()-tic))
    with np.load(H/'nominal_s232_l1_r81001.npz') as z:actions=z['actions']
    x=np.linalg.norm(actions,axis=1);var=np.var(x)
    funcs=dict(MG=lambda:estimate(x,MGCFG,seed=123).MG,entropy=lambda:entropy(x),
        recurrence=lambda:min(np.mean((x[l:]-x[:-l])**2)/(2*var) for l in range(20,251)),
        increments=lambda:np.mean(np.diff(x)**2)/(2*var),J1=lambda:np.mean(np.diff(actions,axis=0)**2))
    for fn in funcs.values():fn()
    for rep in range(7):
        for name in np.random.default_rng(rep).permutation(list(funcs)):
            batch=20 if name in ['entropy','increments','J1'] else 1
            tic=time.perf_counter()
            for _ in range(batch):funcs[name]()
            times.append(dict(name=name,repeat=rep,seconds=(time.perf_counter()-tic)/batch))
    pd.DataFrame(times).to_csv(H/'timing_raw.csv',index=False)
    pd.DataFrame(times).groupby('name').seconds.agg(['median','min','max']).to_csv(H/'timing_summary.csv')
    (H/'audit.json').write_text(json.dumps(dict(checks=checks,platform=platform.platform(),python=platform.python_version(),
        torch=torch.__version__,threads=1,cpu='Intel Core i5-12500H',timing_repetitions=3,
        feature_timing_repetitions=7,paired_confirmation_seeds=[232,233,234,235],
        archive_contains_training=False),indent=2))
    (H/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(checks)
if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):main()
