"""Check custom final-policy observation loop against stock VecNormalize."""
import json
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import gymnasium as gym
from motion import Actor,make_env,H,RESETS
from threadpoolctl import threadpool_limits

def main():
    rows=[]
    for reset in RESETS:
        root=H/'seed200/step4194304';actor=Actor(root);single=make_env();obs,_=single.reset(seed=reset)
        raw=DummyVecEnv([make_env]);vec=VecNormalize.load(root/'normalize.pkl',raw)
        vec.training=False;vec.norm_reward=False;vec.seed(reset);vobs=vec.reset();err=0.
        for t in range(4608):
            action=actor(obs);standard,_=actor.model.predict(vobs,deterministic=True)
            np.testing.assert_array_equal(action,standard[0]);err=max(err,float(np.max(np.abs(action-standard[0]))))
            obs,reward,terminated,truncated,_=single.step(action);vobs,vr,done,infos=vec.step(standard)
            np.testing.assert_allclose(reward,vr[0],rtol=1e-6,atol=1e-6)
            assert bool(done[0])==bool(terminated or truncated)
            if terminated or truncated:break
        stored=json.loads((root/f'reset{reset}/metrics.json').read_text());assert stored['steps']==t+1
        rows.append(dict(reset=reset,steps=t+1,action_difference=err,terminal_height=float(single.data.qpos[1]),terminal_angle=float(single.data.qpos[2])))
        single.close();vec.close()
    (H/'standard_evaluation_audit.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
