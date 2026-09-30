import json
import numpy as np
import gymnasium as gym
from train import OrbitReward,H
from motion import reduced

def run():
    base=gym.make('Walker2d-v5');zero=OrbitReward(gym.make('Walker2d-v5'),0);shaped=OrbitReward(gym.make('Walker2d-v5'),3)
    rng=np.random.default_rng(92);steps=0;max_penalty_error=0.
    for seed in range(4):
        obs=[e.reset(seed=seed)[0] for e in [base,zero,shaped]]
        for i in range(150):
            action=rng.uniform(-1,1,6);a,b,c=[e.step(action) for e in [base,zero,shaped]]
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[0],c[0]);assert a[1]==b[1]
            e=shaped.unwrapped;x=reduced(e.data.qpos,e.data.qvel)
            dist=np.sqrt(np.min(np.sum((shaped.orbit.points-x)**2,axis=1)))
            expected=a[1]-3*(1-np.exp(-dist**2/(2*shaped.orbit.sigma2)))
            max_penalty_error=max(max_penalty_error,abs(expected-c[1]));assert abs(expected-c[1])<1e-12
            assert a[2:4]==b[2:4]==c[2:4];steps+=1
            if a[2] or a[3]:break
    for e in [base,zero,shaped]:e.close()
    result=dict(passed=True,steps_checked=steps,zero_wrapper_identical=True,all_physics_identical=True,post_step_reward_checked=True,max_penalty_error=max_penalty_error)
    (H/'wrapper_test.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':run()
