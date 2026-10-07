"""Meaningful regression: lambda0 must reproduce standard PPO bit-for-bit."""
import json
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from threadpoolctl import threadpool_limits
from smooth_ppo import SmoothPPO
from train import H,ANCHOR

def one(cls,coef):
    env=VecNormalize.load(ANCHOR/'normalize.pkl',make_vec_env('Walker2d-v5',n_envs=8,seed=219,env_kwargs=dict(max_episode_steps=5000)))
    env.training=False;env.norm_reward=True
    m=cls.load(ANCHOR/'policy.zip',env=env,device='cpu',n_steps=512,batch_size=256,n_epochs=5,
        learning_rate=lambda p:3e-5*(.1+.9*p),clip_range=.1,target_kl=.01)
    m.smooth_coef=coef;m.pair_rng=np.random.default_rng(100219);m.set_random_seed(219);m.num_timesteps=0;m.ep_info_buffer.clear()
    m.learn(total_timesteps=8192,reset_num_timesteps=True)
    result={k:v.detach().clone() for k,v in m.policy.state_dict().items()};log=dict(m.logger.name_to_value);env.close();return result,log

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        a,_=one(PPO,0);b,log=one(SmoothPPO,0);c,slog=one(SmoothPPO,1)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    delta=max(float(torch.max(abs(a[k]-c[k]))) for k in a)
    assert delta>0 and slog['train/smooth_loss']>0
    result=dict(lambda0_bitwise_identical=True,regularization_changes_parameters=delta,auxiliary_loss=float(slog['train/smooth_loss']),steps=8192)
    (H/'training_implementation_audit.json').write_text(json.dumps(result,indent=2));print(result)
