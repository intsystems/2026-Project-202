from pathlib import Path
import pickle,json
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from environment import H,PhaseWalker

def prepare():
    anchor=H/'anchor';anchor.mkdir(exist_ok=True);old=H.parent/'research_walker_repair/anchor'
    model=PPO.load(old/'policy.zip',device='cpu');raw=make_vec_env(PhaseWalker,n_envs=8,seed=260)
    with (old/'normalize.pkl').open('rb') as f:norm=pickle.load(f)
    norm.observation_space=raw.observation_space;norm.obs_rms.mean=np.r_[norm.obs_rms.mean,0.,0.];norm.obs_rms.var=np.r_[norm.obs_rms.var,1.,1.];norm.set_venv(raw);norm.training=False;norm.norm_reward=True
    new=PPO('MlpPolicy',norm,n_steps=512,batch_size=256,n_epochs=5,learning_rate=3e-5,clip_range=.1,target_kl=.01,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh),device='cpu',seed=260)
    source=model.policy.state_dict();dest=new.policy.state_dict()
    for key,val in source.items():
        if val.shape==dest[key].shape:dest[key]=val.clone()
        else:
            assert key in ['mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight']
            dest[key]=torch.zeros_like(dest[key]);dest[key][:,:17]=val
    new.policy.load_state_dict(dest)
    rng=np.random.default_rng(37);x=torch.tensor(rng.normal(size=(32,17)),dtype=torch.float32);phase=torch.tensor(rng.uniform(-1,1,(32,2)),dtype=torch.float32);xx=torch.cat([x,phase],1)
    with torch.no_grad():
        a=model.policy.get_distribution(x).distribution.mean;b=new.policy.get_distribution(xx).distribution.mean
        v=model.policy.predict_values(x);w=new.policy.predict_values(xx)
    torch.testing.assert_close(a,b,rtol=0,atol=1e-6);torch.testing.assert_close(v,w,rtol=0,atol=1e-6)
    new.save(anchor/'policy.zip');norm.save(anchor/'normalize.pkl');norm.close()
    (H/'initialization_audit.json').write_text(json.dumps(dict(passed=True,actor_max_error=float(abs(a-b).max()),critic_max_error=float(abs(v-w).max()),optimizer_reset_both_arms=True),indent=2))
    print('PHASE ANCHOR READY')

if __name__=='__main__':torch.set_num_threads(1);prepare()
