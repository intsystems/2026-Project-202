"""Copy numerical helpers; generate a pinned PPO variant with one auxiliary term."""
from pathlib import Path
import inspect,textwrap,hashlib,json
from stable_baselines3 import PPO
H=Path(__file__).resolve().parent;old=H.parent/'research_walker_repair'
for name in ['motion.py','build_report.py']:
    (H/name).write_bytes((old/name).read_bytes())
source=inspect.getsource(PPO.train)
original=source
source=source.replace('        # Switch to train mode', '''        # Pair observations BEFORE RolloutBuffer.get flattens its arrays.
        obs = self.rollout_buffer.observations
        valid = self.rollout_buffer.episode_starts[1:] == 0
        pair_a = th.as_tensor(obs[:-1][valid], device=self.device)
        pair_b = th.as_tensor(obs[1:][valid], device=self.device)
        smooth_values = []
        # Switch to train mode''')
needle='                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'
assert source.count(needle)==1
source=source.replace(needle,needle+'''
                smooth_loss = th.zeros((), device=self.device)
                if self.smooth_coef > 0 and len(pair_a):
                    idx = self.pair_rng.integers(0, len(pair_a), size=len(actions))
                    a = self.policy.get_distribution(pair_a[idx]).distribution.mean.clamp(-1, 1)
                    b = self.policy.get_distribution(pair_b[idx]).distribution.mean.clamp(-1, 1)
                    smooth_loss = (a - b).square().mean()
                    loss = loss + self.smooth_coef * smooth_loss
                smooth_values.append(float(smooth_loss.detach()))''')
source=source.replace('        # Logs','        self.logger.record("train/smooth_loss", np.mean(smooth_values))\n        # Logs')
header='''# PPO.train from installed SB3, minimal audited temporal regularization patch.
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
class SmoothPPO(PPO):
    smooth_coef = 0.
'''
(H/'smooth_ppo.py').write_text(header+source,encoding='utf-8')
(H/'ppo_source.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),patched_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2))
# Preserve integration-state/actor implementation; add only action logging and J metrics.
s=(H/'motion.py').read_text()
s=s.replace('rewards=[];velocities=[];saved=[]','rewards=[];velocities=[];actions=[];saved=[]')
s=s.replace('action=actor(obs);obs,reward,terminated,truncated,info=env.step(action)', 'action=actor(obs)\n        if step>=BURN:actions.append(action.copy())\n        obs,reward,terminated,truncated,info=env.step(action)')
s=s.replace('if eligible:\n        started=', '''result['padded_reward']=float(np.sum(rewards)/N)
    result['J1']=float(np.mean(np.diff(actions,axis=0)**2)) if len(actions)>1 else None
    result['J2']=float(np.mean(np.diff(actions,n=2,axis=0)**2)) if len(actions)>2 else None
    if eligible:
        started=''')
s=s.replace("anchor_ids=np.array(anchor_ids),integration_states", "actions=np.array(actions),anchor_ids=np.array(anchor_ids),integration_states")
(H/'motion.py').write_text(s)
# Local probe copy: fixed physical horizon, fixed phase radius, fixed terminal interval.
s=(old/'perturb.py').read_text().replace('4*period+1','601').replace('t<4*period','t<600')
s=s.replace('horizon=4*period','horizon=600').replace('period//2','75').replace('dist[i,-period:]','dist[i,-150:]')
s=s.replace("tag='perturb'","tag='fixed600'").replace('epsilon=EPS,probes=', 'epsilon=EPS,phase_radius=75,terminal_steps=150,probes=')
(H/'perturb_fixed.py').write_text(s)
print('Prepared numerical helpers and PPO variant')
