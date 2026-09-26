import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
import research_sync_control as s
from actdim.estimator.config import EstimatorConfig
from actdim.estimator.windows import score
from actdim.estimator.diagnostics import diagnose
A=s.graph(32); omega=np.load('research_sync_control_results/omega.npy')
for name,g in [('before',0),('mid',0.816381),('after',1.81418)]:
 rows=s.simulate(np.random.default_rng(900).uniform(-np.pi,np.pi,32),omega,g,steps=18000,dt=.03,A=A)
 th=rows[2000:,:32]
 rng=np.random.default_rng(1234); a=rng.normal(size=32); b=rng.normal(size=32)
 sensors={'local':rows[2000:,32], 'random_sin':np.sin(th)@a, 'random_both':np.sin(th)@a+np.cos(th)@b, 'order':rows[2000:,-1]}
 for sn,x in sensors.items():
  x=x[-8192:]; cfg=EstimatorConfig(max_E=12,tau=20,k_neighbors=8,theiler=700,theiler_cap=700,window=8192,stride=4096)
  try:
   e=score(x,cfg,seed=0); d=diagnose(x,cfg,seed=0)
   print(name,sn,'MG',e['MG'],'LB',e['LB'],'PR',e['PRdelay'],'ratio',d.identifiability_ratio,'cross',d.trend_crossings,'std',np.std(x))
  except Exception as ex: print(name,sn,type(ex).__name__,ex)
