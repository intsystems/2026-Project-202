import sys, numpy as np
sys.path.insert(0,'.')
import research_sync_control as s
A=s.graph(32); om=1+np.random.default_rng(0).normal(0,.2,32); t0=np.random.default_rng(900).uniform(-np.pi,np.pi,32)
for g in [0,.2,.5,.8,1,1.2,1.5,2,3,5,8,12]:
 rows=s.simulate(t0,om,g,steps=20000,dt=.03,A=A)
 o=rows[2000:,-1]
 print(g, o[-5000:].mean(),o[-5000:].std(),o[-1])
