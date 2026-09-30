import json
import numpy as np
from environment import H,PhaseWalker,RHO
from evaluate import strobe

def run():
    a=PhaseWalker(0);b=PhaseWalker(3);rng=np.random.default_rng(73);count=0
    for seed in range(3):
        x,_=a.reset(seed=seed);y,_=b.reset(seed=seed);np.testing.assert_array_equal(x,y)
        for i in range(100):
            action=rng.uniform(-.5,.5,6);aa=a.step(action);bb=b.step(action)
            np.testing.assert_array_equal(aa[0],bb[0]);np.testing.assert_allclose(bb[1],aa[1]-3*bb[4]['orbit_penalty'],atol=1e-12)
            assert aa[2:4]==bb[2:4];count+=1
            if aa[2]:break
    phase=np.mod(np.arange(4096)*RHO,152);theta=2*np.pi*phase/152;qp=np.zeros((4096,9));qv=np.zeros((4096,9));qp[:,1]=np.cos(theta);qp[:,2]=np.sin(theta)
    m,_=strobe(dict(phase=phase,qpos=qp,qvel=qv));assert m['section_count']>=25 and m['D_strobe']<1e-6
    qp[:,3]=np.arange(4096)/4096;n,_=strobe(dict(phase=phase,qpos=qp,qvel=qv));assert n['D_strobe']>.01
    a.close();b.close();result=dict(passed=True,wrapper_steps=count,periodic_section=m,drifting_section=n)
    (H/'phase_test.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':run()
