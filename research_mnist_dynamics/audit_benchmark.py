from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from run import H,data,model
from reference import pr,residual
from analyze import mg,cheap

if __name__=='__main__':
    root=H/'confirmation';rows=[];acc=[]
    x,y,tx,ty,_=data();rng=np.random.default_rng(10299)
    train_ids=np.concatenate([rng.choice(np.flatnonzero(y.numpy()==c),10,False) for c in range(10)])
    valid_ids=np.concatenate([rng.choice(np.flatnonzero(ty.numpy()==c),10,False) for c in range(10)])
    test_ids=np.setdiff1d(np.arange(len(tx)),valid_ids);px=tx[valid_ids];py=ty[valid_ids]
    net=model();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Load lazy sklearn/scipy libraries BEFORE fixing their thread pools.
    mg(np.sin(np.arange(512)/17)+np.cos(np.arange(512)/9))
    with threadpool_limits(limits=1):
        for s in range(1,6):
            base=np.load(root/f'seed_{s}/base/trajectory.npy',mmap_mode='r')
            drop=np.load(root/f'seed_{s}/drop/trajectory.npy',mmap_mode='r')
            assert np.array_equal(base[:2048],drop[:2048])
            for arm in ['base','drop']:
                d=root/f'seed_{s}'/arm;net.load_state_dict(torch.load(d/'final.pt',weights_only=True))
                with torch.no_grad():accuracy=float((net(tx[test_ids]).argmax(1)==ty[test_ids]).float().mean())
                acc.append(dict(seed=s,arm=arm,unseen_test_accuracy=accuracy,n=9900))
        d=root/'seed_1/drop';traj=np.load(d/'trajectory.npy',mmap_mode='r');w=traj[-512:]
        series=pd.read_csv(d/'probe.csv').test_probe.to_numpy()[-512:]
        idx=np.random.default_rng(718).choice(traj.shape[1],128,False)
        observer=np.random.default_rng(1122).choice([-1.,1.],size=traj.shape[1])/np.sqrt(traj.shape[1])
        eig=np.load(d/'spectrum_4096.npy');v,_=pr(w)
        assert np.isclose(v,eig.sum()**2/np.sum(eig**2),rtol=1e-10)
        # PR scale/orthogonal invariance checked without materializing P x P matrix.
        assert np.isclose(v,pr(10*w.astype(float))[0],rtol=1e-9)
        t=np.arange(512,dtype=float);q=np.stack([np.sin(t/15),np.cos(t/7)],1)
        assert np.isclose(pr(q)[0],pr(q@np.array([[0.,-1.],[1.,0.]]))[0])
        pr(w);mg(series)
        for rep in range(7):
            t=time.perf_counter();full,_=pr(w);fulltime=time.perf_counter()-t
            t=time.perf_counter();small,_=pr(w[:,idx]);smalltime=time.perf_counter()-t
            m=mg(series);cheapstats=cheap(series)
            # Cost per single fixed100-image observation, inference only.
            t=time.perf_counter()
            with torch.no_grad():
                for _ in range(100):out=torch.nn.functional.cross_entropy(net(px),py)
            forward=(time.perf_counter()-t)/100
            params=list(net.parameters());buffer=np.empty((512,traj.shape[1]),np.float32)
            short=np.empty((512,128),np.float32)
            t=time.perf_counter()
            for i in range(512):buffer[i]=torch.cat([p.detach().flatten() for p in params]).numpy()
            full_record=time.perf_counter()-t
            t=time.perf_counter()
            for i in range(512):short[i]=torch.cat([p.detach().flatten() for p in params]).numpy()[idx]
            small_record=time.perf_counter()-t
            rows.append(dict(repeat=rep,full_pr=full,small_pr=small,full_seconds=fulltime,
                small_seconds=smalltime,probe_forward_seconds=forward,full_record512_seconds=full_record,
                small_record512_seconds=small_record,**m,cheap_seconds=cheapstats['cheap_seconds']))
        pd.DataFrame(rows).to_csv(root/'benchmark.csv',index=False)
        pd.DataFrame(acc).to_csv(root/'unseen_accuracy.csv',index=False)
        pools=threadpool_info()
        assert all(p['num_threads']==1 for p in pools)
        audit=dict(paired_prefix_identical=True,PR_matches_eigen_formula=True,PR_scale_rotation_invariance=True,
            unseen_test_n=9900,validation_n=100,torch_threads=torch.get_num_threads(),threadpools=pools,
            evaluation_note='9900 images excluded from fixed probe and gradients; aggregate test accuracy was logged earlier, so not a blind final test')
        (root/'audit.json').write_text(json.dumps(audit,indent=2))
    print('Audit and benchmark complete.')
