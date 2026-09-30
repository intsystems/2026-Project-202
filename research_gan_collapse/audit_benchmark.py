"""Verify frozen output and benchmark useful comparators after all training."""
import json
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from model import Generator,Classifier,configure
from prepare import H
from run import metrics
from analyze import score,cheap

if __name__=='__main__':
    configure()
    initial=torch.load(H/'mnist_pilot/base_long/checkpoint_12288.pt',weights_only=False)
    final=torch.load(H/'mnist_branches/frozen/checkpoint_16384.pt',weights_only=False)
    assert all(torch.equal(v,final['g'][k]) for k,v in initial['g'].items())
    a=np.load(H/'mnist_pilot/base_long/evaluation_12288.npz')
    for f in (H/'mnist_branches/frozen').glob('evaluation_*.npz'):
        b=np.load(f);assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['confidence'],b['confidence'])
    g=Generator(1);g.load_state_dict(initial['g']);g.eval()
    c=Classifier();c.load_state_dict(torch.load(H/'classifier.pt',weights_only=True));c.eval()
    z=torch.randn(10000,64,generator=torch.Generator().manual_seed(7711))
    counts=[]
    with threadpool_limits(limits=1):
        # Exact tensor equality on arbitrary test z, not just evaluator labels.
        other=Generator(1);other.load_state_dict(final['g']);other.eval()
        with torch.no_grad():assert torch.equal(g(z[:512]),other(z[:512]))
        logs=pd.read_csv(H/'mnist_branches/base/logs.csv').g_loss.to_numpy()
        score(logs[-512:]);metrics(g,c,z[:512],1)
        timings=[]
        for rep in range(5):
            t=time.perf_counter();small,_,_=metrics(g,c,z[:512],1);small_t=time.perf_counter()-t
            t=time.perf_counter();full,_,_=metrics(g,c,z,1);full_t=time.perf_counter()-t
            v=score(logs[-512:]);q=cheap(logs[-512:])
            timings.append(dict(repeat=rep,small_seconds=small_t,full_seconds=full_t,**v,**q))
        # Sampling variation on the BEFORE generator, distinct from training seeds.
        for seed in range(5):
            zz=torch.randn(10000,64,generator=torch.Generator().manual_seed(8820+seed))
            v,_,_=metrics(g,c,zz,1);counts.append(dict(latent_seed=8820+seed,**v))
        pools=threadpool_info();threads=torch.get_num_threads()
    pd.DataFrame(timings).to_csv(H/'mnist_branches/benchmark.csv',index=False)
    pd.DataFrame(counts).to_csv(H/'mnist_branches/fresh_latents.csv',index=False)
    (H/'mnist_branches/audit.json').write_text(json.dumps(dict(frozen_parameters_and_BN_identical=True,
        evaluator_outputs_identical_all_saved_steps=True,arbitrary_images_bitwise_identical=True,
        actual_torch_threads=threads,threadpools=pools,reference_has_no_confirmed_collapse=True),indent=2))
    print('Frozen tensors/images/evaluator audit passed; repeated timings and fresh-latent evaluation saved.')
