"""Independent reference sanity checks and dataset characterization."""
import argparse,json,copy
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run import H,load_data,TextVAE,reference

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(H/'pilot_seed0'));args=p.parse_args()
    from pathlib import Path
    root=Path(args.root);torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        train,val,vocab=load_data();net=TextVAE(vocab)
        net.load_state_dict(torch.load(root/'regularized/final.pt',weights_only=True))
        # Known q(z|x)=prior: MI, KL and matched-noise shuffling response must vanish.
        null=copy.deepcopy(net)
        with torch.no_grad():
            for module in [null.mu,null.lv]:module.weight.zero_();module.bias.zero_()
        eps=torch.randn(4,len(val),16,generator=torch.Generator().manual_seed(830))
        nul=reference(null,val,eps)
        assert abs(nul['MI'])<1e-5 and nul['KL']==0 and nul['shuffle_symkl']==0
        # Large finite-mixture MI is bounded by log(number of evaluation sentences).
        for arm in ['base','regularized']:
            r=pd.read_csv(root/arm/'reference.csv')
            assert r.MI.max()<=np.log(len(val))+1e-4
            assert (r.KL>=-1e-6).all() and (r.shuffle_symkl>=-1e-6).all()
        counts=torch.bincount(train.flatten(),minlength=vocab).float()+1
        counts[0]=0;prob=counts/counts.sum();tokens=val[val!=0]
        baseline=float(-prob[tokens].log().mean())
        a=pd.read_csv(root/'base/logs.csv');b=pd.read_csv(root/'regularized/logs.csv')
        assert np.array_equal(a.probe_nll[:1024],b.probe_nll[:1024])
        assert np.isclose(a.probe_nll[1024],b.probe_nll[1024])
        (root/'audit.json').write_text(json.dumps(dict(reference_null=nul,
            all_MI_within_empirical_upper_bound=True,paired_prefix_and_branch_equal=True,
            train_sentences=len(train),validation_sentences=len(val),vocabulary=vocab,
            mean_train_tokens=float((train!=0).sum(1).float().mean()),
            validation_unk_fraction=float((tokens==3).float().mean()),
            smoothed_training_unigram_validation_nll=baseline),indent=2))
        print('Reference-null and paired-observation audits passed; unigram NLL:',baseline)
