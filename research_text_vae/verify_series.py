"""Audit all planned raw logs and paired summaries, without retraining."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent

if __name__=='__main__':
    roots=[H/'pilot_seed0']+[H/f'confirmation_seed{s}' for s in range(1,10)]
    summary=pd.read_csv(H/'all_seeds.csv').set_index('seed');assert list(summary.index)==list(range(10))
    rows=[];final_hashes=set();corpus=json.loads((H/'data/selection.json').read_text())['hashes']
    for split,digest in corpus.items():
        assert hashlib.sha256((H/f'data/ptb.{split}.txt').read_bytes()).hexdigest()==digest
    for seed,r in enumerate(roots):
        for arm in ['base','regularized']:
            d=r/arm;l=pd.read_csv(d/'logs.csv');ref=pd.read_csv(d/'reference.csv');meta=json.loads((d/'meta.json').read_text())
            assert np.array_equal(l.step,np.arange(3072))
            assert np.array_equal(ref.step,np.arange(0,3073,128))
            assert meta['seed']==seed and meta['parameters']==276016 and meta['switch']==1024
            assert np.isfinite(l[['probe_nll','train_nll','train_KL']].to_numpy()).all()
            assert np.isfinite(ref[['MI','KL','shuffle_symkl','nll']].to_numpy()).all()
            digest=hashlib.sha256((d/'final.pt').read_bytes()).hexdigest();assert digest not in final_hashes;final_hashes.add(digest)
        a=pd.read_csv(r/'base/logs.csv');b=pd.read_csv(r/'regularized/logs.csv')
        for col in ['probe_nll','train_nll','train_KL']:
            assert np.allclose(a[col].iloc[:1024],b[col].iloc[:1024],rtol=0,atol=1e-12)
        assert np.isclose(a.probe_nll.iloc[1024],b.probe_nll.iloc[1024],rtol=0,atol=1e-12)
        assert (a.beta==.01).all() and (b.beta.iloc[:1024]==.01).all() and (b.beta.iloc[1024:]==1).all()
        w=pd.read_csv(r/'windows.csv').query('window==512 and tau==1')
        assert len(w)==42
        ratios={}
        for arm in ['base','regularized']:
            ww=w[w.arm==arm];pre=ww[ww.end.between(512,1024)].MG;post=ww[ww.end.between(2048,3072)].MG
            assert len(pre)==5 and len(post)==9
            ratios[arm]=post.median()/pre.median()
        q=ratios['regularized']/ratios['base'];assert np.isclose(q,summary.loc[seed,'MG_paired'],rtol=1e-12)
        refs={arm:pd.read_csv(r/arm/'reference.csv') for arm in ['base','regularized']}
        decreases={}
        for metric in ['MI','shuffle_symkl']:
            ratios={arm:tab[tab.step.between(2048,3072)][metric].median()/tab[tab.step.between(512,1024)][metric].median()
                for arm,tab in refs.items()}
            decreases[metric]=ratios['regularized']/ratios['base']
        early=refs['base'].query('512<=step<=1024')
        event=decreases['MI']<.5 and decreases['shuffle_symkl']<.5 and early.MI.median()>.1 and early.shuffle_symkl.median()>1e-4
        assert bool(event)==bool(summary.loc[seed,'event'])
        audit=json.loads((r/'audit.json').read_text());assert audit['paired_prefix_and_branch_equal']
        assert abs(audit['reference_null']['MI'])<1e-5 and audit['reference_null']['shuffle_symkl']==0
        rows.append(dict(seed=seed,rows_per_arm=3072,reference_checkpoints=25,q_recomputed=q,
            independent_event_recomputed=bool(event),passed=True))
    (H/'series_audit.json').write_text(json.dumps(dict(planned_seeds=list(range(10)),all_complete=True,
        unique_final_checkpoint_files=len(final_hashes),corpus_sha256=corpus,
        training_code_sha256=hashlib.sha256((H/'run.py').read_bytes()).hexdigest(),per_seed=rows),indent=2))
    print('All10 seeds: 20 complete arms, matching prefixes, reference null controls and recomputed q verified.')
