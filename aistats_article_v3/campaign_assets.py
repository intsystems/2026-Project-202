"""Verify archived campaign results and generate manuscript tables, without training."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
E = H / 'evidence/oct07_campaign'
SOURCE = H.parent / 'research_mg_wins'

def table(name, headers, rows):
    out = [r'\begin{tabular}{l' + 'r'*(len(headers)-1) + '}', r'\toprule',
           ' & '.join(headers) + r' \\', r'\midrule']
    out += [' & '.join(map(str,row)) + r' \\' for row in rows]
    out += [r'\bottomrule',r'\end{tabular}']
    (H/'tables'/name).write_text('\n'.join(out)+'\n',encoding='utf-8')

def main():
    if not E.exists():
        paths = ['FINAL_CAMPAIGN_ru.md', 'ssl/results_main/test_final.csv',
                 'ssl/results_confirm/confirm_final.csv','ssl/results_confirm/T1_confirm.csv',
                 'ssl/results_main/T1_config_ranking.csv', 'ssl/results_main/costs_ms.json',
                 'ssl/results_main/T2_bad_run_auc.csv','ssl/results_main/T3_early_warning_auc.csv',
                 'ssl/results_main/T4_checkpoint_regret.csv', 'spikes/results_test.csv',
                 'plasticity/results/closed_loop_summary.csv','lrdecay/results/test_table.csv',
                 'games/results/results.csv','transfer/results/table_overall.csv']
        for setting in ['ssl','spikes','plasticity','lrdecay','games','transfer']:
            paths += [str(p.relative_to(SOURCE)) for p in (SOURCE/setting).glob('*.py')]
            paths += [setting+'/REPORT_ru.md']
        for rel in paths:
            dst=E/rel; dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/rel,dst)
    old=pd.read_csv(E/'ssl/results_main/test_final.csv')
    new=pd.read_csv(E/'ssl/results_confirm/confirm_final.csv')
    data=pd.concat([old,new],ignore_index=True)
    assert set(data.seed)=={10,11,12,20,21} and len(data)==145
    assert data.groupby('seed').size().eq(29).all()
    assert not data.duplicated(['seed','name']).any()
    rules=pd.read_csv(E/'ssl/results_main/T1_config_ranking.csv').dropna(subset=['col']).set_index('stat')
    reported=pd.read_csv(E/'ssl/results_confirm/T1_confirm.csv')
    stats=[]
    for method, rule in rules.iterrows():
        for seed,g in data.groupby('seed'):
            v=g[rule.col].to_numpy(float)*rule.sign;ok=np.isfinite(v)
            if ok.sum()<5: continue
            v[~ok]=v[ok].min()-1
            stats.append(dict(method=method,seed=int(seed),rho=float(spearmanr(v,g.acc).statistic),
                              regret=float(g.acc.max()-g.acc.iloc[np.argsort(-v)[0]])))
    per=pd.DataFrame(stats);per.to_csv(E/'ssl/recomputed_per_seed.csv',index=False)
    for row in reported.itertuples():
        if row.stat=='random_choice':continue
        seeds=[20,21] if row.set=='new_20_21' else [10,11,12,20,21]
        g=per[(per.method==row.stat)&per.seed.isin(seeds)]
        assert np.isclose(g.rho.mean(),row.rho,atol=1e-10),row.stat
        assert np.isclose(g.regret.mean(),row.regret1,atol=1e-10),row.stat
    labels={'MG':'MG (gradient norm)','out_std':'Embedding spread', 'spectral_entropy':'Spectral entropy',
            'roughness':'Roughness','linear_pr':'Delay covariance PR','corr_dim':'Correlation dimension',
            'self_repeat':'Recurrence error','rankme_h':r'RankMe ($h$)', 'rankme_z':r'RankMe ($z$)',
            'alpha_h':r'$\alpha$-ReQ slope','lidar_z':r'LiDAR ($z$)'}
    rows=[]
    for key,label in labels.items():
        g=per[per.method==key]
        rows.append([label,f'{g[g.seed<20].rho.mean():.3f}',f'{g[g.seed>=20].rho.mean():.3f}',
                     f'{g.rho.mean():.3f}',f'{100*g.regret.mean():.2f}'])
    table('ssl_campaign.tex',['Score','Seeds 10--12','Seeds 20--21','All five',r'Regret (pp)'],rows)
    pooled=reported[reported['set']=='pooled_10_12_20_21'].set_index('stat')
    table('ssl_intervals.tex',['Comparison',r'$\Delta\rho$ lower',r'$\Delta\rho$ upper'],
          [[v,f'{pooled.loc[k,"diff_lo"]:.3f}',f'{pooled.loc[k,"diff_hi"]:.3f}']
           for k,v in labels.items() if k!='MG'])
    table('ssl_all.tex',['Statistic','Log',r'Mean $\rho$',r'Regret (pp)'],
          [[str(k).replace('_',r'\_'),str(v.col).split('|')[-1].replace('_',r'\_'),
            f'{per[per.method==k].rho.mean():.3f}',f'{100*per[per.method==k].regret.mean():.2f}']
           for k,v in rules.iterrows()])
    spikes=pd.read_csv(E/'spikes/results_test.csv');primary=spikes[spikes.budget==2].set_index('method')
    table('spikes_campaign.tex',['Score','Hits',r'FP/10k','AUROC'],
          [[str(k).replace('_',r'\_'),v.test_hits,f'{v.test_fa10k:.2f}',f'{v.test_AUROC:.3f}']
           for k,v in primary.iterrows() if '|' not in k and k!='best non-MG (any column)'])
    fig,ax=plt.subplots(figsize=(6.8,2.6),layout='constrained')
    keys=['MG','out_std','linear_pr','roughness','rankme_h','rankme_z','alpha_h','lidar_z']
    for i,key in enumerate(keys):
        g=per[per.method==key].sort_values('seed')
        for j,row in enumerate(g.itertuples()):
            ax.scatter(i+(j-2)*.045,row.rho,marker='o' if row.seed<20 else '^',
                       s=25,color='#2563eb' if key=='MG' else '#65758b')
        ax.plot([i-.22,i+.22],[g.rho.mean()]*2,color='black',lw=1.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#65758b',label='Seeds 10–12'),
                       Line2D([],[],marker='^',ls='',color='#65758b',label='New seeds 20–21'),
                       Line2D([],[],color='black',label='Mean over five seeds')],fontsize=8,ncol=3,loc='lower left')
    ax.set(xticks=range(len(keys)),xticklabels=['MG','Embedding\nspread','Delay PR','Roughness','RankMe h','RankMe z',r'$\alpha$-ReQ','LiDAR'],
           ylabel='Spearman correlation',ylim=(-.15,.9))
    ax.tick_params(labelsize=8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    fig.savefig(H/'figures/ssl_campaign.pdf');plt.close(fig)
    # Timing metadata sums BOTH RankMe(h) and RankMe(z); do not claim a single-score speedup.
    costs=json.loads((E/'ssl/results_main/costs_ms.json').read_text())
    result={'ssl_records':len(data),'ssl_seeds':sorted(map(int,data.seed.unique())),
            'all_ranking_and_regret_values_recomputed':True,
            'MG_rho':float(per[per.method=='MG'].rho.mean()),
            'MG_regret_pp':float(100*per[per.method=='MG'].regret.mean()),
            'spike_hits':primary.loc['MG','test_hits'],
            'bootstrap_unit':'configurations, shared draw across seeds; pointwise, not multiplicity-adjusted',
            'timing_caveat':'MG analysis excludes per-step gradient-norm reduction; RankMe timer includes two SVDs',
            'costs_ms':costs}
    (H/'campaign_validation.json').write_text(json.dumps(result,indent=2))
    manifest=[dict(path=str(p.relative_to(E)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in E.rglob('*') if p.is_file() and p.name!='manifest.json']
    (E/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
