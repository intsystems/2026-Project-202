"""Re-score saved detections without fitting rules or rerunning training."""
from pathlib import Path
import json, shutil
import pandas as pd
H=Path(__file__).resolve().parent
ORDER=['MG','abs_diff','norm_diff','perm_entropy','sample_entropy','spectral_entropy','crossings','lag1','det_std','level']
NAMES=['MG','Abs. increments','Norm. increments','Perm. entropy','Sample entropy','Spectral entropy','Trend crossings','Lag-one corr.','Detrended std.','Level']
LABEL=dict(zip(ORDER,NAMES))
STRONG=['lr10','lr100','freeze_head','freeze_bias','prune50','prune80','prune95']
NULL=['base','batch_up','scale','smooth']
def table(name,headers,rows):
    text=[r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    text+=[' & '.join(map(str,row))+r' \\' for row in rows]
    text +=[r'\bottomrule',r'\end{tabular}']
    (H/'tables'/name).write_text('\n'.join(text)+'\n',encoding='utf-8')

def main():
    records=pd.read_csv(H/'new_results/baseline_records.csv')
    summary=pd.read_csv(H/'new_results/baseline_summary.csv')
    output=[]
    for stat in ORDER:
        d=records[(records.stat==stat)&(records.detector=='block')]
        a=d[d.arm.isin(STRONG)];b=d[d.arm.isin(NULL)]
        assert len(a)==28 and len(b)==16
        for horizon in [500,1000,2000,5000]:
            hit=(a.alarm>a.event_step)&(a.alarm<=a.event_step+horizon)
            output.append(dict(stat=stat,horizon=horizon,hits=int(hit.sum()),events=len(a),
                               control_false_positives=int(b.false_alarm.sum()),controls=len(b),
                               delay=float((a.loc[hit,'alarm']-a.loc[hit,'event_step']).median())))
    pd.DataFrame(output).to_csv(H/'new_results/horizon_sensitivity.csv',index=False)
    rows=[]
    for stat in ORDER:
        group=[r for r in output if r['stat']==stat]
        rows.append([LABEL[stat],*[f"{r['hits']}/28" for r in group],f"{group[0]['control_false_positives']}/16"])
    table('horizon_sensitivity.tex',['Feature','500','1,000','2,000','5,000','FP'],rows)
    for kind,filename in [('block','detectors.tex'),('cusum','cusum.tex')]:
        rows=[]
        for stat in ORDER:
            row=summary[(summary.stat==stat)&(summary.detector==kind)].iloc[0]
            fp=int(sum(row['alarms_'+arm] for arm in NULL))
            delay='---' if pd.isna(row.delay) else f'{row.delay:,.0f}'
            rows.append([LABEL[stat],f'{int(row.hits)}/28',f'{fp}/16',delay])
        table(filename,['Feature','Detected','FP','Delay'],rows)
    rules=json.loads((H/'new_results/rules.json').read_text())
    table('baseline_rules.tex',['Feature','$M$','$B$','Sign',r'$\delta$','CUSUM drift','CUSUM $h$'],[
        [LABEL[s],rules[s+'_block']['M'],rules[s+'_block']['B'],rules[s+'_block']['sign'],
         f"{rules[s+'_block']['delta']:.6g}",rules[s+'_cusum']['drift'],f"{rules[s+'_cusum']['delta']:.6g}"] for s in ORDER])
    source=H/'evidence/research_known_modes_results/measurements.csv'
    if not source.exists():
        shutil.copy2(H.parent/'research_known_modes_results/measurements.csv',source)
    data=pd.read_csv(source).query('regime == "switch"')
    rows=[]
    for obs in ['loss','probe_error']:
        for phase in ['before','after']:
            values=data.query('observer == @obs and phase == @phase').MG
            assert len(values)==3
            rows.append([obs.replace('_',' '),phase,f'{values.median():.6f}',f'{values.min():.6f}',f'{values.max():.6f}'])
    table('regression_variation.tex',['Signal','Phase','Median','Minimum','Maximum'],rows)
    mg=[r for r in output if r['stat']=='MG']
    assert [r['hits'] for r in mg]==[8,16,22,27]
    print('MG detection horizons:',[(r['horizon'],r['hits']) for r in mg])
    print('Saved horizon, all-baseline, CUSUM, calibration and regression tables.')

if __name__=='__main__':main()
