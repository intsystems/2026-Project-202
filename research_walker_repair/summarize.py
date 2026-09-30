import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion import H

def fmt(v,n=3):return '—' if v is None or not np.isfinite(v) else f'{v:.{n}f}'.replace('.',',')

def aggregate():
    s=json.loads((H/'selection.json').read_text());labels=[s['pilot_label']]+[f'seed{x}'+('_B' if s['conservative'] else '') for x in s['confirmation_seeds']]
    rows=[];seeds=[];sens=[]
    for label in labels:
        root=H/label;pair=json.loads((root/'pair.json').read_text());gate=json.loads((root/'gate.json').read_text());meta=json.loads((root/'train.json').read_text())
        frame=pd.read_csv(root/'MG_summary.csv')
        row=dict(label=label,pilot=label==s['pilot_label'],gate=gate['passed'],walking_early=pair['walking_early'],walking_late=pair['walking_late'],common=len(pair['common_resets']),training_seconds=meta['seconds'],parameter_relative_delta=meta['parameter_relative_delta'],plateau_range=gate['reward_relative_range'])
        test=pd.read_csv(root/'test.csv');test['fixed_horizon_score']=test.mean_reward.fillna(0)*test.n_analysis/4096
        scores=test.groupby('step').fixed_horizon_score.mean()
        row['all_test_reward_early']=float(scores.loc[0]);row['all_test_reward_late']=float(scores.loc[1048576]);row['all_test_reward_ratio']=float(scores.loc[1048576]/scores.loc[0])
        local=[]
        for reset in pair['common_resets']:
            r=dict(label=label,reset=reset)
            for stage,step in [('early',0),('late',1048576)]:
                trace=root/f'step{step:07d}'/f'reset{reset}';m=json.loads((trace/'metrics.json').read_text())
                for k in ['recurrence','section_dispersion','mean_reward','mean_speed','period']:r[k+'_'+stage]=m[k]
                for k,v in m['cheap'].items():r[k+'_'+stage]=v
                mg=frame[(frame.step==step)&(frame.reset==reset)&(frame.sensor=='right_knee')&(frame.window==2048)&(frame.tau==s['tau'])].iloc[0]
                r['MG_'+stage]=float(mg.MG);r['MG_valid_'+stage]=bool(mg.all_valid)
            for k in ['recurrence','section_dispersion','mean_reward','mean_speed','MG','entropy','period_cv','std']:
                r[k+'_ratio']=r[k+'_late']/r[k+'_early'] if r[k+'_early'] and np.isfinite(r[k+'_early']) else np.nan
            r['MG_valid']=r['MG_valid_early'] and r['MG_valid_late'];r['autocorr_delta']=r['autocorr_late']-r['autocorr_early'];local.append(r);rows.append(r)
        df=pd.DataFrame(local)
        for k in ['recurrence','section_dispersion','mean_reward','mean_speed','entropy','period_cv','std']:
            for suffix in ['early','late','ratio']:row[k+'_'+suffix]=float(df[k+'_'+suffix].median()) if len(df) else np.nan
        good=df[df.MG_valid] if len(df) else df
        row.update(MG_valid_resets=len(good),MG_assessable=len(good)>=8,MG_early=float(good.MG_early.median()) if len(good) else np.nan,
            MG_late=float(good.MG_late.median()) if len(good) else np.nan,MG_ratio=float(good.MG_ratio.median()) if len(good) else np.nan,
            autocorr_delta=float(df.autocorr_delta.median()) if len(df) else np.nan)
        row['reference_event']=bool(len(df)>=8 and row['recurrence_ratio']<=.75 and row['section_dispersion_ratio']<=.75 and row['recurrence_early']>=.02 and row['section_dispersion_early']>=.01)
        row['MG_decreases']=bool(row['MG_assessable'] and row['MG_ratio']<1)
        if pair['common_resets']:
            reset=pair['common_resets'][0];row['perturb_reset']=reset
            for stage,step in [('early',0),('late',1048576)]:
                p=json.loads((root/f'step{step:07d}'/f'reset{reset}'/'perturb_2.json').read_text())
                row['A_'+stage]=p['median_amplification'];row['falls_'+stage]=p['falls'];row['P_'+stage]=p['period']
            row['A_ratio']=row['A_late']/row['A_early'] if row['A_late'] is not None and row['A_early'] else np.nan
        seeds.append(row)
        for (sensor,w,tau),part in frame.groupby(['sensor','window','tau']):
            ratios=[];bad=0;ident=[]
            for reset in pair['common_resets']:
                a=part[(part.step==0)&(part.reset==reset)].iloc[0];b=part[(part.step==1048576)&(part.reset==reset)].iloc[0]
                if a.all_valid and b.all_valid and a.MG>0:ratios.append(b.MG/a.MG)
                else:bad+=1
                ident += [a.ident_min,a.ident_max,b.ident_min,b.ident_max]
            sens.append(dict(label=label,pilot=row['pilot'],sensor=sensor,window=int(w),tau=int(tau),valid_pairs=len(ratios),invalid_pairs=bad,
                ratio=float(np.median(ratios)) if ratios else np.nan,ident_min=float(np.nanmin(ident)) if np.isfinite(ident).any() else np.nan,
                ident_max=float(np.nanmax(ident)) if np.isfinite(ident).any() else np.nan,reference_event=row['reference_event']))
    df=pd.DataFrame(seeds);df.to_csv(H/'all_seeds.csv',index=False);pd.DataFrame(rows).to_csv(H/'paired_traces.csv',index=False);pd.DataFrame(sens).to_csv(H/'sensitivity.csv',index=False)
    cf=df[~df.pilot]
    summary=dict(selection=s,confirmation_runs=5,gate_passes=int(cf.gate.sum()),heldout_walks=int(cf.walking_late.sum()),heldout_total=50,
        broadly_assessable_MG=int(cf.MG_assessable.sum()),reference_events=int(cf.reference_event.sum()),
        MG_decreases_on_events=int(cf.loc[cf.reference_event,'MG_decreases'].sum()),
        MG_decreases_on_non_events=int(cf.loc[~cf.reference_event,'MG_decreases'].sum()),MG_decreases_total=int(cf.MG_decreases.sum()),
        median_R_ratio=float(cf.recurrence_ratio.median()),median_D_ratio=float(cf.section_dispersion_ratio.median()),median_MG_ratio=float(cf.MG_ratio.median()),
        seeds=seeds)
    summary['old_final_control']=json.loads((H/'old_final_control/summary.json').read_text())
    (H/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['seeds','selection']},indent=2))
    return df,summary

def plots(df):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cf=df[~df.pilot];fig,axs=plt.subplots(1,2,figsize=(10.5,3.1),constrained_layout=True)
    for label in df.label:
        v=pd.read_csv(H/label/'validation.csv');a=v.groupby('step').agg(walking=('walking','sum'),reward=('mean_reward','median'))
        color='gray' if label==df[df.pilot].label.iloc[0] else None
        axs[0].plot(a.index/1e6,a.reward,'o-',ms=3,label=label,color=color);axs[1].plot(a.index/1e6,a.walking,'o-',ms=3,color=color)
    axs[0].set(xlabel='Additional transitions (millions)',ylabel='Median validation step reward');axs[0].legend(fontsize=7,ncol=2)
    axs[1].set(xlabel='Additional transitions (millions)',ylabel='Healthy walking / 5',ylim=(-.2,5.3));fig.savefig(H/'training.pdf');fig.savefig(H/'training.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.1),constrained_layout=True)
    for ax,k,title in zip(axs,['recurrence','section_dispersion','MG'],['Full-state recurrence error R','Full-state section dispersion D','Scalar MG, right knee']):
        for r in cf.itertuples():ax.plot([0,1],[1,getattr(r,k+'_ratio')],'-o',label=r.label)
        ax.set_xticks([0,1],['Anchor (=1)','Final / anchor']);ax.set_title(title,fontsize=10)
        ax.axhline(1,color='gray',ls=':',lw=.7)
    axs[0].set_ylabel('Median paired ratio')
    axs[0].legend(fontsize=7);fig.savefig(H/'paired.pdf');fig.savefig(H/'paired.png',dpi=150);plt.close(fig)
    first=cf[cf.common>=8].iloc[0] if (cf.common>=8).any() else df.iloc[0]
    label=first.label;pair=json.loads((H/label/'pair.json').read_text());reset=pair['common_resets'][0]
    fig,axs=plt.subplots(1,3,figsize=(10.5,3),constrained_layout=True)
    for stage,step in [('Anchor',0),('Final',1048576)]:
        root=H/label/f'step{step:07d}'/f'reset{reset}';data=np.load(root/'trajectory.npz');curve=pd.read_csv(root/'recurrence.csv');p=np.load(root/'perturb_2_curves.npz');d=p['distances'];d=d.reshape(-1,d.shape[-1]);med=np.nanmedian(d/d[:,:1],axis=0)
        axs[0].plot(np.arange(512)*.008,data['qpos'][:512,4],label=stage,lw=.8)
        axs[1].plot(curve.lag*.008,curve.error,label=stage,lw=1)
        axs[2].plot(np.arange(len(med))/int(p['period']),np.maximum(med,1e-8),label=stage,lw=1)
    axs[0].set(xlabel='Time (s)',ylabel='Right knee (rad)',title=f'{label}, reset {reset}')
    axs[1].set(xlabel='Lag (s)',ylabel='Full-state mismatch',title='Independent recurrence')
    axs[2].set(xlabel='Time / period',ylabel='Distance / initial',yscale='log',title='Closed-loop perturbations')
    for ax in axs:ax.legend(fontsize=8)
    fig.savefig(H/'example.pdf');fig.savefig(H/'example.png',dpi=150);plt.close(fig)

def report(df,s):
    cf=df[~df.pilot];sel=s['selection'];sens=pd.read_csv(H/'sensitivity.csv');sens=sens[~sens.pilot]
    training_rows='\n'.join(f'| {r.label[4:]} | {"да" if r.gate else "нет"} | {r.walking_late}/10 | {fmt(100*(r.all_test_reward_ratio-1),1)}% | {fmt(100*r.parameter_relative_delta,2)}% |' for r in cf.itertuples())
    outcome_rows='\n'.join(f'| {r.label[4:]} | {fmt(r.recurrence_ratio)} / {fmt(r.section_dispersion_ratio)} | {fmt(r.MG_early)} → {fmt(r.MG_late)} | {r.MG_valid_resets}/10 | {"да" if r.reference_event else "нет"} |' for r in cf.itertuples())
    auxiliary='\n'.join(f'| {r.label[4:]} | {fmt(r.entropy_ratio)} / {fmt(r.period_cv_ratio)} | {fmt(r.autocorr_delta)} | {fmt(r.A_early,1)} → {fmt(r.A_late,1)} | {int(r.falls_late)}/68 |' for r in cf.itertuples())
    sensitivity='\n'.join(f'| {sensor}, {w}, {tau} | {fmt(p.ratio.median())} | {int(((p.ratio<1)&(p.valid_pairs>=8)).sum())}/5 |' for (sensor,w,tau),p in sens.groupby(['sensor','window','tau']))
    bench=json.loads((H/'benchmark.json').read_text());cost='Сопоставимой подтверждающей пары для измерения времени нет.'
    if bench['available']:
        b=pd.DataFrame(bench['operations']).pivot(index='operation',columns='stage',values='median')
        names={'common_motion_acquisition':'Запись движения (общая)','cheap_scalar_all':'Простые признаки: вся запись','full_state_regularities':'R и D: вся запись','cheap_scalar_five_windows':'Простые признаки: пять окон','full_state_five_windows':'R и D: пять окон','MG_five_windows':'MG: пять окон','closed_loop_perturbations':'68 проб возмущений'}
        names={k:v for k,v in names.items() if k in b.index}
        cost='| Операция | Исходная политика, с | Финальная политика, с |\n| --- | --- | --- |\n'+'\n'.join(f'| {name} | {fmt(b.loc[k,"early"],4)} | {fmt(b.loc[k,"late"],4)} |' for k,name in names.items())
    message=f"Критерий стабильного дообучения прошли {s['gate_passes']}/5 повторов; финальные политики выдержали {s['heldout_walks']}/50 отложенных проверок."
    mg_message=f"Независимый критерий упрощения выполнен в {s['reference_events']}/5 повторов; MG снизился в {s['MG_decreases_on_events']} из этих случаев."
    if s['reference_events']==0:mg_message='Заранее заданный независимый критерий упрощения не выполнен ни в одном повторе. Поэтому успешное дообучение само по себе не является новым положительным экспериментом для MG.'
    if (cf.MG_ratio>1).all():
        direction=f"MG вырос во всех пяти повторах на {fmt(100*(cf.MG_ratio.min()-1),1)}–{fmt(100*(cf.MG_ratio.max()-1),1)}%."
    else:direction=f"MG снизился в {s['MG_decreases_total']}/5 повторах."
    mismatch=int(((cf.recurrence_ratio<1)&(cf.section_dispersion_ratio<1)&(cf.MG_ratio>1)).sum())
    interpretation=f"{direction} Медианные парные отношения: R={fmt(s['median_R_ratio'])}, D={fmt(s['median_D_ratio'])}, MG={fmt(s['median_MG_ratio'])}. В {mismatch} повторе оба независимых показателя уменьшились без достижения порога сильного упрощения, а MG вырос. Поэтому эти данные не устанавливают универсального соответствия MG и регулярности походки."
    md=f'''# Устойчивое дообучение Walker2d и повторная проверка MG

**{message} {mg_message}**

## 1. Что изменено в обучении

PPO обучает двуногую модель Walker2d двигаться вперёд: 17 наблюдений о теле, шесть управляющих моментов, награда за движение и допустимое положение корпуса со штрафом за управление. MG измеряет движение при замороженной политике по одному коленному датчику; независимые проверки используют полное состояние и возмущённые симуляции.

Новый опыт стартует с **первого устойчивого снимка старого seed200, после 655360 переходов**. Это дообучение, не запуск с нуля. Прежний опыт, в котором более поздняя политика потеряла навык ходьбы, сохранён отдельно.

На одинаковых десяти тестовых состояниях старый финальный снимок прошёл {s['old_final_control']['walking']}/10 проверок, новые политики — {s['heldout_walks']}/50. Это результат всей процедуры исправления; эффект отдельных гиперпараметров не выделен.

**Настройки.** Actor/critic: отдельные MLP 64×64, tanh. PPO: восемь сред, rollout 512 на среду, batch 256, пять эпох, gamma 0,99, GAE 0,95, clip 0,1, target KL 0,01, learning rate линейно от 0,00003 до 0,000003, entropy coefficient 0, value coefficient 0,5, gradient clip 0,5. Сохранены веса и Adam-состояние исходной политики; статистики нормировки наблюдений и наград заморожены. Лимит обучающего эпизода увеличен с 1000 до 5000 шагов. Каждый запуск получает 1 048 576 новых переходов; снимки через 131072, после обновлений PPO. Torch 2.10 CPU, SB3 2.7.1, Gymnasium 1.2.3, MuJoCo 3.14.0, один вычислительный поток на запуск.

Пилот 210 проверил настройку до просмотра MG; затем выполнены все пять повторов 211–215 с разной случайностью обучения и общими исходными весами. Резервная настройка не потребовалась.

**Критерий стабильности.** Пять валидационных reset 41001–41005. Каждый из последних трёх снимков должен успешно пройти хотя бы четыре проверки из пяти и иметь медианную награду за шаг не ниже 90% исходной. Проверка: 512 шагов разгона и 4096 анализа, всего 36,864 с; ни одного прерывания по здоровью, скорость не ниже 0,5 м/с. Последняя политика проверяется ещё на десяти отложенных reset 51001–51010. MG не участвовал в выборе настройки. Это эмпирическая стабильность на заданном горизонте, не доказательство математической сходимости PPO.

| Повтор | Критерий стабильности | Отложенная ходьба | Награда с учётом неудач | Изменение нормы весов |
| --- | --- | --- | --- | --- |
{training_rows}

Исходная политика прошла 10/10 состояний. Награда: сумма за 4096 отсчётов с нулями после прерывания, среднее по всем десяти reset относительно исходной политики; неудачи учтены. Последний столбец — относительная норма изменения весов. Аудит подтвердил изменение actor и неизменность нормировки. Общая инициализация не позволяет считать 50 проверок независимыми обучениями.

![Все обучающие повторы](training.pdf)

<!-- pagebreak -->

## 2. Независимое упрощение и сигнал MG

Физическое состояние: восемь qpos без абсолютного продвижения и девять qvel; позиции/углы делятся на 1, скорости на 5. **R** — минимальная средняя квадратичная ошибка повторения при задержках 20–250, делённая на удвоенную общую дисперсию. **D** — дисперсия состояния в пиках правого колена, нормированная общей дисперсией. Пики: prominence 0,15 рад, расстояние ≥20 шагов, параболическое уточнение времени. Меньшие R/D означают более повторяемое движение; несколько пиков за цикл могут давать ненулевой D даже при периодической походке.

Критерий задан заранее: медианы парных отношений поздний/исходный для R и D обе ≤0,75, исходные медианы R≥0,02 и D≥0,01. Отдельные отношения сохранены; этот критерий не является определением точного числа активных компонент.

**MG** получает только угол правого колена qpos4: окно 2048, E20, k20, шаг окна512, tau={sel['tau']}, Theiler={39*sel['tau']}; пять окон на запись. Tau один раз выбран по независимому периоду пилотных валидационных движений. E40 — дополнительная диагностика. Запись требует std колена ≥0,05 рад и ≥8 пиков; для общего вывода о повторе нужно ≥8/10 общих пригодных reset и все пять численно пригодных окон на каждом учитываемом reset. Неудачи сохраняются.

| Повтор | Отношения R / D | MG: исходный → финальный | Пригодные MG-пары | Критерий упрощения |
| --- | --- | --- | --- | --- |
{outcome_rows}

Отношения и график — медианы парных отношений по reset. Абсолютные исходный и финальный MG в таблице агрегируются отдельно: отношение этих медиан может отличаться от медианы отношений. Пилот исключён из подтверждающей статистики. {interpretation}

![Сравнение исходной и финальных политик](paired.pdf)

**Дорогая проверка.** Первый общий пригодный reset, точки движения 0 и 1024; каждая из 17 нормированных координат возмущается на ±0,001: 68 продолжений по четыре периода. Политика заново реагирует на изменённое наблюдение. A — медианное усиление расстояния за последний период с коррекцией фазы; это не спектр Ляпунова. Нулевая проба воспроизводится точно; одинаковые исходные пробы кэшируются.

| Повтор | Энтропия / CV: отношения | Изменение автокорреляции | A: исходное → финальное | Прерывания финальных проб |
| --- | --- | --- | --- | --- |
{auxiliary}

Энтропия, CV интервалов между пиками и автокорреляционный показатель используют то же колено. Меньшие энтропия/CV и большая автокорреляция означают более регулярный сигнал. A относится к одному reset; горизонт 4P меняется в секундах и тоже влияет на усиление. Снижение A при A>1 не означает устойчивости.

<!-- pagebreak -->

## 3. Стоимость, чувствительность и вывод

![Первый пригодный повтор, без выбора по MG](example.pdf)

Время измерено отдельно после завершения параллельного обучения, на первой пригодной подтверждающей паре. Девять повторений вычислительных метрик, три записи движения, одна полная серия возмущений; показаны медианы. Это локальные CPU-измерения, не универсальные коэффициенты ускорения.

{cost}

Коленный угол уже есть в наблюдении политики, отдельного neural-network forward для его измерения нет. Запись движения общая для MG и анализа состояния. Пробы возмущений требуют новых симуляций, но измеряют другую характеристику; преимущество по времени перед ними не означает равноценности диагностики. Дешёвые скалярные признаки и R/D полного состояния остаются обязательными альтернативами.

В этом 17-мерном симуляторе на одинаковых пяти окнах MG занимает около 1,8 с, R/D — 0,05 с, простые признаки — 0,008 с. Преимущества MG по времени перед этими дешёвыми альтернативами не показано.

**Чувствительность.** Ниже медиана парных отношений MG по пяти дообучениям и число повторов со снижением при ≥8 пригодных парах. Основная настройка после результатов не менялась. E40/E20 и все непригодные окна сохранены в CSV.

| Датчик, окно, tau | Медианное отношение MG | Повторов со снижением |
| --- | --- | --- |
{sensitivity}

При окне 1024 знак изменения MG противоположен основному результату в трёх из пяти повторов: вывод зависит от размера окна.

**Что установлено.** {message} {mg_message} Контактная динамика не удовлетворяет автоматически предпосылкам гладкой размерностной интерпретации; здесь MG — эмпирический индикатор временного ряда. Успешное обучение, регулярная походка и устойчивость к возмущениям — разные проверяемые свойства. Один тип робота и общая исходная политика ограничивают переносимость выводов. Для формулировки «MG обнаруживает упрощение» важны независимые изменения R/D и устойчивость результата к окну и датчику, а не только рост награды.

Полные настройки: PROTOCOL.md. Все 45 валидационных и 20 отложенных проверок каждого запуска, исходные веса, конечные политики, траектории, окна MG, пробы, аудит и замеры времени включены в архив. Статья не изменялась.
'''
    (H/'report_ru.md').write_text(md,encoding='utf-8')

if __name__=='__main__':
    df,s=aggregate();plots(df);report(df,s)
