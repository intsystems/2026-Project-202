import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion import H,RESETS
def num(x,n=3):return '—' if x is None or not np.isfinite(x) else f'{x:.{n}f}'.replace('.',',')

def main():
    selection=json.loads((H/'selection.json').read_text());tau=selection['tau'];nanchors=len(selection['anchors']);traces=[];seeds=[]
    for seed in [200]+selection['confirmation_seeds']:
        out=H/f'seed{seed}';pair=json.loads((out/'pair.json').read_text());seedrow=dict(seed=seed,usable=pair['usable'],early=pair['early'],late=pair['late'],resets=len(pair['common_resets']))
        if pair['usable']:
            frame=pd.read_csv(out/'MG_summary.csv')
            for reset in pair['common_resets']:
                row=dict(seed=seed,reset=reset,early=pair['early'],late=pair['late'])
                for stage in ['early','late']:
                    root=out/f"step{pair[stage]:07d}"/f'reset{reset}'
                    m=json.loads((root/'metrics.json').read_text());p=json.loads((root/f'perturb_{nanchors}.json').read_text())
                    mg=frame[(frame.step==pair[stage])&(frame.reset==reset)&(frame.window==2048)&(frame.tau==tau)&(frame.sensor=='right_knee')].iloc[0]
                    for key in ['recurrence','section_dispersion','mean_speed','period']:
                        row[f'{key}_{stage}']=m[key]
                    for key in ['std','entropy','autocorr','period_cv']:row[f'{key}_{stage}']=m['cheap'][key]
                    row[f'MG_{stage}']=float(mg.MG);row[f'MG_valid_{stage}']=bool(mg.all_valid)
                    row[f'amplification_{stage}']=p['median_amplification'];row[f'falls_{stage}']=p['falls'];row[f'probes_{stage}']=p['probes']
                for key in ['MG','recurrence','section_dispersion','amplification','entropy','std','period_cv']:
                    a=row[f'{key}_early'];b=row[f'{key}_late'];row[key+'_ratio']=b/a if a is not None and b is not None and a>0 else None
                row['MG_assessable']=row['MG_valid_early'] and row['MG_valid_late'];traces.append(row)
            part=pd.DataFrame([r for r in traces if r['seed']==seed])
            for col in part.columns:
                if col not in ['seed','reset','early','late','MG_assessable','MG_valid_early','MG_valid_late']:seedrow[col]=float(part[col].median()) if part[col].notna().any() else None
            seedrow['MG_assessable']=bool(part.MG_assessable.all())
            seedrow['reference_simplification']=bool(seedrow['recurrence_ratio']<=.75 and seedrow['section_dispersion_ratio']<=.75
                and seedrow['recurrence_early']>=.02 and seedrow['section_dispersion_early']>=.01)
            seedrow['MG_decreases']=bool(seedrow['MG_assessable'] and seedrow['MG_ratio']<1)
        else:seedrow.update(MG_assessable=False,reference_simplification=False,MG_decreases=False)
        seeds.append(seedrow)
    df=pd.DataFrame(seeds);df.to_csv(H/'all_seeds.csv',index=False);pd.DataFrame(traces).to_csv(H/'paired_traces.csv',index=False)
    cf=df[df.seed>200];valid=cf[cf.usable];events=valid[valid.reference_simplification];assessable=events[events.MG_assessable]
    result=dict(selection=selection,planned_confirmation=5,usable_pairs=len(valid),reference_confirmed=len(events),
        reference_events_MG_assessable=len(assessable),MG_decreases_on_confirmed=int(assessable.MG_decreases.sum()),
        non_events=int((~valid.reference_simplification).sum()),MG_decreases_on_non_events=int(valid.loc[~valid.reference_simplification,'MG_decreases'].sum()),
        failed_pair_seeds=cf.loc[~cf.usable,'seed'].tolist(),seeds=seeds)
    if len(assessable):result['confirmed_median_MG_ratio']=float(assessable.MG_ratio.median())
    (H/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['selection','seeds']},indent=2))
    sensitivity(df,selection);plot(df,traces,selection);report(df,result,traces)

def sensitivity(df,selection):
    rows=[]
    for seedrow in df[df.usable].itertuples():
        pair=json.loads((H/f'seed{seedrow.seed}/pair.json').read_text())
        frame=pd.read_csv(H/f'seed{seedrow.seed}/MG_summary.csv')
        for (sensor,w,tau),part in frame.groupby(['sensor','window','tau']):
            ratios=[];valid=True;idents=[]
            for reset in pair['common_resets']:
                a=part[(part.step==pair['early'])&(part.reset==reset)].iloc[0]
                b=part[(part.step==pair['late'])&(part.reset==reset)].iloc[0]
                valid=valid and bool(a.all_valid) and bool(b.all_valid)
                ratios.append(b.MG/a.MG if a.MG>0 else np.nan)
                idents += [a.ident_min,a.ident_max,b.ident_min,b.ident_max]
            rows.append(dict(seed=seedrow.seed,sensor=sensor,window=w,tau=tau,all_valid=valid,
                MG_ratio=float(np.median(ratios)),reference_simplification=seedrow.reference_simplification,
                ident_min=float(np.nanmin(idents)) if np.isfinite(idents).any() else None,
                ident_max=float(np.nanmax(idents)) if np.isfinite(idents).any() else None))
    pd.DataFrame(rows).to_csv(H/'sensitivity.csv',index=False)

def plot(df,traces,selection):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    valid=df[df.usable];cf=valid[valid.seed>200]
    fig,axs=plt.subplots(1,3,figsize=(11,3.2),constrained_layout=True)
    for ax,key,title in zip(axs,['recurrence','section_dispersion','MG'],['Full-state recurrence error','Full-state section dispersion','Scalar MG, right knee']):
        for row in valid.itertuples():
            ax.plot([0,1],[getattr(row,key+'_early'),getattr(row,key+'_late')],'-o',label=str(row.seed),color='gray' if row.seed==200 else None,alpha=.8)
        ax.set_xticks([0,1],['First eligible','Final']);ax.set_title(title)
    axs[0].legend(title='Policy seed',fontsize=7,ncol=2)
    fig.savefig(H/'paired.pdf');fig.savefig(H/'paired.png',dpi=170);plt.close(fig)
    if not len(valid):return
    example=int(cf.seed.min()) if len(cf) else 200;pair=json.loads((H/f'seed{example}/pair.json').read_text());reset=pair['common_resets'][0]
    fig,axs=plt.subplots(2,2,figsize=(10.5,5.6),constrained_layout=True)
    for stage,color in [('early','#d95f02'),('late','#0868ac')]:
        root=H/f'seed{example}'/f"step{pair[stage]:07d}"/f'reset{reset}';data=np.load(root/'trajectory.npz');m=json.loads((root/'metrics.json').read_text())
        axs[0,0].plot(np.arange(512)*.008,data['qpos'][:512,4],label=stage,color=color,lw=.8)
        axs[0,1].plot(data['qpos'][:2048,4],data['qvel'][:2048,4],color=color,lw=.5,label=stage,alpha=.65)
        curve=pd.read_csv(root/'recurrence.csv');axs[1,0].plot(curve.lag*.008,curve.error,label=stage,color=color)
        probe=np.load(root/f"perturb_{len(selection['anchors'])}_curves.npz");a=probe['distances'];norm=a/a[:,:,:1]
        med=np.nanmedian(norm.reshape(-1,norm.shape[-1]),axis=0)
        axs[1,1].plot(np.arange(len(med))/int(probe['period']),np.maximum(med,1e-8),label=stage,color=color)
    axs[0,0].set(xlabel='Time (s)',ylabel='Right knee (rad)',title=f'Frozen policies, seed {example}, reset {reset}')
    axs[0,1].set(xlabel='Right knee (rad)',ylabel='Knee velocity (rad/s)',title='Same sensor phase portrait')
    axs[1,0].set(xlabel='Lag (s)',ylabel='Normalized full-state mismatch',title='Independent full-state recurrence')
    axs[1,1].set(xlabel='Time / estimated period',ylabel='Transverse distance / initial',yscale='log',title='Closed-loop perturbations, median')
    for ax in axs.flat:ax.legend(fontsize=8)
    fig.savefig(H/'example.pdf');fig.savefig(H/'example.png',dpi=170);plt.close(fig)
    (H/'example_selection.json').write_text(json.dumps(dict(seed=example,reset=reset,rule='First usable confirmation seed; pilot fallback only if none usable'),indent=2))

def report(df,result,traces):
    selection=result['selection'];cf=df[df.seed>200];valid=cf[cf.usable];confirmed=valid[valid.reference_simplification]
    if result['reference_confirmed']:
        conclusion=f"Независимый критерий упрощения выполнен в {result['reference_confirmed']} из {result['usable_pairs']} пригодных пар. MG снизился в {result['MG_decreases_on_confirmed']} из {result['reference_events_MG_assessable']} численно пригодных подтверждённых случаев."
    else:conclusion=f"Основной независимый критерий упрощения не выполнен ни в одной из {result['usable_pairs']} пригодных пар. Поэтому эта серия не является подтверждением способности MG обнаруживать заданное упрощение."
    table=[]
    for r in df.itertuples():
        if not r.usable:table.append(f'| {r.seed} | нет пригодной пары | — | — | — |');continue
        status='да' if r.reference_simplification else 'нет'
        table.append(f'| {r.seed}{" (пилот)" if r.seed==200 else ""} | {int(r.early)} / {int(r.late)} | {num(r.recurrence_ratio)} / {num(r.section_dispersion_ratio)} | {num(r.MG_ratio)} | {status} |')
    bench=json.loads((H/'benchmark.json').read_text()) if (H/'benchmark.json').exists() else dict(available=False)
    cost_text='Нет пригодной подтверждающей пары для сопоставимого benchmark.';cost_table='';speed_claim=''
    if bench['available']:
        cost=pd.DataFrame(bench['operations']);names={'common_motion_acquisition':'Общая запись движения','cheap_scalar_all':'Простые признаки одного сустава','full_state_regularities':'Регулярность полного состояния','MG_five_windows':'MG: пять окон','closed_loop_perturbations':'Замкнутые пробы возмущений'}
        costs=cost.pivot(index='operation',columns='stage',values='median')
        cost_table='| Измерение | Ранний этап, с | Финальный этап, с |\n| --- | --- | --- |\n'+'\n'.join(f'| {name} | {num(costs.loc[key,"early"],4)} | {num(costs.loc[key,"late"],4)} |' for key,name in names.items())
        cost_text=f"Последовательный CPU-benchmark на seed {bench['seed']}, reset {bench['reset']}, по одному потоку. Для MG и простых метрик — девять повторов, для записи движения — три. Полный набор возмущений рассчитан по одному разу на этап: его время описательное, без надёжной оценки разброса."
        speed_claim='Время получения исходной траектории общее и прибавляется к обоим методам. Экономия относительно проб возмущений не означает равную диагностическую силу: MG не измеряет коэффициент восстановления. Быстрые признаки и метрики полного состояния также включены в сравнение.'
    cheap_rows=[]
    for r in valid.itertuples():
        cheap_rows.append(f'| {r.seed} | {num(r.entropy_ratio)} | {num(r.period_cv_ratio)} | {num(r.autocorr_late-r.autocorr_early)} | {num(r.amplification_ratio)} |')
    cheap_table='| Seed | Энтропия: отношение | CV периода: отношение | Изменение автокорреляции | A: отношение |\n| --- | --- | --- | --- | --- |\n'+'\n'.join(cheap_rows)
    sens=pd.read_csv(H/'sensitivity.csv')
    sens=sens[sens.seed>200]
    sens_rows=[]
    for (sensor,w,t),part in sens.groupby(['sensor','window','tau']):
        event=part[part.reference_simplification];good=event[event.all_valid]
        sens_rows.append(f'{sensor}, W={w}, tau={t}: снижение в {int((good.MG_ratio<1).sum())}/{len(good)} пригодных подтверждённых случаев (всего событий {len(event)})')
    sensitivity_text='; '.join(sens_rows)+'.' if sens_rows else 'Пригодных подтверждающих пар нет.'
    failed='нет' if not result['failed_pair_seeds'] else ', '.join(map(str,result['failed_pair_seeds']))
    horizon=selection['horizon'];pilot_extended=json.loads((H/'seed200'/f'train_{horizon}.json').read_text())['resume']>0
    text=f'''# MG при обучении двуногой модели устойчивой походке

Эксперимент начат 29 сентября 2026. Walker2d-v5 в MuJoCo; пилот 200 и пять новых обучений 201–205. **{conclusion}**

## 1. Что учится, что измеряется и зачем

Нейросетевая политика получает 17 наблюдений о положении и скорости тела и выдаёт шесть управляющих моментов. PPO обучает её двигаться вперёд и сохранять здоровое положение корпуса, учитывая штраф за управление. Измеряем движение робота с фиксированными снимками политики, а не временной ряд loss оптимизатора. Проверяем, сопровождается ли формирование более регулярной походки снижением MG по одному датчику. Это симулятор, не физический робот.

**Обучение.** Stable-Baselines3 2.7.1, Gymnasium 1.2.3, MuJoCo 3.14.0; CPU. Отдельные actor/critic MLP 64×64, tanh; PPO: восемь сред, rollout 256 шагов на среду, minibatch 64, десять эпох, learning rate 0,0003, gamma 0,99, GAE 0,95, clip 0,2, entropy coefficient 0, value coefficient 0,5, clipping нормы градиента 0,5. Наблюдения и награды нормируются VecNormalize; эпизоды обучения ограничены 1000 шагами. Всего {horizon} переходов на seed, снимки через 131072 перехода. {'Пилот продлён один раз с 2M до 4M из-за непригодности финального этапа, до анализа MG; подтверждающие seed обучались сразу на полный выбранный горизонт.' if pilot_extended else 'Продление пилота не потребовалось.'}

**Проверка ходьбы.** Политика детерминированная, статистики нормировки заморожены. Три фиксированных начальных seed: 31001–31003. Шаг управления 0,008 с; после 512 шагов разгона записываются 4096 шагов, всего 36,864 с. Пригодная запись не содержит падения, средняя скорость не ниже 0,5 м/с, отклонение правого колена не ниже 0,05 рад и есть не менее восьми его пиков. Этап требует двух пригодных записей из трёх.

Сравниваем первый пригодный снимок с финальным, используя не менее двух общих пригодных начальных состояний. Отбор не использует MG и не выбирает самый регулярный поздний этап. Пригодных пар среди пяти подтверждающих seed: {result['usable_pairs']}; без пары: {failed}. Повторы начальных состояний не считаются новыми обученными политиками.

## 2. Независимое подтверждение регулярности

**R — ошибка повторяемости полного состояния.** Используются восемь координат положения без абсолютного перемещения вперёд и девять скоростей; скорости делятся на 5. Минимизируем средний квадрат различий состояний при задержках 20–250 шагов (0,16–2 с), делённый на удвоенную временную дисперсию состояния. Скорость движения вперёд сохранена. Минимизирующая задержка задаёт номинальный период P; это оценка, а не доказательство существования цикла.

**D — разброс полного состояния в сходных фазах шага.** Отмечаем пики правого колена (prominence 0,15 рад, расстояние не менее 20 шагов), уточняем время параболической интерполяцией и измеряем разброс всех 17 координат в этих точках, нормированный общей дисперсией. У сложной походки несколько различных пиков за период могут поддерживать ненулевой D даже при повторяемом движении.

Основной критерий зафиксирован заранее: медианы парных отношений R и D «поздний/ранний» обе не выше 0,75, а исходные медианы не ниже 0,02 и 0,01 соответственно. Последние условия исключают переинтерпретацию малых колебаний уже почти периодической походки. Это операционное определение, не теорема о размерности.

| Seed | Этапы обучения | Отношения R / D | Отношение MG | Критерий упрощения |
| --- | --- | --- | --- | --- |
{chr(10).join(table)}

Отношения сначала считаются для каждого общего начального состояния, затем берётся медиана внутри seed. Пилот исключён из подтверждающей статистики. При непригодных окнах числовой MG не используется для подтверждающего вывода; подробные флаги сохранены в CSV.

<!-- pagebreak -->

## 3. Движение и реакция на возмущения

![Первый пригодный подтверждающий seed](example.pdf)

Сверху — один и тот же сустав и его фазовый портрет; снизу — независимая повторяемость полного состояния и отклик замкнутой системы на возмущения. Оранжевый — первый пригодный этап, синий — финальный. Пример выбирается по пригодности, не по направлению изменения MG.

**Проверка восстановления.** В {len(selection['anchors'])} заранее заданных точках траектории возмущаем по очереди каждую из 17 нормированных координат на ±0,001: {34*len(selection['anchors'])} продолжений на запись. Политика заново вычисляет действия по возмущённым наблюдениям. Продолжение длится четыре номинальных периода; падения сохраняются. Расстояние до невозмущённой траектории минимизируется вдоль линейных сегментов в пределах половины периода, чтобы отделить сдвиг фазы от изменения формы движения. Показатель A — медианное расстояние за последний период, делённое на начальное поперечное расстояние. A<1 означает затухание в этом конечном опыте; A>1 — рост. Это не полный спектр Ляпунова и не доказательство асимптотической устойчивости.

Реализовано точное восстановление полного интеграционного состояния, включая warm start решателя. Нулевая проба воспроизводит исходную траекторию. Первый пробный пакетный расчёт действий не прошёл этот контроль из-за накопления различий округления; он сохранён отдельно и исключён. Все используемые пробы пересчитаны с тем же одиночным вычислением действий, что и исходная траектория.

**MG.** Основной датчик — правое колено, выбранное до запусков. Окно 2048 отсчёта, вложение 20, задержка {selection['tau']}, 20 соседей, Theiler {39*selection['tau']}; пять окон со сдвигом 512. Задержка выбрана один раз по независимому периоду пилотной походки, без просмотра MG, и заморожена для подтверждений. Проверены также другое колено, окна 1024/4096, половинная и двойная задержки; все варианты сохранены, основной после просмотра не менялся. E40 — диагностическая проверка, не замена основного результата.

![Все пригодные пары](paired.pdf)

Каждая линия — одна обученная политика; серый — пилот. Показаны медианы по начальным состояниям. Снижение MG при отсутствии основного независимого критерия наблюдалось в {result['MG_decreases_on_non_events']} из {result['non_events']} пригодных неподтверждённых случаев; это не считается доказательством упрощения.

<!-- pagebreak -->

## 4. Стоимость и границы применения

**Дешёвые альтернативы и устойчивость.** Для энтропии и CV меньшие значения означают более регулярный сигнал; для автокорреляции — большие. A характеризует конечный отклик на возмущения; уменьшение A ещё не означает затухание, если абсолютное A остаётся больше единицы. Отношения — поздний этап к раннему; медианы парных отношений по reset, без пилота.

{cheap_table}

**Чувствительность MG.** {sensitivity_text} Полные результаты и диагностика E40/E20 сохранены в sensitivity.csv; настройки не выбирались по благоприятности результата.

{cost_text}

{cost_table}

{speed_claim}

Угол колена уже присутствует в наблюдении политики. Для MG не выполняется отдельный neural-network forward ради получения скалярного loss. Однако сама запись движения требует симуляции и работы политики: эти затраты не объявляются экономией MG. Обычный коэффициент вариации периода, автокорреляция и спектральная энтропия используют тот же датчик и остаются обязательными конкурентами.

**Как читать результат.** {conclusion} Пять инициализаций относятся к одной задаче и одному расписанию обучения. Неудачные политики и непригодные сравнения не заменялись новыми seed. Улучшение награды или скорости не тождественно регулярности, а регулярность одного датчика не гарантирует устойчивости полного движения.

Контакты делают динамику гибридной; условия гладкой размерностной интерпретации здесь не установлены. Значения MG рассматриваются как эмпирический индикатор формы временного ряда, а не точное число степеней свободы. Период и амплитуда могут меняться вместе с обучением. Времена относятся к этой небольшой модели на CPU, не к большому роботу или распределённой системе.

Все политики, статистики нормировки, траектории, непринятые записи, индивидуальные пробы, окна MG и альтернативные метрики сохранены. PROTOCOL.md описывает предварительные критерии; RESOURCE_AMENDMENT.md — только изменение параллельности вычислений; NUMERICAL_AUDIT.md — исправление численного контроля. Основная статья не изменялась.
'''
    (H/'report_ru.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
