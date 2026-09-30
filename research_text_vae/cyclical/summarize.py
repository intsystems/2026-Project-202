from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
NAMES={'MG':'MG','std':'Отклонение loss','entropy':'Спектр. энтропия','KL':'KL обучения','beta':'Расписание beta','periodic':'Равные интервалы'}
COLORS={'MG':'#0868ac','std':'#d95f02','entropy':'#7570b3','KL':'#1b9e77','beta':'#999999','periodic':'#252525'}
def num(x,n=2):return '—' if x is None or not np.isfinite(x) else f'{x:.{n}f}'.replace('.',',')

def main():
    evaluations={s:json.loads((H/f'seed{s}/evaluation.json').read_text()) for s in range(100,110)}
    all_rows=pd.concat([pd.read_csv(H/f'seed{s}/scores.csv') for s in range(100,110)],ignore_index=True)
    all_rows.to_csv(H/'all_scores.csv',index=False);cf=all_rows[all_rows.seed>100]
    assert cf.suitable.all(),'A confirmation has unsuitable initial state; handle explicitly in report.'
    rng=np.random.default_rng(20260929);grouped=[]
    for (budget,method),g in cf.groupby(['budget','method'],sort=False):
        delays=[]
        for seed in range(101,110):
            d=next(p for p in evaluations[seed]['policies'] if p['budget']==budget and p['method']==method)
            delays+=d['score']['delays']
        rr=g.hits.to_numpy()/g.events.to_numpy()
        boot=np.mean(rng.choice(rr,size=(20000,len(rr)),replace=True),axis=1)
        grouped.append(dict(budget=int(budget),method=method,hits=int(g.hits.sum()),events=int(g.events.sum()),
            pooled_recall=float(g.hits.sum()/g.events.sum()),macro_recall=float(rr.mean()),
            macro_recall_ci95=np.quantile(boot,[.025,.975]).tolist(),
            checks=int(g.checks.sum()),mean_checks=float(g.checks.mean()),unproductive=int(g.unproductive.sum()),
            mean_delay=float(np.mean(delays)) if delays else None,
            matched_periodic_hits=int(g.matched_periodic_hits.sum()),
            loss_hits=int(g.loss_hits.sum()),loss_events=int(g.loss_events.sum()),
            recovery_hits=int(g.recovery_hits.sum()),recovery_events=int(g.recovery_events.sum())))
    primary={g['method']:g for g in grouped if g['budget']==12}
    mg=primary['MG'];periodic=primary['periodic']
    diff=cf[cf.budget==12].pivot(index='seed',columns='method',values='recall')
    delta=(diff.MG-diff.periodic).to_numpy()
    ci=np.quantile(np.mean(rng.choice(delta,size=(20000,len(delta)),replace=True),axis=1),[.025,.975]).tolist()
    truth_summary=[dict(seed=s,suitable=e['truth']['suitable'],MI0=e['truth']['MI0'],response0=e['truth']['response0'],
        events=e['dense']['events'],censored=e['dense']['censored_events'],loss=e['dense']['loss_events'],recovery=e['dense']['recovery_events'])
        for s,e in evaluations.items()]
    pd.DataFrame(truth_summary).to_csv(H/'events_by_seed.csv',index=False)
    bench=json.loads((H/'benchmark.json').read_text());op=bench['operations'];costs=[]
    for g in grouped:
        m=g['method'];acquisition=7168*op['probe']['median'] if m in ['MG','std','entropy'] else 0.
        if m in ['MG','std','entropy']:feature=105*op[m]['median']
        elif m in ['KL','beta']:feature=op[m+'_all105']['median']
        else:feature=0.
        processing=feature+(op['policy_all96']['median'] if m!='periodic' else 0.)
        calls=g['mean_checks']+1;refcost=calls*op['reference']['median']
        cost=acquisition+processing+refcost;cached=processing+refcost
        dense=97*op['reference']['median']
        break_even=(acquisition+processing)/(97-calls) if calls<97 else None
        costs.append(dict(budget=g['budget'],method=m,probe_seconds=acquisition,processing_seconds=processing,
            reference_seconds=refcost,total_seconds=cost,cached_probe_total_seconds=cached,dense_seconds=dense,
            relative_to_dense=cost/dense,cached_relative_to_dense=cached/dense,
            break_even_reference_seconds=break_even))
    pd.DataFrame(costs).to_csv(H/'costs.csv',index=False)
    summary=dict(pilot_seed=100,confirmation_seeds=list(range(101,110)),primary=primary,all_budgets=grouped,
        truth=truth_summary,MG_minus_periodic_macro_recall=float(delta.mean()),MG_minus_periodic_ci95=ci,
        MG_better_seeds=int((delta>0).sum()),MG_equal_seeds=int((delta==0).sum()),MG_worse_seeds=int((delta<0).sum()),
        costs=costs,degenerate_windows=sum(int(pd.read_csv(H/f'seed{s}/features.csv').degenerate.sum()) for s in range(101,110)))
    (H/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(primary=primary,delta=summary['MG_minus_periodic_macro_recall'],ci=ci),indent=2))
    plot_example(evaluations[101]);plot_cohort(grouped,costs)
    write_report(summary,all_rows,bench)

def plot_example(evaluation):
    seed=101;out=H/f'seed{seed}';ref=pd.read_csv(out/'reference.csv');feat=pd.read_csv(out/'features.csv');log=pd.read_csv(out/'logs.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,1,figsize=(10.6,7),sharex=True,constrained_layout=True)
    truth=evaluation['truth'];a=ref[ref.step>=1024]
    axs[0].plot(a.step,a.MI/truth['MI0'],label='MI / initial',color='#7b3294')
    axs[0].plot(a.step,a.shuffle_symkl/truth['response0'],label='Shuffle response / initial',color='#008837')
    axs[0].axhline(.5,color='gray',ls=':',lw=.8);axs[0].axhline(.65,color='gray',ls=':',lw=.8)
    axs[0].set_title('Independent reference: information and code use (seed 101)');axs[0].legend(fontsize=8)
    for method in ['MG','std']:
        baseline=feat[feat.end.isin([512,640,768,896,1024])][method].median()
        f=feat[feat.end>=1024];axs[1].plot(f.end,f[method]/baseline,label=method+' / initial',color=COLORS[method])
    axs[1].set_title('Causal scalar features, W=512');axs[1].legend(fontsize=8)
    methods=['MG','std','entropy','KL','beta','periodic']
    for y,m in enumerate(methods):
        d=next(p for p in evaluation['policies'] if p['budget']==12 and p['method']==m)
        axs[2].scatter(d['checks'],[y]*len(d['checks']),marker='|',s=100,color=COLORS[m])
        hits=[r['step'] for r in d['score']['records'] if r['scored_hit']]
        axs[2].scatter(hits,[y]*len(hits),marker='o',s=35,facecolors='none',edgecolors=COLORS[m])
    axs[2].set_yticks(range(6),methods);axs[2].invert_yaxis()
    axs[2].set_title('Requested reference checks (cap 12); circles mark scored detections')
    axs[2].set_xlabel('Optimizer updates');axs[2].set_xlim(1024,7168)
    for ax in axs:
        for e in truth['events']:ax.axvline(e['step'],color='#b2182b' if e['direction']=='loss' else '#1b7837',ls='--',lw=.8,alpha=.7)
    fig.savefig(H/'example.pdf');fig.savefig(H/'example.png',dpi=170);plt.close(fig)

def plot_cohort(grouped,costs):
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.4),constrained_layout=True)
    for m in ['MG','std','entropy','KL','beta','periodic']:
        rows=sorted([g for g in grouped if g['method']==m],key=lambda x:x['budget'])
        axs[0].plot([g['budget'] for g in rows],[g['pooled_recall'] for g in rows],'-o',label=m,color=COLORS[m],ms=4)
    axs[0].set(xlabel='Maximum additional reference checks',ylabel='Event recall, seeds 101-109',xticks=[6,12,24],ylim=(-.02,1.02))
    axs[0].legend(ncol=2,fontsize=8)
    data=[c for c in costs if c['budget']==12];xx=np.arange(len(data))
    axs[1].bar(xx-.18,[c['total_seconds'] for c in data],width=.36,label='Probe acquisition charged')
    axs[1].bar(xx+.18,[c['cached_probe_total_seconds'] for c in data],width=.36,label='Identical probe already logged')
    axs[1].axhline(data[0]['dense_seconds'],color='black',ls='--',lw=1,label='Dense reference')
    axs[1].set_xticks(xx,[c['method'] for c in data],rotation=20)
    axs[1].set(ylabel='Estimated monitoring seconds / run',title='Full monitoring cost, cap 12');axs[1].legend(fontsize=7)
    fig.savefig(H/'cohort.pdf');fig.savefig(H/'cohort.png',dpi=170);plt.close(fig)

def write_report(s,df,bench):
    p=s['primary'];mg=p['MG'];periodic=p['periodic'];cost={c['method']:c for c in s['costs'] if c['budget']==12}
    eventcount=mg['events'];censored=sum(x['censored'] for x in s['truth'] if x['seed']>100)
    if mg['hits']>periodic['hits']:
        result=f"MG обнаружил больше событий, чем проверки через равные интервалы: {mg['hits']} против {periodic['hits']} из {eventcount}."
    elif mg['hits']==periodic['hits']:
        result=f"MG и проверки через равные интервалы обнаружили одинаковое число событий: {mg['hits']} из {eventcount}."
    else:result=f"MG обнаружил меньше событий, чем проверки через равные интервалы: {mg['hits']} против {periodic['hits']} из {eventcount}."
    table='\n'.join(f"| {NAMES[m]} | {g['hits']}/{g['events']} ({num(100*g['pooled_recall'],1)}%) | {num(g['mean_checks'],1)} | {num(g['mean_delay'],0)} | {g['unproductive']} |" for m,g in p.items())
    costtable='\n'.join(f"| {NAMES[m]} | {num(c['probe_seconds'])} | {num(c['processing_seconds']+c['reference_seconds'])} | {num(c['total_seconds'])} | {num(c['cached_probe_total_seconds'])} |" for m,c in cost.items())
    seedtable=[]
    for seed in range(101,110):
        part=df[(df.seed==seed)&(df.budget==12)].set_index('method')
        seedtable.append(f"| {seed} | {int(part.loc['MG','events'])} | {int(part.loc['MG','hits'])}/{int(part.loc['MG','checks'])} | {int(part.loc['periodic','hits'])}/12 | {int(part.loc['MG','matched_periodic_hits'])}/{int(part.loc['MG','checks'])} |")
    b=bench['operations'];thresholds=json.loads((H/'selection.json').read_text())['settings']['12']
    thresholds_text='; '.join(f"{m}: {num(v['threshold'])}" for m,v in thresholds.items())
    small={r['method']:r for r in s['all_budgets'] if r['budget']==6}
    other_budgets=(f"При лимите 6 MG находит {small['MG']['hits']}/{eventcount}, равные интервалы — {small['periodic']['hits']}/{eventcount}, "
        f"но бесплатное расписание beta даёт {small['beta']['hits']}/{eventcount}. Преимущество MG над периодической схемой здесь не означает преимущества над всеми простыми альтернативами.")
    practical=('Практическое преимущество выбранного MG-монитора в основном сравнении не подтверждено. '
        if mg['hits']<=periodic['hits'] and cost['MG']['total_seconds']>=cost['periodic']['total_seconds'] else '')
    text=f'''# MG как триггер диагностики при циклическом обучении текстового VAE

29 сентября 2026. Пилот 100; девять новых инициализаций 101–109. Код и настройки предыдущих экспериментов сохранены.

**Главный результат.** {practical}{result} У MG среднее число дополнительных проверок — {num(mg['mean_checks'],1)} при лимите 12. С учётом получения скалярного ряда оценка стоимости MG — {num(cost['MG']['total_seconds'])} с на запуск, равномерных проверок — {num(cost['periodic']['total_seconds'])} с. Числа ниже относятся только к новым подтверждающим seed, не к пилоту.

## 1. Кто учится и зачем нужен мониторинг

Обучаем текстовый VAE восстанавливать предложения Penn Treebank. Encoder читает всё предложение и задаёт распределение кода; decoder предсказывает слова по предыдущим истинным словам и выборке кода. В такой модели код может потерять информацию и перестать существенно влиять на предсказания. Практический вопрос: можно ли по дешёвому наблюдению выбирать моменты, когда следует проверить использование кода дорогим прямым измерением?

**Модель и данные.** Те же фиксированные 6000 обучающих и 256 проверочных предложений, длина 8–24 слова плюс EOS, словарь 2000 по обучающему корпусу. Embedding 48, encoder/decoder GRU по 64 скрытых координаты, Gaussian-код 16; 276016 параметров. Код подаётся в начальное состояние decoder и на каждом шаге. Teacher forcing, без dropout. Adam: шаг 0,001, batch 32, clipping нормы градиента 5. Это небольшой NLP-эксперимент, не обучение LLM.

**Обучение.** Loss — средняя по батчу сумма NLL слов плюс beta, умноженный на KL posterior к стандартному Gaussian prior. Первые 1024 обновления: beta=0,01. Далее три цикла по 2048 обновлений: за первую половину beta линейно растёт с 0 до 1, вторую половину равен 1; в начале следующего цикла сбрасывается. Итого 7168 обновлений. Циклическое расписание адаптировано из Fu et al., NAACL 2019, N19-1021; совпадение их результатов не заявляется. Монитор не меняет обучение — решения проверяются причинным воспроизведением записанного хода обучения.

**Независимые события.** Каждые 64 обновления считаем информацию предложения и кода для смеси 256 posterior (4 MC-выборки, потолок log 256 нат) и симметризованный KL предсказаний при перестановке posterior между предложениями с согласованным шумом. Потеря: оба показателя не выше 50% исходной медианы; восстановление: оба не ниже 65%. Исходная медиана берётся на шагах 512, 640, 768, 896, 1024. Переход подтверждается двумя последовательными измерениями; время события — второе измерение. Это частичное восстановление, не возврат всей информации. Изменение beta само по себе событием не считается.

## 2. Обнаружение при лимите 12 дорогих проверок

Всего {eventcount} событий с полным горизонтом обнаружения 512 шагов; ещё {censored} поздних событий отмечены отдельно. Проверка засчитывается, если независимо подтверждает текущее событие в пределах 512 шагов и до противоположного перехода; повторные проверки одного события не дают новых попаданий. Задержка считается от подтверждения события, без переноса его времени назад. Ранние предупреждения до этого момента не засчитывались: качество прогнозирования не проверялось.

| Способ | Найдено событий | Проверок на seed | Задержка, шаги | Проверок без нового события |
| --- | --- | --- | --- | --- |
{table}

Одна общая начальная проверка на шаге 1024 добавляется ко всем способам и учтена в стоимости. Задержка усредняется только по обнаруженным событиям, поэтому малая задержка при больших пропусках не означает хороший монитор. Проверки без нового события суммированы по девяти seed; это затраченные вызовы диагностики, не независимые статистические false positive.

<!-- pagebreak -->

## 3. Как выглядят события и запросы диагностики

![Заранее выбранный пример seed 101](example.pdf)

Seed 101 выбран заранее как первый подтверждающий запуск. Сверху — два независимых показателя, делённые на исходный уровень; в центре — MG и отклонение loss; внизу — запросы диагностики. Красные вертикали — подтверждённая потеря информации, зелёные — восстановление. Кружки отмечают зачтённые обнаружения. Последующие значения ряда при принятии решения не используются.

**Вход мониторов.** MG получает только reconstruction NLL/token на фиксированных 16 проверочных предложениях и фиксированном Gaussian noise, по одному наблюдению до каждого обновления. Окно 512, вложение 20, delay 1, 20 соседей, Theiler 39; без удаления тренда. Отклонение loss и нормированная спектральная энтропия получают то же окно. KL — среднее последних 64 обучающих KL, уже вычисляемых в loss. Beta — известное расписание. Встреченные численно вырожденные окна MG в подтверждающих запусках: {s['degenerate_windows']}.

## 4. Полная стоимость, включая получение входного ряда

Операции измерены после прогрева в одном процессе, 40 чередующихся повторов на CPU: forward probe — {num(1000*b['probe']['median'],2)} мс; полная диагностика — {num(1000*b['reference']['median'],2)} мс; MG одного окна — {num(1000*b['MG']['median'],2)} мс. Torch использует 2 потока, оцениватель — 1. Времена параллельного обучения не используются. На машине может работать другая задача, поэтому это локальная оценка стоимости компонентов, не точный benchmark изолированного сервера.

| Способ | Получение ряда, с | Обработка + проверки, с | Всего, с | Если ряд уже записан, с |
| --- | --- | --- | --- | --- |
{costtable}

В стоимость MG/std/entropy включены все 7168 дополнительных forward probe, 105 вычислений признака и выбранные вызовы диагностики. Последний столбец применим только когда именно этот фиксированный probe уже записывался для другой цели. Обычный KL уже входит в обучение и не требует дополнительного forward. Плотная схема: 97 диагностик, {num(cost['MG']['dense_seconds'])} с, все подтверждённые события обнаруживаются по построению эталона. Сокращение числа вызовов нельзя приравнивать к ускорению: нужно оплатить получение признака и учитывать пропущенные события.

Для MG без заранее записанного probe окупаемость относительно плотной схемы начинается, если один вызов независимой диагностики стоит более {num(1000*cost['MG']['break_even_reference_seconds'],1)} мс при неизменных остальных затратах. Это расчёт порога, не экспериментальное подтверждение ускорения на более крупной модели.

<!-- pagebreak -->

## 5. Все seed, перенос порога и границы результата

| Seed | Событий | MG: найдено / проверок | Равные интервалы: найдено / 12 | Интервалы с числом проверок MG |
| --- | --- | --- | --- | --- |
{chr(10).join(seedtable)}

При точном совпадении числа проверок по каждому seed MG обнаруживает {mg['hits']} событий, периодическая схема — {mg['matched_periodic_hits']}. Этот дополнительный периодический контроль знает итоговое число запросов, но не времена событий; это сопоставление затрат после эксперимента, не заранее выбранное число запросов. При общем лимите 12 MG лучше периодической схемы на {s['MG_better_seeds']} seed, равен на {s['MG_equal_seeds']}, хуже на {s['MG_worse_seeds']}.

![Бюджеты и стоимость](cohort.pdf)

Слева доля найденных событий при лимитах 6, 12 и 24; справа полная стоимость при лимите 12. Пороги каждого способа и бюджета выбраны только на пилоте 100 и затем заморожены. Средняя по seed разность recall MG минус равные интервалы при лимите 12 — {num(100*s['MG_minus_periodic_macro_recall'],1)} процентного пункта; 95% парный bootstrap по девяти seed — [{num(100*s['MG_minus_periodic_ci95'][0],1)}; {num(100*s['MG_minus_periodic_ci95'][1],1)}], 20000 повторов. Окна и события внутри seed не считались независимыми выборками. Все запуски используют один корпус и probe.

**Точная политика.** На шагах 1088, 1152, ... сравниваем текущий признак с его значением при последнем запросе (изначально — медиана пяти предшествующих окон). Запрос срабатывает, когда абсолютное значение логарифма отношения достигает порога, прошло не менее 128 шагов и бюджет не исчерпан. Пороги при лимите 12: {thresholds_text}. Перебор 14 заранее заданных порогов на пилоте максимизирует число обнаружений, затем минимизирует бесполезные проверки и задержку. Для всех признаков применяется одна общая схема; это сравнение конкретных политик, а не доказательство оптимальности MG или слабости всех возможных KL-мониторов. При маленьком пороге бюджет может закончиться рано.

**Другой бюджет.** {other_budgets}

**Вывод для статьи.** {practical}Это отдельная проверка полезности мониторинга, а не опровержение ранее наблюдавшегося согласования MG с потерей информации. Следует учитывать обнаружения, задержку и полную стоимость одновременно. Циклы одинаковы во всех запусках, что благоприятно для периодических проверок; перенос на неизвестные расписания, другие корпуса и LLM не проверен. Точная размерность и улучшение генерации текста не оценивались.

Все логи, веса, события, решения, пилотный перебор, замороженные пороги и независимые проверки сохранены. Подробности: PROTOCOL.md и README.md. Результаты прежних парных и защищённых режимов не подменяются новой серией; их отчёты сохранены отдельно.
'''
    (H/'report_ru.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
