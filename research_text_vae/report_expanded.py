"""Expanded report; all planned seeds required, pilot excluded from main statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent

def num(x,n=3):return f'{x:.{n}f}'.replace('.',',')

def cohort_stats(df):
    q=df.MG_paired.to_numpy();rng=np.random.default_rng(20260929)
    medians=np.median(rng.choice(q,(20000,len(q)),replace=True),axis=1)
    return dict(n=len(df),reference_events=int(df.event.sum()),
        primary_agreement=int(((df.MG_paired<1)&df.event&df.primary_valid).sum()),
        raw_MG_decreases=int((df.MG_ratio_regularized<1).sum()),
        base_MG_increases=int((df.MG_ratio_base>1).sum()),
        median_paired_MG=float(np.median(q)),range_paired_MG=[float(q.min()),float(q.max())],
        bootstrap_median95=[float(x) for x in np.quantile(medians,[.025,.975])],
        bootstrap_resamples=20000,bootstrap_seed=20260929,
        failed_seeds=df.loc[~(df.event & (df.MG_paired<1) & df.primary_valid),'seed'].tolist())

def plot_cohort(roots,df):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(10,5.6),constrained_layout=True)
    for arm,color in [('base','#777777'),('regularized','#0868ac')]:
        for ax,filename,col,timecol in [(axs[0,0],'reference.csv','MI','step'),
                (axs[0,1],'reference.csv','shuffle_symkl','step'),
                (axs[1,0],'logs.csv','probe_nll','step'),(axs[1,1],'windows.csv','MG','end')]:
            series=[]
            for root in roots[1:]:
                f=root/arm/filename if filename!='windows.csv' else root/filename
                x=pd.read_csv(f)
                if filename=='windows.csv':x=x[(x.arm==arm)&(x.window==512)&(x.tau==1)]
                series.append(x.set_index(timecol)[col])
            data=pd.concat(series,axis=1)
            assert not data.isna().any().any()
            t=data.index.to_numpy();v=data.to_numpy();med=np.median(v,axis=1)
            ax.plot(t,med,label=arm,color=color)
            lo,hi=np.quantile(v,[.25,.75],axis=1);ax.fill_between(t,lo,hi,color=color,alpha=.18)
    titles=['Independent: sentence-code MI (nats)','Independent: response to code shuffling',
        'Observed scalar: reconstruction NLL/token','MG, W=512, delay=1']
    for ax,title in zip(axs.flat,titles):
        ax.set_title(title);ax.set_xlabel('Optimizer step');ax.axvline(1024,ls='--',lw=1,color='#b35806');ax.legend(fontsize=8)
    fig.savefig(H/'expanded_figure.pdf');fig.savefig(H/'expanded_figure.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.5,2.8),constrained_layout=True)
    ax.scatter(df.seed,df.MG_paired,c=['#999999' if s==0 else '#0868ac' for s in df.seed],s=45)
    ax.axhline(1,color='#aa3333',ls='--',lw=1);ax.axvline(2.5,color='#888888',ls=':',lw=1)
    ax.set(xlabel='Seed (0: pilot; 1-2: initial confirmation; 3-9: expansion)',ylabel='Paired MG ratio q',xticks=range(10),ylim=(0,1.08))
    fig.savefig(H/'seed_effects.pdf');fig.savefig(H/'seed_effects.png',dpi=180);plt.close(fig)

def main():
    roots=[H/'pilot_seed0']+[H/f'confirmation_seed{s}' for s in range(1,10)]
    rows=[];data=[];primary_windows=[]
    for seed,r in enumerate(roots):
        for name in ['summary.json','secondary_summary.json','audit.json','windows.csv']:
            if not (r/name).exists():raise RuntimeError(f'Incomplete planned seed{seed}: {name}')
        s=json.loads((r/'summary.json').read_text());p=next(x for x in s['scalar'] if x['window']==512 and x['tau']==1)
        sec=json.loads((r/'secondary_summary.json').read_text());data.append(s);ref=s['reference']
        w=pd.read_csv(r/'windows.csv').query('window==512 and tau==1').copy();w['seed']=seed;primary_windows.append(w)
        row=dict(seed=seed,role='pilot' if seed==0 else ('initial_confirmation' if seed<=2 else 'expansion'),
            event=s['independent_event'],primary_valid=bool(not w.degenerate.any() and np.isfinite(w.MG).all()),
            MI_before=ref['base']['MI']['before'],MI_after=ref['regularized']['MI']['after'],MI_paired=s['reference_paired']['MI'],
            shuffle_before=ref['base']['shuffle_symkl']['before'],shuffle_after=ref['regularized']['shuffle_symkl']['after'],shuffle_paired=s['reference_paired']['shuffle_symkl'],
            MG_before=p['arms']['base']['MG']['before'],MG_base=p['arms']['base']['MG']['after'],
            MG_regularized=p['arms']['regularized']['MG']['after'],MG_ratio_regularized=p['arms']['regularized']['MG']['ratio'],
            MG_ratio_base=p['arms']['base']['MG']['ratio'],MG_paired=p['paired']['MG'],
            std_paired=p['paired']['std'],entropy_paired=p['paired']['entropy'],train_MG_paired=sec['paired'])
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(H/'all_seeds.csv',index=False)
    cf=df[df.seed>0];ex=df[df.seed>=3];c=cohort_stats(cf);e=cohort_stats(ex);a=cohort_stats(df)
    summary=dict(planned_seeds=list(range(10)),total=a,confirmation=c,expansion=e,
        primary_agreement=a['primary_agreement'],fresh_agreement=c['primary_agreement'],
        median_paired_MG=c['median_paired_MG'],all_independent_events=bool(df.event.all()),seeds=rows)
    (H/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:summary[k] for k in ['total','confirmation','expansion']},indent=2))
    plot_cohort(roots,df)
    sensitivity=[];sens_rows=[]
    for window,tau in [(256,1),(512,1),(1024,1),(512,4)]:
        vals=[]
        for seed,s in enumerate(data):
            q=next(x for x in s['scalar'] if x['window']==window and x['tau']==tau)['paired']['MG']
            sens_rows.append(dict(seed=seed,window=window,tau=tau,paired_MG=q))
            if seed>0:vals.append(q)
        sensitivity.append(f'| {window}, {tau} | {num(np.median(vals))} [{num(min(vals))}; {num(max(vals))}] | {sum(q<1 for q in vals)} / 9 |')
    pd.DataFrame(sens_rows).to_csv(H/'all_seeds_sensitivity.csv',index=False)
    table='\n'.join(f'| {x.seed}{" (пилот)" if x.seed==0 else (" (новый)" if x.seed>=3 else "")} | {num(x.MI_before,2)} / {num(x.MI_after,2)} | {num(x.MG_before,2)} / {num(x.MG_regularized,2)} | {num(x.MG_base,2)} | {num(x.MG_paired)} |' for x in df.itertuples())
    windows=pd.concat(primary_windows);windows=windows[windows.seed>0]
    decrease=100*(1-cf.MG_ratio_regularized);base_growth=100*(cf.MG_ratio_base-1)
    fail='нет' if not c['failed_seeds'] else ', '.join(map(str,c['failed_seeds']))
    ci=c['bootstrap_median95'];qr=c['range_paired_MG']
    mi_reduction=100*(1-cf.MI_after/cf.MI_before)
    sh_reduction=100*(1-cf.shuffle_after/cf.shuffle_before)
    text=f'''# MG при потере информации в текстовом VAE: 10 инициализаций

Расширение серии, 29 сентября 2026. Пилот seed 0; подтверждение seeds 1–9, включая семь добавленных запусков 3–9. Все пары и исходные логи сохранены.

**Основной результат:** независимое событие подтверждено в {c['reference_events']}/9 подтверждающих запусков. MG на фиксированной ошибке восстановления показал более сильное снижение, чем контроль, в {c['primary_agreement']}/9, в том числе в {e['primary_agreement']}/7 добавленных запусков. Медиана парного отношения $q={num(c['median_paired_MG'])}$; диапазон {num(qr[0])}–{num(qr[1])}. Пилот исключён из этих чисел. Неудачные seed по основному критерию: {fail}.

## 1. Что учится и что считаем упрощением

Текстовый VAE восстанавливает предложения: encoder преобразует целое предложение в распределение скрытого кода, decoder предсказывает слова по предыдущим словам и этому коду. Проверяем ослабление роли кода при обучении. Если в коде становится мало информации о предложении, а его подмена почти не меняет предсказания, эта часть модели фактически перестаёт решать исходную задачу представления текста. Это операционное упрощение латентного механизма, не доказательство уменьшения размерности всей системы.

**Данные и модель.** Penn Treebank, фиксированные 6000 обучающих и 256 проверочных предложений, длина 8–24 слова плюс EOS. Словарь 2000 токенов строится по обучающей части; в проверке около 18,4% токенов представлены UNK. Embedding 48, encoder GRU 64, Gaussian latent 16, decoder GRU 64; 276016 параметров. Код подаётся в начальное состояние decoder и на каждом шаге. Teacher forcing, без dropout. Adam: learning rate 0,001, batch 32, clipping нормы градиента 5.

Loss — сумма NLL слов предложения плюс $\\beta\\,\\mathrm{{KL}}(q(z\\mid x)\\Vert\\mathcal N(0,I))$, затем среднее по батчу. После общих 1024 обновлений при $\\beta=0{{,}}01$ создаются две ветви: **base** сохраняет коэффициент, **regularized** повышает его до 1. Обе обучаются до 3072 шагов. Веса, Adam и генератор батчей/латентного шума совпадают при развилке. Learning rate и размер кода не меняются; координаты вручную не выключаются.

## 2. Независимое подтверждение и наблюдатель MG

На фиксированных 256 предложениях каждые 128 шагов измеряем **взаимную информацию предложения и кода**: Monte Carlo, четыре фиксированные выборки на предложение, явная смесь 256 posterior. Это информация для конечной эмпирической смеси с потолком $\\log256\\approx5{{,}}545$ нат, не неизвестная популяционная величина. Вторая проверка — **перестановка кодов между предложениями** при согласованном шуме: симметризованный KL предсказаний и ухудшение reconstruction NLL.

Событие считается независимо подтверждённым, если информация и реакция на перестановку обе снижаются относительно base более чем вдвое; исходно информация выше 0,1 нат, а KL предсказаний выше 0,0001 нат/токен. Критерий не использует MG. Контроль с одинаковыми нулевыми posterior даёт нулевую информацию и нулевую реакцию на перестановку.

**Вход MG:** reconstruction NLL на токен на фиксированных 16 предложениях и фиксированном Gaussian noise. KL, информация и коэффициент $\\beta$ в ряд не входят. Окно 512, stride 128, embedding 20, delay 1, 20 соседей, Theiler 39; тренд не удаляется. Префиксы и первое наблюдение при развилке совпадают.

Сравниваем медианы пяти окон с концами 512–1024 и девяти поздних окон с концами 2048–3072. Поздние окна целиком лежат после перехода. Парный показатель $q$ — отношение поздней медианы MG в regularized к поздней медиане в base; при общей предыстории это также отношение двух изменений «после/до». $q<1$ означает более сильное снижение относительно контроля.

**Что варьировали.** Только инициализацию и случайные батчи/латентный шум обучения. Корпус, probe, модель, расписание и анализ сохранены. Seed 0 — исследовательский пилот; seeds 1–2 — первоначальное подтверждение; seeds 3–9 добавлены после запроса расширить серию. Запуски не отбирались по исходу. Статистика ниже относится к seeds 1–9; перекрывающиеся окна не считаются независимыми повторами.

<!-- pagebreak -->

## 3. Все десять пар и разброс эффекта

| Seed | Информация: до / после, нат | MG: до / после | MG после в base | $q$ |
| --- | --- | --- | --- | --- |
{table}

«После» без уточнения относится к regularized. В девяти подтверждающих запусках информация уменьшается на {num(mi_reduction.min(),1)}–{num(mi_reduction.max(),1)}%, реакция на перестановку — на {num(sh_reduction.min(),1)}–{num(sh_reduction.max(),1)}%. MG в regularized снижается на {num(decrease.min(),1)}–{num(decrease.max(),1)}%; изменение MG в base составляет +{num(base_growth.min(),1)}–+{num(base_growth.max(),1)}%.

![Девять подтверждающих инициализаций](expanded_figure.pdf)

Линия — медиана по seeds 1–9, полоса — межквартильный диапазон, не доверительный интервал. Серый — base, синий — regularized; вертикаль — развилка. Сверху независимые показатели, снизу входной скаляр и MG. На каждом шаге объединяются одни и те же девять seed; пилот не включён. Отдельные графики каждой пары сохранены.

**Оценка разброса:** медиана $q$ по девяти подтверждающим seed — {num(c['median_paired_MG'])}; 95% percentile bootstrap-интервал медианы — [{num(ci[0])}; {num(ci[1])}], 20000 ресэмплирований пар seed. Для семи добавленных seed медиана — {num(e['median_paired_MG'])}. Интервал описывает вариацию инициализаций/батчей при фиксированных данных и probe; перенос на другие корпуса им не проверяется.

<!-- pagebreak -->

## 4. Устойчивость, альтернативы и границы вывода

| Окно, delay | Медиана $q$ [минимум; максимум] | $q<1$, seeds 1–9 |
| --- | --- | --- |
{chr(10).join(sensitivity)}

Основной вариант — 512, delay 1. Окно 1024 имеет только одно окно до перехода, поэтому это более слабая проверка. При delay 4 эффект может существенно ослабевать. Ни один вариант не заменял основной после просмотра результатов; полная таблица по каждому seed сохранена отдельно.

![Парный эффект каждого seed](seed_effects.pdf)

Каждая точка — одна парная инициализация. Seed 0 показан серым как пилот; пунктир после seed 2 отделяет семь добавленных запусков. Красная линия $q=1$ соответствует отсутствию дополнительного снижения относительно base.

**Наблюдатель важен.** Для обычного minibatch loss диапазон $q$ равен {num(cf.train_MG_paired.min())}–{num(cf.train_MG_paired.max())}; $q<1$ в {int((cf.train_MG_paired<1).sum())}/9 случаев. Эти значения нельзя подменять результатом фиксированного probe. Для стандартного отклонения того же фиксированного loss медиана парного отношения — {num(cf.std_paired.median())}, снижение относительно base в {int((cf.std_paired<1).sum())}/9; для спектральной энтропии — {num(cf.entropy_paired.median())}, в {int((cf.entropy_paired<1).sum())}/9. Это дешёвые конкуренты, поэтому преимущество MG перед существующей диагностикой пока не установлено.

**Не абсолютная размерность.** В основном варианте у подтверждающих seed численно вырожденных окон: {int(windows.degenerate.sum())}. Отношение оценок embedding 40/20: {num(windows.ident.min(),2)}–{num(windows.ident.max(),2)}. Численная невырожденность сама по себе не подтверждает условия размерностной интерпретации. Рекуррентность здесь не установлена. Проверка масштаба x10 пройдена; IAAFT-суррогаты не дают основания приписывать сигналу доказанное восстановление особой нелинейной геометрии.

**Что подтверждено:** воспроизводимость сопутствующего сигнала MG при независимо измеренном ослаблении использования кода в данной NLP-постановке. **Что ещё не проверено:** специфичность детектора, преимущество перед KL и простыми статистиками, улучшение генеративного качества, другие корпуса и LLM. Reconstruction NLL не является marginal likelihood или perplexity генерации из prior: encoder видит восстанавливаемое предложение. Следующий содержательный контроль — то же изменение регуляризации с защитой от потери использования кода.

Расчёты выполнены на CPU; часть запусков шла параллельно, поэтому времена этих запусков не используются для сравнительного benchmark. Дополнительный forward фиксированного probe имеет стоимость. Корпус и наблюдатель во всей серии одни и те же: десять seed расширяют проверку воспроизводимости, но не число независимых задач.

Источники постановки: Bowman et al., Generating Sentences from a Continuous Space, CoNLL 2016, K16-1002; He et al., Lagging Inference Networks and Posterior Collapse in Variational Autoencoders, ICLR 2019, arXiv 1901.05534. Все численные результаты получены нашими запусками. Исходный отчёт по трём seed сохранён в initial_three; основной manuscript не менялся.
'''
    if (H/'protection/summary.json').exists():
        from protection.report_integration import integrate
        text=integrate(text)
    (H/'report_ru.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
