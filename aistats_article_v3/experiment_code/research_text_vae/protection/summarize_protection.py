from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent

def num(x,n=3):return f'{x:.{n}f}'.replace('.',',')
def src(seed):return H.parent/('pilot_seed0' if seed==0 else f'confirmation_seed{seed}')

def main():
    selection=json.loads((H/'selection.json').read_text());mode=selection['mode'];rows=[]
    comparisons=[]
    for seed in range(10):
        out=H/mode/f'seed{seed}';d=json.loads((out/'comparison.json').read_text());comparisons.append(d)
        primary=d['scalar'][0];ref=d['reference'];keep=d['retention']['reference']
        row=dict(seed=seed,protection_valid=d['retention']['protection_valid'],q_regularized=primary['q_regularized'],
            q_protected=primary['q_protected'],R=primary['protected_vs_regularized'],closer_to_base=primary['closer_to_base'],
            MI_retained_fraction=keep['MI']['retained_fraction'],shuffle_retained_fraction=keep['shuffle_symkl']['retained_fraction'])
        for arm in ['base','regularized','protected']:
            for metric in ['MI','shuffle_symkl','shuffle_nll_gap','KL']:
                row[f'{metric}_{arm}']=ref[arm][metric]['after']
            row[f'MG_{arm}']=primary['arms'][arm]['MG']['after']
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(H/'all_seeds.csv',index=False)
    conf=df[df.seed>0];valid=conf[conf.protection_valid]
    rng=np.random.default_rng(20260929)
    def interval(values):
        if not len(values):return None
        a=np.asarray(values);return np.quantile(np.median(rng.choice(a,(20000,len(a)),replace=True),axis=1),[.025,.975]).tolist()
    summary=dict(mode=mode,n_confirmation=9,protection_valid_count=len(valid),
        protected_MG_higher_count=int((valid.R>1).sum()),closer_to_base_count=int(valid.closer_to_base.sum()),
        median_R_all=float(conf.R.median()),median_R_valid=float(valid.R.median()) if len(valid) else None,
        median_R_valid_bootstrap95=interval(valid.R),median_q_protected=float(conf.q_protected.median()),
        median_q_regularized=float(conf.q_regularized.median()),
        failed_protection_seeds=conf.loc[~conf.protection_valid,'seed'].tolist(),seeds=rows)
    sensitivity=[];cheap=[]
    for window,tau in [(512,1),(256,1),(1024,1),(512,4)]:
        values=[]
        for d in comparisons[1:]:
            setting=next(x for x in d['scalar'] if x['window']==window and x['tau']==tau)
            values.append(setting['protected_vs_regularized'])
        sensitivity.append(dict(window=window,tau=tau,median_R=float(np.median(values)),
            min_R=float(min(values)),max_R=float(max(values)),R_above_one=int(sum(x>1 for x in values))))
    for metric in ['std','entropy']:
        values=[d['scalar'][0]['arms']['protected'][metric]['after']/d['scalar'][0]['arms']['regularized'][metric]['after'] for d in comparisons[1:]]
        cheap.append(dict(metric=metric,median_R=float(np.median(values)),R_above_one=int(sum(x>1 for x in values))))
    summary['sensitivity']=sensitivity;summary['cheap_baselines']=cheap
    pd.DataFrame(sensitivity).to_csv(H/'sensitivity.csv',index=False)
    (H/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='seeds'},indent=2))
    plot(df,mode)
    table='\n'.join(f'| {x.seed}{" (пилот)" if x.seed==0 else ("*" if not x.protection_valid else "")} | {num(x.MI_regularized,2)} / {num(x.MI_protected,2)} | {num(x.shuffle_symkl_regularized,3)} / {num(x.shuffle_symkl_protected,3)} | {num(x.MG_regularized,2)} / {num(x.MG_protected,2)} | {num(x.R)} |' for x in df.itertuples())
    pilot_encoder=json.loads((H/'encoder5/seed0/retention.json').read_text())
    er=pilot_encoder['reference']
    if mode=='freebits':
        method='''Использован заранее предусмотренный запасной вариант **free bits**: для каждой из 16 координат считаем средний по батчу KL к prior, заменяем его максимумом с 0,5 нат и суммируем. При этом внешний коэффициент beta равен 1, как в ветви с потерей информации. Это не жёсткое ограничение на информацию: код должен использоваться ради восстановления текста. Выполняется одно обычное обновление Adam на шаг.

Free bits меняет форму функции потерь. Поэтому этот контроль слабее по изоляции причины, чем сохранение исходной функции потерь и дополнительные шаги encoder. Он позволяет сравнить два режима с одинаковым внешним beta, но не устраняет все различия динамики оптимизации.'''
    else:
        method='''В защищённой ветви перед каждым обычным обновлением выполняются пять дополнительных обновлений encoder при beta=1. Обновляются GRU encoder и параметры posterior; decoder и общие embedding заморожены на этих внутренних шагах. Градиент реконструкции проходит через decoder к коду. Это наша фиксированная адаптация идеи aggressive inference training, не точное воспроизведение алгоритма He et al.'''
    ci=summary['median_R_valid_bootstrap95'];ci_text='не определён' if ci is None else f'[{num(ci[0])}; {num(ci[1])}]'
    failed='нет' if not summary['failed_protection_seeds'] else ', '.join(map(str,summary['failed_protection_seeds']))
    sensitivity_text='; '.join(f"W={r['window']}, delay={r['tau']}: медиана R={num(r['median_R'])}, R>1 в {r['R_above_one']}/9" for r in sensitivity)
    cheap_text='; '.join(f"{'стандартное отклонение' if r['metric']=='std' else 'спектральная энтропия'}: отношение protected/regularized {num(r['median_R'])}, выше в {r['R_above_one']}/9" for r in cheap)
    text=f'''# Проверка MG при сохранении информации в текстовом коде

29 сентября 2026. Третья ветвь добавлена к тем же десяти сохранённым состояниям: пилот 0 и подтверждающие seeds 1–9. Исходные ветви и их результаты не изменялись.

**Критерий сохранения кода выполнен в {len(valid)}/9 подтверждающих seed.** Среди этих seed MG в защищённой ветви выше, чем в ветви с потерей информации, в {summary['protected_MG_higher_count']}/{len(valid)}; ближе к исходному контролю — в {summary['closer_to_base_count']}/{len(valid)}. Медианное отношение MG protected/regularized по всем девяти seed — {num(summary['median_R_all'])}. Таблица включает все seed, без отбора по результату.

## 1. Какой вопрос проверяем

Текстовый VAE учится восстанавливать предложения: encoder читает предложение и строит Gaussian-распределение кода, decoder предсказывает слова по предыдущим истинным словам и выборке этого кода. При усилении KL-регуляризации модель может перестать использовать код. Проверяем, различает ли MG потерю информации и сохранение её существенной части при одинаковом внешнем коэффициенте регуляризации.

В каждой тройке **base** оставляет beta=0,01, **regularized** меняет beta на 1, **protected** делает то же изменение с защитой. Все ветви стартуют из одинаковых весов, Adam и состояния генератора после 1024 обновлений. Конец — 3072 обновления. Основные батчи и латентный шум совпадают.

**Данные и модель:** Penn Treebank, 6000 обучающих и 256 проверочных предложений; длина 8–24 слова плюс EOS; словарь 2000. Embedding размерности 48, encoder и decoder GRU по 64 скрытых координаты, код размерности 16; всего 276016 параметров. Teacher forcing, без dropout. Adam с шагом 0,001, batch 32, ограничение нормы градиента 5. Исходный loss — средняя по батчу сумма NLL слов предложения плюс beta, умноженный на KL posterior к стандартному Gaussian prior. Два исходных продолжения переиспользованы из архивов.

## 2. Как выбиралась защита

Сначала на pilot0 проверены пять дополнительных шагов encoder на каждое обычное обновление. Decoder и общие embedding на внутренних шагах действительно неизменны, включая их Adam-состояния. Поздняя информация составила {num(er['MI']['protected'])} нат против {num(er['MI']['before'])} до вмешательства; реакция на перестановку — {num(er['shuffle_symkl']['protected'])} против {num(er['shuffle_symkl']['before'])}. Эта попытка не прошла критерий сохранения кода; она сохранена отдельно и не используется как проверка специфичности MG.

{method}

**Критерий сохранения кода задан до запусков:** поздние медианы (шаги 2048–3072) взаимной информации и реакции предсказаний на перестановку кода должны каждая составлять не менее половины исходной медианы (шаги 512–1024) и быть хотя бы вдвое выше regularized. Сначала проверялся этот критерий на pilot 0, без расчёта его MG; затем выбранная защита зафиксирована для всех seeds 1–9 независимо от результата MG. Seed с неудачной защитой: {failed}.

**Измерения.** Информация — MC с четырьмя выборками на предложение для смеси 256 posterior (потолок log 256 нат). Реакция на код — симметризованный KL предсказаний при циклической перестановке распределений encoder между предложениями и согласованном шуме, в нат/токен. Эти проверки выполняются каждые 128 шагов. MG получает только reconstruction NLL/token на первых 16 проверочных предложениях и фиксированном Gaussian noise. Окно 512, сдвиг 128, размерность вложения 20, задержка 1, 20 соседей, Theiler 39; тренд не удаляется. Поздние окна заканчиваются на шагах 2048–3072 и не пересекают развилку.

<!-- pagebreak -->

## 3. Все тройки и сравнение MG

| Seed | Информация reg / prot | Реакция на код reg / prot | MG reg / prot | $R$ |
| --- | --- | --- | --- | --- |
{table}

Все значения — поздние медианы. $R=\\mathrm{{MG}}_{{protected}}/\\mathrm{{MG}}_{{regularized}}$: больше1 означает более высокий MG при защите. Звёздочка отмечает непрошедший независимый критерий защиты; такие seed не скрыты. Основная статистика исключает pilot0. По прошедшим критериям seed 95% bootstrap-интервал медианы R — {ci_text} (20000 ресэмплирований пар seed, условно на фиксированном корпусе).

![Все подтверждающие тройки](comparison_cohort.pdf)

Медиана и межквартильный диапазон по всем seeds1–9, включая неудачные защиты. Серый — base, синий — regularized, оранжевый — protected. Сверху независимые характеристики кода, снизу наблюдаемый loss и MG. Оси времени одинаковы; дополнительных внутренних обновлений encoder на оси нет.

**Устойчивость:** {sensitivity_text}. Это заранее сохранённые дополнительные настройки; основной вариант не менялся. **Дешёвые альтернативы на том же входе:** {cheap_text}. Преимущество MG перед ними этим контролем не установлено.

**Граница вывода.** Защита сохраняет значительную, но не всю информацию. Медиана protected/base для MG — {num(summary['median_q_protected'])}, regularized/base — {num(summary['median_q_regularized'])}: полного возвращения к исходному контролю нет. Более высокий MG при защите согласуется с меньшей потерей информации в данной постановке, но free bits также меняет динамику оптимизации. Контроль не доказывает ни точную размерность, ни универсальную причинную специфичность MG. KL и простые статистики loss остаются конкурентами; преимущества в стоимости здесь не заявляем.

Протокол, неудачная попытка encoder5, все финальные веса, диагностические CSV и варианты окон сохранены. Это дополнительные продолжения прежних инициализаций, не новая серия независимых данных.
'''
    import re
    text=re.sub(r'([А-Яа-яЁё])(\d)',r'\1 \2',text);text=re.sub(r'(\d)([А-Яа-яЁё])',r'\1 \2',text)
    text=re.sub(r'\b(seeds?|embedding|latent|GRU|batch|pilot|log|Theiler)(?=\d)',r'\1 ',text)
    (H/'report_ru.md').write_text(text,encoding='utf-8')

def plot(df,mode):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(10,5.7),constrained_layout=True)
    for arm,color in [('base','#777777'),('regularized','#0868ac'),('protected','#d95f02')]:
        for ax,file,col,tm in [(axs[0,0],'reference.csv','MI','step'),(axs[0,1],'reference.csv','shuffle_symkl','step'),
            (axs[1,0],'logs.csv','probe_nll','step'),(axs[1,1],'windows','MG','end')]:
            cols=[]
            for seed in range(1,10):
                root=H/mode/f'seed{seed}';original=src(seed)
                if file=='windows':
                    d=pd.read_csv(root/'three_arm_windows.csv');d=d[(d.arm==arm)&(d.window==512)&(d.tau==1)]
                else:d=pd.read_csv((root if arm=='protected' else original/arm)/file)
                cols.append(d.set_index(tm)[col])
            d=pd.concat(cols,axis=1);assert not d.isna().any().any();t=d.index.to_numpy();v=d.to_numpy()
            lo,hi=np.quantile(v,[.25,.75],axis=1);ax.plot(t,np.median(v,axis=1),label=arm,color=color);ax.fill_between(t,lo,hi,color=color,alpha=.13)
    for ax,title in zip(axs.flat,['Sentence-code MI (nats)','Response to code shuffling','Observed reconstruction NLL/token','MG, W=512, delay=1']):
        ax.set_title(title);ax.set_xlabel('Outer step');ax.axvline(1024,color='black',ls=':',lw=1);ax.legend(fontsize=7)
    fig.savefig(H/'comparison_cohort.pdf');fig.savefig(H/'comparison_cohort.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
