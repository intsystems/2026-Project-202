"""Report the planned stopping outcome if pilot eligibility fails at 4M."""
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion import H,RESETS,Actor,state_reference
from perturb import zero_audit
import torch
from threadpoolctl import threadpool_limits

def main():
    out=H/'seed200';pair=json.loads((out/'pair.json').read_text());assert not pair['usable'] and pair['horizon']==4194304
    assert not (H/'selection.json').exists() and not list(out.glob('step*/reset*/MG_windows.csv'))
    frame=pd.read_csv(out/'evaluation.csv');counts=frame.groupby('step').eligible.sum()
    assert counts.index.tolist()==list(range(0,4194305,131072))
    assert frame.groupby('step')['reset'].apply(lambda s:sorted(s)==RESETS).all()
    training=[json.loads((out/f'train_{n}.json').read_text()) for n in [2097152,4194304]]
    protocol_hash=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()
    assert all(m['protocol_sha256']==protocol_hash for m in training)
    final=frame[frame.step==4194304];first=frame[(frame.step==pair['early'])&frame.eligible]
    for row in first.itertuples():
        root=out/f'step{row.step:07d}'/f'reset{row.reset}';data=np.load(root/'trajectory.npz')
        m=json.loads((root/'metrics.json').read_text());ref,_=state_reference(data['qpos'],data['qvel'])
        for k in ['recurrence','section_dispersion','period']:np.testing.assert_allclose(m[k],ref[k],rtol=1e-12)
    example=out/f"step{pair['early']:07d}"/'reset31001'
    data=np.load(example/'trajectory.npz');m=json.loads((example/'metrics.json').read_text())
    error=zero_audit(Actor(example.parent),data,0,m['period'])
    probe=json.loads((example/'perturb_4.json').read_text());assert probe['inference_mode']=='single_observation_as_nominal'
    progress=pd.read_json(out/'progress.jsonl',lines=True)
    fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True,constrained_layout=True)
    axs[0].plot(progress.step/1e6,progress.recent_return,'o-',ms=3);axs[0].set_ylabel('Training episode return')
    axs[1].step(counts.index/1e6,counts.values,where='mid');axs[1].set(ylabel='Eligible resets / 3',yticks=[0,1,2,3])
    for reset in RESETS:
        part=frame[frame.reset==reset];axs[2].plot(part.step/1e6,part.steps*.008,'o-',ms=2,label=str(reset))
    axs[2].axhline(36.864,ls='--',color='gray');axs[2].set(xlabel='Training transitions (millions)',ylabel='Evaluation survival (s)');axs[2].legend(title='Reset',fontsize=8)
    for ax in axs:ax.axvline(2.097152,ls=':',color='gray');ax.spines[['top','right']].set_visible(False)
    fig.savefig(H/'feasibility.pdf');fig.savefig(H/'feasibility.png',dpi=160);plt.close(fig)
    failures=[]
    for r in final.itertuples():
        reasons=[]
        if not r.complete:reasons.append('прерывание по здоровью')
        if np.isfinite(r.mean_speed) and r.mean_speed<.5:reasons.append('скорость ниже 0,5')
        if np.isfinite(r.knee_std) and r.knee_std<.05:reasons.append('отклонение колена ниже 0,05')
        if r.peaks<8:reasons.append('меньше восьми пиков')
        failures.append(dict(reset=int(r.reset),reasons=reasons))
    final_rows='\n'.join(f'| {int(r.reset)} | {int(r.steps)} ({r.steps*.008:.2f} с) | {", ".join(f["reasons"]) if f["reasons"] else "пригодна"} |' for r,f in zip(final.itertuples(),failures))
    checkpoints=counts[counts>=2].index.tolist()
    summary=dict(outcome='pilot_feasibility_failure',pilot=200,transitions=4194304,pair=pair,
        eligible_checkpoints=checkpoints,final_eligible_resets=int(final.eligible.sum()),
        final_survival_steps=final.steps.astype(int).tolist(),final_complete_resets=int(final.complete.sum()),final_failure_reasons=failures,training_seconds=sum(m['seconds'] for m in training),
        MG_computed=False,confirmation_seeds_run=0,zero_replay_error=error)
    (H/'summary.json').write_text(json.dumps(summary,indent=2))
    standard=json.loads((H/'standard_evaluation_audit.json').read_text()) if (H/'standard_evaluation_audit.json').exists() else None
    if standard:assert standard['passed']
    (H/'audit.json').write_text(json.dumps(dict(passed=True,type='feasibility_only',all_33_checkpoints_three_resets=True,
        unchanged_protocol_hash=protocol_hash,no_MG_or_confirmation=True,zero_replay_error=error,standard_evaluation=standard),indent=2))
    text=f'''# Walker2d: проверка нового эксперимента для MG

Протокол от 29 сентября 2026. **Пилот не прошёл проверку пригодности: финальная политика после 4 194 304 переходов имеет {int(final.eligible.sum())} пригодных записей из трёх. Сравнение MG и подтверждающая серия не запускались. Это не отрицательный результат самого MG: надёжной пары движений для запланированной проверки не получено.**

## 1. Что учится и зачем это проверять

Двуногая модель Walker2d в MuJoCo учится двигаться вперёд. Нейросеть получает 17 наблюдений о положении и скорости тела и выдаёт шесть управляющих моментов. PPO максимизирует стандартную награду среды: скорость вперёд и здоровое положение корпуса с учётом штрафа за управление. Никакой периодической эталонной траектории политике не сообщается.

Идея эксперимента: при обучении нерегулярное движение может смениться повторяемой походкой. Тогда MG одного датчика сустава потенциально позволит наблюдать упрощение движения без анализа полного состояния или множества возмущённых симуляций. Это гипотеза: награда за продвижение сама по себе не требует регулярной походки. Измеряется движение при замороженной политике, а не динамика параметров или loss во время оптимизации. Это симулятор, не физический робот.

**Точная постановка обучения.** Gymnasium 1.2.3 Walker2d-v5, MuJoCo 3.14.0, Stable-Baselines3 2.7.1, seed 200. Actor и critic — отдельные MLP 64×64 с tanh. Восемь сред, по 256 шагов на сбор данных; PPO: batch 64, десять эпох, learning rate 0,0003, gamma 0,99, GAE 0,95, clip 0,2, entropy coefficient 0, value coefficient 0,5, ограничение нормы градиента 0,5. VecNormalize нормирует наблюдения и награды; обучение на CPU, один поток Torch. Эпизод обучения ограничен 1000 шагами. Снимки политики и нормировки сохраняются каждые 131072 перехода.

Сначала выполнено 2 097 152 перехода. Финальная политика не прошла критерий ходьбы, поэтому сделано заранее разрешённое единственное продление до 4 194 304. Сохранялись веса, оптимизатор и нормировка, а эпизоды симулятора при продолжении были перезапущены. Гиперпараметры не подбирались по результату. Суммарное время обучения: {summary['training_seconds']/60:.1f} мин; оно не является стоимостью MG.

## 2. Независимая проверка движения

На каждом из 33 снимков политика детерминированная, нормировка заморожена. Использованы три начальных seed: 31001–31003. Шаг управления 0,008 с. После 512 шагов разгона требуются 4096 шагов анализа: вместе 4608 шагов, или 36,864 с непрерывной ходьбы. Ограничение времени эпизода снято, критерий нездорового состояния сохранён.

Пригодная запись: ни одного падения за весь интервал, средняя скорость не ниже 0,5 м/с, стандартное отклонение правого колена не ниже 0,05 рад, не менее восьми пиков (prominence 0,15 рад, расстояние не менее 20 шагов). Этап пригоден при двух таких записях из трёх. Сравнивать запланировано **первый пригодный этап и финальный**, минимум на двух общих начальных seed. Выбирать самый красивый поздний снимок не разрешалось; MG в отборе не участвует.

Первый пригодный этап: {pair['early']} переходов. Всего пригодных этапов: {len(checkpoints)} из 33. Их точный список и все 99 исходов записаны в summary.json и seed200/evaluation.csv.

| Начальный seed | Длительность финальной проверки | Результат |
| --- | --- | --- |
{final_rows}

<!-- pagebreak -->

## 3. Что получилось и что из этого следует

![Все этапы, без выбора удачных](feasibility.pdf)

Сверху — средняя недавняя награда обучающих эпизодов; в середине — число пригодных длинных проверок; снизу — длительности всех трёх проверок. Вертикальный пунктир отмечает продление с 2M до 4M. Финальная политика завершает движение уже через 0,976–1,208 с: высота корпуса опускается ниже допустимых 0,8 м. Поэтому неудачу нельзя объяснить только тем, что проверка длиннее обучающего эпизода. Стандартная оценка через VecNormalize дала побитно те же действия и те же моменты завершения, что и наш код; результаты сохранены в standard_evaluation_audit.json.

Более ранние политики проходили все 36,864 с, поэтому наблюдается деградация поведения при продолжении обучения. Обучающая награда усредняет недавние эпизоды со случайными действиями PPO; здесь проверяется детерминированная политика после обновления. Это разные измерения. Причина деградации не установлена отдельным причинным экспериментом.

**Как должно было подтверждаться упрощение.** По всем 17 координатам физического состояния, кроме абсолютного продвижения вперёд, считались два независимых показателя. R — минимальная нормированная ошибка повторения состояния при задержках 20–250 шагов; D — нормированный разброс полного состояния в моменты максимумов колена. Координаты скоростей делятся на 5, позиции и углы — на 1. Критерий: оба отношения «поздний/ранний» не выше 0,75 при исходных R не ниже 0,02 и D не ниже 0,01. Эти числа задают операционную проверку регулярности, а не теорему о размерности. Для всех пригодных промежуточных записей R, D и полные кривые задержек сохранены.

Отдельно реализована дорогая проверка: в четырёх точках движения возмущается каждая из 17 нормированных координат на ±0,001, после чего политика заново управляет роботом в течение четырёх номинальных периодов. Всего 136 продолжений. Расстояние до исходного движения корректируется на сдвиг фазы. На первом пригодном этапе, reset 31001, нулевая проба воспроизвела траекторию точно; медианное усиление возмущения составило {probe['median_amplification']:.1f}, падений {probe['falls']}/{probe['probes']}. Эта единственная проверка заняла {probe['seconds']:.1f} с; она подтверждает работоспособность инструмента, но не улучшение устойчивости при обучении. Это не спектр Ляпунова.

**MG пока не оценивался.** Заранее выбран правый коленный угол, основное окно 2048, размерность вложения 20, 20 соседей; задержка должна была один раз калиброваться по независимому периоду пригодной пилотной пары. Такой пары нет. Поэтому не вводилась произвольная задержка и не запускались пять подтверждающих обучений. Нет ни подтверждения, ни опровержения способности MG обнаруживать упрощение в этой задаче; нет измеренного преимущества по времени перед альтернативами.

**Решение по постановке.** В текущем виде этот опыт не подходит как положительный эксперимент статьи. Проблема возникла до сравнения методов: финальная политика не обеспечивает даже короткую здоровую ходьбу. Если развивать именно эту задачу, сначала нужно стабилизировать PPO и подтвердить длительную ходьбу на отдельных начальных состояниях; простое увеличение окна оценки проблему не исправит. Снижение learning rate или изменение горизонта обучения — только гипотезы для нового пилота, не выполненные исправления. Текущий отрицательный результат сохранён полностью.

В архиве: протокол, исходники, все снимки и нормировки, 99 проверок, полные состояния, независимые метрики, численный контроль и график. NUMERICAL_AUDIT.md объясняет исправление пакетного вычисления действий: из-за округления оно не воспроизводило одиночную исходную траекторию; использованные пробы пересчитаны с точным одиночным вычислением. Статья не изменялась.
'''
    (H/'report_ru.md').write_text(text,encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
