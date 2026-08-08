# Пул одномерных наблюдателей для EDM-предсказания grokking

## Цель

Вместо выбора одного удобного лога после просмотра успешного запуска нужен заранее фиксированный пул одномерных наблюдателей. Каждый наблюдатель должен быть:

- дешёвым или умеренно дешёвым;
- применимым к разным архитектурам;
- вычисляемым только по train-состоянию модели;
- достаточно гладким для delay embedding;
- воспроизводимым при повторном измерении одного checkpoint;
- проверяемым на grokking и non-grokking runs одним протоколом.

`val_loss` и `val_acc` сохраняются, но используются для разметки результата и как сильный baseline. Они не входят в основной train-only predictor.

---

## 1. Почему текущие градиентные проекции оказались слабыми

В текущем логировании записывается величина

$$
s_{g,r}(t)=\langle g(\theta_t,B_t),v_r\rangle,
$$

где $B_t$ — случайный mini-batch, а $v_r$ — фиксированное случайное направление.

Это не просто наблюдение состояния $\theta_t$. Оно зависит от нового случайного batch на каждом шаге:

$$
g(\theta_t,B_t)
=
\nabla L(\theta_t)+\xi_t,
$$

где $\xi_t$ — stochastic-gradient noise. После проекции:

$$
s_{g,r}(t)
=
\langle\nabla L(\theta_t),v_r\rangle
+
\langle\xi_t,v_r\rangle.
$$

Когда средний градиент мал, второй член доминирует. Большая стохастичность не улучшает реконструкцию динамики: она добавляет внешнее случайное возбуждение, которое не определяется состоянием модели.

Условие generic observable в delay-embedding аргументах означает, что наблюдатель не вырожден относительно состояния системы. Оно не означает устойчивость к независимому шуму произвольной мощности.

### Наблюдение в имеющемся логе

В memorization gap запуска `p=211, WD=0.5` получены следующие медианные характеристики:

| Семейство | Lag-1 autocorrelation | Roughness $\operatorname{std}(\Delta x)/\operatorname{std}(x)$ |
|---|---:|---:|
| gradient projections | -0.015 | 1.426 |
| update projections | -0.016 | 1.425 |
| displacement projections | -0.097 | 1.481 |
| activation projections | 0.863 | 0.523 |
| logit projections | 0.832 | 0.579 |
| weight projections | 0.983 | 0.181 |

Для белого шума ожидаются autocorrelation около нуля и roughness около $\sqrt{2}\approx1.414$. Текущие gradient/update projections практически совпадают с этой картиной. EDM по ним в основном оценивает геометрию batch noise.

### Дополнительные причины

1. В PDF показан только `r0`, хотя в CSV записаны `r0`, `r1`, `r2`. Одна проекция может быть почти ортогональна важному направлению.
2. Три направления всё ещё слишком мало для устойчивого projection consensus.
3. Single-step update содержит тот же высокочастотный шум, преобразованный Adam moments.
4. Локальный linear detrending удаляет медленный дрейф, который для параметров может быть главным фазовым сигналом.
5. Окно $W=200$ подписано по центру, что визуально сдвигает сигнал примерно на 5 000 шагов в прошлое. Для прогноза нужен `window_end_step`.
6. Delay vectors строились без Theiler exclusion и сильно перекрывались во времени.

---

## 2. Почему нельзя просто перейти к weight norm

В успешном `WD=0.5` run размерность `weight_norm` визуально падает перед переходом. Однако в `WD=0` non-grokking run тот же observer показал ещё более сильные dimension drops без генерализации.

В причинном расчёте с timestamp конца окна максимальная относительная амплитуда для `weight_norm` была приблизительно:

```text
WD=0.5 grokking run:     0.14 до onset
WD=0 non-grokking run:   0.41 без последующего grokking
```

Следовательно, `weight_norm` чувствителен к фазам оптимизации, но пока не специфичен к grokking. Он остаётся полезным context observer, но не может быть единственным primary predictor.

Аналогичная проблема обнаружилась для `parameter participation ratio`, `train entropy` и `train hidden norm`: в длинном WD=0 run возникали крупные падения без grokking.

---

## 3. Принцип нового пула

Основной predictor должен использовать только train-side observers. Validation metrics нужны для определения $T_{onset}$ и проверки прогноза.

Пул делится на четыре уровня:

1. **Primary representation observers** — главные кандидаты для dimension-drop hypothesis.
2. **Primary trajectory observers** — независимое подтверждение со стороны параметров и движения оптимизатора.
3. **Context observers** — помогают объяснить фазу, но не обязаны падать.
4. **Archive observers** — сохраняются на будущее, но не участвуют в текущем frozen-rule.

Главное правило: нельзя выбрать лучшую случайную проекцию после просмотра test-run. Сначала рассчитываются несколько фиксированных направлений, затем используется агрегат семейства.

---

## 4. Primary representation observers

### 4.1. Fixed-probe activation sketches

Берётся один фиксированный train probe set $P$ для всего запуска. Для каждого примера $x_i\in P$ модель выдаёт последний скрытый вектор $h_i(t)$.

Для направления $r$ заранее генерируются:

- фиксированное feature direction $u_r$;
- фиксированные знаки примеров $q_{i,r}\in\{-1,+1\}$.

Одномерный observer:

$$
a_r(t)
=
\frac{1}{\sqrt{|P|}}
\sum_{i\in P}
q_{i,r}\langle h_i(t),u_r\rangle.
$$

Преимущества:

- observer является гладкой функцией параметров модели;
- измерение детерминировано при `model.eval()`;
- mini-batch noise отсутствует;
- способ не зависит от конкретного вида слоя;
- фиксированные знаки не дают усреднить разные примеры в один mean vector.

В новых экспериментах рекомендуется $R=16$ направлений. Текущие три направления сохраняются, но недостаточны для окончательного вывода.

### 4.2. Fixed-probe logit sketches

Аналогично строятся проекции логитов:

$$
z_r(t)
=
\frac{1}{\sqrt{|P|C}}
\sum_{i\in P}q_{i,r}\langle f_{\theta_t}(x_i),w_r\rangle.
$$

Они ближе к выходу модели и могут раньше отражать появление правила. При этом они не используют validation labels.

### 4.3. Train prediction distribution

На том же фиксированном train probe сохраняются:

- mean entropy;
- median entropy;
- mean top-1/top-2 margin;
- 10%, 50%, 90% quantiles margin;
- mean confidence;
- hidden-vector norm quantiles.

Эти метрики являются context для activation/logit sketches. Они могут насыщаться после меморизации, поэтому не назначаются единственным observer.

---

## 5. Primary trajectory observers

### 5.1. Нормированные проекции направления весов

Сырые weight projections смешивают изменение нормы и направления. Для каждого слоя $\ell$ следует логировать отдельно:

$$
n_{\ell}(t)=\log(\lVert\theta_{\ell}(t)\rVert_2+\varepsilon),
$$

$$
w_{\ell,r}^{dir}(t)
=
\left\langle
\frac{\theta_{\ell}(t)}{\lVert\theta_{\ell}(t)\rVert_2+\varepsilon},
v_{\ell,r}
\right\rangle.
$$

Так radial shrinkage от weight decay отделяется от изменения структуры решения.

Рекомендуется $R=16$ направлений на слой. Для общего family score сначала агрегируются направления внутри слоя, затем слои; нельзя выбирать наиболее красивый слой на test-run.

### 5.2. Длинногоризонтное перемещение

Single-step update слишком шумен. Вместо него используется

$$
\Delta_K\theta_t=\theta_t-\theta_{t-K},
$$

где $K$ соответствует 500–2 000 optimizer steps.

Логируются:

$$
\lVert\Delta_K\theta_t\rVert_2,
$$

$$
\left\langle
\frac{\Delta_K\theta_t}{\lVert\Delta_K\theta_t\rVert_2+\varepsilon},v_r
\right\rangle,
$$

и coherence соседних перемещений:

$$
c_K(t)
=
\cos(\Delta_K\theta_t,\Delta_K\theta_{t-K}).
$$

Интегрирование по $K$ шагам подавляет независимый mini-batch noise и сохраняет медленный дрейф траектории.

### 5.3. Path efficiency

За интервал $K$:

$$
\eta_{path}(t)
=
\frac{
\lVert\theta_t-\theta_{t-K}\rVert_2
}
{
\sum_{s=t-K+1}^{t}\lVert\theta_s-\theta_{s-1}\rVert_2+\varepsilon
}.
$$

Значение около нуля соответствует блужданию с взаимно компенсирующимися update; большее значение — согласованному направленному движению. Это generic scalar observer оптимизационного режима.

---

## 6. Исправленные gradient observers

Raw `gradient_norm`, `gradproj` и `updateproj` продолжают логироваться, но не участвуют в primary dimension-drop rule.

Вместо них добавляются оценки среднего градиента и gradient noise.

### 6.1. EMA gradient

$$
m_t=\beta m_{t-1}+(1-\beta)g_t.
$$

Логируются:

- $\lVert m_t\rVert_2$;
- participation ratio $m_t$;
- нормированные projections $\langle m_t/\lVert m_t\rVert,v_r\rangle$;
- cosine $\cos(m_t,m_{t-K})$.

Для Adam можно использовать bias-corrected first moment самого optimizer, не выполняя дополнительный backward.

### 6.2. Gradient signal-to-noise ratio

На $M$ последовательных mini-batches:

$$
\bar g_t=\frac{1}{M}\sum_{j=1}^{M}g_{t,j},
$$

$$
SNR_g(t)
=
\frac{\lVert\bar g_t\rVert_2^2}
{\frac{1}{M}\sum_j\lVert g_{t,j}\rVert_2^2-\lVert\bar g_t\rVert_2^2+\varepsilon}.
$$

Это превращает стохастику из помехи в измеряемую характеристику режима.

### 6.3. Fixed-probe gradient

Если дополнительный backward допустим, раз в `diagnostic_every` вычисляется gradient одного и того же train probe. Его projections являются детерминированной функцией checkpoint и значительно лучше подходят для EDM, чем gradients случайных batches.

---

## 7. Context observers

Эти метрики сохраняются и используются для интерпретации, но не являются достаточным warning по отдельности:

- total и layer-wise weight norm;
- parameter participation ratio;
- gradient norm;
- update norm;
- gradient cosine;
- gradient participation ratio;
- train loss и batch loss;
- train entropy, margin и confidence;
- learning rate, weight decay, batch size;
- optimizer first/second moment norms;
- activation norm;
- elapsed time и examples seen.

Task-specific Fourier metrics для modular arithmetic также сохраняются, но не входят в generic predictor.

---

## 8. Outcome и benchmark metrics

Следующие величины не входят в основной train-only score:

- `val_acc` — определяет onset и completion;
- `val_loss` — сильный benchmark predictor;
- validation entropy, margin и confidence — target-adjacent benchmarks;
- validation activation/logit sketches — exploratory upper bound.

Они нужны, чтобы ответить на вопрос: даёт ли train-only EDM сигнал раньше, чем сама validation trajectory начинает изменяться.

---

## 9. Как агрегировать случайные проекции

Для каждой проекции $r$ одного семейства $f$ рассчитывается dimension-drop score $A_{f,r}(t)$.

Family amplitude:

$$
A_f(t)=\operatorname{median}_{r=1}^{R}A_{f,r}(t).
$$

Projection consensus:

$$
C_f(t)
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbf 1\{A_{f,r}(t)\ge\delta_f\}.
$$

Семейство активирует warning только при выполнении обоих условий:

$$
A_f(t)\ge\delta_{family},
$$

$$
C_f(t)\ge0.6.
$$

Это защищает от выбора одного удачного `r0` или `r2`. В отчёте показываются медиана и диапазон по всем направлениям, а не только первая проекция.

---

## 10. Предлагаемый frozen primary pool

Для новых запусков основной пул состоит из трёх независимых семейств:

| Семейство | Observer | Роль |
|---|---|---|
| Representation | fixed-train-probe activation sketches, $R=16$ | внутренняя геометрия представления |
| Parameter direction | normalized per-layer weight-direction sketches, $R=16$ | изменение структуры весов без radial shrinkage |
| Trajectory | $K$-step displacement sketches, coherence и path efficiency | согласованность движения оптимизатора |

Gradient EMA/SNR используется как подтверждающий context family, но от него заранее не требуется падение размерности: при переходе его сложность может увеличиваться.

Первичный warning разумно определить как согласованный dimension drop хотя бы в двух из трёх primary families. Порог и lead horizon калибруются только на development runs.

Для уже существующих логов этот primary pool полностью восстановить нельзя: имеются только три activation projections, сырые weight projections и слишком короткие displacement intervals. Поэтому старый `p=211 WD=0.5` report остаётся exploratory evidence, а не тестом frozen predictor.

---

## 11. Предварительная проверка пригодности observer до EDM

Перед расчётом intrinsic dimension каждый scalar observer проходит quality gate.

### 11.1. Повторяемость checkpoint

Один checkpoint измеряется несколько раз. Для deterministic observer относительный разброс должен быть практически нулевым. Для stochastic observer оценивается noise floor.

### 11.2. Временная связность

Базовый фильтр:

$$
\rho_1=\operatorname{corr}(x_t,x_{t-1})>0.5.
$$

Это не универсальная теорема, а практическая защита от применения delay embedding к почти белому шуму.

### 11.3. Отсутствие saturation

Observer исключается из текущего окна, если его variance близка к машинному noise floor или большая часть значений совпадает.

### 11.4. EDM stability

Знак dimension change должен сохраняться при разумных изменениях:

- окна $W$;
- $k$;
- embedding dimension $E$;
- Theiler exclusion;
- raw/detrended preprocessing.

### 11.5. Специфичность

Warning должен быть редким в matched intervals non-grokking runs. Красивое падение в одном grokking run недостаточно.

---

## 12. Исправления аналитического framework

1. Использовать только trailing windows.
2. Timestamp каждой ID-оценки — `window_end_step`.
3. Добавить Theiler exclusion не меньше $(E-1)\tau$.
4. Не считать перекрывающиеся windows независимыми samples.
5. Показывать все projection directions или family aggregate.
6. Хранить raw и detrended отдельно; primary preprocessing зафиксировать заранее.
7. Сравнивать с raw-trend baseline того же observer.
8. Подбирать thresholds на development runs и замораживать до test.
9. Считать false warnings на WD=0, early-generalization и censored runs.
10. Разделять predictor metrics и outcome metrics.

---

## 13. Что продолжать записывать в лог

Даже метрики, не вошедшие в primary predictor, лучше не удалять. Стоимость их записи уже невелика, а будущие гипотезы могут измениться.

Минимальный постоянный лог:

```text
train/validation loss and accuracy
train/validation entropy, margin, confidence
total and layer-wise weight norms
gradient and update norms
gradient cosine and participation ratios
optimizer moment norms
fixed train-probe activation/logit sketches
normalized weight-direction sketches
K-step displacement sketches and path efficiency
EMA-gradient projections and gradient SNR
raw gradient/update projections
learning rate, weight decay, examples seen
phase flags and stopping metadata
```

Validation-side метрики сохраняются для outcome labeling и benchmark, даже если не входят в основной прогноз.

## Итог

Провал текущих gradient/update projections не противоречит их generic характеру: измерялся случайный mini-batch gradient, а не гладкая функция состояния модели. Random projection не удаляет batch noise и одна проекция не гарантирует попадание в переходное направление.

Для следующей серии основной train-only пул следует строить из fixed-probe activation sketches, нормированных weight-direction sketches и длинногоризонтных displacement/coherence observers. Градиенты нужно сначала усреднить во времени, взять из Adam EMA или вычислить на fixed probe. `val_loss` остаётся benchmark, а weight norm — context observer, но ни одна из них не назначается единственным predictor.
