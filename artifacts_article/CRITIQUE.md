# Разбор `icomp_artifacts.pdf` (сборка 01.09.2026, 34 стр.)

Ссылки даются на **номера строк на полях PDF** (сквозная нумерация 1–915) — они
однозначнее номеров страниц. Приложения обозначены буквами, как в статье.

Разбор разделён на три части: **фактические расхождения**, **пробелы в
аргументации** и **стиль**. Внутри каждой — по убыванию важности.

---

## Часть 0. Одной страницей

Статья технически честная: она сама называет почти все свои слабые места. Проблема
в **том, где именно она их называет**. Каждое из пяти самых серьёзных ограничений
живёт в приложении или в придаточном предложении, а abstract и introduction
сформулированы так, будто их нет. Рецензент, читающий подряд, споткнётся не о
недостаток корректности, а о несоответствие между заявкой и её собственными
оговорками.

Пять мест, где это критично:

1. **«шесть систем»** в abstract — четыре из них статья сама аннулирует (стр. 211).
2. **«constructed ground truth»** — головная система скорится не против построенной
   истины, а против *измеренного* participation ratio (табл. 11), который §6.4
   отдельно объявляет другой величиной.
3. **Все валидационные системы внешне возбуждены, обучение — нет** (стр. 230–233).
   Сказано один раз, в конце §6.1, и не вынесено ни в abstract, ни в выводы.
4. **Отношение идентифицируемости считается непоследовательно** между эталонной
   полосой и тестируемыми случаями (стр. 779–782). Это подрывает §7.1 — раздел, на
   котором держится решение уйти к сохранённой траектории.
5. **Приложение M** содержит два результата *против* оценщика (стр. 895–902), из
   которых один означает, что на окружности оценка воспроизводится из дробных
   долей частоты без всякого поиска соседей. В основном тексте этого нет.

Плюс один стилистический дефект, заметный сразу: **12 из 25 подписей к рисункам и
таблицам — косвенные вопросы** («Whether…» ×8, «Why…» ×2, «How…» ×2). Подпись
должна называть содержимое, а не задавать вопрос.

---

## Часть 1. Фактические расхождения

### 1.1. Abstract обещает шесть систем; статья аннулирует четыре из них

**Где:** abstract, стр. 9–10; §1 Contributions, стр. 43–44; §5, стр. 184; против
§5.2, стр. 210–214 и табл. 4.

Abstract: «On six constructed systems of known active dimension it is accurate to
about one component, up to a ceiling near eight.»

§5.2: «It invalidated four of our six systems and the function-subspace variant
besides, each of which returns its reported estimate with the parameters frozen».

Табл. 4, колонка «requirements failed», построчно: *oscillating matrix* — не
применимо; *online linear regression* — две; *logistic regression* — две; *frozen
nonlinear decoder* — одна; *perceptron in a k-subspace* — одна; *image data,
parameter subspace 1–8* — **none**; *image data 1–20* — одна; *image data,
function subspace* — **all of them**.

То есть требованиям, которые статья зафиксировала *до* экспериментов (§4),
удовлетворяет **ровно одна строка из восьми**. Заявлять по такому основанию
точность «на шести системах» нельзя: читатель понимает это как шесть независимых
подтверждений, а подтверждение одно.

**Как исправить.** В abstract и в Contributions писать то, что есть на самом
деле, и это не слабее — это просто честнее:

> On a linear head trained inside a k-dimensional subspace over a frozen image
> network — the one system of six that meets every requirement fixed in advance —
> the estimate is accurate to about one component up to a ceiling near eight. The
> other five fail at least one requirement; §5.2 and table 4 say which.

Слово «шесть» при этом остаётся, но перестаёт работать как аргумент от
количества.

### 1.2. «Constructed ground truth» — головная система скорится против измеренного PR

**Где:** §1, стр. 43 («against a constructed ground truth»); §4, стр. 172–174;
табл. 11 (подпись); §6.4, стр. 249–254.

§4 определяет протокол: «the ground truth follows from the construction rather
than from a fit. Every system is built with r independent quasiperiodic phases, so
that dact(R) = r; the effective rank of the recorded trajectory **must then
confirm** that all r are comparably excited.»

Подпись табл. 11: «Every estimate on this system is **scored against these
values**, never against r.»

Это разные вещи. По §4 PR — *проверка*; по табл. 11 PR — *мишень*. А §6.4 целиком
посвящён доказательству того, что PR и активная размерность — **не одна и та же
величина**, и что оценка следует размерности, а не PR. Скорить против PR и
одновременно утверждать, что оценщик меряет не PR, — внутреннее противоречие.

Численно для рекуррентной ветви это ничего не меняет (табл. 11: PR = 1.00, 2.00,
4.00, 6.00, 8.00 против r = 1, 2, 4, 6, 8), но для `qp_nopre` (3.73 при r = 4,
5.98 при 6, 7.77 при 8) и для шумных ветвей — меняет.

**Как исправить.** Одно предложение в §5.2 и одно в подписи табл. 11:

> Where the two agree to two decimals (table 11, `qp`) the choice is immaterial;
> where they do not, we score against the measured value and say so, because the
> construction there did not excite what it was asked to.

И убрать «constructed ground truth» из §1 — заменить на «ground truth fixed by
the construction and verified on the recorded trajectory».

### 1.3. Все валидационные системы внешне возбуждены; обучение — нет

**Где:** §6.1, стр. 230–233. Больше нигде.

> «Every system evaluated above is externally driven, whereas training is not, so
> the evaluation establishes only a conditional result.»

Это **главное ограничение всей работы**, и оно стоит одним предложением в конце
подраздела, между разбором Тейлеровского исключения и приложением K. Ни abstract,
ни Contributions, ни Conclusion, ни Limitations его не повторяют. Limitations
(стр. 325–334) перечисляют семь других вещей и эту — нет.

**Как исправить.** Вынести в Limitations первой строкой и одной фразой в abstract.
Формулировка уже написана автором, её надо просто переставить.

### 1.4. Отношение идентифицируемости посчитано непоследовательно

**Где:** приложение I.1, стр. 777–782. В основном тексте — нет.

> «The five records the cap already binds on escape it, since both estimates there
> sit at 150. It bites on the remaining two and, **more consequentially, on the
> constructed systems of section 5: the reference band of section 6.4 was itself
> measured with 76 against 150. A ratio built from two differently excluded
> estimates is not the quantity section 3.4 describes.**»

Последнее предложение написано самим автором, и оно означает, что ρ_ident в §6.4
(эталон) и ρ_ident в §7.1 (тест) — **не одна и та же статистика**. А §7.1 на
основании их сравнения отвергает обе постановки grokking'а и обосновывает переход
к сохранённой траектории. То есть под самым сюжетообразующим решением статьи лежит
несогласованное измерение, и сказано об этом в приложении I.1 на 27-й странице.

Рядом, стр. 768–770:

> «it happens on five of the seven parameter-norm records of section 7. Those five
> were scored at WT = 150 rather than at the value the rule asks for, **in the
> direction that inflates the estimate**.»

Пять из семи записей, на которых основан §7.1, посчитаны с нарушением
собственного правила, в сторону, завышающую оценку. §7.1 об этом молчит.

**Как исправить.** Либо пересчитать (предпочтительно — это, судя по табл. 17,
доступно), либо вынести в §7.1 явную оговорку и ослабить формулировку вывода: не
«The level is therefore not a dimension in either setting», а «The level fails the
diagnostic we can compute; appendix I.1 shows that diagnostic is itself
inconsistent between the reference and the test, so the rejection rests on the
trend-crossing count alone in five of seven runs.»

### 1.5. Приложение M содержит два результата против оценщика

**Где:** приложение M, стр. 895–902. В основном тексте — нет.

> «Two findings count against the estimator. The linear participation ratio is
> higher on the circle than on the four-torus (table 23), so anyone treating it as
> a count of phases would order the two backwards. On the circle **the estimate is
> also not geometric**: … two of the eight seeds drew a frequency whose multiples
> cluster into near-duplicates; **their estimate is reproducible from the
> fractional parts of those multiples alone, with no signal and no neighbour
> search.**»

Второй пункт — сильное утверждение: в четверти сидов число, которое статья
называет измерением геометрии, вычисляется из арифметики частоты. Это надо в
Limitations основного текста.

*(Первый пункт, кстати, — лучший из имеющихся аргументов «PR ≠ размерность», и он
в §6.4 не использован. Там аргумент строится на убывании амплитуд; здесь PR
**переставляет порядок** — это резче.)*

### 1.6. Потолок называется тремя разными числами

**Где:** abstract, стр. 10 («a ceiling near eight»); §5.2, стр. 204–205 («saturates
above about eight… at k = 20 it tracks the rank to about ten»); §5.2, стр. 206
(«never exceeds about eleven»); приложение L, стр. 861 («It never passes 11.2»).

Три величины (8 / 10 / 11.2) относятся к трём разным вещам: к замороженной
восьмикомпонентной конфигурации, к двадцатикомпонентной, и к максимуму по всему
свипу. Статья нигде не разводит их явно, и читатель видит противоречие.

**Как исправить.** Ввести одно предложение в §5.2: «Three ceilings must be kept
apart: the frozen eight-component configuration saturates near eight, the
twenty-component one near ten, and no configuration we swept tracks beyond 11.2
(appendix L).»

### 1.7. Потолок объявлен загадкой, хотя приложение L даёт ответ

**Где:** §5.2, стр. 206–209 («One caveat: …the limit **may be** the number of
components the delay window resolves linearly, with no geometry involved») против
приложения L, стр. 869–872 («A purely linear statistic reproduces the whole
dependence… **Most of the ceiling is therefore the number of components the delay
window resolves linearly.**»).

В основном тексте это гипотеза в оговорке, в приложении — установленный вывод.
Надо согласовать, и лучше в сторону приложения: результат сильнее и он измерен.

### 1.8. Подгонка в табл. 22 подана как объяснение

**Где:** приложение L, стр. 866–868 и табл. 22.

> «a form logarithmic in both variables fits **an order of magnitude more closely
> than the spread of the data itself**.»

`8.61 log10 Emax + 0.90 log10 N − 5.31`, RMSE = 0.19 при разбросе данных 2.00.
Это три свободных параметра на 13 ячеек (6 в свипе Emax + 7 в свипе N), без
отложенной проверки. RMSE в десять раз меньше разброса данных — это не
подтверждение формы, это признак того, что параметров хватило. Подано же как
довод.

**Как исправить.** Либо добавить отложенную ячейку, либо переписать без претензии
на объяснение: «A two-term logarithmic form absorbs the residual variation; with
three parameters on thirteen cells we do not read it as a mechanism.»

### 1.9. Abstract утверждает больше, чем дают суррогаты

**Где:** abstract, стр. 17–19 («falls at the same step in those four runs and in
neither of the other two, **more steeply than in randomised surrogates of each
run**») против табл. 16.

p-значения генерализующих запусков: 0.025 / 0.025 / 0.025; 0.050 / 0.050 / 0.025;
0.050 / **0.075** / 0.025; 0.050 / 0.050 / 0.050. При 39 суррогатах минимально
достижимое p равно 0.025, то есть половина ячеек стоит на разрешающей способности
теста, а одна выходит за 0.05.

Формулировка «more steeply than in randomised surrogates» читается как
установленная значимость. Точнее — и по-прежнему в пользу работы — так:

> the fall is in the top 2.5–7.5 % of what surrogates of the same run produce, at
> every smoothing length, in all four runs and in neither of the other two.

### 1.10. «Among the largest» против квантилей

**Где:** §7.3, стр. 302–304 («That drop is among the largest any of the four ever
shows») против табл. 16, колонка quantile: 0.99, 0.93, 0.92, **0.89**.

0.89 — это «в верхних 11 %», что «среди самых больших» описывает щедро. Напишите
диапазон: «between the 89th and 99th percentile of that run's own window-to-window
changes».

### 1.11. Терминология: `drive` против `forcing`

**Где:** везде. Подсчёт по тексту: `drive/driven/drives/driving` — **37**
вхождений, `forced/forcing` — **6**.

При этом теорема 3 в приложении A названа **«Delay embedding for forced systems;
Stark, 1999»** — то есть при цитировании источника используется его термин, а во
всём остальном тексте — свой. Stark говорит о *forced systems*; «drive» в
нелинейной динамике обычно означает нечто иное.

**Как исправить.** Один термин на всю работу — `forcing`, как у Stark. Это ровно
то, что уже сделано в слайдах доклада (`phases forced`), так что расхождение
сейчас ещё и между статьёй и докладом.

### 1.12. Мелочи

| Где | Что | Как |
|---|---|---|
| стр. 4 (abstract) | «how much of it does one such scalar carry?» — «it» отсылает то ли к размерности, то ли к траектории; «carry much of a dimension» не говорят | «how much of that dimension survives in one such scalar?» |
| стр. 25 | «Both questions are argued with whichever count of degrees of freedom is least expensive to compute» — размашистое утверждение обо всей области без единой ссылки | либо снабдить ссылками, либо снять |
| стр. 26 | «the available dimension of the subspace the parameters are confined to [Li et al., 2018]» — Li et al. меряют *наименьшее* k, решающее задачу; §2 стр. 61 это и говорит, а §1 — нет | согласовать §1 с §2 |
| стр. 89 | «We call k the available dimension» — вводится как своё определение, хотя в §1 приписано Li et al. | «Following Li et al. [2018] we call k…» |
| стр. 153 | «The value returned here is invalid rather than a dimension of one» — двойная отрицательная конструкция на ровном месте | «What the estimator returns here is a statistic of the window.» |
| стр. 271–273 | «A change within a single run could still be one» — «one» отсылает через два предложения к «a dimension» | повторить существительное |
| табл. 5 | три разных числа (0.90 / 0.48 / 0.382) для одного измерения — **это сделано хорошо и честно**, но в abstract попадает только худшее | оставить как есть, упомянуть в §5.2, что цифра — худшая из трёх агрегаций |

---

## Часть 2. Пробелы в аргументации

### 2.1. Из четырёх условий валидности проверяются два

§3.3 (стр. 139–146) перечисляет условия, при которых оценка есть размерность:

1. окно внутри одного режима;
2. детерминированное отображение на компактном инвариантном множестве;
3. запись покрывает множество достаточно плотно, чтобы после исключения Тейлера у
   каждой точки остался настоящий возврат;
4. радиусы соседей покрывают диапазон постоянного показателя.

§3.4 даёт диагностики для (1)–(2) и прямо признаёт: «Neither tests recurrence
itself» — то есть (3) без диагностики. Limitations (стр. 326–328) признаёт, что
(4) **не проверяется никогда**: «they are whatever the twenty nearest neighbours
give».

При этом в приложении I.2 (стр. 801–806) диагностика возвратности предлагается
(отношение d̂MG при WT = 150 к WT = 20) и тут же обесценивается автором:
«The principled version is older than either: a space–time separation plot
[Provenzale et al., 1992] … and the exclusion should be chosen from one».

**Итог:** статья формулирует четыре условия, снабжает диагностиками два, для
третьего держит в приложении вариант, который сама называет неправильным, и
четвёртое не проверяет. Читатель обнаруживает это, только собрав три места
воедино.

**Как исправить.** Табличка на четыре строки в §3.4: условие → диагностика →
где измерена → чего не хватает. Это одновременно закроет упрёк и покажет, что
работа знает границы собственного протокола.

### 2.2. §7.1 и §7.3 отвечают на разные вопросы, и переход между ними размыт

§7.1 отвергает обе постановки на окне в треть записи. §7.3 запускает тот же
оценщик на окне в 60 отсчётов и получает сигнал. Между ними стр. 271–273 («This
window cannot show it: it spans tens of thousands of optimiser steps against a
transition a few hundred steps wide») и стр. 297–301.

Логически всё сходится, но читатель успевает подумать, что статья сначала
отвергла метод, а потом им же получила результат. Не хватает одного предложения на
стыке — того, что §7.3 говорит только в конце (стр. 310–311): «A window as short
as the transition need not lie inside one regime, so the fall is a signal that
tracks the transition and not a measurement of equation (1).» Это надо сказать
**до** результата, а не после.

### 2.3. Приложение H.2 подрывает §7.3 сильнее, чем признаёт §7.3

Приложение H.2 (стр. 707–713): в длинном запуске при p = 211 без weight decay
оценка падает через десятки тысяч шагов после запоминания, и механизм —
утроившаяся норма параметров, то есть ровно тот confound из §6.3, который §7.3
называет «the confound that matters most».

§7.3 отвечает на это (стр. 714–716) тем, что там другое окно и другая
конфигурация. Ответ верный, но короткий, а сам факт — что падение оценки не
специфично для генерализации — в основном тексте не назван. В Limitations о нём
есть только «The drop at the matched window is close to the drop a ramped gain
alone produces».

### 2.4. Табл. 20: два сида расходятся, и это не откомментировано

Приложение K, табл. 20: при 2.5·10⁶ и 2.8·10⁶ один сид расходится, при 3·10⁶ —
оба. В колонке `return` при 2·10⁶ стоит «0.120, 1.013» — то есть один сид
возвращается (0.120), другой нет (1.013), при одинаковой скорости обучения. Текст
(стр. 840–842) говорит: «only five of them return close to states already passed
through. The return column of table 20 separates those two groups by a factor of
eight with nothing in between.» Разрыв 0.120 → 1.007 — это фактор восемь, верно,
но то, что **на одной и той же скорости обучения два сида попадают в разные
группы**, куда интереснее и никак не прокомментировано.

---

## Часть 3. Стиль и читаемость

Общий диагноз: текст **плотный и точный, но однообразный**. Он написан одной
интонацией — короткое утверждение, затем оговорка, — и на 34 страницах это
утомляет. Ниже конкретные шаблоны с подсчётами.

### 3.1. Подписи-вопросы: 12 из 25

**«Whether…» ×8:** рис. 4, 6, 7, 8, 13; табл. 8, 10, 15.
**«Why…» ×2:** табл. 5, рис. 12. **«How…» ×2:** табл. 4, рис. 10.

Подпись к рисунку должна отвечать, а не спрашивать. Читатель, листающий статью по
картинкам (а рецензенты листают именно так), получает восемь вопросов подряд и ни
одного вывода.

| было | стало |
|---|---|
| Whether any grokking log lies where the estimate was accurate | No grokking log lies where the estimate was accurate |
| Whether the estimate follows the active dimension or the effective rank | The estimate follows the active dimension, not the effective rank |
| Whether the full-batch statistic has any range at longer windows | The full-batch statistic leaves its floor only above 10⁴ steps |
| Whether a change of the active dimension is recovered at both levels and after it reverses | A change of the active dimension is recovered at both levels and after it reverses |
| How accurately the estimator recovers the active dimension on each system | Recovery on each system, with the requirements each fails |

Оставить вопросительную форму имеет смысл там, где ответ отрицательный и в этом
соль (рис. 7, 13) — но не восемь раз.

### 3.2. `rather than` — 21 раз

Это английский эквивалент «а не», на который уже указывали в докладе. Часть
употреблений законна (противопоставление — суть предложения), но 21 на 18 тысяч
слов — это тик, а не приём. Примеры, где конструкция ничего не даёт:

- стр. 143 «is a statistic of the window and not a dimension» → «…is a statistic
  of the window»; отрицание уже несёт заголовок раздела;
- стр. 153 «invalid rather than a dimension of one» → см. 1.12;
- стр. 172 «follows from the construction rather than from a fit» → «follows from
  the construction»;
- стр. 622 «report an outcome rather than a setting» → «report an outcome»;
- стр. 681 «gives a limit of the measurement rather than a negative finding» →
  «gives a limit of the measurement».

Правило простое: если вторая половина противопоставления — то, что читателю и в
голову не пришло, её надо убрать.

### 3.3. Отрицания: около 90 на 18 тысяч слов

`does not` 24, `never` 18, `neither` 11+2, `cannot` 9, `nor` 6+3, `is not` 9,
`do not` 7, `are not` 3. Плотность — примерно одно отрицание на 200 слов, и они
кучкуются: §3.3, §6.1, Limitations читаются как перечень того, чего работа не
делает.

Особенно тяжёлые места — двойные отрицания:

- стр. 863 «Nor is E > 2d in the form we used it» — начинать предложение с `Nor`
  трижды в статье (стр. 696, 720, 863) уже много;
- стр. 756 «neither run is a negative result» — после трёх отрицаний в том же
  абзаце;
- стр. 202 «for no scored observer does the roughness of the log order them» →
  «the roughness orders the ranks for no observer we scored» или, лучше, прямо:
  «every observer that orders the ranks does so through the estimate, not through
  roughness» (и это как раз тот случай, где противопоставление уместно).

### 3.4. `, and` — 68 раз

Наибольший вред — там, где `and` соединяет положительное и разрушительное:

> стр. 219–220: «Noise at a fortieth of the drive amplitude **nearly preserves the
> ordering and** moves the estimate at r = 1 from 1.1 to 11.3.»

Сохранение порядка и рост оценки в десять раз — это не однородные члены. Нужно
двоеточие или точка: «Noise at a fortieth of the drive amplitude leaves the
ordering nearly intact. It also moves the estimate at r = 1 from 1.1 to 11.3.»

### 3.5. `One caveat:` — дважды, и оба раза это не caveat

Стр. 206 и стр. 287. В обоих случаях за двоеточием стоит не оговорка, а
альтернативное объяснение результата — то есть возражение по существу. Подавать
возражение как «одна оговорка» — значит его прятать. См. 1.7.

### 3.6. Цифры в местах, где они не работают

Претензия, уже звучавшая по слайдам, в статье тоже есть. Примеры, где число можно
снять без потери:

- стр. 266–267 «their estimates ranging from four to thirteen components across
  configurations» — здесь число как раз работает, оставить;
- стр. 594 «every rank above one returns about 2.5» — это ключ к рис. 5, оставить;
- стр. 657–658 «That disagreement is a per cent of the value at the median and
  under nine per cent at its worst, smaller than every effect section 7.2
  reports» — три числа, из которых работает последнее сравнение; первые два можно
  оставить только в подписи к таблице;
- стр. 719–720 «a window of a third of each record, 39 990 optimiser steps,
  advanced by 10 000 steps at a time. That gives nine windows per run, localising
  a feature only to ±19 995 steps» — четыре числа в двух предложениях, из которых
  читателю нужно одно: разрешение ±20 тысяч шагов против перехода в несколько
  сотен. Остальное — в таблицу.

### 3.7. Что написано хорошо

Чтобы правка не съела сильные места:

- **§3.1** (стр. 84–119) — определение активной размерности. Строгое, введено по
  порядку, каждое обозначение определено до использования. Это лучший раздел
  статьи.
- **Табл. 5** — «Why three different errors are quoted for one measurement».
  Редко кто выписывает свои три разные цифры в таблицу и объясняет, откуда каждая.
  Это надо сохранить и на это надо ссылаться.
- **Стр. 143–144** — «The estimator itself would accept points drawn from
  independent runs; recurrence is needed because our entire sample is one
  trajectory.» Точное объяснение в двух строках.
- **Стр. 175–177** («Frozen configuration») — формулировка того, почему нельзя
  искать настройки системы и оценщика одновременно. Готовая цитата.
- **Приложение I.2** — разбор того, почему исключение Тейлера задаёт значение на
  транзиенте. Механизм показан, а не назван.
- **§4** целиком — протокол, зафиксированный до экспериментов, с явной таблицей
  того, кто чему не удовлетворяет. Именно это делает часть 1 настоящего разбора
  возможной: статья честна, надо лишь переставить акценты.

---

## Часть 4. Порядок правок

**Обязательно перед подачей:**

1. Abstract и Contributions: снять «six systems» и «constructed ground truth»
   (1.1, 1.2).
2. Вынести «every system is externally driven, training is not» в Limitations и
   abstract (1.3).
3. Оговорка о непоследовательном ρ_ident — в §7.1, не в приложении I.1 (1.4).
4. Два результата приложения M — в Limitations (1.5).
5. Согласовать три потолка (1.6) и снять «One caveat» вокруг линейного объяснения
   (1.7).

**Сильно улучшит текст:**

6. Переписать 12 подписей-вопросов (3.1).
7. Сократить `rather than` вдвое и разредить отрицания в §3.3, §6.1, Limitations
   (3.2, 3.3).
8. Таблица «условие → диагностика → где измерена» в §3.4 (2.1).
9. Перенести предложение про «signal, not a measurement» в начало §7.3 (2.2).

**По возможности:**

10. Единый термин `forcing` вместо `drive` (1.11) — заодно сойдётся со слайдами.
11. Ослабить формулировку про суррогаты и квантили (1.9, 1.10).
12. Убрать претензию на объяснение из табл. 22 (1.8).
13. Прокомментировать расхождение сидов в табл. 20 (2.4).
