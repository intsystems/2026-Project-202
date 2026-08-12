# Рецензия на `report.pdf`

**Версия:** PDF от 11.08.2026, 24 страницы (9 страниц основной части + references/appendices).  
**Предмет оценки:** научная аргументация, валидность экспериментов, воспроизводимость и подача. Замечания привязаны к страницам PDF и, где полезно, к LaTeX-разделам.

## Краткий вердикт

Это необычно добросовестная работа: она сама показывает, что скалярный delay-embedding в двух реальных настройках grokking **нельзя** интерпретировать как размерность, и документирует ряд отрицательных контролей. Сильные стороны — чёткое различение available / functional / active dimension, продуманные контроли нулевой скорости обучения и явное описание ограничений.

Однако центральный положительный результат пока существенно уже, чем сформулировано в title, abstract и conclusion: он получен на **искусственно квазипериодически возбуждаемой** и специально изотропизированной системе. В этой конструкции manifold dimension delay-реконструкции намеренно совпадает с participation ratio траектории. На реальных training logs метод не работает как измеритель целевой величины. Кроме того, нет независимой валидации диагностик, полноценной оценки неопределённости и достаточно близких контролей для заявления о bottleneck при generalisation.

**Оценка в текущем виде:** сильная техническая/отрицательная работа, но риск отказа высокий из-за чрезмерно широкого framing и недостаточно независимой эмпирической валидации. Для сильной conference-подачи нужно либо сузить claim, либо добавить независимые эксперименты, перечисленные в первых пунктах ниже.

---

## Критические замечания (исправить до подачи)

### C1. Целевая величина и то, что оценивает метод, различны
- **Где:** стр. 1, abstract и introduction; §3.1, стр. 3; §6.4, стр. 8; conclusion, стр. 9.
- **Проблема:** `d^pos` определён как participation ratio (PR) ковариации — зависящий от спектра effective rank. Delay embedding + Levina–Bickel оценивает локальную геометрическую/intrinsic dimension реконструированного множества. В §6.4 авторы сами показывают расхождение: при анизотропной амплитуде PR меняется от 4 до 1.65, а estimate остаётся около 4. Значит метод в общем случае не измеряет определённую в eq. (2) *active dimension*.
- **Почему это критично:** основной claim «single scalar log recovers active dimension» верен лишь в специально построенном изотропном случае, а не для определённой в статье величины. Положительный результат частично встроен в конструкцию ground truth.
- **Как исправить:**
  1. В title/abstract/contributions/conclusion заменить общий claim на: «в изотропно возбуждаемых квазипериодических системах scalar estimator recovers the **resolvable mode count**, which coincides with covariance PR by construction».
  2. Вынести `mode count` в отдельную формальную target quantity; не называть его `d^pos`.
  3. Если цель всё же PR, разработать/валидировать estimator именно PR из scalar series либо показать recovery на широкой сетке анизотропий с заранее заданной допустимой ошибкой. Одного опыта §6.4 недостаточно.

### C2. Положительная валидация не демонстрирует применимость к обычному обучению
- **Где:** abstract, стр. 1; §5.3, стр. 6–7; Fig. 1; §8, стр. 9.
- **Проблема:** единственный «complete» recovery result строится внешним квазипериодическим drive через веса loss; направления градиента затем ортогонализируются и компенсируются (Appendix E). Это демонстрирует распознавание специально введённого тора, а не естественно возникающей низкоразмерной динамики optimiser-а. В двух реальных grokking settings (§7) scalar method объявлен inadmissible.
- **Почему это критично:** название и введение обещают инструмент для *training run*, но эксперимент доказывает значительно более узкое утверждение о forced system.
- **Как исправить:** добавить хотя бы одну независимую, не сконструированную под тор, систему с доступной trajectory ground truth: например детерминированный optimizer с проверяемым предельным циклом/квазипериодической динамикой, либо заранее зафиксированная реальная training setup с разными контролируемыми rank-ограничениями без внешнего синусоидального teacher. Иначе честно переименовать статью/abstract в работу о границах scalar delay estimation на *synthetically forced optimisation systems*.

### C3. «Диагностики correctly identify / flag» не валидированы как диагностики
- **Где:** abstract, стр. 1; contribution 3, стр. 2; §3.4, стр. 4; §6.4, стр. 8; Fig. 2 и caption, стр. 18 приложения.
- **Проблема:** пороги `rho_ident <= 1.10` и `> 8 crossings` описывают диапазоны на тех ��е сконструированных режимах, на которых проверяется метод. В тексте прямо сказано «calibrated against no wider family» и «not a validated classifier». При этом abstract говорит, что они «identify correctly», а conclusion — что они «flag ... regimes».
- **Почему это критично:** это leakage между разработкой диагностик и их оценкой; неизвестны sensitivity, specificity, ложные срабатывания и поведение на других transients/noise/observers/длинах записей.
- **Как исправить:** сформировать независимый тестовый набор систем и условий, не использованный при выборе порогов: иные спектры шума, transient shapes, record lengths, lags, observers, амплитуды, нелинейности. До начала теста зафиксировать rule и дать confusion matrix, ROC/PR-кривую или хотя бы sensitivity/specificity с доверительными интервалами. Пока этого нет, во всех ключевых ме��тах называть их *heuristic warning indicators calibrated on the tested constructions*.

### C4. Настройка estimator-а хрупка и частично оптимизирована под тестовые семейства
- **Где:** §4 requirement 2, стр. 5; §5.3, стр. 6; §6.2, стр. 7; Appendix B, стр. 13–14; Appendix K/L о Theiler, стр. 22–23.
- **Проблема:** Appendix B сообщает, что обе frozen configurations находятся на границе grid; для 8-direction selection error меняется в ~5 раз (0.324–1.499) на тех же logs. Для 20 directions нет withheld seeds и result признан exploratory. Лаг `tau=4` переносится из drive period 16 на grokking logs с ACF 161–858; сам текст показывает, что от одного `tau` estimate меняется от 2.60 до 63.35.
- **Почему это критично:** заявленная ошибка 0.87 не демонстрирует устойчивость к realistic preprocessing/configuration choices; применимость к новому log фактически требует неизвестного matched lag.
- **Как исправить:**
  1. Добавить sensitivity table/figure по `tau`, `Emax`, `m`, window length и Theiler для **основного 1–8 result**, не только отдельные illustrative sweeps.
  2. Использовать заранее определённую data-only процедуру выбора lag (сравнить MI/ACF/FNN и robustness criterion) либо явно сказать, что метода выбора lag нет и claim относится только к известной частоте drive.
  3. Перезапустить 20-direction validation с disjoint seeds и Theiler window не меньше embedding span; не использовать её для claims до этого.
  4. В abstract убрать «orders it to twenty» либо пометить как exploratory.

### C5. Неправильная/неполная статистическая единица анализа и нет uncertainty
- **Где:** Table 2, стр. 6; §5.3, стр. 6–7; Fig. 1, стр. 7; §7.3, стр. 9; Fig. 3, стр. 19.
- **Проблема:** многие окна одной траектории, observers одной run и ranks одной конструкции зависимы. Table 2 даёт MAE/корреляцию без CI; Fig. 1 показывает median без error bars; основная grokking находка основана на 4 generalising runs и 2 controls. «No overlap» для четырёх пар также не заменяет оценки неопределённости.
- **Почему это критично:** window-level/observer-level агрегация может существенно завысить ощущение объёма данных. Нельзя оценить, насколько стабильны slope, MAE, момент dip и различие с control.
- **Как исправить:** считать независимой единицей seed/run, а не window; дать число runs, seed-level scatter и 95% bootstrap/permutation CI для MAE, slopes, Spearman и effect size. Для Fig. 3 показать все отдельные runs, интервалы по sketch/seed и предзаданный statistic (min PR в фиксированном around-`t_gen` окне) с paired test против controls. Указать, какие агрегирования были определены заранее.

### C6. Прямое измерение bottleneck не поддерживает причинную или достаточно специфичную интерпретацию
- **Где:** abstract; §7.3, стр. 9; §8, стр. 9; Appendix G, стр. 19–20; Fig. 3, стр. 19.
- **Проблема:** controls — runs без weight decay — одновременно отличаются регуляризацией, generalisation outcome и, вероятно, scale/geometry траектории. Это не matched control для вопроса «связан ли dip именно с generalisation». Окно центрировано и шириной 600 steps, а parameter-space dip отстоит от `t_gen` на 600–1000 steps; измеряются detrended positions, а не тот же `d^pos`, который валидировался ранее. Не показана robustness к длине окна, detrending rule, sketch dimension и probe set именно для transformer result.
- **Почему это критично:** наблюдение интересно, но сейчас это корреляция в малой convenience sample, а wording «near onset» и «bottleneck» сильнее данных.
- **Как исправить:**
  1. Сформулировать результат как exploratory association, не как signature of generalisation.
  2. Добавить controls, меняющие weight decay/seed/task так, чтобы отделить generalisation от decay; желательно runs с сопоставимым training dynamics, но различным outcome.
  3. До анализа определить time-alignment и statistic; провести sensitivity к window 300/600/1200, stride, detrend order, параметрическому/функциональному space.
  4. Показать raw trajectory PR вместе с detrended PR и доказать, что dip не является фильтрационным артефактом.

### C7. CountSketch validation недостаточна в области, где делается главный вывод
- **Где:** Appendix G, стр. 19–20; Table 12.
- **Проблема:** synthetic validation покрывает rank только до 10, тогда как plateau/function-space values на Fig. 3 значительно выше. Две hash families измеряют их взаимное расхождение, но не systematic bias, общий для обоих sketches. Exact comparison на реальной траектории сделан лишь для другого, малого perceptron architecture.
- **Как исправить:** валидировать на synthetic spectra/ranks, covering observed range (минимум до максимума Fig. 3), включая анизотропные и temporal trajectories; выполнить exact или гораздо более широкую-sketch comparison на subset реального transformer trajectory; показать результат для нескольких sketch dimensions (например 512/1024/2048/4096). Заменить «direct estimate of the error» на «empirical repeatability diagnostic / lower bound on sketch uncertainty».

---

## Существенные методологические и логические замечания

### M1. Definition `active dimension` зависит от параметризации и метрики
- **Где:** §3.1, стр. 3; §7.3, стр. 9; Appendix G, стр. 19.
- **Проблема:** covariance PR меняется при неортогональном reparameterization/rescaling weights; параметрические направления в нейросетях не имеют канонической евклидовой метрики. В function space введены centring и normalization logits, что тоже задаёт произвольную метрику.
- **Исправление:** явно вынести metric dependence в definition и limitations; мотивировать выбранную метрику функционально или показать invariance/sensitivity к разумным reparameterizations, layerwise scaling и choice of probe set. Не утверждать прямую сопоставимость между архитектурами без этого.

### M2. Валидация `d^pos`, а результат grokking — `d^det`/`d^upd`
- **Где:** §3.1, стр. 3; §5, стр. 5–7; §7.3, стр. 9; Appendix G.
- **Проблема:** основной benchmark scores `d^pos`; dip использует detrended-position PR и дополнительно increments PR. Это разные статистики с разной интерпретацией, особенно для drifting/nonstationary trajectory.
- **Исправление:** сделать отдельную валидацию каждого endpoint на known systems и на транзиентах; в title/result не называть их одинаковым измерением. Везде рядом с числом писать `d^pos`, `d^det` или `d^upd`.

### M3. Zero-learning-rate control сформулирован некорректно для постоянной траектории
- **Где:** §4 requirement 4, стр. 5; §5.2–5.3, стр. 6; Appendix E/Table 10, стр. 17.
- **Проблема:** при `eta=0` параметрическая covariance равна нулю; PR (`0/0`) не определён. Table 10 всё ещё печатает «active dimension 1.00», хотя README верно отмечает, что это floating-point artefact of centering a constant trajectory.
- **Исправление:** определить constant trajectory как отдельный degenerate case, не присваивать ему dimension 1; в protocol требовать, чтобы observer/estimator был flagged/undefined или не менялся с nominal rank при `eta=0`. Исправить текст и Table 10.

### M4. Theiler exclusion противоречит необходимому embedding span в части заявлений
- **Где:** Algorithm 1 и Appendix B; §5.3, стр. 7; Appendix K/L.
- **Проблема:** `W_T` cap=150 задан из вычислительной стоимости. В 20-direction configuration embedding span 624, но Theiler 150; статья признаёт это для exploratory result. Сама процедура тогда допускает перекрывающиеся delay vectors как neighbours.
- **Исправление:** убрать cap либо использовать ускоренный nearest-neighbour search/subsampling; везде, где cap активен, считать output inadmissible. Перепроверить, активен ли cap в main 1–8 results, и показать sensitivity к `W_T` ≥ span.

### M5. Обоснование «training log is weak deterministic signal inside stochastic gradient fluctuation» — не измерено
- **Где:** §6.1, стр. 7; §7.1, стр. 8.
- **Проблема:** это объяснительная гипотеза, но данные показывают лишь изменение estimator-а и heuristic diagnostics. Не разложены signal/noise, не проверены correlation time и sample-size scaling на actual logs.
- **Исправление:** пометить как hypothesis; добавить controlled batch-size/noise-scale sweep на той же задаче, spectral/ACF analyses и prediction заранее оговорённого изменения `rho_ident`.

### M6. «Transient is a curve» нельзя безоговорочно переносить на весь training window
- **Где:** Table 1, стр. 4; §3.3; §6.1; §7.1–7.2.
- **Проблема:** одна детерминированная trajectory как отображение времени — кривая, но конечная delay cloud, её curvature, nonstationarity, sampling и observation могут давать сложную geometry; statement о том, что estimate «follows shape of curve», правдоподобно, но не количественно доказано для всех представленных full-batch logs.
- **Исправление:** ослабить категоричность («in our constructed transient and the examined logs»), дать synthetic transient family с разной curvature/decay law и доказать, что trend crossings действительно отделяют нужный класс.

### M7. Нет убедительного сравнения с baseline methods / простыми predictors
- **Где:** §2, стр. 2; §5.3; Appendix C.
- **Проблема:** упоминаются PRdelay, spectral PR, roughness, LB и альтернативные ID estimators, но основной текст не даёт полн��го честного comparison: одинаковые splits, CI и простая baseline модель (roughness/ACF/trend) для recovery/admissibility. Для нескольких parameter norms автор сам признаёт возможное spectral ordering.
- **Исправление:** добавить в main table all baselines on identical held-out runs, с MAE/Spearman/CI; включить lagged PCA/SSA, correlation dimension/Facco where applicable, и простой spectral-regression baseline. Если scalar MG не превосходит их, claim должен быть about diagnostics/limitations, а не superior measurement.

### M8. «Observers fixed before run» и независимость observer split описаны недостаточно
- **Где:** §3.2, стр. 3; §5.3, стр. 6; Appendix B/C.
- **Проблема:** observers принадлежат пятью семействам и часть из них конструктивно близка к drive (loss/gradient/projections). «out-of-sample in observer for six of ten» не исключает family-level leakage; aggregate включает observers, использованные для selection.
- **Исправление:** перечислить до экспе��имента полный observer registry, какие 4 selection observers и какие 6 test observers; в основной таблице дать primary result только на entirely unseen observers, а all-observer result — отдельно. Тестировать на типичных logs (loss, accuracy, norm), а не на специально удобных projections.

### M9. Внешняя валидность ограничена маленьким frozen digits head
- **Где:** §5.3, стр. 6; Table 2.
- **Проблема:** backbone trained/frozen, обучается только linear head в 10/20-dimensional affine subspace на sklearn digits. Это уже не полноценная end-to-end neural network training и не transformer/grokking dynamics.
- **Исправление:** назвать систему именно constrained linear-head benchmark во всех claims; добавить хотя бы один benchmark с обучаемым nonlinear backbone либо существенно сузить generalisation language.

### M10. Сильные формулировки превышают объём grokking evidence
- **Где:** abstract; §7.3–7.4, стр. 9; §8.
- **Проблема:** «active dimension collapses ... near onset ... and not in controls» читатель легко интерпретирует как механизм/маркер generalisation. Но 4 vs 2 runs, controls отличаются weight decay, second setting does not reproduce, а scalar analysis полностью inadmissible.
- **Исправление:** headline conclusion: «we observe an exploratory, regularization-associated dip in directly measured detrended trajectory PR in four runs». Убрать/смягчить causal vocabulary `bottleneck`, `near onset`, `signature` до появления предрегистрации и matched controls.

### M11. Нет проверки multiple comparisons / post-selection для dip
- **Где:** §7.3, Fig. 3, Appendix G/H/I.
- **Проблема:** анализирует множество времен, пространств, пяти statistics, seed/task/controls и window sizes. Минимум/падение около `t_gen` может быть выбран после просмотра траекторий.
- **Исправление:** указать, что именно было pre-specified; применить held-out seeds или permutation test с max-statistic across time; сообщить все просмотренные analyses в supplement, а не только удачный endpoint.

### M12. «Under one component» не является достаточной абсолютной характеристикой качества
- **Где:** abstract и contribution 2, стр. 1–2; §5.3; Table 2.
- **Проблема:** MAE 0.87 усредняется по диапазону 1–8, скрывает rank-dependent error, observer dependence и saturation. Ошибка в 0.87 огромна для rank 1–2 и мала для rank 8.
- **Исправление:** давать per-rank bias/MAE/CI, relative error, calibration plot с y=x and error bars; заменять «accurate to under one component» более информативной формулировкой.

---

## Воспроизводимость и отчётность

### R1. Нет самодостаточного анонимного artifact-а
- **Где:** README, раздел “Where the numbers come from”; PDF в целом.
- **Проблема:** источники результатов ссылаются на соседние локальные каталоги `../code/...`, которых нет в `icomp_v2`, и в PDF нет repository/DOI/archived data. Рецензент не может воспроизвести числа и проверить split/configuration.
- **Исправление:** подготовить anonymized artifact: code, exact environment/lockfile, configs, raw or sufficient intermediate outputs, script `make all`, seed list и checksum результатов. В paper добавить anonymous URL и statement о доступности.

### R2. Не хватает полного протокола данных и training hyperparameters в основной статье
- **Где:** §5.3, §7, Appendix L run inventory.
- **Проблема:** читателю сложно быстро восстановить batch sizes, sample counts, logger cadence, optimizer schedule, exact generalisation criterion, number of unsuccessful/censored runs и правила исключения windows.
- **Исправление:** добавить компактную reproducibility table: architecture, data split, optimizer/LR/WD/batch, steps, logging, seeds, observer, window/stride, inclusion/exclusion. Explicitly define `t_gen`, `memorises`, censoring and validation threshold.

### R3. Исключения и агрегации могут создавать selection bias
- **Где:** §5.3 (instantaneous loss excluded); Algorithm 1; §7/Table 11.
- **Проблема:** исключение loss обоснованно control-ом, но другие наблюдатели/degenerate windows отбрасываются при summary; не ясно количество исключений по каждому result и были ли правила до просмотра outcome.
- **Исправление:** для каждого table/figure дать flow: total windows/runs → degenerate → excluded by protocol → analysed; публиковать результаты и с excluded observer, пометив их invalid, а не просто убирать из aggregate.

### R4. Формальное расхождение между equation (4) и реализованным estimator-ом
- **Где:** eq. (4), стр. 3; Algorithm 1, Appendix A, стр. 12.
- **Проблема:** eq. (4) задаёт `N(m−1)/S`, в code/algorithm возвращается `(N(m−1)−1)/S`. Appendix объясняет, что разница мала, но основной mathematical definition и implementируемый endpoint различны.
- **Исправление:** записать unbiased pooled formula уже в §3.2 и отдельно дать asymptotic/MG variant; использовать одно имя для одного estimator-а во всех tables.

### R5. Параметры numerical floors/dither нуждаются в sensitivity analysis
- **Где:** Appendix A, стр. 12; §3.4.
- **Проблема:** `sigma=1e−9`, distance/ratio floors и 1% flag могут существенно влиять именно на nearly constant/transient logs; фиксированы, но не проверены.
- **Исправление:** sweep минимум на порядок в обе стороны; показать, что main recovery и regime labels устойчивы. Документировать random seed dither-а или использовать deterministic tie-breaking.

---

## Подача, структура и визуальное оформление

### P1. Framing и title обещают больше, чем даёт текст
- **Где:** title; abstract, стр. 1; §8, стр. 9.
- **Проблема:** заголовок «Counting the Active Degrees of Freedom of a Training Run» звучит как универсальный способ измерения. Почти половина работы убедительно доказывает, что на естественных logs и обоих grokking setups это не так.
- **Исправление:** либо добавить admissible real training success case, либо переименовать, например: *“When can a scalar training log reveal excited modes? A controlled delay-embedding study and cautions for grokking”*. В abstract первым результатом сделать scope limitation, а не общий recovery claim.

### P2. Главный результат об admissibility спрятан в appendix
- **Где:** Fig. 2 / regime map, стр. 18 приложения; §7.1, стр. 8.
- **Проблема:** именно Fig. 2 наглядно подтверждает, что grokking logs лежат вне calibration region, но читатель main paper его не видит. Зато Fig. 1 занимает заметное место и требует много расшифровки.
- **Исправление:** перенести regime map в main text; сократить related work/часть описаний synthetic ladder или вынести менее существенный table в appendix.

### P3. Figure 1 перегружен и не показывает неопределённость
- **Где:** стр. 7.
- **��роблема:** маленькие подписи, много типов линий/маркеров, legend и annotation; правый panel неочевиден без долгого чтения; нет seed-level spread/error bars. «Admissible» shaded area визуально похожа на установленную decision boundary, хотя это только post-hoc calibration description.
- **Исправление:** разделить на две фигуры либо упростить до основных режимов; добавить CI/individual seed points; явно подписать «calibration range, not validated threshold» внутри panel; увеличь шрифт.

### P4. В PDF видны красные рамки вокруг cross-references
- **Где:** например Fig. 1 caption, стр. 7; references throughout PDF.
- **Проблема:** красные прямоугольники вокруг «5.3», «3.4», «2», «7», «A», «D» выглядят как следы draft/debug и ухудшают читаемость.
- **Исправление:** в `hyperref` включить `hidelinks` или `colorlinks=true` с ненавязчивым единым цветом без рамок; пересоб��ать и визуально проверить весь PDF.

### P5. Термины «dimension», «count», «effective rank», «mode count» всё ещё смешиваются
- **Где:** abstract; §3.1; §5; §6.4; §8.
- **Проблема:** текст признаёт различие, но затем возвращается к «active dimension ... count», «recover number of directions», «one-dimensional bottleneck» для PR. Это оставляет концептуальную неоднозначность у читателя.
- **Исправление:** ввести boxed terminology table в §3: quantity / symbol / geometry / metric dependence / estimator / when comparable. Использовать строго: covariance PR = effective rank; delay ID = resolvable mode/manifold count; `d^det` = detrended covariance PR.

### P6. Слишком много самозащитного текста, а важные проверки трудно найти
- **Где:** §§5–6, Appendices B–L.
- **Проблема:** честные caveats повторяются в introduction, validation, validity, conclusion; при этом reader трудно составить единый список: какие results are confirmatory, exploratory, invalid, or negative.
- **Исправление:** добавить одну summary table в main text: `claim | target | system | split | status | limitation`. Удалить повторяющиеся фразы, освободив место для uncertainty и Fig. 2.

### P7. Ключевые negative results должны быть оформлены как результаты, не как оправдания
- **Где:** §5.2, §6.1–6.2, §7.1–7.4.
- **Проблема:** сильная сторона статьи — обнаружение failure modes, но prose иногда выглядит как заранее ожидаемое объяснение, почему основной инструмент не применим.
- **Исправление:** заранее сформулировать falsifiable hypotheses и success/failure criteria; показать единый decision protocol, затем применить его на grokking logs без ручной интерпретации. Это сделает отрицательный вклад более убедительным.

### P8. Ограничение в девять страниц не должно диктовать научную полноту
- **Где:** README “Page budget”; основная часть, стр. 1–9.
- **Проблема:** критические regime map, per-observer behaviour и protocol details вынесены из основной аргументации ради лимита, хотя без них claims нельзя быстро проверить.
- **Исправление:** радикально сократить related-work prose и повторяющиеся disclaimer-ы; перенести в main как минимум Fig. 2, split/status table и uncertainty. Appendix не заменяет ясного core evidence.

---

## Рекомендуемый порядок правок

1. **Исправить claims:** разделить PR/effective rank и mode/manifold count; сузить title, abstract, contributions, §8.
2. **Исправить доказательность:** независимая validation diagnostics; чистые splits; sensitivity к estimator parameters; CI на уровне seeds/runs.
3. **Укрепить direct grokking result:** matched controls, pre-specified test, robustness к окнами/detrending/sketch, или маркировать его exploratory.
4. **Убрать формальные дефекты:** zero-LR `PR=1`, eq. (4) vs Algorithm 1, Theiler cap для всех заявленных results.
5. **Сделать reproducible:** anonymized artifact и единая таблица run/configuration/exclusions.
6. **Перестроить main text:** вынести Fig. 2 и status table, упростить Fig. 1, исправить hyperlink boxes.

## Что уже сделано хорошо и стоит сохранить

- Авторы не скрывают неудачные режимы и zero-LR failure; это повышает доверие.
- §6.2 честно признаёт mismatch delay lag на grokking logs вместо подгонки объяснения.
- Разделение available / functional / trajectory effective dimension полезно и потенциально является самостоятельным вкладом.
- Appendix G корректно не заявляет спектральную гарантию CountSketch; эту аккуратность стоит сохранить, но усилить validation.
- Формулировка о том, что scalar fall не доказывает generalisation (§7.4), правильна; её следует сделать центральным, а не побочным выводом.
