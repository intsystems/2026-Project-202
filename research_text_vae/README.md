# Текстовый VAE: независимая проверка сигнала MG

Новая NLP-постановка, отдельно от CV и изменения learning rate.
Основной документ: `report_ru.pdf`; исходный текст: `report_ru.md`.
Все запуски используют реальные предложения Penn Treebank, два GRU и обучаемый
Gaussian latent. При усилении KL-регуляризации encoder передаёт decoder намного
меньше информации о предложении. MG получает только reconstruction loss на
фиксированной выборке и не получает независимые диагностические показатели.

`PROTOCOL.md` содержит заранее выбранные настройки и записи о первоначальном
подтверждении (seeds1–2) и расширении на seeds3–9. Всего десять пар: pilot0 и
confirmation1–9. Основные параметры не подбирались по новым seed.
Инициализации/батчи разные, но во всех десяти запусках одинаковые данные и
validation probe. Пять/девять перекрывающихся окон не являются независимыми
повторами. Не заявляется точная размерность всей системы или доказанное
преимущество перед обычной VAE-диагностикой.

Добавлен контроль `protection/`: третья ветвь с сохранением значительной части
информации при том же внешнем коэффициенте KL. Он продолжает те же десять
инициализаций. Отдельный отчёт и точный порядок выбора защиты находятся в этой
папке; основной PDF включает дополнительный раздел. Неудачная попытка с пятью
дополнительными шагами encoder также сохранена. Это проверка различения режимов,
а не доказательство универсальной специфичности MG. Предыдущий отчёт по десяти
парам сохранён в `before_protection/`.

Следующая проверка практической полезности находится в `cyclical/`: новое обучение
с циклической KL-регуляризацией и сравнение способов выбирать моменты дорогой
диагностики. Отдельный отчёт `cyclical/report_ru.pdf` учитывает пропуски событий,
задержки, простые альтернативы и стоимость получения входного ряда. Эта серия
не заменяет результаты парного эксперимента и контроля с сохранением кода.

## Файлы

- `all_seeds.csv`, `summary.json`: все результаты с метками pilot,
  initial_confirmation и expansion; основная статистика исключает pilot0.
- `pilot_seed0`, `confirmation_seed1` ... `confirmation_seed9`: десять парных запусков.
- `all_seeds_sensitivity.csv`: варианты окна/delay отдельно для каждого seed.
- `expanded_figure.pdf`: медиана и межквартильный диапазон по confirmation1–9;
  `seed_effects.pdf`: парный эффект всех десяти инициализаций.
- `expansion_status.json`: состояние семи добавленных запусков.
- `initial_three/`: исходные PDF/Markdown/числа отчёта по трём запускам.
- `base/logs.csv`, `regularized/logs.csv`: все3072 наблюдения каждой ветви;
  `step=t` — до обновленияt (нумерация с0).
- `reference.csv`: независимые измерения после `step` обновлений, включая3072.
- `windows.csv`: MG, E40/E20, численные флаги, std, FFT entropy; `end` — число
  наблюдений, окно использует `[end-window:end]`. Основной вариантW512,tau1.
- `minibatch_windows.csv`: исследовательская проверка обычного trainNLL,
  не замена основного наблюдателя.
- `observer_controls.csv`: масштаб x10 и три IAAFT-суррогата.
- `audit.json`: проверки на известном нулевом posterior и совпадение ветвей.
- `branch.pt`: одинаковые веса/Adam/RNG перед вмешательством; `final.pt`:
  финальные веса каждой ветви.
- `data/selection.json`: словарь, точные номера строк и SHA256 исходных файлов.
- `confirmation_figure.pdf`: прежний график первого подтверждающего seed1;
  отдельные `overview.pdf` есть для каждой из десяти пар.

Смысл независимых показателей:
`MI` — MC information для равномерной смеси256 posterior, максимумlog256;
`KL` — средний analytic KL posterior к стандартному Gaussian prior;
`shuffle_symkl` — средний по непустым токенам симметризованный KL предсказаний
после циклической перестановки encoder distributions между предложениями;
`shuffle_nll_gap` — NLL с чужим кодом минус NLL с собственным кодом;
`active_units` — число Var(mu_j)>.01, только вспомогательная характеристика.
Для shuffled-кода переставляются mu и variance, но epsilon остаётся согласованным
с исходным примером. При одинаковых posterior это даёт строго нулевой эффект.
Симметризованный KL — половина суммы двух KL, не Jensen–Shannon divergence.
При четырёх MC draws MI имеет шум и может немного превышать analytic KL;
это не строгая оценка сверху/снизу популяционной информации.

`probe_nll` — reconstruction NLL/token на16 фиксированныхvalidation предложениях
и одном фиксированном epsilon на каждое. Это не generative perplexity на новых
текстах без encoder. `train_nll` — реконструкция меняющегосябатча без KL и beta.
`train_KL` и `beta` сохраняются, но никогда не подаются в основной MG.

## Воспроизведение

Python3.13 CPU; точные версии — `environment.json`. Установить:

```powershell
python -m pip install -r research_text_vae/requirements.txt
```

Выполнять из корня проекта, родителя `research_text_vae`. Все файлы и относительные
пути также сохранены в архиве. GPU не требуется. Каждая пара занимает несколько
минут на использованном CPU; точное время зависит от нагрузки. Часть запусков
выполнялась параллельно, поэтому времена не используются для сравнительного
benchmark методов. В расширенной серии используется максимум три работника.

Полный повтор (перезапишет папки соответствующих запусков):

```powershell
python research_text_vae/run.py --seed 0 --out research_text_vae/pilot_seed0
python research_text_vae/run.py --seed 1 --out research_text_vae/confirmation_seed1
python research_text_vae/run.py --seed 2 --out research_text_vae/confirmation_seed2
python research_text_vae/expand.py --seeds 3 4 5 6 7 8 9 --workers 3
```

Для КАЖДОЙ из первоначальных трёх папок выполнить следующие команды, заменив
путь `pilot_seed0`. `expand.py` сам выполняет эти стадии для seeds3–9 и сохраняет
их логи; полностью готовые seed повторно не обучаются.

```powershell
python research_text_vae/analyze.py --root research_text_vae/pilot_seed0
python research_text_vae/extras.py --root research_text_vae/pilot_seed0
python research_text_vae/audit.py --root research_text_vae/pilot_seed0
```

Сводки, PDF и архив:

```powershell
python research_text_vae/make_report.py
python research_text_vae/build_report.py
python research_text_vae/package_results.py
```

`make_report.py` вызывает `report_expanded.py` и требует все десять завершённых
пар, чтобы случайно не собрать выборочный отчёт. Основной bootstrap-интервал
медианы q рассчитан по девяти подтверждающим парным seed (20000 ресэмплирований,
seed20260929); интервал не распространяется на смену корпуса/модели.
После расчёта можно запустить `python research_text_vae/verify_series.py` для
проверки количества строк, совпадения префиксов и повторного расчёта q из CSV.

XeLaTeX и Times New Roman требуются только для PDF. В отличие от основного
manuscript это самостоятельный технический отчёт, без blue/claude markup.
Для пересчёта MG/таблиц веса и загрузка корпуса не нужны: достаточно приложенных
CSV. Для нового обучения и reference audit `run.py` скачивает Penn Treebank
из `tomsercu/lstm` и сверяет/записывает SHA256; текстовый корпус в архив не включён.

## Литература и область вывода

Bowman et al., *Generating Sentences from a Continuous Space*, CoNLL2016,
K16-1002, раздел3.1 описывает случай, когда decoder игнорирует latent и KL падает.
He et al., *Lagging Inference Networks and Posterior Collapse in Variational
Autoencoders*, ICLR2019, arXiv1901.05534: информация latent как диагностика.
Нашу архитектуру и вмешательство не следует считать точным воспроизведением этих
статей. Здесь проверяется сопряжённое изменение скалярного сигнала и независимо
измеренной роли латентного кода. Условие рекуррентности не установлено, точная
active dimension не измеряется. Улучшение существующих способов детектирования
collapse, генеративного качества, или перенос на LLM ещё не проверены.
