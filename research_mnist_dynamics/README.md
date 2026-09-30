# MNIST: изменение динамики обучения и скалярный MG

Главный документ: `report_ru.pdf` (4 страницы); редактируемый текст — `report_ru.md`.
Эксперимент завершён на CPU. Это частично положительный, но недостаточно устойчивый
пример: основной MG даёт нужное отличие от контроля в 4/5 пар, сильный независимый
сигнал относится только к ковариации колебаний весов после удаления тренда.
Вариант W1024 даёт 5/5 в дополнительной проверке. Преимущество над дешёвыми
альтернативами не установлено. Основная статья не изменялась.

## Как читать результаты

- `PROTOCOL.md`: порядок выбора пилота, наблюдателя и подтверждающей серии;
- `confirmation/paired_summary.csv`: все пять пар, без исключений;
- `confirmation/summary.json`: числа отчёта и расчёт накладных расходов;
- `confirmation/probe_windows.csv`: основной `signal=test_probe, preprocess=raw`;
- `confirmation/probe_sensitivity.csv`: дополнительные окна и delay;
- `confirmation/benchmark.csv`: семь повторов прогретого сравнения;
- `confirmation/audit.json`: совпадение префиксов, формулы PR, масштаб, потоки;
- `confirmation/unseen_accuracy.csv`: 9900 изображений вне probe (название историческое:
  полная тестовая accuracy раньше логировалась, это не слепой итоговый тест);
- `seed_N/base` и `seed_N/drop`: логи обучения, loss фиксированных выборок, PR,
  финальные веса, спектры, параметры запуска;
- `pilot/`: все три исходных режима, в том числе неудачные.

В `reference.csv` поле `pr` — ковариационный PR после удаления линейного тренда;
`raw_pr` — только центрирование; `update_pr` — PR разностей весов с удалением тренда.
`small_pr` использует фиксированные 128 координат (seed718).
`probe.csv:test_probe` — в действительности validation loss на фиксированных100
изображениях, НЕ minibatch loss. `probe.csv:seconds` включает ОБА probe на200
изображениях; стоимость одного100-image probe отдельно измерена в `benchmark.csv`.
`checks_seconds` — общее время E20+E40, не время добавочного вызова E40.
`trajectory.npy[t]` — веса перед обновлением с индексом t (нумерация с0),
`logs.csv:step=t+1` и `probe.csv:step=t+1`. Accuracy записывается после `step`
обновлений. Финальные веса после4096 обновлений находятся в `final.pt`.

## Воспроизведение

Команды выполняются из корня проекта (родителя `research_mnist_dynamics`).
В архиве сохранена такая же структура и необходимый код `code/actdim`.
Python3.13; точные версии библиотек записаны в `environment.json`.
Для установки: `python -m pip install -r research_mnist_dynamics/requirements.txt`.
GPU не требуется. Для полных траекторий всех запусков нужно около4GB диска,
а для ковариационных окон — несколько сотен MB RAM. MNIST скачивается torchvision
в `research_gan_collapse/data`; это лишь общая папка данных, GAN-код не нужен.

Быстро перестроить сводки и графики из CSV, входящих в архив:

```powershell
python research_mnist_dynamics/summarize.py
python research_mnist_dynamics/build_report.py
```

Вторая команда требует XeLaTeX и Times New Roman. Числа в тексте отчёта относятся
к приложенному запуску, а не автоматически переписываются при новых измерениях.

Повторить весь эксперимент (перезапишет результаты в папках pilot/confirmation):

```powershell
python research_mnist_dynamics/pilot.py
python research_mnist_dynamics/analyze.py --root research_mnist_dynamics/pilot
python research_mnist_dynamics/probe.py --root research_mnist_dynamics/pilot/sgd
python research_mnist_dynamics/confirm.py
python research_mnist_dynamics/probe_controls.py
python research_mnist_dynamics/audit_benchmark.py
python research_mnist_dynamics/summarize.py
```

`confirm.py` включает обучение, независимый анализ, fixed-probe и исходные scalar
channels. Наблюдатель и настройки для seeds1–5 указаны в протоколе. При повторном
расчёте probe скаляры заново вычисляются по весам, старый кэш не используется.
Пилотная коррекция была исследовательской; новые seeds не подбирались по исходу.
Случайные индексы train/probe/weights сохранены в `selection_ids.npz`.

Собрать компактный архив:

```powershell
python research_mnist_dynamics/package_results.py
```

В него входят все CSV/JSON, небольшие веса и спектры, код, PDF и исходник отчёта.
Исключены большие `trajectory.npy`, кэш MNIST, промежуточные LaTeX-файлы и превью.
Полные траектории остаются локально; их можно заново получить командами выше.
Без них можно построить готовые сводки/графики, но нельзя пересчитать независимый PR
из весов или повторить аппаратный benchmark. SHA256 содержимого есть в
`MANIFEST.sha256`; это контроль целостности, не независимое воспроизведение.
