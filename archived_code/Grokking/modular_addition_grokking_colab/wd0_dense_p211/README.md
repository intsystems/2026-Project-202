# Dense WD=0, p=211

Файлы:
- `colab_p211_wd0_dense.ipynb` — запуск в Google Colab.
- `prime_sweep_omnigrok_dense.py` — версия генератора с потоковой записью CSV.

Особенности:
- `log_every=1`: EDM, loss, gradient/update/displacement и random-projection метрики пишутся на каждом optimizer step;
- CSV дописывается пакетами по 100 строк, а не пересоздаётся целиком — это важно для 500k–1M шагов;
- TensorBoard остаётся включённым;
- текстовый `training.log` выводит сводную строку раз в 1000 шагов;
- `weight_decay=0`, `max_steps=500_000`.

Для 1M шагов замените `max_steps=500_000` на `max_steps=1_000_000`.

Проверяйте сначала 1–2 тысячи шагов: при `log_every=1` каждая итерация включает мониторинговую оценку, поэтому запуск заметно медленнее обычного режима.
