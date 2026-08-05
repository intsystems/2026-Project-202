# Omnigrok: sweep по простым модулям

`prime_sweep_omnigrok.py` обучает однослойный Omnigrok Transformer решать
`y = (a + b) mod p` для 15 простых модулей:

`101, 113, 211, 307, 431, 607, 857, 1201, 1693, 2371, 3323, 4651, 6521, 9151, 9973`.

Во всех экспериментах, включая малые модули, используется mini-batch.

## Воспроизводимый базовый режим

Настройки повторяют рабочий эксперимент проекта
`generator_logs_to_flat_grokking_with_stochastic.ipynb`:

- `p=113` в первом пилотном запуске;
- `batch_size=256`;
- `float64` с инициализацией сразу в выбранном dtype;
- исходная Omnigrok-архитектура без LayerNorm и dropout;
- AdamW: `lr=1e-3`, `weight_decay=1`, `betas=(0.9, 0.98)`;
- ??? `p=113` ??????????? 30% train; ??? ??????? `p` ????? train-????????
  ??????????? ?? ???????? 34 ????????? ?? ?????? ????? ??????;
- `DataLoader(..., shuffle=True, drop_last=False)`;
- точка логирования каждые 10 optimizer steps.

## Запуск в Google Colab

Откройте `colab_prime_sweep_omnigrok.ipynb` и выполните ячейки сверху вниз.
Сначала рекомендуется пилот:

```python
PRIMES = (113, 211, 307)

cfg = Config(
    primes=PRIMES,
    protocol_name="omnigrok_prime_sweep_v4_class_exposure_normalized",
    batch_size=256,
    model_dtype="float64",
    normalize_train_examples_per_class=True,
    train_examples_per_class=34.0,
    log_every=10,
    text_log_every=1_000,
    required_gap_steps=1_000,
    post_grok_steps=5_000,
    force_restart=True,
)
```

После проверки замените `PRIMES` на `DEFAULT_PRIMES`.

## Данные

При `p² <= 500000` используется полная таблица пар и тот же `torch.randperm`,
что в исходном notebook. Для больших модулей строится фиксированное множество
не более чем из 500 000 непересекающихся пар, которое делится на train/validation
в отношении 30/70. Фактические размеры и покрытие записываются в метаданные.

## Автоматическая остановка

Мемоизация и генерализация должны сохраняться `patience_logs` точек подряд.
Запуск считается настоящим гроккингом, только если между оценёнными началами
мемоизации и генерализации не меньше `required_gap_steps`.

После **обнаружения** такого события обучение выполняет ещё `post_grok_steps`
новых optimizer steps и останавливается. Поэтому хвост не теряется из-за того,
что начало генерализации оценивается задним числом по окну стабильности.

## Логи и TensorBoard

Для каждого `(p, seed)` сохраняются:

- `training_log.csv` — полный численный лог;
- `training.log` — построчный UTF-8 лог, обновляемый в реальном времени;
- `config.json` и `COMPLETED.json`;
- `checkpoint.pt` и финальный `checkpoint_final.pt`;
- TensorBoard events в подпапке `tensorboard`.

В лог входят train/validation loss и accuracy, нормы и participation ratio,
фиксированные случайные проекции лоссов, логитов, активаций, весов, градиентов,
обновлений и смещений параметров, а также entropy, confidence, margin,
hidden norm и Fourier-метрики embedding/unembedding.

TensorBoard нужно запустить до обучения:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/grokking_prime_sweep --reload_interval 5
```

События `stable_memorization`, `stable_generalization`, `genuine_grokking`,
планируемая остановка и чекпоинты выводятся под ячейкой и в `training.log`.
