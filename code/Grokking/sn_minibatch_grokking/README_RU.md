# Minibatch-grokking на `S_5`, `S_6`, `S_7` в Kaggle

Эксперимент использует архитектуру из Stander et al., ICML 2024:

```text
opaque left ID  -> left embedding  --\
                                      concat -> Linear -> ReLU -> Linear -> product ID
opaque right ID -> right embedding --/
```

- `S_5`: embedding 256, hidden 128 — как в статье;
- `S_6`: embedding 512, hidden 256 — как в статье;
- `S_7`: embedding 512, hidden 512 — экстраполяция, в статье не проверялась;
- 40% пар относятся к train по детерминированному hash split;
- `AdamW(lr=1e-3, betas=(0.9,0.98), weight_decay=1.0)` в minibatch-версии;
- в отличие от статьи, здесь выполняются настоящие minibatch-обновления.

> Почему v2: буквальный `torch.optim.Adam(weight_decay=1)` из full-batch протокола нельзя без изменений переносить в minibatch. Coupled L2 после Adam-preconditioning схлопывал параметры почти к нулю. Поэтому для minibatch используется decoupled `AdamW`. Это осознанная адаптация, а не точная реплика статьи.

Статья: <https://proceedings.mlr.press/v235/stander24a.html>

## Рекомендуемый воспроизводимый режим

После двух неудачных minibatch-переносов основной запуск перенесён в отдельный точный full-batch протокол:

- `train_sn_fullbatch.py`;
- `kaggle_sn_fullbatch_grokking.ipynb`;
- bias-free MLP из открытой реализации;
- `torch.randperm` split 40/60;
- full-batch `AdamW`, FP64 cross-entropy и AMP выключен;
- канонические логи `train_loss`, `full_train_loss`, `val_loss`, `train_acc`, `full_train_acc`, `val_acc`, `weight_norm`, `grad_norm`.

Для итоговых экспериментов используйте именно этот режим. Minibatch-файл сохранён только как неудачная экспериментальная ветка.

## Быстрый запуск в Kaggle

1. Создать новый Kaggle notebook и включить GPU.
2. Upload `train_sn_minibatch.py` как notebook file или вставить его содержимое.
3. Запустить:

```python
import sys
sys.path.insert(0, "/kaggle/working")

from train_sn_minibatch import Config, run

config = Config(
    output_root="/kaggle/working/sn_minibatch_grokking",
    protocol_name="stander_mlp_minibatch_v2_adamw",
    n_values=(5, 6, 7),
    seeds=(42,),
)
run(config)
```

Рекомендуется сначала запускать отдельно:

```python
config.n_values = (5,)
run(config)
```

затем `S_6`, и лишь потом `S_7`. `S_7` существенно дороже из-за 5040 выходных
классов и не имеет опубликованной гарантии grokking.

## Настройки minibatch

По умолчанию:

```python
batch_size_by_n = {5: 4096, 6: 4096, 7: 2048}
```

Если не хватает VRAM, уменьшить только соответствующий batch:

```python
config.batch_size_by_n[7] = 1024
```

Уменьшение batch меняет динамику и может уничтожить memorization gap. Поэтому
размер batch записывается в `metadata.json` и считается частью протокола.

## Какие файлы появляются

```text
sn_minibatch_grokking/
  stander_mlp_minibatch_v2_adamw/
    S_5/seed_42/
      metadata.json
      training_log.csv
      checkpoint.pt
      COMPLETED.json
```

В `training_log.csv` пишутся:

- train/validation loss и accuracy;
- три фиксированные случайные проекции поэлементных train/validation losses:
  `monitor_train_lossproj_r*` и `val_lossproj_r*`;
- `weight_norm`, `grad_norm`;
- число просмотренных примеров и эквивалент эпохи;
- три фиксированные случайные проекции весов и градиентов **каждого слоя** на
  diagnostic-шагах; между ними — проекция left embedding.

Последние ряды предназначены для последующего layerwise EDM-анализа как более
generic 1-D observers, чем weight norm.

Loss projection определяется на неизменном monitor-set как

```text
lossproj_r(t) = sum_i sign_r(i) * CE_i(t) / sqrt(N),  sign_r(i) in {-1,+1}.
```

Знаки детерминированы ID пары и остаются одинаковыми на всех шагах. В отличие
от среднего `val_loss`, такая проекция не уничтожает разнонаправленные изменения
loss отдельных примеров. Для EDM каждый `val_lossproj_r*` анализируется как
отдельный одномерный временной ряд.

## Продолжение в следующей Kaggle-сессии

1. Нажать **Save Version -> Save & Run All**.
2. В новом notebook выбрать **Add Input -> Your Work** и добавить output
   предыдущей версии.
3. Оставить в конфигурации:

```python
resume_search_roots=("/kaggle/input",)
```

Скрипт найдёт совпадающий `checkpoint.pt`, скопирует его в writable
`/kaggle/working` и продолжит обучение.

## Важное ограничение

Опубликованный результат `S_5/S_6` получен в full-batch режиме. Здесь сохраняются
задача, MLP, размеры модели, fraction и optimizer family, но batch-режим намеренно
изменён. Поэтому сначала нужен pilot на `S_5`, а успех `S_6/S_7` является
экспериментальным результатом, а не гарантией статьи.
