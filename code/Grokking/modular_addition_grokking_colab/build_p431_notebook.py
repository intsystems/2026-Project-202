"""Build a clean Yandex DataSphere/JupyterLab notebook for p=431.

The generated notebook intentionally does not install packages, mount drives,
scan the whole filesystem, or start background services.  Upload this notebook
and ``prime_sweep_omnigrok.py`` to the same DataSphere dataset/work directory.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "colab_p431_omnigrok.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    markdown(
        """# Omnigrok modular addition: p = 431

Версия для Yandex DataSphere / обычного JupyterLab.

Перед запуском положите `prime_sweep_omnigrok.py` в ту же рабочую папку или
укажите её путь в `PROJECT_DIR`/переменной окружения `OMNIGROK_PROJECT_DIR`.
Никаких `google.colab`, Google Drive, Kaggle/Terra API, установки PyTorch или
фоновых процессов notebook не использует.

Результаты пишутся в отдельный каталог `outputs_p431` (или в путь из
`OMNIGROK_OUTPUT_ROOT`). В каталоге запуска будут CSV для EDM, текстовый лог,
TensorBoard events, конфигурация, summary и checkpoints.

Порядок: выполните проверку окружения, затем конфигурацию, затем запуск.
"""
    ),
    markdown(
        """## 1. Пути и проверка окружения

По умолчанию используется текущая папка JupyterLab. Поиск ограничен несколькими
явно заданными кандидатами и не делает рекурсивный обход большого датасета.
Если файл лежит в другом месте, задайте `PROJECT_DIR` вручную.
"""
    ),
    code(
        """from pathlib import Path
import os
import shutil
import sys
import torch

# Если notebook и trainer лежат рядом, оставьте None.
# Пример: PROJECT_DIR = Path('/home/jupyter/work/resources/grokking')
PROJECT_DIR = None
if PROJECT_DIR is None:
    env_project = os.environ.get('OMNIGROK_PROJECT_DIR')
    PROJECT_DIR = Path(env_project).expanduser() if env_project else Path.cwd()
PROJECT_DIR = Path(PROJECT_DIR).resolve()
MODULE_PATH = PROJECT_DIR / 'prime_sweep_omnigrok.py'

if not MODULE_PATH.is_file():
    candidates = [Path.cwd(), Path('/home/jupyter/work/resources'), Path('/home/jupyter/work')]
    found = next((p / 'prime_sweep_omnigrok.py' for p in candidates if (p / 'prime_sweep_omnigrok.py').is_file()), None)
    if found is not None:
        MODULE_PATH = found.resolve()
        PROJECT_DIR = MODULE_PATH.parent
    else:
        raise FileNotFoundError(
            f'Не найден {MODULE_PATH}. Загрузите prime_sweep_omnigrok.py рядом с notebook '
            'или задайте PROJECT_DIR/OMNIGROK_PROJECT_DIR.'
        )

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Датасет может быть read-only; результаты по умолчанию сохраняем в рабочей
# директории JupyterLab. При необходимости задайте OMNIGROK_OUTPUT_ROOT явно.
OUTPUT_ROOT = Path(os.environ.get('OMNIGROK_OUTPUT_ROOT', str(Path.cwd() / 'outputs_p431'))).expanduser()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print('Project directory:', PROJECT_DIR)
print('Trainer:', MODULE_PATH)
print('Output directory:', OUTPUT_ROOT)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('TensorBoard executable:', shutil.which('tensorboard') or 'not found (CSV logging still works)')
if not torch.cuda.is_available():
    raise RuntimeError('GPU не обнаружен. Выберите GPU-конфигурацию DataSphere и перезапустите kernel.')
print('GPU:', torch.cuda.get_device_name(0))
"""
    ),
    markdown(
        """## 2. Конфигурация

Mini-batch Omnigrok для `p=431`: `float32`, batch 512 и `weight_decay=0.5`.
Чтобы начать новый эксперимент, измените `protocol_name`; завершённые запуски
с тем же именем будут пропущены.
"""
    ),
    code(
        """from prime_sweep_omnigrok import Config, run_sweep

CONFIG = Config(
    output_root=str(OUTPUT_ROOT),
    protocol_name='yandex_p431_omnigrok_wd05_v1',
    primes=(431,),
    seeds=(42,),

    train_fraction=0.30,
    normalize_train_examples_per_class=True,
    train_examples_per_class=34.0,
    max_sampled_pairs=500_000,

    d_model=128,
    d_mlp=512,
    num_heads=4,
    d_head=32,
    batch_size=512,
    batch_size_by_p={431: 512},
    model_dtype='float32',

    learning_rate=1e-3,
    betas=(0.9, 0.98),
    delayed_weight_decay=False,
    weight_decay=0.5,
    weight_decay_by_p={431: 0.5},

    max_steps=400_000,
    log_every=50,
    diagnostic_every=500,
    checkpoint_every=5_000,
    tensorboard_enabled=True,
    tensorboard_flush_secs=5,
    tensorboard_histogram_every=5_000,
    text_log_enabled=True,
    text_log_filename='training.log',
    text_log_every=1_000,

    monitor_train_pairs=2_048,
    monitor_val_pairs=2_048,
    eval_batch_size=2_048,
    projection_count=3,
    target_train_acc=0.99,
    target_val_acc=0.95,
    patience_logs=5,
    required_gap_steps=10_000,
    post_grok_steps=5_000,

    force_restart=False,
    skip_completed=True,
    device='auto',
    fused_adamw=True,
)
print(CONFIG)
"""
    ),
    markdown(
        """## 3. TensorBoard (необязательно)

Логирование в CSV и `training.log` работает независимо от TensorBoard. После
конфигурации можно открыть отдельный Terminal в JupyterLab и запустить:

```bash
tensorboard --logdir <путь_к_outputs_p431/протоколу> --host 0.0.0.0 --port 6006 --reload_interval 5
```

Сервис запускается вручную, поэтому он не блокирует kernel и не создаёт
неожиданных фоновых процессов.
"""
    ),
    code(
        """TB_DIR = OUTPUT_ROOT / CONFIG.protocol_name
print('Команда для Terminal JupyterLab:')
print(f'tensorboard --logdir "{TB_DIR}" --host 0.0.0.0 --port 6006 --reload_interval 5')
"""
    ),
    markdown(
        """## 4. Запуск обучения

Trainer остановится после обнаруженного grokking через `post_grok_steps`.
Если требуемый gap не найден, обучение дойдёт до `max_steps`; это будет указано
в `COMPLETED.json`.
"""
    ),
    code("summaries = run_sweep(CONFIG)\nsummaries\n"),
    markdown("""## 5. Проверка файлов и последних строк лога"""),
    code(
        """import json
import pandas as pd

run_root = OUTPUT_ROOT / CONFIG.protocol_name / 'p_431' / 'seed_42'
print('Run directory:', run_root)
for name in ('training_log.csv', 'training.log', 'COMPLETED.json', 'config.json',
             'checkpoint.pt', 'checkpoint_final.pt', 'tensorboard'):
    path = run_root / name
    print(f'{name:20s} exists={path.exists()}  path={path}')

csv_path = run_root / 'training_log.csv'
if csv_path.exists():
    display(pd.read_csv(csv_path).tail())
summary_path = run_root / 'COMPLETED.json'
if summary_path.exists():
    print(json.dumps(json.loads(summary_path.read_text(encoding='utf-8')), indent=2, ensure_ascii=False))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(OUT)
