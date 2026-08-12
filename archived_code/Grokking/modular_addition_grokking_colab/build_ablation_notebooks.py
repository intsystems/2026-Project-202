"""Create Colab notebooks for no-weight-decay ablations at p=211 and p=431."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}


def make(p: int, max_steps: int = 300_000) -> None:
    protocol = f"omnigrok_p{p}_v1_wd0_ablation"
    cells = [
        md(f"""# Ablation: Omnigrok modular addition p={p}, weight decay = 0\n\nКонтрольный запуск для проверки роли AdamW weight decay. Архитектура, split, mini-batch, learning rate и логирование совпадают с соответствующим WD>0 протоколом; изменён только `weight_decay=0.0`.\n\nЭто отдельный protocol name, поэтому TensorBoard и checkpoints не смешиваются с другими экспериментами."""),
        md("""## Что проверяет абляция\n\nСравниваются два режима:\n\n- baseline с фиксированным ненулевым WD;\n- этот запуск с `WD=0`.\n\nИнтерпретировать результат нужно по train/validation accuracy, loss, gap, weight norm, gradient/update norms, participation ratios, random projections и Fourier diagnostics. Если при WD=0 validation не генерализуется, это отрицательный контроль, а не ошибка эксперимента."""),
        code("""from google.colab import drive\ndrive.mount('/content/drive')\n%cd /content/drive/MyDrive/grokking_prediction_original/2026-Project-202/code/Grokking/modular_addition_grokking_colab"""),
        code("""!pip install -q tensorboard"""),
        code(f"""from prime_sweep_omnigrok import Config, run_sweep\n\nCONFIG = Config(\n    output_root='/content/drive/MyDrive/grokking_prime_sweep',\n    protocol_name='{protocol}',\n    primes=({p},),\n    seeds=(42,),\n\n    train_fraction=0.30,\n    normalize_train_examples_per_class=True,\n    train_examples_per_class=34.0,\n    max_sampled_pairs=500_000,\n\n    d_model=128,\n    d_mlp=512,\n    num_heads=4,\n    d_head=32,\n    batch_size=512,\n    batch_size_by_p={{{p}: 512}},\n    model_dtype='float32',\n\n    learning_rate=1e-3,\n    betas=(0.9, 0.98),\n    delayed_weight_decay=False,\n    weight_decay=0.0,\n    weight_decay_by_p={{{p}: 0.0}},\n\n    max_steps={max_steps:,},\n    log_every=50,\n    diagnostic_every=500,\n    checkpoint_every=5_000,\n    tensorboard_enabled=True,\n    tensorboard_flush_secs=5,\n    tensorboard_histogram_every=5_000,\n    text_log_enabled=True,\n    text_log_filename='training.log',\n    text_log_every=1_000,\n\n    monitor_train_pairs=2_048,\n    monitor_val_pairs=2_048,\n    eval_batch_size=2_048,\n    projection_count=3,\n    target_train_acc=0.99,\n    target_val_acc=0.95,\n    patience_logs=5,\n    required_gap_steps=10_000,\n    post_grok_steps=5_000,\n    force_restart=True,\n    skip_completed=False,\n    device='auto',\n    fused_adamw=True,\n)\nCONFIG"""),
        md("""## Live TensorBoard\n\nЗапусти эту ячейку до обучения."""),
        code(f"""%load_ext tensorboard\n%tensorboard --logdir /content/drive/MyDrive/grokking_prime_sweep/{protocol} --reload_interval 5"""),
        md("""## Запуск обучения\n\nВыполни следующую ячейку и дождись `summary` либо останови run вручную после достаточного числа шагов."""),
        code("""summaries = run_sweep(CONFIG)\nsummaries"""),
        code(f"""from pathlib import Path\nimport pandas as pd\n\nrun_root = Path(CONFIG.output_root) / CONFIG.protocol_name / 'p_{p}' / 'seed_42'\nprint('run directory:', run_root)\nfor name in ('training_log.csv', 'training.log', 'summary.json', 'run_config.json'):\n    path = run_root / name\n    print(name, path.exists(), path)\nif (run_root / 'training_log.csv').exists():\n    display(pd.read_csv(run_root / 'training_log.csv').tail())"""),
        md("""Для абляционного сравнения скачай `training_log.csv`, `summary.json` и `run_config.json`. Не объединяй WD=0 и WD>0 в один TensorBoard logdir без явного разделения тегов."""),
    ]
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"colab_p{p}_omnigrok_wd0_ablation.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    make(211, 200_000)
    make(431, 300_000)
