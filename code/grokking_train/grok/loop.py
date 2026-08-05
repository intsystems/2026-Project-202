"""The training loop that produces the CSV logs analysed by ``../grokking_analysis``.

One entry point, :func:`train`, driven entirely by a :class:`~grok.config.RunConfig`.
Every row of the output is one *logged* optimization step; the analysis package
slides its window over that row grid, so ``log_every`` sets the sampling rate of
the reconstructed attractor and must stay constant within a run.
"""

import contextlib
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from . import metrics, models, tasks

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def resolve_device(spec="auto"):
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@contextlib.contextmanager
def default_dtype(name):
    """Temporarily set the global default dtype (the notebooks set float64 globally)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(DTYPES[name])
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_optimizer(config, model):
    if config.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay, betas=tuple(config.betas))
    if config.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr,
                         momentum=config.momentum, weight_decay=config.weight_decay)
    raise KeyError(f"unknown optimizer '{config.optimizer}'. Available: adamw, sgd")


def _batches(task, config):
    """Endless stream of training batches; the whole split at once if ``batch_size`` is None."""
    if config.batch_size is None:
        while True:
            yield task.X_train, task.Y_train

    loader = DataLoader(
        TensorDataset(task.X_train, task.Y_train),
        batch_size=min(config.batch_size, len(task.X_train)),
        shuffle=True,
    )
    while True:
        for batch in loader:
            yield batch


def train(config, outdir=None, progress=True, overwrite=False):
    """Run ``config`` and return ``(DataFrame, written_path_or_None)``.

    ``outdir`` is where the CSV lands (``config.csv_name``); pass ``None`` to keep
    the log in memory.  An existing file is left alone unless ``overwrite=True`` --
    these runs take minutes to hours, so silently clobbering one is expensive.
    """
    path = None
    if outdir is not None:
        path = Path(outdir) / config.csv_name
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists (pass overwrite=True / --force)")

    device = resolve_device(config.device)
    columns = list(config.columns)
    logs = {name: [] for name in columns}

    with default_dtype(config.dtype):
        task = tasks.from_config(config, device)
        model = models.build(config, task.vocab_size, n_ctx=task.n_ctx).to(device)
        optimizer = build_optimizer(config, model)
        criterion = nn.CrossEntropyLoss()
        probe = metrics.GradientProbe() if config.needs_grad_probe else None

        val_x = task.X_val if config.val_batch_size is None else task.X_val[:config.val_batch_size]
        val_y = task.Y_val if config.val_batch_size is None else task.Y_val[:config.val_batch_size]
        if len(val_x) == 0:
            raise ValueError("empty validation split -- lower `fraction`")

        stream = _batches(task, config)
        bar = tqdm(total=config.max_steps, desc=config.key, disable=not progress)
        started = time.perf_counter()

        for step in range(config.max_steps):
            batch_x, batch_y = next(stream)

            model.train()
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            if probe is not None:
                probe.update(model)
            if config.double_step:
                optimizer.step()          # see RunConfig.double_step -- reproduces a known bug

            if step % config.log_every == 0:
                row = _observe(model, criterion, columns, step, loss, logits, batch_y,
                               val_x, val_y, probe)
                for name in columns:
                    logs[name].append(row[name])
                if progress and step % (config.log_every * 50) == 0:
                    bar.set_postfix(
                        loss=f"{row.get('train_loss', float('nan')):.3f}",
                        v_acc=f"{row.get('val_acc', float('nan')):.2f}",
                    )

            bar.update(1)

        bar.close()

    df = pd.DataFrame(logs, columns=columns)
    elapsed = time.perf_counter() - started

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    _report(config, task, df, device, elapsed, path)
    return df, path


@torch.no_grad()
def _observe(model, criterion, columns, step, loss, logits, targets, val_x, val_y, probe):
    """Collect one log row.  Only the requested columns are computed."""
    model.eval()
    row = {"step": step}

    if "train_loss" in columns:
        row["train_loss"] = loss.item()
    if "train_acc" in columns:
        row["train_acc"] = metrics.accuracy(logits, targets)

    val_logits = model(val_x)                       # val_acc is a mandatory column
    row["val_acc"] = metrics.accuracy(val_logits, val_y)
    if "val_loss" in columns:
        row["val_loss"] = criterion(val_logits, val_y).item()

    if "weight_norm" in columns:
        row["weight_norm"] = metrics.weight_norm(model)
    if probe is not None:
        row.update({k: v for k, v in probe.values.items() if k in columns})

    return row


def _report(config, task, df, device, elapsed, path):
    grokked = df[df["val_acc"] >= 0.95]["step"] if "val_acc" in df else []
    print(f"[{config.key}] {config.description or config.summary()}")
    print(f"    task     : {task} on {device} ({config.dtype})")
    print(f"    log      : {len(df)} rows x {len(df.columns)} columns, "
          f"steps {df['step'].min():.0f}..{df['step'].max():.0f}")
    print(f"    grokking : {'not reached' if len(grokked) == 0 else f'step {grokked.iloc[0]:.0f}'}"
          "  (first step with val_acc >= 0.95)")
    if "weight_norm" in df:
        print(f"    ||w||_2  : {df['weight_norm'].iloc[0]:.2f} -> {df['weight_norm'].iloc[-1]:.2f}")
    if path is not None:
        print(f"    wrote    : {path}")
    print(f"    took     : {elapsed:.1f}s")
