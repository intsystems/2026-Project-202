# Edge of stability data

Full-batch perceptron runs at several learning rates. These files support the article's analysis of stability, oscillation, and recurrence.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`eos_runs.csv`](eos_runs.csv) | Learning rate, outcome, generalization steps, and stability summaries for each run. |
| [`eos_lr2e+06_s1_train.csv`](eos_lr2e+06_s1_train.csv), `archived__eos_*_train.csv` (16 files, [example](archived__eos_lr1.5e+06_s1_train.csv)) | Recorded training and validation metrics, loss, and weight norm. |
| `eos_*_sharp.csv` (8 files, [example](eos_lr1.5e+06_s1_sharp.csv)) | Maximum Hessian eigenvalue and stability ratio by step. |
| [`eos_diagnostics.csv`](eos_diagnostics.csv), [`eos_diagnostics_summary.csv`](eos_diagnostics_summary.csv) | Window diagnostics and aggregates by scalar signal, segment, and logging frequency. |
| [`eos_recurrence.csv`](eos_recurrence.csv) | Return distances relative to travel and scale, with oscillation and stability measurements. |
