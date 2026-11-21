# Full-batch perceptron grokking

Compare modular arithmetic and polynomial targets, including label-matched controls and the window-length sweep.

## In the article

- [Section 4.4 · Detecting changes during grokking](../../../../article/manuscript/ReadMe.md#section-44--detecting-changes-during-grokking). The full-batch comparison.
- [Appendix E · Additional learning settings](../../../../article/manuscript/ReadMe.md#appendix-e--additional-learning-settings). The unregularized quadratic perceptron in the grokking paragraph.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Grokking and optimization stability](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.
- [recompute_metrics.py](recompute_metrics.py). Recompute the article's metrics after collecting the upstream runs.

- [Training loop](train.py)
- [Training configurations](run_configs.py)
- [Grokking and representation analysis](../analyze_training.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --device cuda:0
python recompute_metrics.py --device cpu
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

The diagnostic map uses both perceptron and transformer estimates. The effective-rank collapse did not reproduce in the full-batch setting.
