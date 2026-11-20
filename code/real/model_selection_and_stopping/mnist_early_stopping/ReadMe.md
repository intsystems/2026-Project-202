# Noisy MNIST early stopping

Use rules fixed on development runs to select stopping checkpoints in eight confirmation runs with 40% label noise.

## On the poster

[Block 07 · Select models and stop training](../../../../article/poster/ReadMe.md#block-07--select-models-and-stop-training) · MNIST classification with 40% noisy labels. This experiment belongs to the poster.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Model selection and early stopping](../ReadMe.md)

- [frozen_rules.csv](logs/frozen_rules.csv). Fixed checkpoint rules used for evaluation.
- [run.py](run.py). Original source. Train and record features and classification metrics.
- [control.py](control.py). Original source. Score the frozen checkpoint decisions.
- [collect_logs.py](collect_logs.py). Collect MNIST training and feature logs.
- [analyze_logs.py](analyze_logs.py). Rebuild the poster panel from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python collect_logs.py --help
python analyze_logs.py
```
