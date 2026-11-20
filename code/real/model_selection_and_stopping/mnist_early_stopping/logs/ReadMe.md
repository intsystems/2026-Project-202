# Noisy MNIST early stopping data

MNIST confirmation runs with 40% noisy labels. Logs cover seeds 10–17.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`decisions_10_17.csv`](decisions_10_17.csv) | Stopping checkpoint, held-out accuracy, test accuracy, and compute fraction by seed and rule. |
| [`frozen_rules.csv`](frozen_rules.csv) | Scalar signal and threshold, or fixed checkpoint, for each decision rule. |
| `seed*_logs.csv` (8 files, [example](seed10_logs.csv)) | Training loss, weight norm, probe loss, and measured times by step. |
| `seed*_features.csv` (8 files, [example](seed10_features.csv)) | Windowed MG, roughness-related features, and statistics for the rule inputs. |
| `seed*_metrics.csv` (8 files, [example](seed10_metrics.csv)) | Train, validation, and test accuracy at evaluation checkpoints. |
