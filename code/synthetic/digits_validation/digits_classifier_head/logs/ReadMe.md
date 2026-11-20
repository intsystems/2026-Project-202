# Digits parameter subspace data

Estimates and reference ranks for a constrained classifier head on handwritten digits. The files compare recurrent, noisy, and transient runs.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`sweep_raw.csv`](sweep_raw.csv) | Full parameter-subspace sweep by forcing condition, seed, rank, and scalar signal. |
| [`ground_truth_PR.csv`](ground_truth_PR.csv) | Measured trajectory participation ratio by forcing condition and reference rank. |
| [`observer_scores.csv`](observer_scores.csv) | Error, rank correlation, and identifiability by forcing condition and scalar signal. |
| [`calibrated_mae.csv`](calibrated_mae.csv) | Mean absolute error and bias before and after calibration. |
| [`atlas_raw.csv`](atlas_raw.csv) | Per-run regime comparison with known state ranks and estimator diagnostics. |
| [`identifiability_ratio.csv`](identifiability_ratio.csv) | Ratio of the MG estimates at two embedding dimensions by run. |
