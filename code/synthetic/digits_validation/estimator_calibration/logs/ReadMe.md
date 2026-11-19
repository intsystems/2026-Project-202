# Estimator calibration data

Development results for selecting the eight- and twenty-direction settings. The fixed settings used by the scripts are in [settings.py](../settings.py).

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`config_ranking.csv`](config_ranking.csv) | Eight-direction candidate settings ranked by error, rank correlation, and spread. |
| [`calibration_scores.csv`](calibration_scores.csv) | Scores for each candidate setting and scalar signal. |
| [`scores_frozen.csv`](scores_frozen.csv) | Per-run results using the fixed twenty-direction configuration. |
| [`frozen_per_r.csv`](frozen_per_r.csv) | Fixed-configuration estimates and reference statistics by rank. |
| [`heldout_qp_summary.csv`](heldout_qp_summary.csv) | Held-out quasiperiodic runs scored before and after calibration. |
| [`invariance_controls.csv`](invariance_controls.csv) | Change in MG under the invariance controls. |
