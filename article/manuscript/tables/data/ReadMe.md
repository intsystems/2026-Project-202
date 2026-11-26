# Data for reported results

These files preserve the numerical inputs behind the manuscript's tables. The paths below work in a fresh checkout.

## CNN interventions

- [Detector parameters](cnn/rules.json). Full-precision calibration for the small CNN.
- [Fixed intervention records](cnn/baseline_records.csv). Run-level alarms and delays by statistic.
- [Shifted intervention records](cnn/fresh_cnn_records.csv) and [summary](cnn/fresh_cnn_summary.csv). Evaluation with varied intervention times.
- [Detection horizons](cnn/horizon_sensitivity.csv). Results by time allowed after an intervention.

The [ResNet experiment](../../../../code/real/scalar_log_diagnostics/resnet_freeze/ReadMe.md) provides its inputs and a script for replaying the transferred detector.

## Self-supervised encoder selection

- [Calibrated rankings](ssl/T1_config_ranking.csv). Column and direction picked for each statistic.
- [Evaluation seeds 10–12](ssl/test_final.csv) and [evaluation seeds 20–21](ssl/confirm_final.csv). Feature scores and linear-classifier test accuracy for each configuration.
- [Pooled comparison](ssl/T1_confirm.csv). Use rows labeled `pooled_10_12_20_21` for the rank correlations, accuracy gaps, and paired bootstrap intervals.

Run `python article/manuscript/tables/data/ssl/replay.py` from the repository root to check the saved rankings and accuracy gaps. For the abstract's result, select the configuration with the highest `MG|grad_norm` within each seed. Subtract its `acc` from the highest `acc` in the same seed and average the five differences. The result is 0.408 percentage points. Full encoder training logs are separate inputs.

## Transformer loss spikes

- [Calibration scores](spikes/results_cal_percolumn.csv). Selected warning rules and full-precision thresholds per statistic and input.
- [Held-out results](spikes/results_test.csv). Detections, false alarms, lead times, and discrimination metrics.
- [Event onsets](spikes/onsets.json). Onset steps grouped by configuration and seed.
- [Window scores](spikes/window_scores.csv.gz). Per-run values for all statistics selected by the calibration procedures.

Run `python article/manuscript/tables/data/spikes/replay.py` from the repository root to replay the frozen rules and check the saved results. The paper uses the budget of 2 false alarms per 10,000 scored updates. The [figure data](../../figures/ReadMe.md#rebuild-the-selection-and-warning-figures) include the underlying loss envelope for the example in the paper.
