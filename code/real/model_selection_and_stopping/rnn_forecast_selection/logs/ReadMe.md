# RNN forecast selection data

Forecaster selection on single-neuron confirmation traces. The comparison includes full MG, MG with fewer queries, and validation.

[Experiment overview](../ReadMe.md)

| CSV | What it contains |
| --- | --- |
| [`mg_calibration.csv`](mg_calibration.csv) | MG and candidate errors for neuron 2 and horizon 32 in the 18 calibration repeats (seeds 601–618). These rows fit the decision tree used by the MG rules. |
| `fresh_seed*__*.csv` (160 files, [example](fresh_seed681__H2.csv)) | One scalar activity window per seed and generator condition, containing forecast input and target samples. |
| [`query_fresh.csv`](query_fresh.csv) | Reduced-query MG value, model selected, forecast error, and elapsed time by seed, condition, and query budget. |
| [`query_fresh_comparisons.csv`](query_fresh_comparisons.csv) | Chosen model and forecast error by selection method, seed, and condition. |
| [`query_paired_summary.csv`](query_paired_summary.csv) | Time and speedups for paired full-MG, reduced-query, and validation comparisons. |
| [`query_paired_timing.csv`](query_paired_timing.csv) | Raw timing repetitions by method, seed, and condition. |
| [`query_intervals.csv`](query_intervals.csv) | Mean improvements, confidence intervals, and win and tie counts against baselines. |
