# Measuring MG from a scalar record

The paper's Section 2.4 describes the protocol; the appendix supplies the numerical algorithm. `measurement_code/` is a snapshot of the estimator actually used in the project. `mg_pipeline.py` is a new example wrapper around that unchanged kernel, not the historical experiment runner.

1. Define a task-specific simplification and its independent reference. Select one observable on calibration runs (parameter norm, a fixed-probe loss, or fixed-policy activity). Keep logging interval, probe inputs and sampling noise fixed. Measure probe acquisition cost separately.
2. Calibrate window, stride, delay, embedding dimension, neighbor number, exclusion and any detector on separate runs. Freeze the values or the adaptive rule before testing. Choose a raw-amplitude floor based on precision and measurement noise; there is no universal numerical value valid for every observable.
3. Measure completed windows and retain their raw standard deviations, floor flags, valid-vector numbers and doubled-embedding checks. NaN, insufficient neighbors, constant records and numerical-floor domination are unavailable measurements, not zero dimension. Report their frequency and use the same eligibility rule for every condition. Do not discard unsuccessful windows retrospectively.
4. For recurrent systems, a dimension interpretation additionally requires stationary occupation statistics, an appropriate embedding, sufficient repeated visits and resolved local scaling. Stability under 2E is only a diagnostic. For SGD/transients, calibrate relative changes against an independent reference and competing scalar statistics; an MG drop alone neither proves loss of active dimension nor predicts better generalization.

Example for an **already logged**, regularly sampled CNN parameter norm:

```sh
python mg_pipeline.py log.csv --column param_norm --window 1000 --stride 500 --E 20 --tau 1 --k 20 --theiler 19 --min-std 0 --out mg_windows.csv
```

`--min-std 0` only excludes exact constants in this example. Set a meaningful measurement floor for deployment. This explicit gate is recommended practice and is not retroactively applied to the manuscript's historical results. The wrapper keeps the same exclusion for E and 2E, and marks failed 2E checks as missing. It never emits an automatic “dimension valid” verdict.

Dependencies: Python, NumPy and scikit-learn. Saved numerical outputs and analysis scripts elsewhere in the bundle reproduce the paper's tables. Training scripts are supplied under `experiment_code/` with original relative layout and a source hash manifest; external data and all original checkpoints are not bundled. See REPRODUCIBILITY.md for commands and limits.
