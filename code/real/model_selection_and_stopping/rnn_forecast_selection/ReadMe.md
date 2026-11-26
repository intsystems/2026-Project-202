# RNN forecast selection

Compare reduced-query MG, full MG and held-out validation for choosing a model to forecast one neuron in trained recurrent generators.

## In the article

- [Section 4.4 · Model selection from scalar logs](../../../../article/manuscript/ReadMe.md#section-44--model-selection-from-scalar-logs). Forecast error and time to select and fit a forecaster.
- [Appendix H · Forecast selection and sampled-query estimation](../../../../article/manuscript/ReadMe.md#appendix-h--forecast-selection-and-sampled-query-estimation). Full and sampled-query MG, forecasts, and timing comparisons.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Model selection and early stopping](../ReadMe.md)

- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.
- [fresh_generators.py](fresh_generators.py). Train the generators and record one neuron's activation.
- [query_budget.py](query_budget.py). Estimate MG from a subset of the query points.
- [query_compare.py](query_compare.py). Fit the MG decision tree on the calibration set and compare it against rolling validation.
- [collect_logs.py](collect_logs.py). Generate new forecast windows.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python query_compare.py
```

The plotter verifies agreement on all 160 saved choices and draws the timing comparison from the included CSV files. The scoring script independently recomputes the choices and errors of full MG, sampled MG, and rolling validation.

To generate and score new windows for one repeat, run:

```bash
python collect_logs.py 681 --output ../../../runs/rnn_forecast_selection/fresh
python query_compare.py 681 --input ../../../runs/rnn_forecast_selection/fresh
```

Pass `--arms M4 T4` to collect only those conditions. The generator and scoring code run on CPU.

The timing comparison measures selection plus fitting the chosen forecaster. The panel compares model choices for confirmation seeds 681–700.
