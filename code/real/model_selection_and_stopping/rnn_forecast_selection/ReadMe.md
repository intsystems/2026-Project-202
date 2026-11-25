# RNN forecast selection

Compare reduced-query MG, full MG and held-out validation for choosing a model to forecast one neuron in trained recurrent generators.

## In the article

- [Section 4.8 · Forecast model selection from single-neuron recordings](../../../../article/manuscript/ReadMe.md#section-48--forecast-model-selection-from-single-neuron-recordings). Choosing forecasters from single-neuron activity windows.
- [Appendix H · Forecast selection and sampled-query estimation](../../../../article/manuscript/ReadMe.md#appendix-h--forecast-selection-and-sampled-query-estimation). Full and sampled-query MG, forecasts, and timing comparisons.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Model selection and early stopping](../ReadMe.md)

- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.
- [fresh_generators.py](fresh_generators.py). Original source. Original training and rollout generation.
- [query_budget.py](query_budget.py). Original source. Reduced-query MG computation.
- [query_compare.py](query_compare.py). Original source. Decision, timing and model comparison.
- [collect_logs.py](collect_logs.py). Run original collection with the local historical inputs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
```

The plotter verifies agreement on all 160 saved choices and draws the timing comparison from the included CSV files. Fresh generation uses the scripts and frozen models in `research_mg_real_forecast/` and the generator implementation in `research_generator/` at commit `f6b579b1`. From a checkout of that commit with the common dependencies installed, run `python research_mg_real_forecast/fresh_generators.py 681` to produce one new repeat.

The timing comparison measures selection plus fitting the chosen forecaster. The panel compares model choices for confirmation seeds 681–700.
