# RNN forecast selection

Compare reduced-query MG, full MG and held-out validation for choosing a model to forecast one neuron in trained recurrent generators.

## On the poster

[Block 07 · Select models and stop training](../../../../article/poster/ReadMe.md#block-07--select-models-and-stop-training) · Forecasting one neuron in a trained RNN. This experiment belongs to the poster.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Model selection and early stopping](../ReadMe.md)

- [analyze_logs.py](analyze_logs.py). Rebuild the poster panel from the included CSVs.
- [fresh_generators.py](fresh_generators.py). Original source. Original training and rollout generation.
- [query_budget.py](query_budget.py). Original source. Reduced-query MG computation.
- [query_compare.py](query_compare.py). Original source. Decision, timing and model comparison.
- [collect_logs.py](collect_logs.py). Run original collection with the local historical inputs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py 681
```

The local plotting entry point uses only the included CSV files. Run the historical collection pipeline from its original archival layout for frozen model files and other upstream inputs.

The timing comparison measures selection plus fitting the chosen forecaster. The panel compares model choices for confirmation seeds 681–700.
