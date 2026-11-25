# Walker2d policy selection

Select policies for transfer to actuator lag and noise using nominal reward, MG, entropy or a fixed policy.

## In the article

- [Appendix F.2 · Selecting policies for actuator shifts](../../../../article/manuscript/ReadMe.md#appendix-f2--selecting-policies-for-actuator-shifts). Nominal acquisition, target evaluation, and comparisons with simple selection rules.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Model selection and early stopping](../ReadMe.md)

- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.
- [checkpoint_selection.py](checkpoint_selection.py). Original source. Evaluate historical policies and collect episode metrics.
- [analyze.py](analyze.py). Original source. Score and summarize selections.
- [collect_logs.py](collect_logs.py). Run original collection with the local historical inputs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py 232
```

The local plotting entry point uses only the included CSV files. Run the historical collection pipeline from its original archival layout for frozen model files and other upstream inputs.

Full reevaluation needs the original Walker2d checkpoints, normalization files and simulation environment. The included tables are sufficient to redraw the panel.
