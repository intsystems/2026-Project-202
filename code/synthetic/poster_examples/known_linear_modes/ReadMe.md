# Known modes in linear learning

Heavy-ball linear learning excites a controlled number of independent modes. The analysis uses the probe error.

## In the article

The current manuscript does not report this experiment.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Linear learning and recurrent dynamics](../ReadMe.md)

- [research_known_modes.py](research_known_modes.py). Collection and scoring source. Runs the deterministic linear experiment and measurements.
- [collect_logs.py](collect_logs.py). Run the linear simulation and score its scalar records.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py
```
