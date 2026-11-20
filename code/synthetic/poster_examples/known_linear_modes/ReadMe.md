# Known modes in linear learning

Heavy-ball linear learning excites a controlled number of independent modes. The poster panel uses the probe error.

## On the poster

[Block 04 · Synthetic tests with a known active dimension](../../../../article/poster/ReadMe.md#block-04--synthetic-tests-with-a-known-active-dimension) · Linear learning. This experiment belongs to the poster.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Poster demonstrations](../ReadMe.md)

- [research_known_modes.py](research_known_modes.py). Collection and scoring source. Runs the deterministic linear experiment and measurements.
- [collect_logs.py](collect_logs.py). Run the linear simulation and score its scalar records.
- [analyze_logs.py](analyze_logs.py). Rebuild the poster panel from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py
```
