# Four-tone RNN learning

Train recurrent generators on four tones that have one, two or four independent phases. Record individual neuron activity after training.

## On the poster

[Block 04 · Synthetic tests with a known active dimension](../../../../article/poster/ReadMe.md#block-04--synthetic-tests-with-a-known-active-dimension) · RNNs learning four-tone signals. This experiment belongs to the poster.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Poster demonstrations](../ReadMe.md)

- [run.py](run.py). Original source. Training, recurrence and MG scoring.
- [generator.py](generator.py). Original source. Training, recurrence and MG scoring.
- [analyse.py](analyse.py). Original source. Training, recurrence and MG scoring.
- [collect_logs.py](collect_logs.py). Train and record the three poster conditions.
- [analyze_logs.py](analyze_logs.py). Rebuild the poster panel from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --help
```
