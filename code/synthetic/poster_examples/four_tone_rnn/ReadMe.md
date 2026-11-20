# Four-tone RNN learning

Train recurrent generators on four tones that have one, two or four independent phases. Record individual neuron activity after training.

## In the article

The current manuscript does not report this experiment.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Linear learning and recurrent dynamics](../ReadMe.md)

- [run.py](run.py). Original source. Training, recurrence and MG scoring.
- [generator.py](generator.py). Original source. Training, recurrence and MG scoring.
- [analyse.py](analyse.py). Original source. Training, recurrence and MG scoring.
- [collect_logs.py](collect_logs.py). Train and record the one-, two-, and four-rhythm conditions.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --help
```
