# Four-tone RNN learning

Train recurrent generators on four tones that have one, two or four independent phases. Record individual neuron activity after training.

## In the article

- [Section 3.2 · Learned generators distinguish phases from harmonics](../../../../article/manuscript/ReadMe.md#section-32--learned-generators-distinguish-phases-from-harmonics). The H4, M4, and T4 cases of the generator study. The folder contains those three conditions.
- [Appendix C.3 · Learned autonomous generators](../../../../article/manuscript/ReadMe.md#appendix-c3--learned-autonomous-generators). Architecture, settings, and the independent Lyapunov reference.

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
