# Four-tone RNN learning

Each output learns one sinusoid of unit amplitude. In our task labels, H denotes harmonics of one fundamental, M mixes independent fundamentals with their harmonics, and T denotes a torus target with rationally independent frequencies. The digit counts sinusoidal outputs. The bundled neuron recordings cover H4, M4 and T4, which have one, two and four independent phases, respectively.

## In the article

- [Section 3.1 · Independent oscillations and harmonics](../../../../article/manuscript/ReadMe.md#section-31--independent-oscillations-and-harmonics). Single-neuron MG for targets with the same number of spectral lines.
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
