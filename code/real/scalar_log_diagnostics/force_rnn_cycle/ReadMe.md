# FORCE RNN cycle

Compare one neuron's autonomous activity before and after learning a figure-eight reference. Evaluation holds weights fixed.

## In the article

- [Section 3.2 · Tracking the transition to a periodic orbit](../../../../article/manuscript/ReadMe.md#section-32--tracking-the-transition-to-a-periodic-orbit). Single-neuron MG during figure-eight learning and independent checks of the full hidden state.
- [Appendix C.3 · Learned autonomous generators](../../../../article/manuscript/ReadMe.md#appendix-c3--learned-autonomous-generators). Checkpoint evaluation details and the Lyapunov-based selection of chaotic initializations.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Diagnostics of model dynamics](../ReadMe.md)

- [run.py](run.py). Original source. Train and evaluate the FORCE recurrent model or score its rollouts.
- [analyze.py](analyze.py). Original source. Train and evaluate the FORCE recurrent model or score its rollouts.
- [collect_logs.py](collect_logs.py). Train the FORCE model and record autonomous rollouts.
- [analyze_logs.py](analyze_logs.py). Regenerate the figure from the included CSVs.

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python collect_logs.py
python analyze_logs.py
```
