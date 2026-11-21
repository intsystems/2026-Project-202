# FORCE RNN cycle

Compare one neuron's autonomous activity before and after learning a figure-eight reference. Evaluation holds weights fixed.

## In the article

- [Section 3.3 · FORCE training from chaos to periodic output](../../../../article/manuscript/ReadMe.md#section-33--force-training-from-chaos-to-periodic-output). The figure-eight intervention and pre/post-training single-neuron analysis.
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
