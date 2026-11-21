# Delay coordinate reconstruction

Example scalar observations and delay clouds for recurrent, transient and mini-batch cases.

## In the article

- [Section 2.2 · Delay reconstruction and dynamical-systems theory](../../../../article/manuscript/ReadMe.md#section-22--delay-reconstruction-and-dynamical-systems-theory). Figure 1 illustrates the scalar samples, delay vectors, and admissible neighbors.
- [Appendix A.2 · Recurrent, transient, and stochastic regimes](../../../../article/manuscript/ReadMe.md#appendix-a2--recurrent-transient-and-stochastic-regimes). Delay reconstructions and scalar traces for the recurrent, transient, and stochastic examples.

## Files and commands

- [CSV logs and results](logs/ReadMe.md)
- [PDF figures](figures/ReadMe.md)
- [Delay embedding and estimator limits](../ReadMe.md)

- [collect_logs.py](collect_logs.py). Run the registered simulation or training pipeline and collect fresh outputs under code/runs.
- [analyze_logs.py](analyze_logs.py). Regenerate the PDF figures from the included CSV files.

- [Log and delay cloud generation](collect_curves.py)

Run from this directory after installing the [common dependencies](../../../ReadMe.md#run-the-code).

```bash
python analyze_logs.py
python collect_logs.py --fast --jobs 1
```

The PDF exports ship in figures/. Redrawn figures go to code/runs/redrawn/ and leave the released PDFs in place.

This simulation or control records and scores its observations in the same registered command.
