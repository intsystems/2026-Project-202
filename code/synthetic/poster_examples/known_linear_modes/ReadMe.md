# Known modes in linear learning

Heavy-ball linear learning excites a controlled number of independent modes. The analysis uses the probe error.

## In the article

- [Section 3.1 · Analytic oscillators and learning](../../../../article/manuscript/ReadMe.md#section-31--analytic-oscillators-and-learning). Dense regression with independently controlled oscillatory modes.
- [Appendix C.2 · Dense regression with endogenous modes](../../../../article/manuscript/ReadMe.md#appendix-c2--dense-regression-with-endogenous-modes). The known-mode table and the separate selective-damping comparison.

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
