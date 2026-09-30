# CIFAR-10 neural-collapse / scalar MG experiment

Read `report_ru.pdf` or `report_ru.md` for the completed Russian report. This is a
CPU subset pilot, not full-data CIFAR training. Its conclusion is negative for
using scalar-loss MG as a reliable neural-collapse monitor in this setting.

All commands run from `2026-Project-202`. Point `--data` to the extracted official
`cifar-10-batches-py` directory containing data_batch_1..5 and test_batch.

```powershell
python research_neural_collapse/run.py --data 'PATH_TO/cifar-10-batches-py' --seeds 0 1 2
python research_neural_collapse/run.py --data 'PATH_TO/cifar-10-batches-py' --arm frozen --seeds 0
python research_neural_collapse/analyze.py
python research_neural_collapse/benchmark.py --data 'PATH_TO/cifar-10-batches-py'
python research_neural_collapse/build_report.py
```

Training skips runs that already have metadata.json; use a new `--out` directory
to actually rerun training. Analysis accepts `--root`. Source package `code/actdim`
is required. Libraries: torch, numpy, scipy, pandas, sklearn, threadpoolctl,
matplotlib. PDF requires XeLaTeX, Times New Roman, polyglossia and standard packages.

`PROTOCOL.md` predates MG inspection. `results/*/features/` contains real 64-D
features from every checkpoint, enough to recompute independent NC metrics without
training. `audit.json` verifies ideal-simplex/rotation/scale invariance and frozen
feature identity. `mg_windows.csv` includes all settings, diagnostics and repeat
timings. `correlations.csv` includes raw trends and first differences. Neither
overlapping windows nor the three seeds on one data split are independent datasets.

Timing distinction: per-step scalar-probe acquisition versus per-checkpoint NC
measurement is intentional. `timing.csv` estimates the cost of one scalar probe
from warmed forward measurements. Actual joint acquisition for both probes is
recorded in each run's metadata. `warmed_benchmark.csv` includes an efficient
100-image direct NC competitor, not only the full-data reference. Negative control
still trains its linear head; only NC1/NC2 feature geometry is guaranteed constant.

No user manuscript sources or PDF builds were changed.
