# EDM analysis of the Muon/nanoGPT log

This directory turns `icomp_article/message (2).txt` into reproducible tabular
logs and applies sliding-window empirical dynamic modeling to the dense training
loss trajectory. The original message file remains unchanged.

Run from the repository root:

```powershell
python code/Grokking/nanogpt_muon_edm/parse_nanogpt_log.py `
  "icomp_article/message (2).txt" `
  code/Grokking/nanogpt_muon_edm/parsed

python code/Grokking/nanogpt_muon_edm/analyze_edm.py `
  code/Grokking/nanogpt_muon_edm/parsed `
  code/Grokking/nanogpt_muon_edm/results

python code/Grokking/nanogpt_muon_edm/analyze_all_edm_tau1.py `
  code/Grokking/nanogpt_muon_edm/parsed `
  code/Grokking/nanogpt_muon_edm/results_tau1

python code/Grokking/nanogpt_muon_edm/analyze_batch_tau1.py `
  code/Grokking/nanogpt_muon_edm `
  code/Grokking/nanogpt_muon_edm/results_batch_tau1
```

To rebuild the Russian PDF report, run from this directory:

```powershell
New-Item -ItemType Directory -Force .texmf-var | Out-Null
$env:TEXMFVAR=(Resolve-Path .texmf-var).Path
pdflatex -interaction=nonstopmode -halt-on-error report_ru.tex
pdflatex -interaction=nonstopmode -halt-on-error report_ru.tex
```

The primary estimate is the Levina–Bickel intrinsic dimension of a delay
embedding. Delay is selected separately in every window using the first local
minimum of delayed mutual information. Temporally adjacent neighbours are
excluded with a Theiler window. Both the raw loss and a locally linearly
detrended version are reported because a monotonic training trend can itself
look like a low-dimensional manifold.

For comparison, the analysis also reproduces the estimator from the existing
`search_for_optimal_parameters.py` (`E=15`, `k=5`, fixed `tau=1`, no Theiler
window). It gives higher absolute values but the same qualitative trajectory.

`analyze_all_edm_tau1.py` is the fixed-delay report pipeline. It runs all four
dimension methods already used in the project: FNN, Cao, simplex projection and
Levina--Bickel MLE. Its source report is `REPORT_TAU1_RU.md`; the compiled PDF
is `report_tau1_ru.pdf`.

Each supplied run has 2330 training-loss observations. Validation loss has only
11 observations per run and is used as an external progress indicator.

The audited eight-run conclusion is in `BATCH_AUDIT.md`. In brief, the weak FNN
decrease is not corroborated by the other estimators: simplex and MLE increase
from early to late windows for all eight optimizers, both before and after local
detrending. Treat this as a descriptive negative/control result because there
is one run per optimizer and the sliding windows overlap by 90%.

## Consolidated regime report

Run `python build_final_report.py` after `analyze_batch_tau1.py`. The consolidated
Russian comparison of the logged `lmo` (`lr=0.06`) and `sign` (`lr=0.03`)
regimes is `FINAL_COMPARATIVE_REPORT_RU.md`. The machine-readable aggregate is
`results_batch_tau1/training_regime_summary.csv`.
