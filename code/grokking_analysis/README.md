# Grokking Analysis — script-based reproduction pipeline

Modernized, notebook-free version of the analysis half of [`../Grokking/`](../Grokking/).
Everything is plain `.py`: one importable package (`edm/`), one figure registry
(`experiments.py`) and one CLI (`reproduce_figures.py`) that regenerates the figures of
[`icomp_article/grokking_en.tex`](../../icomp_article/grokking_en.tex) from the raw training logs.

Nothing here trains a network — the CSV logs in `grokking_logs/` are the input. Training
code still lives in `../Grokking/` (`grokking_model.py`, `grokking_utils.py` and the
`generator_logs_*.ipynb` notebooks).

## Quick start

```bash
pip install -r requirements.txt

python reproduce_figures.py --list      # what can be built
python reproduce_figures.py             # build everything into ./figures
python reproduce_figures.py s5_wd1      # build one figure
python test_edm.py                      # sanity checks (also runs under pytest)
```

Total runtime for all eight figures is about 15 s on a laptop CPU.

## Layout

| Path | Contents |
| --- | --- |
| `edm/embedding.py` | Delay-coordinate embedding (Takens / Stark) and `tau` selection via delayed mutual information |
| `edm/dimension.py` | Dimension estimators: Levina–Bickel **MLE** (primary), simplex projection, Cao's method, classical FNN |
| `edm/sliding.py` | Log loading and the sliding-window driver that produces `d_hat(t)` as a `DimensionTrace` |
| `edm/plots.py` | The three figure families used in the paper |
| `experiments.py` | Registry mapping each article figure to its log, observable and window parameters |
| `reproduce_figures.py` | CLI |
| `test_edm.py` | Estimator sanity checks (Lorenz-63 vs. white noise) plus registry/claim checks |
| `grokking_logs/` | The five CSV logs the article's figures are built from |
| `figures/` | Output |

## Figures

| Key | Article figure | Log | Observable | Window |
| --- | --- | --- | --- | --- |
| `mod_wd1` | Fig. 1 top row | modular addition, WD=1.0 | `weight_norm` | W=300, S=50 |
| `mod_wd0` | Fig. 1 bottom row | modular addition, WD=0.0 | `weight_norm` | W=300, S=50 |
| `s5_wd1` | Fig. 2 top row | $S_5$ composition, WD=0.2 | `weight_norm` | W=300, S=50 |
| `s5_wd0` | Fig. 2 bottom row | $S_5$ composition, WD=0.0 | `weight_norm` | W=300, S=50 |
| `s5_wd1_val_loss` | Fig. 3a | $S_5$, WD=0.2 | `val_loss` | W=300, S=50 |
| `s5_wd0_val_loss` | Fig. 3b | $S_5$, WD=0.0 | `val_loss` | W=300, S=50 |
| `grokking_dimension` | Fig. 5b (App. B) | full-batch GD baseline | `train_loss` | W=1500, S=300 |
| `grokking_accuracy` | Fig. 5a (App. B) | full-batch GD baseline | — | smoothing 150 |

`mod_wd*` / `s5_wd*` each emit three PNGs (`_acc`, `_loss`, `_norm`) sharing one `d_hat(t)`
curve, so the collapse can be read against accuracy, loss and the weight norm independently.

Regenerated output is pixel-identical to `icomp_article/images/` for every `_acc` / `_loss` /
`_norm` panel. The two appendix images in the article are crops of these figures (the title
band was trimmed by hand), so they match in content but not in canvas size.

## Log files

| CSV | Task | Optimizer | Grokking |
| --- | --- | --- | --- |
| `..._to_flat_grokking_with_stochastic.csv` | modular addition, $p=113$ | AdamW, WD=1.0, batch 256 | step 13810 |
| `..._to_flat_grokking_with_stochastic_without_wight_decay.csv` | modular addition, $p=113$ | AdamW, WD=0.0 | never |
| `..._S_5_with_stochastic.csv` | $S_5$ composition | AdamW, WD=0.2, batch 256 | step 6735 |
| `..._S_5_with_stochastic_without_wight_decay.csv` | $S_5$ composition | AdamW, WD=0.0 | never |
| `grokking_modular_addition_logs.csv` | modular addition, $p=97$ | full-batch GD, WD=1.0 | step 3270 |

The remaining logs of the original folder ($S_6$, small weight decay, non-stochastic variants)
are exploratory runs that no article figure depends on; they stay in `../Grokking/grokking_logs/`.

## Using the package directly

```python
from edm import load_logs, sliding_dimension, plot_presentation_panels

df = load_logs("grokking_logs/grokking_modular_addition_logs_S_5_with_stochastic.csv")
trace = sliding_dimension(df, target_metric="weight_norm", method="mle",
                          window_size=300, step_size=50, seed=0)
plot_presentation_panels(trace, df, outdir="figures", prefix="s5_wd1")
```

`method` accepts `"mle"`, `"fnn"`, `"cao"` and `"simplex"`; `tau_selector` accepts `"fixed"`
(the paper's default, $\tau=1$) and `"dmi"`.

## Notes on reproducibility

* All estimators dither their input with 1e-9-scale Gaussian noise to break KD-tree ties.
  The driver seeds this (`--seed`, default 0), so repeated runs are bit-identical; the
  original notebooks used unseeded `np.random` and drift by <0.01 in `E` between runs.
* Windows whose standard deviation falls below `1e-6` are treated as degenerate observables
  and reported as `E = 1` — this is the "degenerate observable" failure mode discussed in
  Sec. 5 of the paper, not a numerical accident.
* `include_last_window=False` reproduces the original loop `range(0, n - W, S)`, which drops
  the final full window. It is set for the figures that were produced with the old
  `analyze_grokking_dimensionality` helper so their x-extent matches the published images.
