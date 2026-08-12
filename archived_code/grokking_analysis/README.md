# Grokking Analysis — script-based reproduction pipeline

Modernized, notebook-free version of the analysis half of [`../Grokking/`](../Grokking/).
Everything is plain `.py`: one importable package (`edm/`), one figure registry
(`experiments.py`) and one CLI (`reproduce_figures.py`) that regenerates the figures of
[`icomp_article/grokking_en.tex`](../../icomp_article/grokking_en.tex) from the raw training logs.

Nothing here trains a network — the CSV logs in `grokking_logs/` are the input. They are
produced by [`../grokking_train/`](../grokking_train/), whose run registry uses the same keys
as `experiments.py` (`python train.py --article --outdir ../grokking_analysis/grokking_logs`).
The original notebook versions of both halves are kept in `../Grokking/`.

## Quick start

```bash
pip install -r requirements.txt

python reproduce_figures.py --list      # what can be built
python reproduce_figures.py             # build everything into ./figures
python reproduce_figures.py s5_wd1      # build one figure
python reproduce_figures.py --article-exact   # drop the annotation, match the paper byte for byte
python reproduce_figures.py --compare mle_mg mle_mg_theiler   # overlay the corrected curves
python best_practice.py                 # tuned re-analysis -> figures/best_practice
python local_dimension.py               # local trajectory dimension over ~100 iterations
python test_edm.py                      # sanity checks (also runs under pytest)
```

Total runtime for all eight figures is about 15 s on a laptop CPU.

## Layout

| Path | Contents |
| --- | --- |
| `edm/embedding.py` | Delay-coordinate embedding (Takens / Stark), `tau` selection via delayed mutual information, autocorrelation time |
| `edm/dimension.py` | Dimension estimators: Levina–Bickel **MLE** (primary) with the MacKay–Ghahramani correction and the Theiler window, simplex projection, Cao's method, classical FNN |
| `edm/sliding.py` | Log loading and the sliding-window driver that produces `d_hat(t)` as a `DimensionTrace` |
| `edm/plots.py` | The three figure families used in the paper |
| `experiments.py` | Registry mapping each article figure to its log, observable and window parameters |
| `reproduce_figures.py` | CLI |
| `best_practice.py` | Re-analysis with every knob tuned, plus line-constant and identifiability diagnostics |
| `local_dimension.py` | Local singular-spectrum dimension over ~100-iteration segments |
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

Every figure states, on the y-axis, which 1D series the dimension was estimated from — the
published panels showed a dimension curve next to accuracy with nothing saying it came from
$\lVert w
Vert_2$. Wherever validation accuracy is drawn, training accuracy is drawn with it, so
the memorization point is visible next to the generalization point. Both apply to
`best_practice.py` and `local_dimension.py` as well.

`--article-exact` drops the annotation and reproduces the published figures pixel-identically for
every `_acc` / `_loss` / `_norm` panel. The two appendix images in the article are crops of these figures (the title
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

## The MacKay–Ghahramani correction

Both estimators share the per-point statistic $S_i = \sum_{j=1}^{k-1}\log(r_k/r_j)$, for which
$d\,S_i \sim \Gamma(k-1, 1)$, so the local estimate is $\hat d_i = (k-1)/S_i$.

* **Levina–Bickel (2004)** — the paper's estimator — averages the local estimates,
  $\hat d = \frac1n\sum_i \hat d_i$. Since $\mathbb{E}[1/S_i] = d/(k-2)$ rather than $d/(k-1)$,
  each $\hat d_i$ is inflated by $(k-1)/(k-2)$ (33 % at $k=5$), and the arithmetic mean is then
  dominated by the few points with anomalously small $S_i$.
* **MacKay–Ghahramani (2005)** point out that all points share one $d$, so the likelihood should
  be pooled *before* inverting: $\hat d = \bigl(n(k-1) - 1\bigr) / \sum_i S_i$. For large $n$ this
  is the inverse of the average of $1/\hat d_i$ — a harmonic mean of the local estimates — so by
  AM–HM it can never exceed the Levina–Bickel value.

Run `python reproduce_figures.py --compare mle_mg` to plot both curves on the same axes; output
goes to `figures/comparison/` so the article-faithful set in `figures/` is left untouched.

### What it changes

| Figure | Observable | Levina–Bickel | MacKay–Ghahramani | median MG/LB | corr |
| --- | --- | --- | --- | --- | --- |
| `mod_wd1` | `weight_norm` | peak 3.46 | peak 2.99 | 0.91 | 0.99 |
| `mod_wd0` | `weight_norm` | flat 1.33–1.38 | flat 1.33–1.38 | 1.00 | — |
| `s5_wd1` | `weight_norm` | peak 3.94, tail 1.47 | peak 3.08, tail 1.46 | 0.94 | 0.96 |
| `s5_wd0` | `weight_norm` | flat 1.36–1.40 | flat 1.36–1.39 | 1.00 | — |
| `s5_wd1_val_loss` | `val_loss` | peak 13.32 | peak 9.34 | 0.80 | 0.98 |
| `s5_wd0_val_loss` | `val_loss` | 3.15 → 8.61 | 2.41 → 6.11 | 0.73 | 0.97 |
| `grokking_dimension` | `train_loss` | 4.93 → 2.29 | 2.65 → 1.88 | 0.76 | 0.75 |

Every **qualitative** conclusion of the paper survives: the rise-then-collapse under weight decay,
the flat trace without it, and the growth of $E(\mathcal{L}_{val})$ under eternal memorization are
all reproduced, with correlations of 0.96–0.99 between the two curves on the mini-batch runs.

Two **quantitative** caveats:

1. The claim in Sec. 4.2 / 5 that the $S_5$ collapse plateaus at $E \approx 4$, matching the
   minimal faithful irreducible representation of $S_5$, is estimator-dependent. Over the window
   $t \in [6500, 7500]$ the Levina–Bickel curve reads 4.8–5.8 while MacKay–Ghahramani reads
   2.6–3.3, i.e. below the algebraic floor of 4 rather than at it.
2. The correction removes ceiling artifacts. On the full-batch baseline with the DMI-selected
   delay, Levina–Bickel saturates the $E = 15$ clamp in 4 of 45 windows while MacKay–Ghahramani
   peaks at 2.13 — those spikes were the arithmetic mean chasing a handful of near-duplicate
   points, not a genuine high-dimensional excursion. The `grokking_dimension` figure avoids them
   only because it uses $\tau = 1$.

### Applicability

The correction is exact for the model Levina–Bickel already assume (i.i.d. samples from a density
on a $d$-manifold, Poisson-approximated neighbour counts). It makes no extra assumption, so it is
applicable wherever the base estimator is. The one caveat is shared by both: delay vectors from a
trajectory are serially correlated, so the effective sample size is below $n$ — this makes the
$-1$ term in the numerator meaningless at these window sizes ($n(k-1) \approx 1200$–$6000$), but
it does not affect the pooling argument itself. That serial correlation is what the Theiler window
below addresses.

## The Theiler window

Theiler (1986) observed that in an oversampled trajectory the nearest neighbours of $x_t$ are
$x_{t\pm1},\dots,x_{t\pm k}$ — close in *time*, hence close in space by continuity alone. They
sample the local **tangent direction**, not the attractor's invariant measure. The fix is to
exclude every candidate with $|i-j| \le W$ from the neighbour search.

`theiler_window` accepts `0` (off), `"embedding"` — the span of one delay vector,
$(E_{max}-1)\tau = 14$ here — `"autocorr"`, or an explicit lag. `mle_mg_theiler` is
MacKay–Ghahramani pooling with `"embedding"`; `mle_mg_theiler_acf` uses `"autocorr"`.

```bash
python reproduce_figures.py --compare mle_mg mle_mg_theiler
```

### Why it is not optional here

Section 3.2 of the paper rejects FNN and Cao's method with precisely Theiler's argument: *"In the
absence of recurrence, the nearest spatial neighbor of a point $x_t$ becomes its own temporal
neighbor $x_{t-1}$ […] yielding a trivial estimate of $\hat d = 1$."* It then credits MLE with
being immune because it *"does not require global recurrence"*. But absence of recurrence is
exactly the regime in which all $k$ neighbours are temporal. Levina–Bickel does not need
recurrence to **return a number** — it needs recurrence for that number to be **about the
attractor**.

The failure mode is quantitative, not vague. For a locally straight, uniformly sampled trajectory
every $r_j \propto j$, so

$$\hat d \;=\; (k-1)\Big/\textstyle\sum_{j=1}^{k-1}\log(k/j) \;=\; 1.227 \quad (k=5).$$

The $WD=0$ weight-norm controls in the paper sit at $E = 1.33$–$1.40$ — the tangent-line constant,
plus a little curvature. Meanwhile the measured autocorrelation time inside each 300-sample window
is **56–66 samples** for `weight_norm` and `val_loss`, against an embedding window of 14: these
series are oversampled by roughly an order of magnitude relative to their own decorrelation time.

### What it changes

| Figure | Observable | LB (paper) | MG | MG + Theiler | corr(LB, MG+Theiler) |
| --- | --- | --- | --- | --- | --- |
| `mod_wd1` | `weight_norm` | 1.39 – 3.46 | 1.39 – 2.99 | 3.11 – 10.18 | **−0.76** |
| `mod_wd0` | `weight_norm` | flat 1.33–1.38 | flat 1.33–1.38 | 8.84 – 12.93 | — |
| `s5_wd1` | `weight_norm` | 1.34 – 3.94 | 1.34 – 3.08 | 6.39 – 13.35 | **−0.67** |
| `s5_wd0` | `weight_norm` | flat 1.36–1.40 | flat 1.36–1.39 | 10.01 – 11.43 | — |
| `s5_wd1_val_loss` | `val_loss` | peak 13.32 | peak 9.34 | peak 12.52 | +0.75 |
| `s5_wd0_val_loss` | `val_loss` | 3.15 → 8.61 | 2.41 → 6.11 | 7.15 – 10.86 | +0.01 |
| `grokking_dimension` | `train_loss` | 4.93 → 2.29 | 2.65 → 1.88 | 5.27 → 2.09 | +0.76 |

Read that table by observable, not by row:

* **`weight_norm` (Figs. 1 and 2 of the paper) does not survive.** The Theiler-corrected curve is
  *anti*-correlated with the published one. In `s5_wd1` the paper's $E$ falls 3.94 → 1.34 across
  grokking; with temporal neighbours excluded it has a local minimum of 6.39 at the grokking step
  and then **rises** to 13.25. The controls move from a dead-flat 1.35 to a high, flat 10–11. So
  the published weight-norm signal tracks how locally *straight* the $\lVert w\rVert_2$ series is
  within 5 samples — which does drop as the norm decays monotonically under weight decay — rather
  than the dimension of a reconstructed attractor.
* **`train_loss` and `val_loss` largely survive.** These series are rough at short lags, so
  temporal neighbours were never the whole story: correlations stay at +0.75, the full-batch
  baseline keeps its 5.3 → 2.1 collapse, and the $WD=0$ control stays high (7.2–10.9) — arguably a
  *cleaner* version of the paper's "escalating chaos of eternal memorization" than the uncorrected
  3.2 → 8.6 ramp.
* **The flat $E=1.00$ tail of Fig. 3a is not a measurement.** After step ~9000 the $S_5$ `val_loss`
  windows have $\sigma \approx 10^{-8}$, below the `DEGENERATE_STD` guard, so `E = 1` is returned
  by definition. That is the degenerate-observable regime the paper discusses in Sec. 5, and it
  should not be read as a collapse to dimension 1.

### The `"autocorr"` variant is a diagnostic, not a recommendation

Sizing $W$ from the decorrelation time (~65) inside a 300-sample window leaves ~155 candidate
neighbours drawn from the far ends of the window. The estimate then rises toward the $E = 30$
clip (2 of 55 windows for `s5_wd1`).

## Is any of this a *measurement*? — `identifiability_ratio`

An intrinsic dimension exists only if the estimate is insensitive to the size of the space it is
measured in. `embedding_dimension_scan` sweeps $E_{max}$ and `identifiability_ratio` reports
$E(2E_{max})/E(E_{max})$: **≈1 means the number is a property of the data, ≈2 means it is a
property of the embedding.**

| series | $E_{max}$=5 | 10 | 15 | 20 | 25 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| Lorenz-63, 11 000 samples (true $d \approx 2.06$) | 2.11 | 2.15 | 2.17 | 2.17 | 2.17 | 2.19 |
| Lorenz-63, 300 samples | 2.80 | 3.40 | 3.99 | 4.02 | 3.97 | 4.16 |
| white noise, 300 samples ($d = \infty$) | 4.76 | 9.32 | 13.10 | 16.31 | 16.31 | 18.95 |
| `s5_wd1` `weight_norm`, 300 samples | 3.31 | 5.49 | 7.32 | 10.07 | 11.78 | 12.66 |
| `s5_wd0` `weight_norm`, 300 samples | 4.59 | 8.07 | 11.16 | 14.44 | 17.30 | 20.25 |

And as a function of how much data you give it:

| window | Lorenz ratio | `s5_wd1` `weight_norm` ratio |
| --- | --- | --- |
| 300 | 1.18 | 1.30 |
| 1200 | 1.09 | 1.99 |
| 2400 | **1.01** | 2.02 |
| 3000 (the entire run) | 1.01 | **2.00** |
| 11 000 | 0.99 | — |

Lorenz-63 converges to a flat scan at $E = 2.07$ once it has ~2400 samples. The weight-norm logs
never do: the ratio sits at 2.0 even when the whole 3000-sample run is used as a single window, so
$E$ is just tracking $E_{max}$.

**So the Theiler-corrected weight-norm numbers (6–13) are not a better estimate — they are the
signature of no resolvable manifold.** The uncorrected numbers (1.2–3.9) looked stable only
because they were pinned to the tangent-line constant. Both are artifacts; neither is a dimension.
Note also that 300 samples cannot resolve even a textbook 2-D attractor (Lorenz gives 2.8–4.2 with
a 1.18 ratio at that length), so the window size is independently too small regardless of the
series.

### Both plateaus are closed-form constants of a straight line

The sharpest version of the point. For a locally straight, uniformly sampled trajectory the $k$
nearest neighbours sit at $|\Delta t| = 1 \dots k$ — or at $W\!+\!1 \dots W\!+\!k$ once a Theiler
window of $W$ is applied — so $r_j$ is a known integer sequence and the estimate is a constant
that depends only on $k$ and $W$, with no input from the data at all:

| | $k$ | closed form | measured on `np.linspace(0,1,2000)` | measured on the WD=0 control runs |
| --- | --- | --- | --- | --- |
| no Theiler | 5 | **1.33** | 1.33 | `mod_wd0` 1.33–1.38, `s5_wd0` 1.36–1.40 |
| no Theiler | 10 | 1.38 | 1.38 | — |
| Theiler $W=14$ | 5 | **10.76** | 10.67 | `s5_wd0` 10.01–11.43, `mod_wd0` 8.84–12.93 |
| Theiler $W=14$ | 10 | 7.85 | 7.72 | — |

A straight line — no dynamics of any kind — reproduces both published plateaus to two decimals.
This is checked in `test_a_straight_line_reproduces_both_published_plateaus`.

It also answers the "why does $E$ grow *after* grokking?" question directly. Post-grokking the
weight norm becomes an almost perfect smooth decay: over `s5_wd1` the detrended residual falls
from 23 % of the window's standard deviation to 3 %. The series gets **straighter**, so the
Theiler estimate climbs toward its line constant of 10.76 and the un-Theilered one settles onto
1.33. Neither movement says anything about the network — it tracks the smoothness of
$\lVert w \rVert_2$.

## Two further correctness issues found by audit

### Centre labelling is not causal

`sliding_dimension` attributes each window to its centre, as the original notebooks do. A window
centred on step $t$ contains data up to $t + W/2$, so as an **early-warning** signal it reads the
future:

| figure | window | sampling | look-ahead | grokking at |
| --- | --- | --- | --- | --- |
| `mod_wd1` | 300 | 10 steps/sample | **+1500 steps** | 13810 |
| `s5_wd1` | 300 | 5 steps/sample | **+750 steps** | 6735 |
| `grokking_dimension` | 1500 | 1 step/sample | **+750 steps** | 3270 |

`label_position="right"` pins each estimate to the last step it actually saw, which is the only
admissible choice for a predictor; `"center"` remains the default so the published figures still
reproduce. On the full-batch baseline the $\mathcal{L}_{train}$ dimension only settles at its floor
around step 3750 with centre labels and 4499 with causal ones — in both cases *after* the grokking
step of 3270, so the appendix's "drop preceding the jump in validation accuracy" does not survive
either convention.

### $\tau = 1$ is a redundant embedding

The paper fixes $\tau = 1$, so 15 delay coordinates span 14 samples against an autocorrelation
time of ~60: the coordinates are nearly collinear, which is the classic redundancy regime that
delayed mutual information exists to avoid. `select_tau_dmi` asks for $\tau = 3$–$8$ on these
windows. Re-running with $\tau = 6$ does not rescue anything — `s5_wd1` `weight_norm` goes to
$E = 15.00$ at an identifiability ratio of 2.00, i.e. straight to the ceiling. The small, stable
numbers at $\tau = 1$ come from the redundancy, not from structure.

### Known-fragile, left as-is for fidelity

* `DEGENERATE_STD` returns a **fabricated** `E = 1.0` for near-constant windows rather than `NaN`,
  which is what draws the flat tail in Fig. 3a.
* `E_FLOOR = 1.0` clips sub-1 estimates, hiding the pathology where a noiseless periodic series
  drives Levina–Bickel to ~0.15.
* The dither is absolute (`1e-9`) rather than relative to the series scale.
* `delay_embedding` uses the forward convention $[X(t), X(t+\tau), \dots]$ where the paper writes
  the backward one; the two differ by a reflection and a time shift, so dimension estimates are
  unaffected.

## Best-practice re-analysis — `best_practice.py`

`reproduce_figures.py` reproduces what was published. `best_practice.py` asks what the data
supports. Since the audit proved that **no** window length available in these runs makes the
absolute dimension identifiable, buying identifiability with a wider window costs the only thing
that matters — temporal localization — and gains nothing. So this module optimises for
**detection**, not measurement: the smallest window that still supports the Theiler exclusion,
every free correction switched on, and the absolute level never quoted.

| knob | paper | `best_practice.py` |
| --- | --- | --- |
| delay $\tau$ | 1 (redundant embedding) | first minimum of the delayed mutual information |
| window $W$ | 300 samples | smallest $W$ leaving $\ge 5k$ candidates after the Theiler exclusion |
| neighbours $k$ | 5 | 10 |
| label position | window centre (acausal) | right edge (causal) |
| tie-breaking | $+\mathcal{N}(0, 10^{-9})$ dither | none; coincident points dropped |
| $\sum\log(r_k/r_j)$ floor | clamped at $10^{-5}$ | none |
| reported $E$ | clipped to $[1, 30]$ | raw |
| variance-free window | fabricated $E = 1$ | `NaN`, absent from the plot |
| control runs | analysed separately | share the treatment run's $\tau$ and $W$ |

```bash
python best_practice.py              # all runs -> figures/best_practice/
python best_practice.py --calibrate  # Lorenz-63 control
```

Each figure carries three diagnostics: dotted **line constants** (what the estimator returns for a
perfectly straight trajectory at this $k$ and $W_{th}$), grey **identifiability shading**, and the
**causal detector** — the first step at which the statistic sits 25 % below the running peak *it
had seen by that step*.

### Control: the pipeline works where an answer exists

Lorenz-63, 11 000 samples, $\tau=11$, $k=10$, true $d \approx 2.06$: LB 2.40, MG 2.18,
MG + Theiler 2.07 with an identifiability ratio of 1.05.

### Detection results

| run | observable | $\tau$ | $W$ | first estimate | LB | MG | MG + Theiler |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mod_wd1` | `weight_norm` | 5 | 200 | step 1990 | **+3180** | **+2700** | +9100 |
| `mod_wd0` | `weight_norm` | 5 | 200 | step 1990 | silent ✓ | silent ✓ | **false positive** |
| `s5_wd1` | `weight_norm` | 2 | 150 | step 745 | **+3980** | **+4040** | +4640 |
| `s5_wd0` | `weight_norm` | 2 | 150 | step 745 | silent ✓ | silent ✓ | silent ✓ |
| `s5_wd1_val_loss` | `val_loss` | 2 | 150 | step 745 | +4100 | +3830 | +3380 |
| `s5_wd0_val_loss` | `val_loss` | 2 | 150 | step 745 | **false positive** | **false positive** | **false positive** |
| `grokking_dimension` | `train_loss` | 12 | 400 | step 399 | +2215 (unstable) | +743 | +2759 (unstable) |

(A signed number is the causal lead over the grokking step; "silent" means the detector never
fired, which is the correct answer for a control run.)

* **Levina–Bickel and MacKay–Ghahramani on the weight norm give a working detector**: 2700–4000
  optimization steps of causal lead on both grokking runs, and neither control fires. This is the
  paper's claim, and at these settings it holds — but see the caveat below about *what* is being
  detected.
* **MG + Theiler is the wrong tool for detection at these window sizes.** It false-positives on
  `mod_wd0` and its range (4.8–29.3) shows it is unstable when only ~64 candidate neighbours
  survive the exclusion. Theiler is right for *measuring*; it costs too many neighbours to be a
  detector in a 200-sample window.
* **`val_loss` is not specific.** It fires early on the grokking runs but also on the $WD=0$
  control, for all three estimators. As an early-warning signal the weight norm is the better
  observable — which is the paper's practical conclusion, reached for a different reason.
* **Removing the clamps exposes a Levina–Bickel instability that MacKay–Ghahramani fixes.** On the
  full-batch baseline, LB exceeds $3\times$ MG in 119 of 913 windows and spikes to 299.8, while MG
  stays inside 1.12–1.62. The published clamps (`E` clipped to $[1, 30]$, $\sum\log$ floored at
  $10^{-5}$) were hiding this, not preventing it.

### The caveat that survives all the tuning

The weight-norm curves still rest on their straight-line constants — 1.27–1.38 against a
no-Theiler constant of 1.38 — and 90–100 % of `s5` windows still fail the identifiability check.
So the detector works, but what it detects is a change in the local *smoothness* of
$\lVert w\rVert_2$, not a collapse of an attractor's dimension. That is a legitimate early-warning
signal with a real lead time and no false positives on the weight-norm controls; it is not
evidence of a topological transition. Both statements are supported by these figures, and only the
first is supported by the data.

## Local trajectory dimension — `local_dimension.py`

All of the above estimates the dimension of the *attractor* the trajectory fills, which needs
recurrence — and these logs have none. A different and better-posed question is local: **over ~100
consecutive iterations, how many independent directions does the optimizer actually use?** That is
defined on a transient, needs no recurrence, and is answered by the singular spectrum of the
delay-embedded segment (Broomhead–King 1986) rather than by nearest neighbours. The summary is the
participation ratio $1/\sum p_i^2$ of the normalised squared singular values.

Switching estimator changes two things:

* **The window can finally be short.** 100–200 iterations is 10–40 logged samples — hopeless for
  any neighbour-based estimate, ample for an SVD.
* **A redundant embedding becomes correct.** $\tau = 1$ with $E = W/2$ is right here, the opposite
  of what Levina–Bickel wants, because the SVD sorts the redundant directions out rather than
  being fooled by them.

And the reference value is exact instead of empirical: a locally straight trajectory gives
**PR = 1**, a planar arc 2, white noise the full embedding dimension. No calibration needed.

```bash
python local_dimension.py              # 200-iteration segments -> figures/local_dimension/
python local_dimension.py --iters 100
```

### Results (segment = 200 iterations, detector = PR > 1.10 sustained for 500 iterations)

| run | observable | samples | PR range | verdict |
| --- | --- | --- | --- | --- |
| `mod_wd1` | `weight_norm` | 20 | 1.00 – 4.78 | fires at 2690, **lead +11 120** |
| `mod_wd0` | `weight_norm` | 20 | 1.00 – 1.03 | **silent** ✓ |
| `s5_wd1` | `weight_norm` | 40 | 1.00 – 7.63 | fires at 1425, **lead +5310** |
| `s5_wd0` | `weight_norm` | 40 | 1.00 – 1.04 | **silent** ✓ |
| `s5_wd1_val_loss` | `val_loss` | 40 | 1.00 – 11.39 | fires at 235, lead +6500 |
| `s5_wd0_val_loss` | `val_loss` | 40 | 1.03 – 10.67 | **false positive** at 1170 |
| `grokking_dimension` | `train_loss` | 200 | 1.00 – 47.29 | fires at 1229, lead +2041 |

This is the cleanest separation anywhere in this project, and it is a *qualitative* one rather
than a threshold on a fitted number:

* **The controls sit at PR = 1.00–1.04 for the entire run.** Without weight decay the weight norm
  is locally a straight line, to four significant figures, for 15 000–20 000 steps.
* **The grokking runs leave PR = 1 thousands of steps before generalization** — `s5_wd1` reaches
  3–7.6 between steps 1500 and 5500 with grokking at 6735 — and then **collapse back to exactly
  1.00** once the network has generalized (from ~7900 in `s5_wd1`).

So the honest statement of the phenomenon is: under weight decay the norm trajectory develops
locally multi-directional structure during the search phase and returns to a one-dimensional path
after grokking; without weight decay it never leaves the one-dimensional path at all. That is a
statement about the optimizer's local exploration, it is exactly measured with a known reference
value, and the WD=0 controls make it specific.

### What this is, and what it is not

It is worth being precise about the method, because the name invites over-claiming.

**It is not SSA.** Singular spectrum analysis builds the trajectory matrix, takes its SVD, *groups
the components, and diagonal-averages them back into subseries* — a decomposition, which is what
SSA is normally used for. Only the first two steps are used here; nothing is grouped or
reconstructed. The correct name is Broomhead–King singular-system analysis, and the statistic is an
**effective rank**.

**Three limits on reading it as a dimension:**

* **It is linear.** The SVD gives the dimension of the smallest linear subspace containing the
  segment, not of the manifold it lies on. A circle is a 1-manifold but scores 2 — which is exactly
  why a sinusoid gives 2.
* **It is energy-weighted, so it is not even the rank.** Two sinusoids of unequal amplitude have
  rank 4 but score ~2.4.
* **The level depends on the embedding.** Mean PR on `s5_wd1` moves 1.27 → 1.35 → 1.46 as the
  embedding goes $W/4 \to W/3 \to W/2$. Only changes are meaningful.

**And the null model wins.** `local_roughness` — the residual of a linear fit, no embedding, no
SVD, three lines — is 0.79–0.95 Spearman-correlated with PR and detects *earlier*:

| run | PR fires | null model fires |
| --- | --- | --- |
| `mod_wd1` | 2690 (lead +11 120) | 2200 (**lead +11 610**) |
| `s5_wd1` | 1425 (lead +5310) | 1175 (**lead +5560**) |
| `mod_wd0`, `s5_wd0` | silent ✓ | silent ✓ |

`local_dimension.py` plots this comparison on every run — three panels: both statistics against
training step with their "locally straight" baselines aligned (PR = 1 coincides with residual = 0,
so one dotted line serves both), the train/val accuracies underneath, and a rank–rank scatter that
shows the Spearman correlation as the rank agreement it actually is. It is also asserted in
`test_local_svd_is_largely_a_roughness_statistic_here`. So the singular spectrum is the
*interpretable* version — it has an exact reference value where a roughness threshold is arbitrary
— but on these logs it is not buying detection power. **The honest description of the phenomenon is
"the weight-norm trajectory departs from local linearity ~5000 steps before grokking, and never
does so without weight decay", not "the attractor dimension rises".**

Two further limits:

1. **`val_loss` is still not specific** — it false-positives on the WD=0 control at every segment
   length tried, as it did for every other estimator here. The weight norm is the observable that
   discriminates.
2. **Logging frequency, not the method, sets the floor.** At a literal 100 iterations `s5_*` still
   works (20 samples, lead +5420) but `mod_*` degrades to 10 samples and the detector fires late.
   Resolving 100 iterations on `mod_*` would need the logs written every step rather than every
   tenth.

## Using the package directly

```python
from edm import load_logs, sliding_dimension, plot_presentation_panels

df = load_logs("grokking_logs/grokking_modular_addition_logs_S_5_with_stochastic.csv")
trace = sliding_dimension(df, target_metric="weight_norm", method="mle",
                          window_size=300, step_size=50, seed=0)
plot_presentation_panels(trace, df, outdir="figures", prefix="s5_wd1")
```

`method` accepts `"mle"`, `"mle_mg"`, `"mle_mg_theiler"`, `"mle_mg_theiler_acf"`, `"fnn"`, `"cao"`
and `"simplex"`; `tau_selector` accepts `"fixed"` (the paper's default, $\tau=1$) and `"dmi"`. For
an arbitrary combination, call the estimator directly — e.g. Levina–Bickel *with* a Theiler window
for an ablation:

```python
from edm import mle_intrinsic_dimension
mle_intrinsic_dimension(series, tau=1, correction="levina_bickel", theiler_window="embedding")
```

Passing a list of traces to any plotting function overlays them on one axis:

```python
traces = [sliding_dimension(df, method=m, seed=0) for m in ("mle", "mle_mg")]
plot_presentation_panels(traces, df, outdir="figures/comparison", prefix="s5_wd1")
```

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
