# E10 — what sets the ceiling: the embedding, or the record?

Section 5.2 of the paper locates a ceiling at about eight directions and leaves two
explanations open, because at the frozen configuration they coincide. The Takens condition
`E > 2d` at `E_max = 20` permits ten components; the Eckmann–Ruelle bound `D ≲ 2 log10 N` at
the frozen window `N = 8000` permits 7.8. Each hypothesis is insensitive to the other's knob,
so two one-dimensional sweeps separate them. This is that experiment: 3 240 scored cells over
`E_max` in {10, 14, 20, 28, 40, 56} at N = 8000, N in {1 000 … 64 000} at `E_max = 20`, and the
record sweep repeated at `E_max = 56` as a control, nine ranks and three seeds throughout.

**Answer.** There are two distinct phenomena and the paper has been calling both "the ceiling".

*The level the estimator reports at high rank is set by the embedding and not by the record.*
At a fixed record of 8 000 samples, raising `E_max` from 10 to 56 lifts the estimate at twenty
directions from 7.31 to 14.51. At a fixed `E_max = 20`, raising the record from 1 000 to 64 000
samples — a factor of 64 — lifts it only from 9.43 to 11.06, and leaves the slope of the
estimate against the truth over the top of the rank range at 0.22 to 0.29 throughout.

*The rank up to which the estimate is actually accurate saturates at about ten components and
neither knob removes it.* Over all sixteen distinct `(E_max, N)` cells the rank at which the
median estimate first falls a full component behind the truth never exceeds 10.4, and its
maximum is attained at the **shortest** record in the grid (`E_max = 56`, N = 1 000).

**H2 is refuted.** The Eckmann–Ruelle bound is exceeded at every cell, most starkly at
`E_max = 56` with N = 1 000, where the estimate reads 13.82 at twenty directions and tracks the
truth to `r ≈ 10.4` against a bound of 6.0. **H1 is right in direction and wrong in form**: the
reported level does rise with `E_max`, but logarithmically, about 2.6 components per doubling,
not as `E_max / 2`. The paper's arithmetic is right at `E_max = 20` by coincidence and wrong at
both ends of the sweep. A third reading fits far better than either and is in section 8. It is
not a comfortable one.

---

## 1. Reproduction check

The published ceiling number is reproduced before anything is swept (`--check`; output in
`check.txt`, per-run values in `check_published.csv`).

Rescoring the twenty-direction frozen configuration (`E_max = 40`, `tau = 16`, `m = 20`, the
autocorrelation Theiler rule, one 8 000-sample window) on the k = 20 trajectories already in
`results/k20_calibration/trajectories`, over the published panel of eight observers and three
seeds:

```
r        2     4     6     8    10    12    14    16    20
MG    2.35  4.41  6.66  7.38  8.99 10.61 11.40 15.40 15.09
truth 2.00  4.00  6.00  8.00 10.00 12.00 14.00 16.00 19.99
```

At r = 20 the estimate is **15.09 against a measured effective rank of 19.99**; the paper
reports 15.1 and 19.99. The check passes.

A second check licenses the design of the record sweep. Each of the 27 long trajectories
written here is compared sample by sample with the corresponding published trajectory over
their common 10 000-sample prefix: **27 of 27 are bit-identical**, maximum absolute difference
exactly zero. The `qp` mode draws no per-step randomness, so lengthening `T` extends the record
without perturbing it, and every `N` in the sweep is a genuine prefix of one run.

## 2. What was swept, and what was held

The system is the one the ceiling result comes from, reused rather than rewritten: the k = 20
adapter on a frozen tanh backbone trained on the scikit-learn digits, driven by `r`
incommensurate loss-weight modulations in a fixed octave at `f0 = 1/16`, the drive whitened by
`pinv(Phi) Q` with the per-mode response gain divided out (`system.py`, `dynamics.py`,
`calibration_k20.make_spec`). The measured effective rank of the trajectory covariance is
recorded for every cell at every record length, and it holds: 2.00, 4.00, 6.00, 8.00, 10.00,
12.00, 14.00, 16.00, 19.99 at N = 8000, and 19.59 at r = 20 even at N = 1000. **The ground
truth is not what moves.**

The estimator is `mg.all_estimators` at the eight-direction frozen configuration of appendix C
— `tau = 4`, `m = 20`, the autocorrelation Theiler rule, dither 1e-9 — with one knob swept at a
time and **no clamping anywhere**. Four observers, one per family (`c_norm`, `g_fro`,
`c_proj1`, `fn_proj1`), three seeds, one window of exactly `N` samples per cell; every number
below is a median over the twelve observer x seed values.

*The record is lengthened, not resampled.* Every `N` is the first `N` post-burn-in samples of
one 64 000-sample run. The sampling rate, the drive frequencies and the delay lag are identical
at every `N`, so a longer record is a longer record and the sweep is not confounded with the
tau-sensitivity of section 6.2.

## 3. Sweep E — `E_max` at fixed N = 8000

Median MG against r, frozen arm:

```
 E_max     r=2   r=4   r=6   r=8  r=10  r=12  r=14  r=16  r=20   Theiler used
    10    2.40  4.26  5.39  5.97  6.18  6.34  6.52  6.58  7.31            36
    14    2.41  4.41  5.71  6.68  7.21  7.45  7.63  7.83  8.71            52
    20    2.36  4.47  5.89  6.94  8.02  8.55  8.82  9.21 10.46            76
    28    2.34  4.60  6.15  7.19  8.44  9.23  9.90 10.37 11.94           108
    40    2.37  4.67  6.37  7.30  8.21  9.64 10.59 11.36 13.34           150
    56    2.37  4.73  6.41  7.44  8.56 10.03 11.08 12.58 14.51           150
```

Three ways of reading a ceiling off those curves:

```
 E_max   Takens E/2   E-R at N=8000   tracking r*   MG at r=20   top slope   PRdelay at r>=16
    10          5.0            7.81          6.55         7.31        0.10               5.33
    14          7.0            7.81          7.38         8.71        0.16               6.90
    20         10.0            7.81          7.87        10.46        0.27               8.51
    28         14.0            7.81          8.50        11.94        0.38              10.12
    40         20.0            7.81          8.56        13.34        0.50              11.99
    56         28.0            7.81          9.00        14.51        0.60              13.53
```

`tracking r*` is the rank at which the median estimate first falls more than one component
below the truth, linearly interpolated between grid points; the 0.5- and 2.0-component versions
are in `ceiling_summary.csv` and behave the same way. `top slope` is the least-squares slope of
the median estimate against r over `r >= 8`: it needs no threshold and no extrapolation, and it
is the least arbitrary of the three, since a true ceiling is a slope of zero and perfect
recovery is a slope of one.

The record is fixed at 8 000 throughout this table, so **the Eckmann–Ruelle bound of 7.81 is a
constant here and cannot explain a column that changes by a factor of six.** Note also that
`tracking r*` rises far less than the reported level does: from 6.55 to 9.00 while `MG at r=20`
doubles. That gap is the two phenomena separating.

## 4. Sweep N — the record at fixed `E_max = 20`

Median MG against r:

```
      N     r=2   r=4   r=6   r=8  r=10  r=12  r=14  r=16  r=20   2log10 N
   1000    2.04  4.21  6.33  6.59  7.48  7.81  8.59  8.28  9.43       6.00
   2000    2.21  4.25  6.59  7.02  7.48  7.73  8.77  8.71  9.73       6.60
   4000    2.40  4.36  6.45  6.60  7.42  8.23  8.65  9.06 10.16       7.20
   8000    2.36  4.47  5.89  6.94  8.02  8.55  8.82  9.21 10.46       7.81
  16000    2.31  4.96  6.47  7.05  8.45  8.57  9.19  9.54 10.64       8.41
  32000    2.34  4.39  6.93  7.64  8.68  8.61  9.27  9.78 10.81       9.01
  64000    2.28  4.44  6.90  7.75  8.90  8.91  9.45  9.95 11.06       9.61
```

```
      N   tracking r*   MG at r=20   top slope   PRdelay at r>=16   rho_ident at r=20
   1000          7.53         9.43        0.22               8.32                1.40
   2000          8.03         9.73        0.23               8.35                1.33
   4000          7.57        10.16        0.29               8.50                1.34
   8000          7.87        10.46        0.27               8.51                1.33
  16000          8.17        10.64        0.27               8.52                1.32
  32000          9.33        10.81        0.25               8.52                1.31
  64000          9.76        11.06        0.25               8.52                1.32
```

A sixty-four-fold change in the record moves the top slope by 0.03, the estimate at twenty
directions by 1.6 components (**0.90 per decade against the 2.0 the bound asserts**) and the
linear delay participation ratio by 0.20. The estimate sits *above* the bound at every record
length: 9.43 against 6.00 at N = 1000, 11.06 against 9.61 at N = 64000.

The tracking ceiling is the one quantity that does respond, rising 1.18 per decade to 9.76 at
N = 64000 — approaching, but not passing, the Takens value of 10 for this `E_max`. On its own
that would look like weak support for a record-length effect saturating against an embedding
limit. Section 5 shows it is not.

## 5. The control that makes the null informative

A flat record sweep at `E_max = 20` proves little on its own, because the Takens bound there is
10 and could be masking a record-length effect. The sweep was therefore repeated at
`E_max = 56`, where the embedding is nowhere near limiting over this rank grid
(`ceiling_raw_e56.csv`, `sweep = N_E56`):

```
      N     r=2   r=4   r=6   r=8  r=10  r=12  r=14  r=16  r=20   2log10 N
   1000    1.93  4.10  5.74  7.55  9.22 10.20 10.45 11.74 13.82       6.00
   2000    2.18  4.38  6.65  7.73  9.06  9.38 10.81 10.73 14.19       6.60
   4000    2.32  4.44  7.00  7.53  9.23  9.83 11.69 12.03 14.23       7.20
   8000    2.37  4.73  6.41  7.44  8.56 10.03 11.08 12.58 14.51       7.81
  16000    2.33  4.54  7.04  7.40  8.62  9.81 11.27 13.09 15.32       8.41

      N   tracking r*   MG at r=20   top slope   PRdelay at r>=16
   1000         10.43        13.82        0.49              13.20
   2000         10.07        14.19        0.50              13.43
   4000         10.33        14.23        0.54              13.55
   8000          9.00        14.51        0.60              13.53
  16000          9.03        15.32        0.68              13.52
```

**At a record of one thousand samples the estimator returns 13.82 at twenty directions and
tracks the truth to within one component up to r = 10.4 — the largest tracking ceiling anywhere
in this study — against an Eckmann–Ruelle bound of 6.0.** Whatever limits this estimator, it is
not the number of points. Over the sixteen-fold record range here the estimate at r = 20 gains
1.10 components per decade and the tracking ceiling *falls* by 1.28 per decade.

Put the two record sweeps together and the picture is unambiguous: the tracking ceiling never
exceeds 10.4 in any of the sixteen cells, and it reaches that value at the shortest record with
the largest embedding. A record-length bound cannot produce that.

## 6. Which functional form fits

Observed slopes beside the slope each hypothesis predicts (`ceiling_slopes.csv`):

```
 sweep         quantity   observed   predicted   units
     E      tracking r*      0.047         0.5   per unit E_max
     E       MG at r=20      0.152         0.5   per unit E_max
     E        top slope      0.011         0.5   per unit E_max
     N      tracking r*      1.175         2.0   per decade of N
     N       MG at r=20      0.895         2.0   per decade of N
     N        top slope      0.017         2.0   per decade of N
 N(E=56)    tracking r*     -1.284         2.0   per decade of N
 N(E=56)     MG at r=20      1.103         2.0   per decade of N
 N(E=56)      top slope      0.158         2.0   per decade of N
```

Fitting the candidate forms to the sixteen distinct `(E_max, N)` cells of the frozen arm,
pooling both sweeps and the control (`ceiling_fits.csv`; RMSE in components, `sd` is the spread
of the quantity being fitted, so a model with RMSE above `sd` is worse than predicting the
mean):

```
 quantity        sd    E/2   2log10N   min   a*(E/2)   b*(2log10N)   both scaled   a*log10E + b*log10N
 tracking r*   1.11  10.76      1.78  1.79      3.44          1.55          1.35                  0.65
 MG at r>=16   2.01   8.54      3.91  3.95      3.57          2.44          1.17                  0.19
 PRdelay       2.64   8.50      3.87  3.81      2.54          3.17          1.14                  0.09
```

Neither hypothesis fits, even with a free multiplicative constant, and neither does their
pointwise minimum — the "both bind" answer, which was a perfectly reasonable prior and is not
what the data shows. What does fit is a form neither hypothesis proposes: the reported level is
**logarithmic in `E_max` and weakly logarithmic in `N`**. For the top-of-grid estimate,
`8.63 log10 E_max + 1.03 log10 N - 5.52`, RMSE 0.19 against a spread of 2.01 — that is +2.60
components per doubling of `E_max` and +1.0 per decade of record. For the tracking ceiling the
same form gives `3.89 log10 E_max + 0.51 log10 N + 1.10`, RMSE 0.65 against a spread of 1.11:
better than either hypothesis, but the residual is over half the spread, which is the
quantitative statement of the accuracy wall.

## 7. Three things that could have produced this and did not

**The Theiler exclusion.** The autocorrelation rule ties the exclusion to `(E-1) tau`, so
sweeping `E_max` sweeps the exclusion too (36 samples at `E_max = 10`, truncated at the
implementation cap of 150 above `E_max = 28`). A second arm holds it fixed at 76 for every
`E_max`. The two arms agree to the second decimal in every cell — top slope
0.10 / 0.16 / 0.27 / 0.38 / 0.50 / 0.60 in both — so the exclusion is inert here, exactly as
appendix C reports for the calibration grid. The realised exclusion is on every row of
`ceiling_raw.csv`.

**Degeneracy.** Zero of 3 240 scored cells raise the degeneracy flag, at any record length or
embedding dimension. No cell was dropped from any median.

**The construction losing its rank.** The measured effective rank equals `r` to within 0.01
components at every `N >= 2000` and to within 0.41 at `N = 1000` (r = 20). The resonance margin
of the frequency set demands a window longer than 1/margin, which is 192 samples at `r <= 6` and
1 274 at `r >= 16`; only the `N = 1000` cells at `r >= 14` fall below that, and they are the
cells the argument least depends on — dropping them strengthens it, since they are the ones
where a short record does best.

## 8. What the answer probably is, and the strongest objection to it

Two observations complicate the clean "it is the embedding dimension" reading. Both are in the
tables above.

**The linear null does the same thing.** `PRdelay`, the participation ratio of the delay
covariance, is a purely linear spectral statistic that knows nothing about manifolds,
neighbours, Takens or Eckmann–Ruelle, and whose only hard limit is `E` itself. Its ceiling rises
with `E_max` exactly as MG's does — 5.33 to 13.53 across the E sweep, against MG's 6.94 to 13.55
— and is flat in `N` to two decimals (8.32 to 8.52 across a 64-fold record; 13.20 to 13.52
across the 16-fold record at `E_max = 56`). It is fitted by `11.09 log10 E_max` with RMSE 0.09,
the tightest fit of anything measured here. So most of what is being called "the ceiling" is
reproduced by counting how many components the delay window resolves *linearly*. This sweep does
not support a geometric reading of the ceiling; it supports a spectral one, and it is the same
objection the audit of `exp10-12` already raised against MG in general.

**`E_max` cannot be moved on its own.** Raising it at fixed `tau` also lengthens the delay span
`(E-1) tau` from 36 to 220 samples, i.e. from about three drive periods to about twenty. A third
arm holds the span at its frozen 76 samples and lets `tau` fall (8, 6, 4, 3, 2, 1) as `E_max`
rises. In that arm the top slope reads 0.19, 0.18, 0.27, 0.30, 0.29, 0.19 — non-monotone, with a
regression slope against `E_max` of 0.000. **At fixed delay span, `E_max` buys nothing.** That
arm has its own defect: `tau = 1` at `E_max = 56` is oversampled and `tau = 8` at `E_max = 10`
is close to aliasing the fastest mode, and those two are exactly the endpoints that fall. It
should not be read as proving the converse. But taken with the other two arms, the operative
variable looks like the temporal extent of the delay window rather than the number of delay
coordinates — and that is not what `E > 2d` is about either.

**The diagnostics behave as designed.** The identifiability ratio
`rho_ident = MG(2 E_max) / MG(E_max)` falls monotonically as the embedding is enlarged — median
over r of 1.22, 1.10, 1.11, 1.07, 1.03, 1.02 for `E_max` = 10 to 56 — and within each `E_max` it
rises with `r`, crossing the 1.15 admissible band at about the rank where the estimate stops
tracking (at `E_max = 20`, N = 8000: 0.96, 1.01, 0.94, 1.12, 1.07, 1.12, 1.13, 1.22, 1.33 for
r = 2 to 20). At r = 20 it is 1.31–1.40 at *every* record length, i.e. the record does not make
an unidentifiable estimate identifiable. A practitioner watching `rho_ident` alone would have
been warned in exactly the cells where the ceiling bites. The trend-crossing count scales
linearly with the record (200, 402, 807, 1 609, 3 234, 6 490, 12 978 for N = 1 000 to 64 000),
confirming the orbit is recurrent at every record length, which is the condition the neighbour
statistic needs.

## 9. What a paper could quote

- **Ceiling per `E_max`** at N = 8 000, tracking to within one component: 6.6, 7.4, 7.9, 8.5,
  8.6, 9.0 at `E_max` = 10, 14, 20, 28, 40, 56. Estimate at twenty directions: 7.3, 8.7, 10.5,
  11.9, 13.3, 14.5. Slope of the estimate against the truth over `r >= 8`: 0.10, 0.16, 0.27,
  0.38, 0.50, 0.60.
- **Ceiling per `N`** at `E_max = 20`: 7.5, 8.0, 7.6, 7.9, 8.2, 9.3, 9.8 at N = 1 000 to 64 000.
  Estimate at twenty directions: 9.4, 9.7, 10.2, 10.5, 10.6, 10.8, 11.1. Slope over `r >= 8`:
  0.22, 0.23, 0.29, 0.27, 0.27, 0.25, 0.25.
- **The control**: at `E_max = 56`, N = 1 000, the estimate is 13.8 at twenty directions and the
  tracking ceiling is 10.4 — its maximum over the whole grid, at the shortest record — against
  an Eckmann–Ruelle bound of 6.0.
- **Observed against predicted slope**: 0.047 components per unit `E_max` where Takens predicts
  0.5; 0.90 components per decade of record for the reported level and 1.18 for the tracking
  ceiling, where Eckmann–Ruelle predicts 2.0; and −1.28 per decade for the tracking ceiling at
  `E_max = 56`.
- **The best-fitting form** for the reported level: `8.63 log10 E_max + 1.03 log10 N - 5.52`,
  RMSE 0.19 on sixteen cells whose spread is 2.01; the Takens form scores RMSE 8.54 and the
  Eckmann–Ruelle form 3.91 on the same points, and their pointwise minimum 3.95.
- **The reproduction**: 15.09 against a measured 19.99 at the published twenty-direction
  configuration, versus the reported 15.1 and 19.99.

## 10. What this does not settle

- **One construction, one drive regime, one observer panel.** Everything here is the fast
  quasiperiodic arm of the digits adapter at `f0 = 1/16` with four observers. The paper's slow
  arm and its real training logs sit on the other side of the tau line, where the delay window
  covers a fraction of a period rather than twenty periods; nothing measured here transfers to
  them, and section 8 suggests the delay span is precisely what matters.
- **`E_max`, `tau` and the delay span are algebraically locked**, so no arm isolates the
  embedding dimension. Section 8 is the best this design can do, and it points away from
  `E > 2d` rather than towards it.
- **The rank grid stops at 20**, because the construction has twenty available directions. At
  `E_max` = 40 and 56 the reported level has not saturated within the grid, so those two columns
  of `MG at r=20` are censored from above and should be read as lower bounds.
- **Why the accuracy wall sits at about ten components is not explained.** It is not `E_max / 2`
  (it is 10.4 at `E_max = 56`), it is not `2 log10 N` (it is 10.4 at N = 1 000), and it is not
  the measured effective rank (which is exact to 0.01). It is the one number in this study
  neither hypothesis and no fitted form accounts for.
- **Whether MG is measuring geometry at all** is not settled here, and is made *less*
  comfortable by this experiment: a linear spectral statistic reproduces the same ceiling, with
  a tighter fit, on the same data.

---

### Files

| file | what |
| --- | --- |
| `check.txt`, `check_published.csv` | the reproduction of the published ceiling and the prefix identity |
| `ceiling_raw.csv` | 2 700 rows, one per (sweep, arm, `E_max`, N, r, seed, observer) |
| `ceiling_raw_e56.csv` | 540 rows: the record sweep repeated at `E_max = 56` |
| `ceiling_cells.csv` | medians per (sweep, arm, `E_max`, N, r) |
| `ceiling_summary.csv` | the ceiling measures per (sweep, arm, `E_max`, N) |
| `ceiling_slopes.csv` | observed slope against the slope each hypothesis predicts |
| `ceiling_fits.csv` | RMSE of each functional form |
| `ceiling.png` | the estimate against the truth for each knob, and the ceilings against the two predictions |
| `trajectories/` | 27 records of 64 000 samples; regenerate with `--simulate` |

Reproduce with, from `code/active_dimension`:

```
python e10_ceiling_sweep.py --check --simulate
python e10_ceiling_sweep.py --score --resume
python e10_ceiling_sweep.py --score --arms "" --n-sweep-E 56 \
    --n-grid 1000,2000,4000,8000,16000 --out ceiling_raw_e56.csv
python e10_ceiling_sweep.py --analyse --figure
```
