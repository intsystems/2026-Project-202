# Does MG measure the active complexity of a training run?

Real data (sklearn digits), a frozen nonlinear backbone, a trainable adapter confined to
k = 10 directions, and an *active* dimension that is set by construction and then **measured**
rather than assumed. Seven excitation arms, 252 sweep runs, 24 transition runs, 60 control
runs, plus a synthetic identifiability atlas and this project's own seven 120 000-step
training logs. Code, CSVs and figures in this directory; `results/tables.txt` regenerates
every table below.

**One-line answer.** MG does measure the active dimension — with MAE 0.87 on held-out r and
held-out seeds — but only for deterministic, recurrent dynamics whose oscillation period the
delay lag has been matched to, and none of those conditions holds for a training log.

The hypothesis under test, as posed: *the MG dimension of a delay reconstruction of a good
1-D observer reflects the number of actively expressed degrees of freedom of the dynamics,
and can serve as an indicator of the system simplifying.*

The answer separates into three claims that behave very differently, and into a prior
question that has to be settled first.

---

## 0. The prior question: when does an active dimension exist to be found?

A delay embedding recovers the dimension of an invariant set. Takens' theorem, and the
Sauer–Yorke–Casdagli refinement, require a **deterministic** flow on a compact invariant set.
Three things an optimiser near a solution can be doing, and what each does to that
requirement:

| the dynamics | the delay-embedded object | the dimension it has |
| --- | --- | --- |
| deterministic and **recurrent** (quasi-periodic on an r-torus) | a diffeomorphic copy of the r-torus | **r** — there is a right answer |
| deterministic and **transient** (descent to a fixed point) | a 1-D curve parameterised by t | **1**, for every r |
| **stochastically** driven (mini-batch or injected gradient noise) | the state *and* the last E−1 innovations | full rank in R^E — no r-manifold exists |

The third row is the decisive one for this project, because mini-batch gradient noise is
exactly that case. Stark, Broomhead, Davies and Huke (*Nonlinear Analysis* 30:5303, 1997;
*J. Nonlinear Sci.* 13:519, 2003) do prove embedding theorems for stochastically forced
systems — but what they embed is the state *together with the noise history*, a skew product,
not an r-dimensional attractor. Osborne and Provenzale (*Physica D* 35:357, 1989) showed that
coloured noise alone yields a finite, reproducible, and entirely spurious correlation
dimension set by the spectral exponent; Theiler (1991) made the same point for filtered
noise. That is the failure mode here.

**E0 measures all three.** Same nominal r, same scalar observer, same matched one-octave
band, Theiler window on, `clamp_to_max_E` off, `tau = 1`, `max_E = 20`:

```
MG against the true active dimension                r=1    2      3      4      5      6      8
deterministic torus     (N = 20 000)               1.38   2.53   2.67   2.82   2.87   3.03   2.85
white-driven OU         (tau_c = 200, N = 20 000)  11.71  11.68  11.64  11.82  11.71  11.74  11.59
band-limited noise      (tau_c = 200, N = 20 000)   3.20   2.94   2.99   3.09   2.97   3.17   3.18

identifiability ratio MG(2 max_E) / MG(max_E)   -- ~1 identifiable, >1.15 not
deterministic torus     1.00   1.06   1.12   1.18   1.16   1.20   1.17
white-driven OU         1.45   1.45   1.43   1.46   1.44   1.44   1.43
band-limited noise      1.34   1.24   1.40   1.20   1.26   1.32   1.29
```

The nominal r on the torus row **is** its measured active dimension: `e0_atlas` records the
state participation ratio at 1.0000, 2.0000, 2.9988, 3.9972, 4.9911, 5.9862, 7.9849 for
r = 1…8, so nothing is being assumed there.

The two stochastic rows are flat in r. Measured spread of MG across r = 1…8 *within* each
(correlation time, record length) cell: 0.20–0.59 for OU in all six cells, and 0.10–0.52 for
the band-limited family in five of six. The exception is the hardest cell — longest
correlation time with the shortest record (`tau_c` = 1000, N = 20 000) — where the spread is
4.91; but the values there are 5.51, 6.53, 5.38, 4.43, 5.36, 6.85, 6.50 across r = 1…8, which
is scatter, not a trend. What the rows *do* move with is `max_E`, `tau_c` and N: white-driven
OU reads 12.0–13.5 at `tau_c` = 50 and 11.0–12.2 at `tau_c` = 1000, with r making no
difference inside either. **For stochastically excited dynamics MG
carries no information about the number of active directions.** The identifiability ratio
says so without needing a ground truth at all: at 1.44 the estimate is a property of the
embedding space, not of the data.

The nulls behave as they should on these rows: `roughness` is flat in r (0.095–0.111 at
`tau_c` = 200) and moves only with `tau_c`, which is the smoothness axis. The spectral
participation ratio is *not* a usable null here — on OU it reads 6–330 and tracks the record
length, because it is counting bandwidth.

### 0.1 The torus row is a statement about tau, not about MG (E6)

The deterministic row saturating at ≈3 is a property of `tau = 1`, not of MG. The delay
window spans `(max_E − 1)·tau` samples and must cover a real fraction of the oscillation
period, or the delay coordinates are collinear and the torus never unfolds. Sweeping tau on
a fixed period-400 torus, `max_E = 20`:

```
tau   delay span      r=1     2      3      4      6      8      (truth 1 2 3 4 6 8)
  1   0.05 periods   1.42   2.44   2.49   2.63   2.44   2.60
  2   0.10           1.42   2.63   2.73   3.10   2.99   3.20
  4   0.19           1.43   2.72   3.31   3.83   3.94   4.29
  8   0.38           1.43   2.77   4.18   6.16   6.57   8.53      <- best
 16   0.76           1.44   3.00   5.92  13.66  12.83  20.76
 32   1.52           1.46   3.06   9.90  33.28  31.02  63.35
```

There *is* a tau that works — the one whose delay window spans about 0.4 of an oscillation
period — and at it MG recovers r to about ±0.8. Either side of it the estimate is wrong by up
to a factor of 24: **across tau alone, with the system unchanged, MG moves by 0.03 at r = 1
and by 61.5 at r = 8.** That is the free-parameter sensitivity `report_0808.md` estimated at
"2.14 components", measured here an order of magnitude larger.

This is the practical obstacle, and it is not a tuning problem that better calibration
solves. The right tau is a function of the signal's oscillation period. In a controlled
experiment that period is known; **in a real training log it is not**, and the estimate at
the wrong tau is not merely noisy but confidently wrong.

### 0.2 The linear nulls: it depends on the observer, and that is the point

`e0b_observer_amplitude.py` runs the **same r-torus** through two scalar observers that
differ only in how evenly they weight the r modes, and scores both statistics against the
measured state participation ratio:

```
observer   MG (best tau)          spectral PR, 256 bands    spectral PR, native res.
equal      MAE 1.19  rho +1.000   MAE 0.35  rho +1.000      MAE 2.52  rho +1.000
generic    MAE 0.34  rho +1.000   MAE 1.61  rho +1.000      MAE 0.87  rho +0.886

per r:                1      2      3      4      6      8
equal    specPR256   1.02   2.15   3.30   4.40   6.57   8.67    <- the null wins here
         MG          1.15   3.21   5.33  10.68   8.93   8.71
generic  MG          1.15   2.22   3.02   3.71   5.23   7.38    <- MG wins here
         specPR256   1.01   1.32   1.59   2.14   3.72   4.55
```

Under an equal-amplitude observer a one-line FFT statistic beats MG by 3.4x; under a generic
one — random weights, which is what a random projection of a network's parameters actually is
— MG beats it by 4.7x. And on the real network, with both calibrated on the same split by the
same objective (§2), MG wins on absolute error by about threefold:

```
                            Spearman vs measured active dim     MAE
MG                                     +0.908                  0.324
spectral PR (native resolution)        +0.892                  0.924
spectral PR (1024 bands)               +0.917                  2.159
linear PR of the delay matrix          +0.783                  2.096
roughness  (must NOT track r)          -0.417                    --
```

So the honest statement is not "a one-line statistic beats MG" but: **which statistic wins
depends on the observer, and the ordering statistic is a near-tie in every case** (Spearman
+1.000 for both in three of the four cells above). What separates them is absolute recovery,
and only after both have been calibrated the same way.
The audit of `exp10`–`exp12` found the linear delay-matrix PR beating MG on *their* data
(MAE 1.03–1.23 against 1.29–1.46, rho 0.994–1.000 against 0.949–0.985) and no experiment in
that suite reported it; here it is reported on every call and it loses.

---

## 1. What the earlier experiments actually established

Three independent audits were run against `../dimension_recovery/exp10`–`exp15` before any
new code. Each verified its claims by executing reduced versions of the shipped code.

**`exp13` and `exp14` contain no information about optimisation.** Re-running `exp14` at its
winning configuration with the learning rate multiplied by **zero** — no training at all —
reproduces the headline: `gradient_fro` MAE 2.06 → 1.87, Spearman unchanged at +0.98, and the
two series correlate at 0.986–0.998 at every k. The observers read the exogenous drive
through the residual. Separately, the trajectory's participation ratio is 2.9–4.3 while the
nominal k is 10–20, so the *active* dimension was never k; and the roughness null, which
`estimators.py` itself declares mandatory, reaches |rho| 0.85–0.93 against MG's 0.92–0.94 and
was never computed.

**`exp10`–`exp12` tuned the level on the answer.** Their calibration grid mixed estimator
parameters with *system* parameters (`cycles_per_window`, and the learning rate in
exp11/12), so choosing a configuration changed the data-generating process; the objective was
error against the known k on the grid that was then reported. MG at k = 20 moves 5.3 → 16.3
across that grid and the winner sits at its boundary. "Held-out" meant held-out seeds, but
`systems.py` ignores its rng under `band_mode="matched"`, so the frequency geometry — the
thing the estimator responds to — was bit-identical across the split. MG there is a concave
monotone function of k crossing the identity near k ≈ 12–13; MAE ≈ 1.37 is a property of the
interval k ∈ [1, 20], and on k ≤ 40 the same pipeline gives MAE ≈ 7.

**`exp15` v1 and v2 fail for identifiable reasons, and v3 does not fail.** v1's Hessian is
exactly `I/N`, so every mode decays at one rate and the trajectory is a straight line —
participation ratio 1.0000 at every k, and MG of a pure linear ramp at that configuration is
1.330, the measured value to three decimals. v2 ran with `theiler=0` (MG 1.23 against 9.58
with the Theiler window on) and frequencies in exact arithmetic progression
(`f0 − 2f1 + f2 = 5e-20`), which collapses the torus to two dimensions. **v3 is sound**:
held-out MAE 0.504, rho 0.988, `functional_rank = trajectory_rank = update_rank = k` verified
for all 64 configurations, trajectory PR 1.00 → 7.58 for k = 1…8, and its roughness null is
flat (rho +0.05 to −0.40 for the good observers). Its limitation is different from a defect:
the k-dimensionality is injected by an external quasi-periodic teacher the network is slaved
to, and it sits at ≈2.5 samples per cycle — the far side of the tau line in §0.1.

Two silent defects in the estimator itself, in `../grokking_analysis/edm/dimension.py`: an
exactly recurrent series drives delay vectors together and the 1e-8 distance floor and 1e-5
log-ratio floor then return **0.08** or **n(k−1)−1 = 399 975** instead of `nan`; and
`clamp_to_max_E` converts a divergent estimate into `max_E`. The clamp did *not* fire in any
of exp10–exp15 — checked — so it explains none of their results.

---

## 2. Design

**The system.** sklearn digits (1024 training examples, 384 held-out probe examples). A tanh
MLP 64→96→96 trained briefly and then **frozen**; on top of it a linear head confined to a
k = 10 dimensional affine subspace, `theta = theta0 + V^T c` with `V` a fixed (10, P)
orthonormal frame. The loss is softmax cross-entropy, so `c → loss` is nonlinear and the
curvature `H(c)` is a real, data-dependent, anisotropic object (eigenvalues 0.017–0.072,
condition number 4.3). All dynamics happen in the 10-dimensional coordinate `c`; the logs are
scalar functions of it.

**The three dimensions, kept apart.**

| | definition | value here |
| --- | --- | --- |
| available | directions the optimiser may move in | 10, by construction |
| functional | rank of ∂(probe logits)/∂c | measured: rank 10, participation ratio 8.40 |
| **active** | directions actually excited over the window | **measured per run** from the trajectory and update covariances |

Every score on the network (E1–E5) is MG against the **measured active** dimension, and
available k is never used. The synthetic sections E0/E0b/E6 index their tables by nominal r,
which is legitimate there and only there: for r equal-amplitude sinusoids the measured state
participation ratio *is* r, verified at 1.0000–7.9849 for r = 1…8 in `atlas_raw.csv`. E6 does
not carry its own PR column, so its ground truth rests on that verification rather than on a
per-run measurement.

**Why the active dimension is not the injected rank.** Near a minimum
`c_{t+1} − c* = (I − ηH)(c_t − c*) − η ξ_t`, and the stationary covariance
`Σ_j (I−ηH)^j η²Σ (I−ηH)^{jT}` is supported on the smallest `(I−ηH)`-invariant subspace
containing `range(Σ)` — the Krylov space, which for a generic `H` is all of R^10 however
small `rank(Σ)` is. Preconditioning the step by `H^{-1}` makes the linearised dynamics
isotropic and only then does rank-r forcing give rank-r motion. Both are run
(`precondition=True/False`) precisely because the difference *is* the available-vs-active
distinction, made by measurement rather than by assertion.

**Making the drive genuinely r-dimensional.** The data is partitioned into 12 fixed groups
(12 for every r, so the partition is not itself a function of r) and the loss weights of r of
them are modulated by r rationally independent sinusoids filling one octave — one octave for
every r, so the bandwidth does not widen with r. That fix matters: under the obvious
construction where adding a mode adds a *higher* frequency, the one-line ratio
`std(Δx)/std(x)` recovers the ordering of r perfectly, and the experiment proves nothing.

Modulating group j tilts the gradient along some direction `phi_j`; the `phi_j` are neither
orthogonal (random data groups have correlated gradients, condition number 1.0 at r=1 rising
to 11.2 at r=8) nor of equal effect. Left alone, that gives a trajectory whose participation
ratio is far below r — exactly the defect measured in `exp14` (PR 2.9–4.3 at nominal k =
10–20). The fix is a mixing matrix `pinv(Phi) Q` with `Q` an orthonormal basis of
`range(Phi)`, with `phi_j` obtained by central differences of the gradient rather than from
a probe trajectory: an SVD of a probe run recovers each direction only up to sign, and an
unknown sign per column destroys the orthogonalisation (measured: achieved PR 1.8 instead of
6). The per-mode scalar response gain `η/|e^{iω} − (1−η)|` is divided out.

Result, verified rather than assumed — trajectory participation ratio at r = 1/2/4/6/8:

```
qp            1.00  2.00  4.00  6.00  8.00      an r-torus, exactly
qp_slow       1.00  2.00  3.98  5.94  7.83      the same at a training-log timescale
noise         1.00  2.00  4.00  5.96  7.92      rank-r injected gradient noise
mixed         1.00  2.00  4.00  5.99  7.96      torus + noise in the SAME r directions
batch_proj    1.00  1.75  3.34  4.42  5.47      real mini-batch noise, projected to rank r
gd            1.04  1.04  1.02  1.02  1.04      a transient is a 1-D curve for every r
batch         5.94  5.93  6.07  5.95  6.09      plain mini-batch SGD: r has no effect at all
```

The last two rows are results, not settings. **The active dimension of a decaying transient
is 1 whatever r is** — which is the whole of why `exp15` v1 measured 1.33 at every k. And
**the noise rank of ordinary mini-batch SGD is not a free parameter**: it is whatever the
data's per-example gradient covariance says, here ≈5.9 regardless of r. Any experiment that
proposes to set r by choosing a batch size is not doing what it thinks.

**Observers.** Twelve 1-D logs in five families — loss (`loss_step`, `loss_full`,
`loss_probe`), norms (`w_fro`, `c_norm`, `fn_fro`), gradient (`g_fro`, `g_proj`), fixed
random projections of the parameters (`c_proj1`), and function-space (`fn_proj1`, `margin`,
`acc_probe`). All random projections are drawn once and held for the whole run. Every one
except `loss_step` is a function of the optimiser state alone; `loss_step` contains the
instantaneous loss weights and is kept in the sweep precisely so that its contamination is
visible rather than assumed away.

**The estimator.** MG from the project's own `edm` package, with `clamp_to_max_E` off, the
Theiler window on, and `degenerate` windows flagged rather than silently floored. Its five
free parameters were chosen once, on seeds 90–92 at r ∈ {2, 4, 6}, from a 48-point grid
containing **only estimator parameters** — and every configuration was scored on the *same*
simulated logs, so choosing a configuration cannot move the data. Every later experiment uses
seeds 0–3 (E2, E3) or 0–4 (E4), and absolute recovery is reported on r ∈ {1, 3, 5, 8}: a
split disjoint in seed **and** in r, because `frequencies` is a deterministic function of r, so holding out seeds
alone would leave the geometry shared. Frozen:
`max_E=20, tau=4, k_neighbors=20, theiler="autocorr", window=8000`.

Three things must be said about that choice. It sits at a **grid boundary in all five swept
parameters**, which is the same warning sign as in `exp10`–`exp12`. MG's error across the 48
configurations runs from 0.32 to 1.50 on identical data — so the level is a choice as much as
a measurement, and only the held-out numbers in §3 are entitled to be read as recovery. And
the winner is a tie: configuration 45, identical but with `theiler="embedding"`, scores
0.324 / 0.908 / 0.757 to the same three decimals, and `"autocorr"` won on sort order alone.
That matters in exactly one place — see the transient arm in §3.2.

## 3. E2 — MG against the measured active dimension

252 runs: 7 excitation arms × seven r values {1, 2, 3, 4, 5, 6, 8} × 4 seeds, plus the
`eta_zero` and no-preconditioner controls. Real data, frozen backbone, 10 available directions throughout. **MG raw, median
over 4 seeds and 12 observers**, against the *measured* active dimension:

```
arm            r=1     2      3      4      5      6      8   | measured active dim
qp            0.90   2.33   3.58   4.63   5.55   6.69   7.40  | 1.0 2.0 3.0 4.0 5.0 6.0 8.0
qp_nopre      0.90   2.27   3.39   4.42   5.59   6.69   8.10  | 1.0 2.0 3.0 4.0 5.0 5.9 7.6
qp_slow       1.12   2.79   3.08   3.32   3.34   3.33   3.42  | 1.0 2.0 3.0 4.0 5.0 6.0 8.0
mixed        12.13  12.08  12.21  12.71  13.95  14.79  15.30  | 1.0 2.0 3.0 4.0 5.0 6.0 8.0
noise        15.08  15.03  15.14  15.11  15.12  15.11  15.13  | 1.0 2.0 3.0 4.0 5.0 6.0 8.0
batch_proj   14.96  15.12  15.07  15.10  15.17  15.15  15.16  | 1.0 1.8 2.6 3.3 4.0 4.5 5.6
batch        15.18  15.14  15.12  15.18  15.15  15.15  15.10  | 6.2 6.2 6.2 6.2 6.2 6.3 6.2
gd           29.03  28.99  28.87  28.70  29.19  29.02  29.02  | 1.05 ... 1.06
```

**Absolute recovery on held-out r ∈ {1, 3, 5, 8} and held-out seeds 0–3**, with any
calibration fitted only on seeds 90–92 at r ∈ {2, 4, 6}. Scored over the **ten usable
state-only observers** — `acc_probe` is excluded because quantisation makes it degenerate in
100 % of windows, and `loss_step` because it fails the `eta_zero` control (below):

```
arm           raw MG   Spearman(MG, active dim)   slope (held-out r)
qp             0.87            +1.000                   0.92
qp_nopre       0.64            +1.000                   1.10
qp_slow        1.59            +1.000                   0.30
mixed          9.20            +0.964                   0.49
noise         10.78            +0.500                   0.004
batch_proj    11.64            +0.750                   0.02
batch          8.96            +0.321                   0.56
gd            27.80            +0.179                   3.21
```

`results/tables.txt` additionally prints affine and isotonic calibrations, restricted to the
four observers E1 actually calibrated (`w_fro`, `c_norm`, `g_fro`, `c_proj1`); on that subset
raw MG scores 0.69 for `qp`. Read those columns with care: **an isotonic fit on an estimator
that is flat in r returns a near-constant predictor**, which scores a small MAE (2.7–5.0 on
the `noise`, `batch_proj` and `gd` rows) while carrying no information at all. That is why
the Spearman column is printed beside it, and why the raw column is the headline. Only the
`qp*` rows have both a low error and an ordering.

### 3.1 What is supported

In the deterministic, recurrent, well-sampled arm MG **does** recover the active dimension of
a real network on real data: **MAE 0.87, Spearman +1.000, slope 0.92** — raw, uncalibrated,
on r values and seeds the estimator's parameters never saw. Every one of the ten usable
state-only observers gives Spearman exactly +1.000 there, and the ordering is not a smoothness
artefact:

```
qp arm, per observer      MAE    slope   Spearman(roughness null, active dim)
loss_probe               0.248   0.908         0.000
margin                   0.327   0.949        -0.179
loss_step                0.452   0.847        -0.179
c_proj1  (fixed proj.)   0.473   0.990        -0.250
g_proj                   0.507   0.930        +0.643
fn_fro                   0.516   0.955        -0.214
c_norm / w_fro           0.541   0.910        +0.714
fn_proj1                 0.576   0.720        +0.143
g_fro                    1.864   1.681        -0.893
loss_full                1.940   1.666        -0.786
acc_probe                  --      --           --   (quantised; degenerate)
```

Loss and function-space observers are the most accurate; the two *gradient-magnitude*-like
observers (`g_fro`, `loss_full`) overshoot by ~70 % in slope. The Frobenius norms, which are
what this project actually logs, sit in the middle at MAE 0.54 — but note their roughness
null reaches +0.714, so for those two observers smoothness is *not* fully controlled and part
of the ordering could be spectral. `acc_probe` is unusable: quantisation makes it degenerate
in 100 % of windows (it is the *only* source of the 4–8 % degenerate fraction reported per
arm; every other observer is at 0.000).

The **`eta_zero` control passes, and it disqualifies one observer** — which is the point of
running it. With the learning rate set to zero, eleven of the twelve observers become
*exactly* constant. The twelfth is `loss_step`, which contains the instantaneous loss weights;
frozen at η = 0, with the measured active dimension identically 1 at every r, its MG still
reads **0.90, 2.22, 3.71, 4.62, 5.45, 5.31, 6.93** across r = 1…8. That is most of the
headline reproduced from the drive alone — exactly the `exp14` failure mode — and it is why
`loss_step` is excluded from every aggregate above despite scoring third best (MAE 0.452) if
it is left in. The other eleven observers cannot do this: they are functions of the optimiser
state, and at η = 0 the state does not move.

### 3.2 What is not supported

**Stochastic excitation: nothing.** `noise` (rank-r injected gradient noise) reads
15.029–15.136 across r = 1…8 while the measured active dimension goes 1.0 → 8.0 — a total
spread of 0.11 against an eightfold change in the truth. `batch_proj` — the brief's
experiment 2 done with the covariance the *data* produces rather than an imposed Gaussian —
reads 14.964–15.172 while its measured active dimension goes 1.0 → 5.6. Plain mini-batch SGD
reads 15.10–15.18 at a measured active dimension of 6.2. The identifiability ratio on
these arms is 1.54–1.57, i.e. the estimate is a property of the embedding space.

**A little noise destroys the reading.** The `mixed` arm is the same r-torus with rank-r
Gaussian noise added *in the same r directions*, at **one quarter of the `noise` arm's
amplitude — a sixteenth of its variance** (`noise_amp` 0.02 against 0.08). Its measured
active dimension is still exactly r. MG goes from 0.90–7.40 to 12.08–15.30 — it stops
reporting the torus and starts reporting the noise floor. The ordering survives (Spearman
+0.964) but the absolute number does not (MAE 9.20). For a real training log, which
is a weak deterministic signal inside stochastic-gradient fluctuation, this is the operative
regime.

**Slow oscillation: order only.** `qp_slow` is the *same system and the same measured active
dimension* as `qp`, differing only in that the drive period is 400 steps instead of 16. MG
saturates at 3.3 from r = 4 onward: slope falls 0.96 → 0.26 and MAE rises 0.87 → 1.59. The
ordering survives, the absolute number does not.

**A transient: a confident wrong answer, and no warning.** In the `gd` arm the measured active
dimension is 1.025–1.063 for every r — a decaying trajectory is a 1-D curve. MG reports
**28.7–29.2**, above `max_E`, at every r. The mechanism is the Theiler window doing its job on
a non-recurrent signal: it forces neighbours to be far apart in time, on a monotone curve they
are then all at comparable distances, the log-ratios collapse and the estimate diverges.
Critically, **the identifiability ratio on this arm is 1.00** — the diagnostic that correctly
flags the stochastic arms does *not* flag this one. A monotone training log can therefore
produce a large, stable, reproducible MG value that has nothing to do with any dimension.

The magnitude here is itself capped. `theiler="autocorr"` asks for a window of ≈ 1680 samples
on this arm; `mg.THEILER_CAP` truncates it to 150, which is why the number is 29 and not
larger. Measured on one window at successive caps: 150 → 27.8, 400 → 43.1, 1000 → 108.8,
uncapped → **172.0**. (The cap binds on only two arms, `gd` at 92 % of windows and
`noise_nopre` at 49 %; every other arm runs at `theiler = 76`, untouched. And the tied
configuration 45, with `theiler="embedding"`, never invokes it at all — so the "29" is a
property of one arbitrary tiebreak, while the *qualitative* failure is not: at every cap from
150 to uncapped, MG on this arm is between 28 and 172 when the truth is 1.)

### 3.3 Available is not active, and the difference is measurable

`qp_nopre` and `noise_nopre` drop the `H^{-1}` preconditioner. The measured active dimension
then falls below the injected rank — 7.55 instead of 8.00 in `qp_nopre`, and **6.43 instead of
8.00 in `noise_nopre`** — because the anisotropic curvature redistributes the excitation
across the available directions. The gap reaches 1.6 components in a system deliberately
built to make injected rank and active rank agree, which is the point: **the two are different
quantities and only one of them is measurable from the trajectory.**

The `qp_nopre` arm is the cleaner illustration of *why* the preconditioner exists rather than
of the scoring difference — scored against nominal r instead of measured PR it gives MAE
0.564, marginally *better* than the 0.644 against the measured value, because its two errors
happen to cancel (at r = 8 it reads 8.10 against a measured 7.55, overshooting where `qp`
undershoots). It also has only two seeds; matched on seeds, `qp` and `qp_nopre` differ by
0.046, well inside the per-observer seed spread of 0.38–0.81. **No conclusion should be drawn
from `qp_nopre` scoring better than `qp`.**

## 4. E3 — a controlled simplification

`r_high → r_low → r_high`, three 15 000-step segments, nothing else changed. Ground truth
re-measured on each segment separately. Detection uses a threshold set from the **pre-switch
segment alone** (its median minus 3× its window-to-window scatter) held for two consecutive
windows — not the midpoint of the two observed levels, which would hand the detector the
answer and, in the arm where the two levels coincide, turn the crossing into a coin flip.

```
arm     true drop   seen drop   level error at r_high   detection rate      median lag
qp        3.99        3.95              0.89            0.94 / 0.94       999 / 1999
noise     3.98        0.15              9.13            0.49 / 0.79      2999 / 1999
```

The detection rates are over all twelve observers, and `acc_probe` detects nothing in either
arm; over the eleven usable observers the `qp` rate is **1.00 in both directions**.

In the `qp` arm the simplification is seen almost exactly (3.95 against 3.99) and the
recovery is essentially perfect: the median `|level_after_recovery − level_before|` is
**0.009**. Level error is 0.89 at the high level and 0.22 at the low — the estimate is more
accurate the fewer directions are active, which is the saturation of §3 seen from another
angle. Per level pair, `w_fro` reads 4.77 → 1.22 → 4.78 for a true 4 → 1 → 4, and
6.90 → 2.12 → 6.91 for 6 → 2 → 6; at 8 → 3 → 8 it reads 6.97 → 3.64 → 6.97, under-reporting
the high level by one component.

The lag needs care. Window 8 000, stride 2 000, switches at 15 000 and 30 000; right edges
fall at 2000k + 7999, so the first one past the down-switch is 15 999 and past the up-switch
31 999. **999 and 1999 are the two floors of the measurement grid, not measured response
times** — 999 is the minimum representable value and it is hit in 91.5 % of `qp` rows, while
1999 is the *only* value `lag_up` ever takes there. What this does show is that MG crosses a
3-sigma threshold on a window that is still 87.5 % pre-switch data. The honest statement is
"below one stride, at a resolution of one stride"; measuring the lag properly would need a
stride far shorter than the window, which costs a full sliding pass per point.

In the `noise` arm the same true change of 4.0 components produces a seen change of **0.15**.
Its "detection rates" of 0.49 and 0.79 are threshold crossings on a flat noisy trace, not
detections — the levels are 15.15 → 15.05 → 15.13. Two observers do move there (`loss_full`
and `loss_step` drop 2.9, `acc_probe` 9.1), but that is the total noise *power* falling when
the noise rank falls, not the dimension being read.

## 5. E4 — what moves MG when r does not

Fixed r = 4, five seeds, twelve observers. A control "fires" if its within-run half-to-half
change exceeds the 97.5th percentile of the baseline's own within-run change, or its level
differs from the baseline level by more than the 97.5th percentile of a *leave-one-out*
baseline null. The reference for effect size is the real 4-component change measured in E3,
3.95.

```
control        |MG shift|   as % of a real 4-component change   fires   what changed
baseline          0.21                  5 %                     0.24   nothing
lr_step           0.16                  4 %                     0.86   learning rate halved
freq_double       0.49                 12 %                     0.26   drive band up an octave
freq_half         0.50                 13 %                     0.25   drive band down an octave
rotate            0.86                 22 %                     0.40   coordinates rotated
obs_scale         1.16                 29 %                     0.49   observer gain ramped 10x
amp_ramp          1.32                 33 %                     0.92   state amplitude ramped 4x
```

**The decision rule's own null rate is 0.24, not the nominal 0.05.** The reason is structural,
not just noise: with five baseline seeds the empirical 97.5th percentile lies strictly below
the maximum, so the largest of the five baseline runs always exceeds it — a deterministic
1-in-5 = 0.20 floor on `fires_within`, which measures 0.186 (qp) and 0.200 (noise). Fixing it
needs ≥ 20 baseline seeds or a bootstrapped quantile; here, every firing rate must simply be
read against 0.24 rather than 0.05. With that correction:

* **The smoothness control passes.** Moving the whole drive band by an octave at fixed r —
  which changes the autocorrelation time from 1 to 4 samples and the roughness from 1.21 to
  0.33 — moves MG by 0.49–0.50 and fires at 0.25–0.26, i.e. at the null rate. This is the
  control that matters most given that `report_0808.md` records Spearman 0.934 between the
  roughness null and the LB estimate on this project's real logs, and it is the one E4 was
  built around. In this design MG is not reading smoothness.
* **Learning rate: the level is safe, the stability is not.** Halving the learning rate moves
  the level by 0.16 (below baseline) but fires the within-run statistic at 0.86. Traversing
  the same manifold at a different speed does not change what MG reports, but it does change
  its window-to-window variability, so a change-detector tuned on variability will fire.
* **Amplitude does move it, and this is the one that matters for a real log.** MG is exactly
  invariant to a constant rescaling, but not to a *ramp*: a 10× gain ramp on the observer's
  fluctuation — which changes nothing about the trajectory at all — moves it 1.16, about a
  third of a real four-component change. A training run whose weight norm grows can therefore
  produce a drift of order one "component" with nothing dimensional happening.
  The `amp_ramp` control is weaker evidence than `obs_scale` and should be read with its
  caveat: ramping the *state* excursion 4× is not quite "at fixed r", because the measured
  active dimension itself drifts 4.00 → 3.74 (minimum 3.49) over the run. MG moves 1.32 where
  the truth moves 0.26 — five times more — so the conclusion survives, but this control is
  not a pure null and `obs_scale`, which is, carries the argument.
* `rotate` is scored on the three projection observers only, since it acts through `R·c` and
  the other nine are invariant by construction. It moves them 0.86: a fixed random projection
  is not a rotation-invariant observer, which is a property of the observer, not of MG.

## 6. E5 — where this project's own logs sit

The seven 120 000-step reruns in `../dimension_recovery/results/extended`, measured with the
frozen configuration. `weight_norm`, the observer this project actually logs:

```
run              MG    ident ratio   roughness   acorr   mean-crossings
grokpos_s0     15.31      1.289        0.030      335         56
lowdata15_s0    4.93      1.599        0.068      434        146
lowdata15_s1    4.14      1.554        0.093      161        150
lowdata15_s2    4.47      1.514        0.149       10        151
lowdata20_s0    3.94      1.366        0.044       66         32
wd0_s0         28.08      0.999        0.000      858          2
wd0_s1         28.00      0.999        0.000      857          2
```

Against the reference values established in E0 and E2 — identifiable ≈ 1.00–1.09, no
dimension exists ≈ 1.44–1.57, monotone transient ≈ 1.00 *with MG ≈ 29* — **none of these logs
is in the identifiable regime.** The five `lowdata`/`grokpos` runs sit at 1.29–1.60, the
stochastic band where E0 and E2 both show MG to be a function of the embedding and of
smoothness rather than of any dimension. The two `wd0` runs sit at 0.999, which looks
identifiable until one notices that their roughness is 0.000, they cross their own trend
twice in a 4 000-sample window, and MG reads 28 — the exact signature of the `gd` arm, where
the true active dimension is 1 and the diagnostic does not fire.

`weight_norm` MG spans **3.94 to 28.08** across seven runs of the same project. And 11 of the
35 run × column cells are outright degenerate, with MG values up to 1.9 × 10⁶ — the
`n(k−1)−1` pathology of `edm.dimension`'s log-ratio floor, firing on quantised accuracy
columns. Those cells are flagged here; nothing in `dimension_recovery` flagged them.

## 7. Verdict

The hypothesis, restated: *the MG dimension of a delay reconstruction of a good 1-D observer
reflects the number of actively expressed degrees of freedom, and can indicate the system
simplifying.* Split into the three claims it contains:

**Claim 1 — MG responds to the number of active components. SUPPORTED, conditionally.**
On real data, a real frozen nonlinear backbone and a real optimiser, with the active
dimension measured rather than assumed, MG recovers it with **MAE 0.87 and Spearman +1.000**
on held-out r and held-out seeds, uncalibrated, over ten state-only observers. All ten give
ρ = +1.000; the roughness null is flat; the `eta_zero` control passes and disqualifies the one
observer that fails it; a controlled 4 → 1 simplification is seen as 3.95 and recovers to
within 0.009. This is a real positive result and it is stronger
than anything in `dimension_recovery` — which is unsurprising, since the audits showed those
experiments' headline numbers survive setting the learning rate to zero.

**Claim 2 — MG estimates their absolute number. SUPPORTED ONLY INSIDE A NARROW REGIME.**
The conditions are: deterministic, recurrent dynamics; a delay window spanning roughly 0.4 of
an oscillation period; and negligible stochastic forcing. Step outside any one of them and
the absolute number goes:

* slower oscillation, same system, same active dimension → slope 0.96 → 0.26, MAE 0.87 → 1.59;
* add noise at a quarter of the pure-noise arm's amplitude → MAE 0.87 → 9.20;
* wrong τ, system unchanged → MG moves by up to 61 components at r = 8;
* across the estimator's own 48-point parameter grid → MAE 0.32 to 1.50 on identical data.

**Claim 3 — it is an indicator of a real training run simplifying. NOT SUPPORTED.**
Three independent reasons, each sufficient:

1. *Real training dynamics are in the wrong class.* Mini-batch gradient noise gives MG no
   information about the active rank (15.03–15.14 across r = 1…8), and its noise rank is not
   even a free parameter — plain mini-batch SGD sits at active PR 6.2 regardless of r. A
   decaying transient has active dimension 1 whatever r is, and MG reports 29 for it (172
   with the Theiler cap removed).
2. *The one free parameter that matters cannot be set.* The correct τ is a function of the
   log's oscillation period. A controlled experiment knows it; a training log does not.
3. *Measured directly, the logs are not in the identifiable regime.* E5 places all seven of
   this project's 120 000-step runs at identifiability ratio 1.29–1.60 or in the monotone
   trap, and their `weight_norm` MG spans 3.94 to 28.08.

**What is worth keeping.** The distinction the brief asks for is real and is measurable:
available (10), functional (rank 10, PR 8.40) and active (measured per run) differ, and the
gap between injected rank and active rank reaches 1.6 components (`noise_nopre`: 6.43 against
an injected 8) in a system built to make them agree. Two
diagnostics earn their place on any future MG number, and neither was computed anywhere in
`dimension_recovery`:

* the **identifiability ratio** E(2·max_E)/E(max_E), measured per arm at r = 2 and 6:

```
qp 1.00 / 1.09     qp_nopre 0.98 / 0.94     qp_slow 0.99 / 1.52     gd 1.00 / 1.00
mixed 1.32 / 1.41  noise 1.57 / 1.56        batch 1.54 / 1.55       batch_proj 1.55 / 1.54
```

  It separates the arms where MG is right from the ones where it is not, and it does so
  *without a ground truth* — including inside a single arm: `qp_slow` reads 0.99 at r = 2,
  where MG is correct (2.79 against 2.00), and 1.52 at r = 6, where it has saturated (3.33
  against 6.00). That is the diagnostic doing exactly what it should.

* the **degenerate flag**, which catches the distance- and log-ratio-floor pathologies that
  produced 1.9 × 10⁶ on real logs — 11 of 35 run × column cells in E5.

The identifiability ratio is **not sufficient on its own**: it reads 1.00 on the `gd`
transient arm, which is where MG is at its most wrong (29 against a true 1). It must be
paired with a check that the window actually contains recurrences — E5's mean-crossing count
is 2 for the two `wd0` runs, and those are exactly the ones that read 28.
