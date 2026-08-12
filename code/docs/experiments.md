# Every number in the article, and what produced it

One experiment per row. `python -m actdim list` prints the same catalogue with what has
run; `python -m actdim plan --all` prints it in dependency order with its cost.

This table is the target. Not all of it is built yet: [`status.md`](status.md) says what is
ported, what is not, and where the archived implementation of each gap is.

Cost is measured: on eight cores where the device is cpu, on a T4 where it is gpu.

## Calibration

Everything downstream depends on these two, and both are committed under `data/frozen/`,
so the rest of the article rebuilds without re-running them. Requirement 2 forbids
reselecting a configuration on any later outcome, so re-run these only to check the
selection, never to improve a result.

| id | article | dev | cost |
| --- | --- | --- | --- |
| `calib.e8` | appendix C, `tab:frozen` — the eight-direction configuration | cpu | 48 min |
| `calib.e20` | appendix C, `tab:frozen`, `tab:k20` — the twenty-direction configuration | cpu | 70 min |

## Section 5 — systems with a known active dimension

Each row of `tab:ladder` is one experiment. The construction fixes the active dimension;
the recorded trajectory's effective rank verifies that all of it is excited.

| id | article | dev | cost |
| --- | --- | --- | --- |
| `sys.matrix` | 5.1, `tab:ladder` row 1 — the oscillating diagonal matrix | cpu | 2 min |
| `sys.matrix.sync` | 5.1 — several coordinates driven from one phase, against the same number of independent oscillators | cpu | 6 min |
| `sys.matrix.k20` | 5.2 — the same system to twenty directions, background to the ceiling | cpu | 17 min |
| `sys.linear` | `tab:ladder` row 2 — online linear regression | cpu | 3 min |
| `sys.logistic` | row 3 — logistic regression | cpu | 2 min |
| `sys.decoder` | row 4 — a frozen nonlinear decoder | cpu | 4 min |
| `sys.subspace` | row 5 — a perceptron confined to a k-subspace | cpu | 2 min |
| `sys.digits.parameter` | rows 6-7, and 5.3, 5.4, 6.1 — a constrained head on a network trained on image data, twelve observers, nine excitation cases | cpu | 143 min |
| `sys.digits.function` | row 8, 5.4 — the same data, function subspace | cpu | 2 min |
| `sys.silence` | 5.3, requirement 4 — the zero-learning-rate control | cpu | 12 min |

`sys.silence` is new. Section 5.3 states that the control invalidated two of the six
systems, and `tab:ladder` marks them accordingly, but no zero-learning-rate arm existed
for those two systems in the archived tree and no result file recorded one (errata 10).
The experiment runs the control so the claim can be kept or withdrawn on evidence.

## Section 6 — conditions of validity

| id | article | dev | cost |
| --- | --- | --- | --- |
| `valid.regime` | 6.1, `fig_regimes`(b) — the identifiability atlas across the three regimes | cpu | 14 min |
| `valid.tau` | 6.2, `fig_tau` — the delay-lag sweep | cpu | 11 min |
| `valid.nuisance` | 6.3 — eight nuisance controls at fixed rank, and their false-alarm rates | cpu | 34 min |
| `valid.anisotropy` | 6.4, `fig_aniso` — the count against the effective rank as the drive is made anisotropic | cpu | 4 min |
| `valid.transitions` | appendix E — detection of a change in the active dimension | cpu | 52 min |
| `valid.theiler.cap` | appendix N — the capped against the full exclusion at the twenty-direction configuration | cpu | 3 min |
| `valid.theiler.contrast` | appendix P, `fig_traces` — the exclusion sweep, recurrent against transient | cpu | 40 min |
| `valid.ceiling` | appendix R, `fig_ceiling` — embedding dimension against record length; refutes both explanations of the ceiling | cpu | 180 min |

## Appendix O — the training runs

The only experiments that need a GPU. Both trainers run in float64, so a CPU fallback is
refused unless `--allow-cpu` is given.

| id | article | dev | cost |
| --- | --- | --- | --- |
| `train.transformer.sketched` | `tab:runs` rows 1-6, with the trajectory sketch attached | gpu | 24 min |
| `train.transformer.extended` | `tab:runs` rows 7-13, the 120,000-step reruns | gpu | 150 min |
| `train.transformer.p211` | `tab:runs` row 14, modular addition at p = 211 | gpu | 120-240 min |
| `train.perceptron.arith` | `tab:runs`, the `a_*` and `x_*` rows | gpu | 36 min |
| `train.perceptron.poly` | `tab:runs`, the `g_*` rows | gpu | 72 min |
| `train.perceptron.sketched` | appendix J's inputs: four runs full batch, four at batch 512, two at p = 23, two long | gpu | 27 min |
| `train.perceptron.eos` | appendix Q — sixteen full-batch runs at eight learning rates, logged every step | gpu | 30 min |

## Section 7 — the application to delayed generalisation

| id | article | dev | cost |
| --- | --- | --- | --- |
| `grok.diagnostics.logs` | 7.1, `fig_map` — the transformer logs on the two diagnostic axes | cpu | 2 min |
| `grok.diagnostics.perceptron` | 7.1, `fig_map` — the perceptron logs, and the label-matched pairs | cpu | 8 min |
| `grok.rank.dip` | 7.2, `fig_dip`, appendix H — the effective-rank collapse, run by run | cpu | 4 min |
| `grok.matched.window` | 7.3, appendix L, `fig_window` — the estimator at the transition's own window, over the whole 36-cell grid | cpu | 5 min |
| `grok.matched.surrogate` | 7.3, appendix L — the shape-preserving surrogate control | cpu | 12 min |
| `grok.extended.outcomes` | appendix G — what the 120,000-step reruns overturn | cpu | 4 min |
| `grok.prwindow` | appendix J, `fig_prwindow` — participation ratio against window length | cpu | 3 min |
| `grok.eos` | appendix Q, `fig_eos` — the two diagnostics on the edge-of-stability logs, across logging strides, and the recurrence statistic | cpu | 65 min |
| `grok.repr` | appendix M — the representation dimension in closed form, and the proof that three tasks lie outside the representable class | cpu | 3 min |

## Checks and cost

| id | article | dev | cost |
| --- | --- | --- | --- |
| `check.sketch.noninvasive` | appendix I — the sketch changes no logged value, bit for bit | gpu | 2 min |
| `check.sketch.cost` | appendix S — what storing the trajectory costs, in bytes and in time | gpu | 3 min |
| `check.tables` | every mechanical table — recomputes each cell from `data/` and diffs it against the printed value, exiting non-zero on a mismatch | cpu | 1 min |

`check.tables` is the release check this repository has failed twice. Run it before
submitting anything.

## Figures

| id | article | dev | cost |
| --- | --- | --- | --- |
| `paper.figures` | all twelve figures, into `../icomp_v2/figures/` | cpu | 2 min |

## Reading this against the archived tree

`../icomp_v2/README.md` carries the old mapping, from article section to a path under
`code/`. Those paths now point into `../archived_code/`. The correspondence is one to one
except where the port merged or split something:

| archived | now |
| --- | --- |
| `active_dimension/e1_calibration.py` | `calib.e8` |
| `active_dimension/calibration_k20.py` + `score_k20_parallel.py` | `calib.e20` (two scorers merged; errata 17) |
| `dimension_recovery/exp9`-`exp15v3` | `sys.*` |
| `active_dimension/e2_rank_sweep.py` + `analyze.py` | `sys.digits.parameter` |
| `active_dimension/e0`, `e3`, `e4`, `e6`, `e7b`, `e8`, `e10`, `e11` | `valid.*` |
| `active_dimension/e9_matched_window.py` + `e9_analyse.py` | `grok.matched.window` |
| `gromov_arithmetic/eos.py` + its copied training loop | `train.perceptron.eos` (one loop; errata E) |
| `gromov_arithmetic/dimension_probe.py` + `gromov_polynomials` | `grok.diagnostics.perceptron` |
| `active_rank/analyze_rank.py` + `dip.py` | `grok.rank.dip` |
| `icomp_v2/make_figures.py` | `paper.figures` |
