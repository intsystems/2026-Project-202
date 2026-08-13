# What the port found

Thirty-eight items, plus one the port introduced and then closed. Four read-only surveys of the archived tree preceded this rewrite, one
per cluster, and the table auditor of `check.tables` found four more by recomputing every
mechanical cell of the article from the data behind it. What
they found is recorded here in full, whether or not the port fixes it, because several
items are statements in `icomp_v2/report.tex` that no committed file supports.

Status means:

* **fixed** — the new code does the right thing; the number it produces will differ from
  the archived one.
* **preserved** — deliberate, documented, and carried forward unchanged.
* **paper** — the code is right and the article's text or caption is not.
* **open** — neither code nor article can settle it without a decision or a re-run.

The article's own `README.md` records two defects it believes are outstanding. Both were
fixed at source before this port began, and that README is stale on the point: see items
24 and 25.

---

## A. Defects that move a published number

| # | what | where | status |
| --- | --- | --- | --- |
| 1 | **The held-out seeds hold nothing out.** `frequencies(..., band_mode="matched")` takes a `seed` and ignores it: the matched branch is `centre * (1 + band*(2*frac-1))`, with no randomness. Every constructed system runs `matched`, so seeds 1 and 2 reuse seed 0's frequency geometry exactly and differ only in phase, amplitude, offset and noise. Every held-out recovery number in section 5 is weaker than the article states, and requirement 3 is not met in the way it claims. | `dimension_recovery/systems.py:72-80` | **fixed** |
| 2 | **The Theiler exclusion is capped at 150 samples by a mutable module global**, written from inside worker processes by three scripts. On the transient arm the autocorrelation rule resolves to about 1600 samples and is clipped, so the published estimate near 29 is the value at the cap rather than at the rule. Appendices N and P disclose the cap; the mechanism was still a global that any import could change. | `active_dimension/mg.py:69`, set at `e7b:42`, `e10:205`, `e11:390` | **fixed** (a config field) |
| 3 | **`tab:alts` aggregates its two columns differently.** The eight-direction column scores the four withheld ranks; the twenty-direction column scores all twenty, including the five the configuration was selected on. Restricting to the ten printed gives MG 1.554 rather than 1.62. The caption does not say so. | `active_dimension/paper_tables.py:88-99` | **paper** |
| 4 | **Degenerate windows drop statistics that do not depend on them.** `summarise` filters every estimator by the degeneracy flag, including roughness, autocorrelation and the spectral participation ratios, which are computed without a neighbour search and remain valid. This depresses the sample count of the null columns in exactly the arms where the nulls matter most. | `active_dimension/mg.py:242-245` | **fixed** |
| 5 | **`a_sum_sq` ran 46,000 steps.** `tab:runs` gives its budget as 100k. It generalised at 8,150, so no claim moves, but the inventory is wrong. The archived cluster report states the truncation openly; the article does not. | `gromov_arithmetic/report.md:118` | **paper** |
| 6 | **`tab:ladder` row 1 prints its two rank correlations in the opposite seed order to its two errors.** The row reads `0.30 / 0.69` and `0.99 / 1.00`; the committed result gives seed 1 error 0.302 with rho 1.000 and seed 2 error 0.693 with rho 0.988, so in the same order the second pair is `1.00 / 0.99`. | `report.tex:820` vs `exp9_frobenius_k10/report_exp9.md:13` | **paper** |
| 7 | **The window-length labels in appendix J are one sample long.** The label is computed as `stride * n * log_every` where the window spans `stride*(n-1)+1` rows, so the row labelled 600 steps covers 550 plus one logging interval. The article quotes those labels. | `gromov_arithmetic/pr_vs_window.py:118` | **fixed** |
| 8 | **`dimension_probe.csv` is not what its own command produces.** The committed file covers four runs and two columns; the script globs all seven `*_train.csv` and defaults to three columns. Re-running it turns `fig_map`'s "perceptron, full batch (10)" into 13. The committed file is right for the article; the command is not. | `gromov_arithmetic/dimension_probe.py:110-116` | **fixed** (the run set is now declared, not globbed) |

## B. Statements the code does not support

| # | what | where | status |
| --- | --- | --- | --- |
| 9 | **`tab:ladder`'s caption describes a sliding-window median for rows that have one window.** It says each side of the error is a median over the run's sliding windows "except the last row". In fact rows 1 to 5 also compute exactly one window per run; only the two rows from the parameter-subspace system slide. | `report.tex:810-813` | **paper** |
| 10 | **The zero-learning-rate control that marks two systems "fails 4" had no implementation.** Section 5.3 says the test invalidated two of the six systems, and `tab:ladder` marks the frozen decoder and the subspace perceptron accordingly. No zero-learning-rate arm existed in either script and no result file recorded one. `sys.silence` implements it, and **it invalidates five systems, not two** — see below. | `report.tex:381-388`; absent from `dimension_recovery/exp13*.py`, `exp14*.py` | **open** — the claim understates its own result |
| 11 | **The sketched perceptron runs are not the table's first four rows.** The article says "the top four perceptron rows are repeated with the trajectory sketch attached"; the four that were sketched are `a_add`, `x_no_grok`, `g_p1`, `g_p1x`. | `report.tex:1949` vs `rank_fb/rank_runs.json` | **paper** |
| 12 | **"Agree to within one or two logged steps" understates the mini-batch arm**, which differs by up to six logged steps, sixty optimiser steps. Full batch agrees exactly. | `report.tex:1951` | **paper** |
| 13 | **Section 5.1's "tracks r across the whole range" covers a range where the estimate is already saturating.** The held-out medians at the top of the first system's range are 8.185 and 8.491 against truths of 9 and 10, with an inversion on the second seed. Section 5.2 concedes the ceiling; section 5.1 does not qualify the claim. | `report.tex:354-363` vs `exp9` held-out results | **paper** |
| 14 | **A comment asserts an identity that does not hold**, and two scripts drop an observer on the strength of it: the parameter norm and the subspace norm are said to return bit-identical estimates at every rank. The largest difference over twenty ranks is 9.3e-5, zero only at rank 1. One is a monotone nonlinear transform of the other, so they agree at the dither and floor level and no further. The exclusion is defensible; this justification for it is not. | `active_dimension/e10_ceiling_sweep.py:111`, `e11_theiler_contrast.py:82` | **fixed** (claim not carried forward) |

## C. Results that cannot be regenerated

| # | what | where | status |
| --- | --- | --- | --- |
| 15 | **Appendix J's input no longer exists.** `pr_vs_window.csv` is committed; the sketches it was computed from were not kept, and the script exits if none is found. The file cannot be rebuilt from the repository at any cost — only by re-training the two runs. | `gromov_arithmetic/results/rank_fb_long/` | **open** — re-run required |
| 16 | **The fine analysis windows cannot be rebuilt either.** `results_fine/` holds no sketches; the six `.npz` exist only under `results/` and are untracked. | `active_rank/results_fine/` | **open** — re-run required |
| 17 | **Two scorers write the same three files** by different aggregation code, and nothing in the files records which produced the committed copy. The same cluster documents an earlier instance of exactly this: code fixed, file not regenerated, committed ranking disagreeing with the committed script. | `active_dimension/calibration_k20.score` and `score_k20_parallel.main` | **fixed** (one scorer) |
| 18 | **`summary.json` was rebuilt from the current registry, not from the run.** Both committed copies show a null wall-clock time, which is how one can tell. Editing a registry entry silently rewrites the recorded configuration of logs produced under the old one. | `gromov_arithmetic/rebuild_summary.py:71-83` | **fixed** (provenance is written by the run) |
| 19 | **The sketched campaigns' metadata cannot tell them apart.** Every configuration field in `rank_runs.json` is null in both the full-batch and the mini-batch directory, including the batch size, which is the only difference between them. A code comment claims this was fixed; the fix is in the code and the committed results predate it. | `gromov_arithmetic/results/rank_{fb,mb}/rank_runs.json` | **fixed** |
| 20 | **No result file anywhere records the code that produced it** — no commit, no version, no timestamp. | whole tree | **fixed** (`provenance.json` per run) |
| 21 | **Raw table row order is nondeterministic**, because results were collected from process pools as they completed. The values are stable; a re-run still cannot be diffed against the committed file. | `dimension_recovery/exp11-15`, `active_dimension` scorers | **fixed** (`map_ordered`) |
| 22 | **The extended reruns cannot be reproduced from the script that plans them.** It declares twelve runs of 200,000 steps and names its control differently from the file that exists; the delivered evidence is seven runs of 120,000 from a shell script that hardcodes the tags. Neither file mentions the other. | `dimension_recovery/extend_runs.py:53-73` vs `launch_extended.sh` | **fixed** (one registry) |
| 23 | **Two different series are described as one run.** The re-trained `mod_wd1` and the earlier canonical log agree to 1e-14 for 198 rows and then diverge under float64 rounding, ending 2.79 apart on the parameter norm. Generalisation is at step 13,700 in the re-run and 13,810 in the canonical log. Appendix O reports 13,700, the archived trainer's registry and README report 13,810. Both are right about their own file. | `active_rank/results*/mod_wd1_train.csv` vs `grokking_analysis/grokking_logs/` | **paper** — section 7.2 rests on the re-run; the article should say so |

## D. Fixed at source before this port; the article's README is stale

| # | what | status |
| --- | --- | --- |
| 24 | `invariance_controls.csv` reporting the constant-rescaling control moving the estimate by 1.68 to 4.96 components. The committed file now reads 0.000 on all eight observers, and 0.000 on seven of eight under rotation, the exception being the fixed random projection, which is not rotation invariant. `icomp_v2/README.md` still lists this as needing fixing at source. | **preserved** (already correct) |
| 25 | `e10_surrogate.py` seeding from `hash()` of a string, which Python salts per interpreter. It now seeds from `zlib.crc32`. Same stale README entry. | **preserved** (already correct) |

## E. Hygiene

Carried here so they are not rediscovered: the archived `run_all.py` was not a driver (two
steps commented out mid-chain, eleven scripts missing, and the true order recoverable only
by reading imports); `dip.py` wrote its figure to a directory fixed next to the code rather
than derived from the results directory, and had already overwritten the coarse figure with
the fine one; `active_rank/report.md`'s memorisation column is stale against its own data by
two orders of magnitude; the zigzag trace of two recovery experiments labels each window by
the truth at its right edge, so a window straddling a switch is scored against the
post-switch rank; four cluster reports that back article rows are in Russian with a UTF-8
byte-order mark while their siblings are in English without one; one superseded figure
script jitters points with an unseeded generator; and three of the estimator's own
statistics were being recomputed from a float32 copy of an array whose exact value was
already stored alongside, which is why one column disagreed with every other file in the
sixth digit.

## F. Found while porting, not by the surveys

| # | what | where | status |
| --- | --- | --- | --- |
| 26 | **The estimator does not standardise; every caller does.** Appendix A's first line standardises the series, and the estimator it describes never did: all five call sites z-score the record before handing it over. The pipeline therefore matched the article while the function alone did not, and a sixth caller would not have known. The port standardises inside the estimator, which also means each window is standardised on its own rather than inheriting the whole record's scale. Measured cost: 1.6e-13 on the twenty-direction windows, up to 2.8e-8 on the transient arm, whose second window has a sixth of the record's spread and so a relatively larger dither. No published digit moves. | `active_dimension/mg.py` against `report.tex` appendix A | **fixed** |
| 27 | **The neighbour kernel's floors hide the case they exist for.** On an exactly recurrent or constant series the distance floor and the log-ratio floor return a finite number -- 0.08, or `N(m-1)-1` -- rather than reporting that the window is degenerate. This is the first of the two silent defects the archived cluster README records without naming. The port counts both floors against their own denominators and raises the degeneracy indicator, still returning the value as appendix A requires. | `grokking_analysis/edm/dimension.py:89,157` | **fixed** |
| 28 | **`clamp_to_max_E` silently returns the embedding dimension** when the estimate exceeds twice it, which converts a divergent estimate into a plausible one. Appendix A says the value is never clamped, and the archived callers were careful to leave the flag off; the default was on. This is the second silent defect. | `grokking_analysis/edm/dimension.py:207` | **fixed** (no clamp exists) |
| 29 | **The trend-crossing count raises from inside a worker on a window containing NaN**, because all three copies of it fit a line with `polyfit`. A closed-form count that returns NaN replaces them. Related: summarising a record shorter than one window took the mean of an empty array, which under this package's test settings is an error rather than a warning. | `e5:48`, `e9:78`, `e10:190`; `mg.py:234` | **fixed** |
| 30 | **The sketch cost measurement records an unresolved device.** It writes `"auto"`, so the committed measurement cannot say whether it ran on a GPU. It also reports a negative overhead, which appendix S handles honestly as a bound rather than a result, and it trains the perceptron while the appendix describes the figure at the transformer's logging stride. | `gromov_arithmetic/sketch_cost.py:95` | **fixed** |

| 31 | **The corrected drive excites its directions less evenly than the published one.** Fixing item 1 changed the frequency layout, and the change has a cost: on the function-subspace system the participation ratio of the trajectory covariance reaches about 0.86 k where the published construction reached 0.94 to 0.96 k. Every hard rank is still exactly k -- the covariance, the update covariance and the map to the outputs -- so the construction excites k directions; what falls short is requirement 1's word *comparably*. The trade runs the other way on rational independence: the corrected resonance margin stays above 1e-3 at every rank where the published one fell to 1.5e-5. Neither construction dominates, so the choice has to be made rather than defaulted. | `actdim/systems/drive.py`; measured against `data/sys.digits.function/rank_diagnostics.csv` | **open** — held by a test so the number cannot drift unnoticed |

## G. Found by the table auditor

`check.tables` recomputes every mechanical cell from `data/` and diffs it against what the
article prints. It confirms items 3, 5, 6, 9 and 11 above, confirms that item 24 is fixed,
and reports these, which no survey found. It checks 21 of the 28 tables, 731 cells and
claims; the seven it cannot check are named in `actdim/tables.py`, and a skipped table
contributes a row saying so, since a report that checked nothing must not look clean.

| # | what | where | status |
| --- | --- | --- | --- |
| 32 | **The record-length scan's slope row is wrong in three of its cells.** At the three record lengths, the article prints 0.24, 0.24 and 0.25 where the file gives 0.2865, 0.2682 and 0.2713. This is not rounding: the printed row rises monotonically with the record length and the measured one does not, so the shape of the row is wrong as well as its values. | `tab:ceiling` against `data/valid.ceiling/ceiling_summary.csv` | **paper** |
| 33 | **The two ceiling scans disagree on the one cell they share.** The embedding scan and the record scan both cover E_max = 20 at N = 8000. The slope row prints 0.27 in the first block and 0.24 in the second. That is a contradiction inside one table and needs no file to settle it; the file gives 0.2682, so the first block is right. Tracking and level agree across both blocks, which puts the fault in the slope row alone. | `tab:ceiling` | **paper** |
| 34 | **A row labelled "median absolute error" reports the mean.** The article prints 2.78 and 3.13; the median over the seven ranks is 2.18 and 2.66. The mean is a defensible statistic and the label is not. | `tab:theiler` against `data/valid.theiler.cap/` | **paper** |
| 35 | **One ground-truth cell is misrounded.** The `noise_nopre` arm at r = 6 prints 5.47 where the file gives 5.464913, which rounds to 5.46. | `tab:gt` | **paper** |

One check the auditor cannot make is worth naming rather than leaving silent: item 12, the
claim that the full-batch and mini-batch arms agree to within one or two logged steps,
needs the milestones of both sketched perceptron campaigns, and neither was promoted. The
auditor records that as a note on `tab:runs` so that its silence is not read as agreement.

## H. Found while wiring section 7

| # | what | where | status |
| --- | --- | --- | --- |
| 36 | **A committed table was computed from a log that is not the one committed beside it.** The perceptron diagnostics were computed from a 60,000-step `x_mix_quad` run, giving four windows; the log committed in the same directory is the full 100,000-step run, which gives seven. Re-running the command on the committed input therefore moves that run's summary from MG 18.57 and 18.86 to 24.35. Nothing published moves — `fig_map` plots the crossing count and the identifiability ratio, which do not change, and `tab:grok-diagnostics` does not list this run — but it is a second instance of item 8, from a different direction: there the command did not produce the committed file, here the committed input does not produce it either. | `gromov_arithmetic/results/arith/` | **fixed** (the run set and the columns are declared) |
| 37 | **One appendix M number depends on a seed the archived code fixed by literal.** The effective rank of the closed-form weights is 148.8 in the article and 147.4 under the port, because the drive phases now come from the run's own stream rather than a hard-coded `seed=0`; across seeds it spans 145.2 to 150.0. Appendix M already declines to treat that number as evidence, which this makes measurable rather than asserted, but it is a published digit and it moves. | `gromov_arithmetic/analytic.py` | **fixed** |
| 38 | **Every number of the two roughness controls moves, because the seeding does.** The archived script drew both clock arms' phases and amplitudes from a bare `np.random.default_rng(seed)` for seeds 0 to 7, which the contract forbids: a stream comes from the run's base seed by the named rule, so that a second process reproduces it and so that adding a draw does not move the draws beside it. `valid.geometry` takes its phases from `clock_phases` and its amplitudes from `clock_amplitudes`, which are different draws, so every cell of all four tables differs. Nothing about the conclusion does. Both pre-specified verdicts still pass and the headline numbers move by less than the seed-to-seed spread: MG 3.953 to 3.976 on the four-clock arm, 1.119 to 1.100 on the one-clock arm, the matched roughness changing by 0.000141 against 0.000673, the observer span 0.841 to 0.834, the estimate's span across the five observer scales 0.104 to 0.101, and its worst distance from four 0.068 to 0.083. Two other differences travel with it and are not the seeding: the truth of a window is now the level it lies *entirely* inside, where the archived version labelled a window by its centre and so scored a window straddling a switch against whichever level its middle sample fell in; and `TwoNN` is left in the table but is unreadable on the one-clock arm for the reason recorded beside it. | `active_dimension/e12_mle_geometry_demo.py:102` | **fixed** |

## The silence control, run for the first time

Requirement 4 asks that setting the learning rate to zero silence the observer: the
parameters cannot move, so an observer of the optimiser must be flat, and a constant log is
the right answer. `sys.silence` runs it on every constructed system and scores the same
observers at the frozen configuration.

Section 5.3 says the control "invalidated two of our six systems". On this evidence it
invalidates **five**, and the two the article names are among them:

| system | observers | still moving at zero learning rate | reproduce their estimate |
| --- | --- | --- | --- |
| oscillating matrix | 13 | 0 | 0 |
| parameter subspace | 12 | 1 | 0 |
| frozen decoder | 7 | 3 | 2 |
| function subspace | 9 | 3 | 2 |
| online linear regression | 6 | 3 | **3** |
| logistic regression | 7 | 3 | **3** |
| subspace perceptron | 7 | 2 | 2 |

The failing observers are the same three everywhere — the loss, the gradient norm and the
gradient projection — and the mechanism is the one section 5.3 already names: the drive
acts on the targets, so the residual keeps moving while the parameters are fixed. What is
new is the extent. The correlation between the trained and the silenced series runs from
0.94 to 0.998, and the estimate is reproduced to within a tenth of a component on nine of
the twelve failing observer-system pairs: the frozen-parameter run returns the same number.

Two qualifications. This is a reduced run, so the cell counts are small and the exact
values will move; the pattern is structural rather than sampled, since an observer that
reads the drive rather than the optimiser does so at any grid size. And a system is only
invalidated for the observers it is *scored* on — a row scored on an observer that does go
silent stands. Which rows those are is a decision for whoever revises `tab:ladder`, and the
per-observer table in `runs/sys.silence/silence.csv` is what it should be decided on.

## I. A defect in the port itself

Recorded because it is the failure this rewrite most nearly repeated, and because the
guard against it is now the thing to preserve.

A `--fast` run writes the right files with the right columns and the wrong numbers. The
frozen configuration is read from the calibration run's output, `runs/` before `data/`, so
a twenty-five-second `--fast` run of `calib.e8` left `max_E = 10` and a different Theiler
rule where every downstream experiment and the whole test suite read the estimator. It
shadowed the committed configuration silently: a configuration file that exists is a
configuration file that loads. An overnight campaign would have run to completion at a
plumbing-check estimator with nothing in any output to say so.

`actdim.runtime.store.is_plumbing_check` is now consulted in three places — promotion into
`data/`, resolution of a downstream input, and the "has this already run" test that decides
what a campaign skips — and `tests/test_runtime.py` holds all three.

## What this means for the article

Items 3, 5, 6, 9, 11, 12, 13, 23, 32, 33, 34 and 35 are text or caption changes with no
re-run behind them. Item 33 is a contradiction inside one table, so it needs correcting
whatever else happens.
Item 1 changes numbers in section 5 and in `tab:ladder` once the constructed systems are
re-run. Item 10 needs the control to be run before section 5.3 can keep its sentence.
Items 15 and 16 need GPU time before appendices H and J can be regenerated at all.
