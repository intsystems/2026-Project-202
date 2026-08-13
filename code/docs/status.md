# What is ported, and what is not

`python -m actdim list` is the live catalogue. This file says what the catalogue cannot:
which parts have been run against real data, which have only been exercised, and what is
known to be missing.

## The port is complete

Every section of the article has an experiment. Forty-one of them, 14.4 hours of CPU work
and 9.1 of GPU, and `python -m actdim plan --all` prints the order.

| area | state |
| --- | --- |
| runtime: device, seeding, provenance, storage, catalogue, CLI | complete |
| the estimator | complete, and equal to the archived implementation on real logged series to 1.6e-13 on the estimate |
| the drive, the observers, all seven ladder systems | complete; the seed defect is fixed and tested |
| the transformer and the perceptron, with the run registry of appendix O | complete: all fourteen transformer rows and all thirteen perceptron rows |
| the trajectory sketch | complete; a test trains the same configuration twice and requires every logged value bit-identical |
| the twelve figures | complete, and pixel-identical to the published ones |
| the table auditor | complete: 21 of 28 tables, 731 cells and claims |
| `calib.*`, `sys.*`, `valid.*`, `train.*`, `grok.*`, `check.*`, `paper.*` | all wired |

401 tests pass. `python -m pytest tests/ -q`.

## How far each part has been verified

Verification here means run against real data and compared with the archived result, not
merely exercised.

**Reproduces the archived result bit for bit.** `grok.extended.outcomes`,
`grok.diagnostics.logs`, `grok.eos` (all three tables), `grok.rank.dip`,
`grok.matched.window`'s headline trace, and the polynomial half of
`grok.diagnostics.perceptron`. On the arithmetic half every shared window agrees to the
bit on all seven statistics.

**Reproduces it with a stated difference.** `grok.matched.window`'s grid tables differ by
at most 4.4e-5, because the archived grids were computed from a copy rounded to five
decimals; all 72 rows align and the verdicts are identical. `grok.matched.surrogate`'s
individual depths differ because the archived stream was seeded from `hash()`, which
Python salts per interpreter; the verdict is unchanged and `tab:matched` reproduces
exactly.

**Run, but not yet compared at full size.** The six ladder rows, `sys.silence`, and the
`valid.*` set apart from `valid.geometry`. `sys.matrix` at full size recovers with a mean
absolute error of 0.27 and a rank correlation of 1.0, against the published 0.30 and 0.69 —
the corrected held-out split did not cost recovery on that system.

**Run at full size and compared.** `valid.geometry`, the last experiment ported. Every cell
of its four tables moved, because its two clock arms are seeded through the contract instead
of from a bare literal (errata 38); both pre-specified verdicts still pass and the six
headline numbers moved by less than the seed-to-seed spread. `python -m actdim diff
valid.geometry` prints the comparison column by column.

**Exercised only.** The `train.*` campaigns and the two `check.sketch.*` experiments have
run at reduced size on a CPU. They need the GPU box.

## What still needs a decision

**Errata item 31.** The corrected drive excites its directions less evenly than the
published one — 0.86 k against 0.94 to 0.96 k on the function-subspace system — while
holding a resonance margin two orders of magnitude better. Every hard rank is still exactly
k. Neither construction dominates. A test holds the measured ratio so it cannot drift while
the question is open.

**The silence control.** Run for the first time, it invalidates five systems where section
5.3 says two. Which rows of `tab:ladder` that changes depends on the observers each row is
scored on, which is an editorial decision; `runs/sys.silence/silence.csv` is the evidence.

## Two things no code can recover

The trajectory sketches behind appendix H's fine windows and appendix J's window-length
sweep were never kept. `grok.rank.dip` and `grok.prwindow` therefore refuse to run until
`train.transformer.sketched` and `train.perceptron.sketched.long` have been trained, and
say so with the command that does it. Everything else in the article rebuilds from `data/`.

## Where the numbers will move

`docs/errata.md` is the register: thirty-eight items, plus one the port introduced and
closed. Twelve are text or caption corrections needing no re-run. Four change a published
value: the seed defect in the drive, the Theiler cap, the null statistics discarded on
degenerate windows, and the window-length labels of appendix J.
