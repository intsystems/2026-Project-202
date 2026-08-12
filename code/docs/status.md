# What is ported, and what is not

The port is not finished. This file says exactly where it stands, so that the gaps are
visible rather than discovered by running something and getting nothing.

`python -m actdim list` is the live version of the top half; this file adds what is
missing, which a catalogue of what exists cannot show.

## Ported, tested, and runnable

| area | state |
| --- | --- |
| runtime: device, seeding, provenance, storage, catalogue, CLI | complete |
| the estimator: embedding, neighbours, MG/LB/TwoNN, companions, diagnostics, surrogates, windows | complete, and equal to the archived implementation on real logged series to 1.6e-13 on the estimate |
| the two frozen configurations | committed under `data/calib.e8/` and `data/calib.e20/`, loaded by `actdim.frozen` |
| the drive, the observers, six of the seven ladder systems | complete; the seed defect is fixed and tested |
| the transformer, its tasks, and the run registry of appendix O | complete, all fourteen rows |
| the perceptron, its tasks, the closed form of appendix M | complete, all thirteen rows |
| the trajectory sketch and its non-invasiveness check | complete; the check is a test that trains twice and requires bit-identical logs |
| the edge-of-stability campaign | complete |
| the twelve figures | complete, and pixel-identical to the published ones |
| the training experiments (`train.*`) | wired |
| the grokking experiments (`grok.*`) | wired, all nine. Seven reproduce their archived table from the logs their `needs` name; `grok.repr` is new and has none to reproduce; `grok.prwindow` cannot be run at all until `train.perceptron.sketched.long` has been trained, as below |
| the ladder experiments (`sys.*`) | wired for the six ported systems |
| the figure experiment (`paper.figures`) | wired |

257 tests pass. `python -m pytest tests/ -q`.

## Not ported

Each of these is a piece of the article that cannot yet be regenerated. The archived
implementation is named so the work can be picked up.

| what | where the archived version is | what depends on it |
| --- | --- | --- |
| **The parameter-subspace system** | `archived_code/active_dimension/{system,dynamics}.py` | rows six and seven of `tab:ladder`, section 5.3's silence control, section 5.4, section 6.1. Named in `actdim.systems.spec.NOT_PORTED`. |
| **The synthetic generators** (quasiperiodic, Ornstein-Uhlenbeck, coloured noise) | `archived_code/active_dimension/generators.py` | `valid.regime`, `valid.tau`, `valid.anisotropy` |
| **The `valid.*` experiments** | `archived_code/active_dimension/e0`, `e3`, `e4`, `e6`, `e7b`, `e8`, `e10_ceiling_sweep` | section 6 entire, appendices E, N, P, R |
| **The `calib.*` experiments** | `archived_code/active_dimension/e1_calibration.py`, `calibration_k20.py` | appendix C. The configurations they select are committed, so only re-selection needs them. |
| **The table auditor** | `archived_code/active_dimension/paper_tables.py` covers two tables | `check.tables`. `actdim/tables.py` exists but is unfinished. |
| **The sketch cost and non-invasiveness experiments** | `archived_code/gromov_arithmetic/sketch_cost.py`, `verify_*.py` | appendix S. The library functions are ported; only the experiment wrappers are missing. |

`actdim.runtime.archive` maps every one of those experiments' outputs to the archived file
it should reproduce, and `data/` is already seeded from them, so the figures and the
article build today. Every seeded file is marked `source: archived` in
`data/manifest.json`; `python -m actdim verify` lists them.

## Two things no code can recover

The trajectory sketches behind appendix H's fine windows and behind appendix J's
window-length sweep were never kept, and the scripts that consumed them exit when they
find none. Those two appendices need GPU time before they can be regenerated at all:
`train.transformer.sketched` and `train.perceptron.sketched.long`. Everything else in the
article can be rebuilt from `data/`.

## Where the numbers will move

`docs/errata.md` is the full register. The four that change a published value are the seed
defect in the drive, the Theiler cap, the discarded null statistics on degenerate windows,
and the window-length labels of appendix J. Item 31 records a cost the drive fix carries
with it, which needs a decision rather than a default.
