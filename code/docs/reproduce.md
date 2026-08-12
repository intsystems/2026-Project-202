# Regenerating the article

Four stages, in order. Each is resumable: an experiment that has already run is skipped
unless it is named on the command line or `--force` is given.

```bash
python -m actdim plan --all      # the whole order, and what it costs, before starting
```

`plan` shows the measured wall time for anything that has already run and the declared
estimate for anything that has not, so the total gets more honest as the campaign proceeds.

## Running it unattended

On a machine with a GPU, the whole thing is one command:

```bash
python -u -m actdim run --all --device cuda:0 --jobs 8 --keep-going --promote \
    2>&1 | tee actdim.log
```

`python -u` matters more than it looks. Redirected stdout is block-buffered, so without it
the log can show nothing for minutes at a time and a hang is indistinguishable from work.
The runner flushes its own progress lines; a library that prints will not.

Four flags matter for an overnight run.

`--keep-going` is the important one. Without it the first failure stops the campaign and
the rest of the night is wasted; with it, every other experiment still runs and the
summary at the end names what failed and prints the command that retries only those.

`--promote` copies each experiment's article-facing outputs into `data/` as it finishes,
so a campaign that dies half way still leaves the half it completed usable. A `--fast` run
is never promoted, whatever the flag says: its outputs have the right columns and the
wrong numbers.

`--jobs` should be the core count less one or two. Every worker pins BLAS to one thread,
so the pool is the only parallelism and over-subscribing makes it slower rather than
faster.

`--device cuda:0` rather than `auto` fails loudly if the GPU is missing, instead of
quietly skipping nine training campaigns and leaving section 7 with nothing to read.

Before starting, the runner prints what the campaign will need — the number of steps, the
CPU and GPU time, and the free disk. A full regeneration writes about 1.2 GB into `runs/`.

Afterwards:

```bash
python -m actdim diff --all      # every number that moved against the archived results
python -m actdim run check.tables   # every table cell against what the article prints
python -m actdim verify          # data/ against the checksums recorded when it was written
```

## 0. Check the machine

```bash
python -m actdim doctor
```

Reports the device, the library versions, the usable cores and the free disk. A full
regeneration writes about 1.2 GB into `runs/`. The library versions matter: the digits
system reads scikit-learn's bundled dataset and fits an L-BFGS model whose low bits move
with the version, and both trainers are float64, where CPU and CUDA differ in the last
bits.

## 1. Calibration — the two frozen configurations

```bash
python -m actdim run calib --jobs 8          # about 1.5 hours on eight cores
```

Everything downstream depends on these. They are also committed, each with the rest of its
calibration run's outputs, as `data/calib.e8/frozen_config.json` and
`data/calib.e20/frozen_k20.json`, so this stage can be skipped entirely and the rest of the
article rebuilt against the configurations it was written from. Re-run it only to check the
selection, and note that requirement 2 forbids reselecting on any later outcome.

A `--fast` calibration is refused as a source: it selects a configuration like a real one
and writes it to the same file, and the configuration it selects is a plumbing check.

## 2. The analysis half — CPU, about twelve core-hours

```bash
python -m actdim run sys --jobs 8            # section 5, the constructed systems
python -m actdim run valid --jobs 8          # section 6, the conditions of validity
```

The two long items are the rank sweep, at about two and a half hours, and the ceiling
scan, at about three. Everything else is minutes. Nothing here needs a GPU or a network.

## 3. The training half — GPU, about eleven T4-hours

```bash
python -m actdim run train --device cuda:0
```

Longer than one Colab session, so run it a group at a time and save after each; see
[`../colab/README.md`](../colab/README.md). On a Linux box with a GPU it is one command.

Two appendices cannot be rebuilt without this stage at any price: the sketches behind the
effective-rank collapse of appendix H and behind the window-length sweep of appendix J
were not kept, and no committed file can regenerate them. Everything else in the article
can be rebuilt from `data/` alone.

## 4. Analysis of the training runs, then figures

```bash
python -m actdim run grok --jobs 8           # section 7 and its appendices
python -m actdim run check --jobs 8          # the sketch checks and the cost measurement
python -m actdim run paper                   # figures into ../icomp_v2/figures/
```

Then rebuild the document:

```bash
cd ../icomp_v2
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

## Promoting and committing

Nothing enters the tracked tree by running an experiment. It enters by promotion:

```bash
python -m actdim promote sys valid grok      # copies declared outputs into data/
python -m actdim verify                      # checks data/ against the manifest
git add data/ && git commit
```

`runs/` can be deleted afterwards. The tracked half is enough to rebuild every figure and
every table.

## What to expect to differ from the archived numbers

The port fixes defects, and fixing them moves values. The full list is in
[`errata.md`](errata.md); the ones that change a number are:

* **Section 5 and `tab:ladder` will move.** The archived held-out seeds reused seed 0's
  frequency geometry, so the recovery errors were measured on a split that held out less
  than the article claims. The corrected construction varies the geometry with the seed.
* **The transient's estimate** was reported at a Theiler exclusion capped for speed rather
  than at the rule the configuration names. The cap is now a stated configuration field,
  so a run at the rule is a run away.
* **Null statistics on degenerate windows** are no longer discarded, which raises their
  sample counts in the arms where windows are marked, the transient one in particular.
* **The window-length labels of appendix J** were one logging interval long.

Nothing else in the port is intended to change a value. Where a re-run disagrees with the
committed file for any other reason, that is a bug: `python -m actdim verify` will show
it, and the provenance records on both sides will say what differed.
