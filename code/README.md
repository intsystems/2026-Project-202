# Counting the active degrees of freedom of a training run

The code behind the article in [`../icomp_v2/`](../icomp_v2/): the estimator, the six
systems whose active dimension is fixed by construction, the two training settings, the
trajectory sketch, and the analysis that turns their logs into every number and figure.

The previous tree is archived unchanged in [`../archived_code/`](../archived_code/). This
one is a port, not a copy: what changed and why is in [`docs/errata.md`](docs/errata.md).

## Getting started

```bash
cd code
pip install -r requirements.txt        # analysis only, no torch, no GPU
python -m actdim doctor                # device, libraries, disk, what has run
python -m actdim list                  # every experiment, its cost, its paper section
```

Nothing needs installing: `python -m actdim` runs from this directory. `pip install -e .`
also works and adds an `actdim` command.

## Running something

```bash
python -m actdim plan sys              # what it would run, in order, and what it costs
python -m actdim run sys.matrix        # one experiment, with its prerequisites
python -m actdim run sys --jobs 8      # a whole group
python -m actdim run sys.matrix --fast # the smallest grid that exercises every branch
```

An experiment writes to `runs/<id>/`, alongside a `provenance.json` recording the commit,
the resolved configuration, every seed, the device, the library versions, and a checksum
of every file it wrote. `python -m actdim promote sys.matrix` copies the outputs the
article reads into `data/`, which is the tracked half.

## Where the work runs

The split is not a Colab special case; it is a property of the experiments.

**The analysis half is CPU and runs anywhere**, including a laptop. That is all of
sections 5 and 6, every diagnostic, and every figure: about twelve core-hours for the
lot, dominated by the rank sweep and the ceiling scan.

**The training half needs a GPU.** Both trainers run in float64, which a CPU will do but
slowly enough that a silent fallback wastes hours, so a GPU experiment refuses a CPU
unless `--allow-cpu` says otherwise. About eleven T4-hours in total.

```bash
python -m actdim run train --device cuda:0     # on a Linux GPU box
```

For Colab, [`colab/`](colab/README.md) holds a notebook and a five-function helper.
Nothing in `actdim` imports it, and deleting the directory changes no result.

## The documentation

| | |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | the module map, the experiment contract, the three rules that hold it together |
| [`docs/experiments.md`](docs/experiments.md) | every section of the article, and the experiment that produced its numbers |
| [`docs/data.md`](docs/data.md) | what is stored, where, in what format, and what is tracked |
| [`docs/reproduce.md`](docs/reproduce.md) | regenerating the article end to end, and what each stage costs |
| [`docs/errata.md`](docs/errata.md) | what the port found wrong in the archived code, and what changed |
| [`docs/status.md`](docs/status.md) | what is ported and what is not, with the archived source of each gap |
| [`colab/README.md`](colab/README.md) | the Colab mode |

## Tests

```bash
python -m pytest tests/ -q
```

The estimator is tested against analytic cases with a known answer, against the archived
implementation on real logged series, and against the article's appendix A line by line.
The trajectory sketch is tested by training the same configuration with and without it
and requiring every logged value to be bit-identical, which is the claim appendix I makes
for it.
