# Architecture

The package is `actdim`. It runs from this directory without being installed:

    python -m actdim list

`pip install -e .` also works and adds an `actdim` command; nothing depends on it.

## The shape of it

    actdim/
      runtime/        device, seeding, provenance, storage, the catalogue, the CLI
      estimator/      the delay reconstruction and the dimension estimate
      systems/        the systems whose active dimension is fixed by construction
      models/         the two networks
      tasks/          the label functions the networks are trained on
      training/       the two training loops and the run registry of appendix O
      sketch/         the trajectory sketch and its non-invasiveness check
      figures/        the article's figures
      experiments/    one module per group; the entry points the CLI runs
    data/             the small tracked subset the article reads
    runs/             raw output, regenerable, not tracked
    tests/
    colab/            the Colab mode, which nothing else imports
    docs/

Three rules hold the layout together.

**Nothing below `experiments/` writes a file.** Library code returns arrays and frames.
The experiment module decides what to store. This is why the estimator can be tested
without a filesystem, and why an experiment's outputs are listed in one place.

**Nothing above `runtime/` resolves a device, derives a seed, or builds an output path.**
Those three were spread across the archived tree and each produced a defect: an
unresolved `"auto"` written into a published measurement, a `hash()`-seeded surrogate
that no second process could reproduce, and a figure written to the wrong directory
because the path was pinned next to the code rather than derived from the run.

**One implementation of anything that appears in more than one place.** The archived tree
had four participation ratios, three trend-crossing counts, three copies of the drive
frequencies with different prime tables, and two IAAFT surrogates with the same name and
different algorithms. Each is now single.

## The experiment contract

An experiment is a function taking a `Context`, registered with a decorator:

```python
from actdim.runtime import CPU, GPU, Context, experiment

@experiment(
    id="sys.matrix",
    title="Recovery on the oscillating diagonal matrix",
    paper=("sec:matrix", "tab:ladder"),
    device=CPU,
    minutes=2,
    needs=("calib.e8",),
    promotes=("recovery.csv", "observer_ranking.csv"),
    tier=1,
)
def run(ctx: Context) -> None:
    cfg = load_frozen(ctx.input("calib.e8", "frozen_config.json"))
    ctx.config(k_max=10, cycles=1000)
    rows = [...]
    ctx.store.table("recovery.csv", pd.DataFrame(rows))
```

* `id` is stable and is what the article's tables cite. Never rename one silently.
* `needs` names upstream experiments; the runner orders them and refuses a cycle.
* `promotes` names the outputs the article reads. `actdim promote` copies exactly those
  into `data/<id>/` and records each with its checksum in `data/manifest.json`.
* `minutes` is the measured cost on eight cores, or on a T4 where `device=GPU`. It only
  has to be right to a factor of two; it exists so `actdim plan --all` can say what a
  regeneration costs before it starts.

### What `Context` gives you

| call | what it does |
| --- | --- |
| `ctx.store.table(name, frame)` | write a CSV, checksummed, rows in the order given |
| `ctx.store.array(name, **arrays)` | write a compressed `.npz` |
| `ctx.store.json(name, obj)` / `.text(name, s)` | write JSON or text |
| `ctx.input(experiment_id, name)` | path to an upstream output, `runs/` then `data/` |
| `ctx.rng(role)` | a NumPy generator for a named stream, recorded |
| `ctx.seed_for(role)` | the integer seed for a named stream |
| `ctx.config(**values)` | record the resolved configuration |
| `ctx.note(key, value)` | record anything else worth keeping |
| `ctx.device`, `ctx.jobs`, `ctx.fast` | resolved device, worker count, smoke-test flag |

`ctx.fast` is set by `--fast`. Under it an experiment runs the smallest grid that
exercises every branch, so the plumbing can be checked in seconds. It must still write
the same files with the same columns; only the number of rows changes.

### Determinism

Every stream comes from `ctx.rng(role)`, seeded by a stable function of the base seed and
the role name. Never `hash()`, never an unseeded default generator, never `np.random.*`
module state.

Results collected from a process pool are sorted before writing. `map_ordered` returns
results in input order for exactly this reason: the archived raw CSVs were written in
pool-completion order, so a re-run produced the same numbers in a different order and no
diff could be taken.

Worker processes call `pin_blas_threads(1)`. Nested parallelism over a threaded BLAS
oversubscribes the machine, and worse, makes the reduction order depend on the thread
count, which moves the low bits of a result between differently loaded machines.

## The data policy

`runs/` is where experiments write. It is not tracked and it is large: the trajectory
sketches alone are hundreds of megabytes.

`data/` is tracked and small. A file gets there only by being named in an experiment's
`promotes` and copied by `actdim promote`. `data/manifest.json` records, per file, the
producing experiment, the command that rebuilds it, its checksum, and the commit the
producing run was at. `actdim verify` checks the tree against it.

The rule this enforces: **no number in the article comes from a file whose producer is
unknown.** The archived tree committed two files that their own scripts could no longer
reproduce, and one that two different scorers both claimed to write.

## Where the article's numbers come from

`docs/experiments.md` carries the full table, section by section. `docs/errata.md`
records what the port found wrong in the archived code and what changed as a result.
