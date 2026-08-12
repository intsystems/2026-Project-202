# Colab mode

A way to borrow a GPU. It is not a mode the package is built around: `actdim` never
imports anything here, and everything in this directory can be deleted without affecting
a single result.

| file | what it is |
| --- | --- |
| `actdim_colab.ipynb` | the notebook to open in Colab |
| `bootstrap.py` | `setup` / `plan` / `run` / `save` / `unpack`, callable from a cell |

## The shape of a session

```python
from colab.bootstrap import setup, plan, run, save
setup()                            # checks the GPU, fails early if there is none
plan("train")                      # what it will run and what it will cost
run("train.perceptron.eos")        # any actdim target
save(only=["train.perceptron"])    # tar runs/ and download it
```

Then, on the machine holding the repository:

```bash
python -c "import sys; sys.path.insert(0,'.'); from colab.bootstrap import unpack; unpack('~/Downloads/actdim_runs.tar.gz')"
python -m actdim promote train
```

`promote` is what moves a result from the untracked `runs/` tree into the tracked `data/`
tree, with its checksum and its provenance. Nothing is committed until then.

## What this replaces, and why it is smaller

The archived tree drove Colab from a PowerShell wrapper around a third-party CLI: a
staged zip of four source directories, an uploaded JSON array of argv arrays, a detached
process group, a pid file, a poller, and a tar fetch. It also needed a companion repair
tool, because a read racing a write in the CLI's session store could persist an empty
session map over a live one, and because a network timeout could make an empty session
listing look like an absent session and orphan a running GPU.

None of that is needed to run a campaign. A cell holds the process; if the notebook dies,
the runtime stops and nothing is left behind to reconcile. What the old layer bought was
the ability to close the laptop during a long job, and Drive buys that back more simply.

## A session is shorter than the full campaign

The training half of appendix O is about eleven T4-hours, longer than one Colab session
will last. Run it a group at a time:

| target | roughly |
| --- | --- |
| `train.perceptron.eos` | 0.5 h |
| `train.perceptron.arith` | 0.6 h |
| `train.perceptron.poly` | 1.2 h |
| `train.perceptron.sketched` | 0.5 h |
| `train.transformer.sketched` | 0.4 h |
| `train.transformer.extended` | 2.5 h |
| `train.transformer.p211` | 2 to 4 h |

Save after each. With Drive mounted, `save(drive_dir=...)` writes there instead, and a
recycled runtime costs nothing.

## The analysis half needs none of this

Everything in sections 5 and 6, and every figure, is CPU work. Run it locally:

```bash
python -m actdim run sys valid --jobs 8
```

`setup(require_gpu=False)` lets it run here too, but there is no reason to.
