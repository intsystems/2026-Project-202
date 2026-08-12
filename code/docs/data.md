# Where the data lives

Two trees, with different rules. The split exists because the article needs about three
megabytes of numbers and the experiments that produce them write about a gigabyte.

## `runs/` — everything an experiment writes

Untracked, regenerable, large. One directory per experiment id:

    runs/sys.matrix/
      recovery.csv
      observer_ranking.csv
      provenance.json

Trajectories and trajectory sketches live here. A single sketched transformer run is 63
to 95 MB of `.npz`; the calibration trajectories of the twenty-direction configuration are
197 MB. None of it is committed, and none of it needs to be: every file is reproduced by
the command its provenance record names.

### `provenance.json`

Written by the runtime, once per run, beside the outputs:

```json
{
  "experiment": "sys.matrix",
  "started_utc": "2026-08-12T18:03:11Z",
  "wall_seconds": 94.2,
  "status": "ok",
  "device": "cpu",
  "git": { "sha": "6f77680...", "dirty": false, "branch": "main" },
  "libraries": { "python": "3.9.13", "numpy": "2.0.2", "sklearn": "1.6.1" },
  "config": { "k_max": 10, "cycles": 1000 },
  "seeds": { "drive_phases": 31, "observer_directions": 555 },
  "inputs": { "calib.e8/frozen_config.json": "..." },
  "outputs": { "recovery.csv": { "sha256": "...", "bytes": 8112, "rows": 240 } }
}
```

The library versions are there because they move numbers. The digits system reads
scikit-learn's bundled dataset and fits an L-BFGS model whose low bits are
library-sensitive; both trainers are float64, where CPU and CUDA disagree in the last
bits. A result whose scikit-learn version is unknown cannot be compared with one whose is.

The archived tree recorded none of this, and the cost was concrete: two committed files
could no longer be reproduced by their own scripts, one file was written by either of two
scorers with no record of which, and one campaign's configuration was overwritten by a
later edit to the registry it was rebuilt from.

## `data/` — the small tracked subset

Committed. A file arrives here only by being named in an experiment's `promotes` and
copied by `python -m actdim promote <id>`. Every one is listed in `data/manifest.json`
with its checksum, the experiment that produced it, the command that rebuilds it, and the
commit the producing run was at.

```bash
python -m actdim verify     # every tracked file against its recorded checksum
```

Three kinds of file live here:

| | |
| --- | --- |
| `data/frozen/` | the two frozen estimator configurations. Committed so the downstream half of the article can be rebuilt without re-running calibration, which is the most expensive root. |
| `data/<experiment id>/` | what the figures read and what the tables quote |
| `data/manifest.json` | the index |

The rule it enforces: no number in the article comes from a file whose producer is
unknown.

## `local/` — source material

Untracked, and nothing reads it. A place to keep the archived result files, a fetched
Colab archive, or anything else worth having on disk and not in the repository.

## Formats

**Tables are CSV**, written with the rows in the order the experiment produced them, which
is input order and not process-pool completion order. This is what makes a re-run
diffable against a committed file.

**Arrays are compressed `.npz`**, with the metadata stored alongside as named arrays rather
than encoded in the filename. The archived scorers recovered a run's experimental arm from
a filename prefix, so renaming a file lost its arm.

**Configurations are JSON**, never a tuple index into a grid defined in source. Three
archived experiments recorded their selected configuration as an integer whose meaning was
a literal in the script, so the configuration could not be read without the version of the
code that wrote it.

## What it costs to keep

Regenerating everything writes roughly 1.2 GB into `runs/`, of which about 3 MB is
promoted. If disk is short, `runs/` can be deleted after promotion: the tracked half is
enough to rebuild every figure and table, and `actdim plan` will say what re-running the
rest would cost.
