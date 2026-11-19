"""Everything an experiment needs that is not about dimension.

Device selection, seeding, provenance, output storage, the experiment catalogue and the
command line. Nothing here knows what the article is about; nothing above here repeats
any of it.
"""
from __future__ import annotations

from experiment_runner.context import Context, build
from experiment_runner.determinism import pin_blas_threads, rng, seed_map, stream_seed
from experiment_runner.device import describe, require_gpu, resolve
from experiment_runner.parallel import default_jobs, map_ordered
from experiment_runner.provenance import Provenance, sha256
from experiment_runner.registry import CPU, GPU, Experiment, experiment, get, order, select
from experiment_runner.store import Store, repo_root, runs_root

__all__ = [
    "Context", "build", "Store", "Provenance", "Experiment", "experiment",
    "CPU", "GPU", "get", "order", "select",
    "resolve", "describe", "require_gpu",
    "rng", "seed_map", "stream_seed", "pin_blas_threads",
    "map_ordered", "default_jobs", "sha256",
    "repo_root", "runs_root",
]
