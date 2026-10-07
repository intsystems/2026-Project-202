"""Seeding, stated once.

Two rules, both learned from the archived tree.

*Never seed from ``hash()``.* Python salts string hashing per interpreter, so a stream
seeded that way is reproducible inside one process and nowhere else. One archived script
seeded surrogates that way and could not regenerate its own committed file.

*Derive every stream from one base seed by an explicit, named rule.* The derivation below
is a stable hash of the role name, so adding a role never moves an existing stream, and
two roles never collide by accident. Roles are named in the provenance record.
"""
from __future__ import annotations

import os
import zlib
from typing import Dict, Iterable, Optional

import numpy as np

# Streams whose seeds appear in published results keep the constants the archived code
# used, so that a re-run reproduces the archived series where nothing else changed.
LEGACY_OFFSETS: Dict[str, int] = {
    "drive_groups": 7717,
    "drive_phases": 31,
    "rotation": 4242,
    "observer_directions": 555,
    "adapter": 7919,
    "sharpness_start": 7_000_000,
}


def stream_seed(base: int, role: str) -> int:
    """A seed for one named stream, derived from the run's base seed.

    ``zlib.crc32`` is used rather than ``hash`` because it is stable across processes,
    interpreters and platforms.
    """
    if role in LEGACY_OFFSETS:
        return int(base) + LEGACY_OFFSETS[role]
    return int((zlib.crc32(role.encode("utf-8")) + 1_000_003 * int(base)) % (2 ** 31 - 1))


def rng(base: int, role: str) -> np.random.Generator:
    """A NumPy generator for one named stream."""
    return np.random.default_rng(stream_seed(base, role))


def seed_map(base: int, roles: Iterable[str]) -> Dict[str, int]:
    """The resolved seeds, for the provenance record."""
    return {role: stream_seed(base, role) for role in roles}


def pin_blas_threads(n: int = 1) -> None:
    """Hold BLAS to ``n`` threads.

    Must run before NumPy is imported to take effect, so worker entry points call it
    first. Nested parallelism -- a process pool over a threaded BLAS -- oversubscribes the
    machine and, worse, makes reduction order depend on the thread count, which moves the
    low bits of a result between runs on differently loaded machines.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))


def seed_torch(base: int, role: str = "torch") -> Optional[int]:
    """Seed torch's global stream, if torch is installed. Returns the seed used."""
    try:
        import torch
    except ImportError:
        return None
    seed = stream_seed(base, role)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
