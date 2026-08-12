"""The compression: feature hashing with random signs, from R^n to R^dim.

Keeping every checkpoint of a 226,816-parameter run at two thousand logged steps is 3.7
gigabytes, so each logged step is compressed instead. A CountSketch (Charikar et al. 2002;
Weinberger et al. 2009) sends each input coordinate to one of ``dim`` buckets with a
random sign, once and for all, so the Gram matrix of the logged points -- and therefore the
covariance spectrum, and therefore the participation ratio the article reads -- survives
the compression up to a bounded error.

The error is measured rather than trusted. ``n_sketch`` independent hash families are
carried side by side and every reported value is their mean, with their disagreement
recorded beside it. Appendix I gives that disagreement over the published transformer
windows: 1.1 per cent of the value at the median, 8.4 per cent at its worst.

The hashes come from NumPy, never from the global torch generator. The training runs seed
one torch stream that the train/validation split, the initial weights and the mini-batch
order all continue, so a single draw taken here would change the run being observed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def hash_family(n_in: int, n_out: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """The bucket index and the +/-1 sign of one CountSketch ``R^n_in -> R^n_out``.

    The two draws and their order are the published ones; changing either gives a
    different, equally valid sketch that no longer reproduces a stored one.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_out, size=n_in)
    sign = rng.integers(0, 2, size=n_in) * 2.0 - 1.0
    return idx, sign


class CountSketch:
    """``n_sketch`` independent hashes of one space, applied to a flat vector.

    The hashes are drawn on construction and stored as NumPy arrays. They are moved to a
    torch device and dtype on first use and cached there, so that a sketch applied
    thousands of times costs one host-to-device copy rather than one per call.
    """

    def __init__(self, n_in: int, dim: int = 1024, n_sketch: int = 2,
                 seeds: Optional[List[int]] = None, seed: int = 0):
        if n_in < 1:
            raise ValueError(f"n_in must be >= 1, got {n_in}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.n_in = int(n_in)
        self.dim = int(dim)
        self.n_sketch = int(n_sketch)
        if seeds is None:
            rng = np.random.default_rng(seed)
            seeds = [int(rng.integers(2 ** 31)) for _ in range(self.n_sketch)]
        if len(seeds) != self.n_sketch:
            raise ValueError(f"expected {self.n_sketch} seeds, got {len(seeds)}")
        self.seeds = [int(s) for s in seeds]
        self._families = [hash_family(self.n_in, self.dim, s) for s in self.seeds]
        self._cache: Dict[Tuple[Any, Any], List[Tuple[Any, Any]]] = {}

    def _on(self, vec: Any) -> List[Tuple[Any, Any]]:
        import torch

        key = (str(vec.device), vec.dtype)
        if key not in self._cache:
            self._cache[key] = [
                (torch.from_numpy(idx).to(device=vec.device, dtype=torch.long),
                 torch.from_numpy(sign).to(device=vec.device, dtype=vec.dtype))
                for idx, sign in self._families
            ]
        return self._cache[key]

    def apply(self, vec: Any) -> Any:
        """Sketch one flat torch vector. Returns ``(n_sketch, dim)`` on the same device."""
        import torch

        if vec.ndim != 1:
            raise ValueError(f"expected a flat vector, got shape {tuple(vec.shape)}")
        if vec.numel() != self.n_in:
            raise ValueError(
                f"sketch was built for {self.n_in} coordinates but was given "
                f"{vec.numel()}. The parameter vector must not change size mid-run.")
        out = torch.zeros(self.n_sketch, self.dim, dtype=vec.dtype, device=vec.device)
        for s, (idx, sign) in enumerate(self._on(vec)):
            out[s].index_add_(0, idx, sign * vec)
        return out

    def apply_numpy(self, vec: np.ndarray) -> np.ndarray:
        """The same sketch on a NumPy vector, for tests and for offline checks."""
        vec = np.asarray(vec, dtype=float).ravel()
        if vec.size != self.n_in:
            raise ValueError(f"sketch was built for {self.n_in} coordinates, got {vec.size}")
        out = np.zeros((self.n_sketch, self.dim), dtype=float)
        for s, (idx, sign) in enumerate(self._families):
            np.add.at(out[s], idx, sign * vec)
        return out
