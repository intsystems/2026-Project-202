"""The modular arithmetic functions of arXiv:2301.02679.

Each entry maps operand arrays to the answer, and the modulus is applied by the caller
that builds the dataset, so one table serves every ``p``. The grouping is the source
paper's own, and it is the reason these particular functions are here: the point of the
reproduction is not that grokking happens, it is that it happens on exactly the
functions the architecture can represent and not on the others.

``GROKKABLE``  Sec. 3: an explicit periodic solution exists (Claim I or Claim II), or
               the paper reports generalisation without giving one (``mul``).
``HARD``       Sec. 4 and App. C: reported to need a training fraction above 0.95, or
               never to exceed one per cent test accuracy.

A run in the second group that quietly generalised would falsify the reproduction
rather than improve it, which is why appendix O lists those rows with their final
validation accuracy rather than dropping them.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

TASKS: Dict[str, Callable[..., np.ndarray]] = {
    # -- with an analytic periodic solution (Claim I, Eqs. 6-7) ----------------
    "add": lambda n, m: n + m,
    "sub": lambda n, m: n - m,
    "sq_sum": lambda n, m: n ** 2 + m ** 2,           # f1(n) + f2(m), Claim II
    "sum_sq": lambda n, m: (n + m) ** 2,              # F(f1 + f2), F not invertible
    # -- learned, no analytic solution given (Sec. 3.3) ------------------------
    "mul": lambda n, m: n * m,
    # -- reported not to grok (Sec. 4, App. C) ---------------------------------
    "mix_quad": lambda n, m: n ** 2 + m ** 2 + n * m,  # needs alpha > 0.95
    "no_grok": lambda n, m: n ** 3 + n * m ** 2 + m,   # never above 1 per cent
}

GROKKABLE: Tuple[str, ...] = ("add", "sub", "sq_sum", "sum_sq", "mul")
HARD: Tuple[str, ...] = ("mix_quad", "no_grok")

DESCRIPTIONS: Dict[str, str] = {
    "add": "n + m mod p -- the headline task, Fig. 0",
    "sub": "n - m mod p",
    "sq_sum": "n^2 + m^2 mod p -- f1(n) + f2(m), Claim II",
    "sum_sq": "(n + m)^2 mod p -- F(f1 + f2) with F not invertible",
    "mul": "n * m mod p -- learnable, no analytic solution given",
    "mix_quad": "n^2 + m^2 + n*m mod p -- not of the form h(f1 + f2); needs alpha > 0.95",
    "no_grok": "n^3 + n*m^2 + m mod p -- App. C: test accuracy never exceeds 1 per cent",
}


def get(name: str) -> Callable[..., np.ndarray]:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}. Known: {sorted(TASKS)}")
    return TASKS[name]


def describe(name: str) -> str:
    return DESCRIPTIONS.get(name, name)


def table(p: int, name: str) -> np.ndarray:
    """The full ``p x p`` answer table, for checking a task definition by eye."""
    n, m = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return np.asarray(get(name)(n, m)) % p
