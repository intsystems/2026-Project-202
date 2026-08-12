"""The two-layer quadratic perceptron of appendix O, and the spectra of its weights.

Reference: A. Gromov, *Grokking modular arithmetic*, arXiv:2301.02679, Sec. 2. The
architecture is Eqs. (1)-(2), and for the quadratic activation the whole network
collapses to Eq. (4):

    f(x) = 1 / (D N) * W2 (W1 x)^2 ,      W1: N x D,  W2: p x N,  D = S p

There are no biases. The input is the one-hot encodings of the ``S`` operands
concatenated, so ``D = S p``; the target is the one-hot encoding of the answer. At
``p = 97`` and width ``500`` that is the 145,500 parameters appendix O quotes.

**Where the normalisation lives, and why it matters.** The weights are drawn from
``N(0, 1)`` and the ``1/(D N)`` is applied in the forward pass. It is not folded into
the initialisation. The two conventions are not cosmetic variants of each other:

* With ``N(0, 1)`` and the prefactor in the forward pass, the output at step zero is
  about zero, so the mean-reduced MSE starts at ``1/p`` -- 0.0105 at ``p = 97``, which
  is exactly what the source paper's Fig. 0a shows -- and the normalised weight norm
  starts at 1.0, again Fig. 0a. Both readings are free calibration checks that the
  implementation matches the paper.
* The reference implementation of Doshi et al. (arXiv:2406.03495) folds the
  normalisation into the initialisation instead. That rescales the loss landscape, and
  with it the usable learning rate, by orders of magnitude.

Mixing the two is not a rounding error, it silently changes the dynamics, and the
archived registry carries two dead run families that were killed by exactly that mix:
an AdamW arm at ``weight_decay = 8.0`` and a paper-faithful arm at ``5.0``, both
numbers taken from the other parametrisation, where decoupled decay removes more of the
norm per step than a task gradient of order ``2/(p^3 N)`` can restore. The weights went
to zero before the task was seen. ``actdim.training.runs_perceptron`` keeps both as
documentation. This module implements one convention and says which; a run that wants
the other needs the other parametrisation written, not its hyperparameters copied.

The learning rate is stated nowhere in the source paper. See ``gd_lr`` in
``actdim.training.runs_perceptron`` for the one this study calibrated and uses.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch

ACTIVATIONS = {
    "quadratic": lambda h: h * h,
    "quartic": lambda h: h ** 4,
    "relu": torch.relu,
    "abs": torch.abs,
    "gelu": torch.nn.functional.gelu,
}


class QuadraticPerceptron(torch.nn.Module):
    """Eqs. (1)-(2): ``f(x) = (1/N) W2 phi(W1 x / sqrt(D))``, no biases.

    The name follows the article, which calls the quadratic activation the defining
    choice; the other activations are here because the source paper sweeps them and a
    run that wants one should not need a second class.
    """

    def __init__(self, p: int, width: int, n_vars: int = 2,
                 activation: str = "quadratic", generator: Optional[Any] = None,
                 dtype: Any = torch.float32, device: Any = "cpu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise KeyError(f"unknown activation {activation!r}. "
                           f"Known: {sorted(ACTIVATIONS)}")
        d_in = n_vars * p
        kw = dict(generator=generator, dtype=dtype, device=device)
        self.W1 = torch.nn.Parameter(torch.randn(width, d_in, **kw))
        self.W2 = torch.nn.Parameter(torch.randn(p, width, **kw))
        self.phi = ACTIVATIONS[activation]
        self.activation = activation
        self.d_in, self.width, self.p, self.n_vars = d_in, width, p, n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x @ self.W1.T) / math.sqrt(self.d_in)
        return (self.phi(h) @ self.W2.T) / self.width


def build(p: int, width: int, n_vars: int = 2, activation: str = "quadratic",
          seed: int = 1, dtype: Any = torch.float32,
          device: Any = "cpu") -> QuadraticPerceptron:
    """A model whose weights come from one named stream and nothing else.

    The generator is dedicated to the initialisation, so that changing the data split,
    the batch order or the sharpness measurement cannot move the weights the run starts
    from. Callers that already hold a generator pass the class directly.
    """
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return QuadraticPerceptron(p, width, n_vars, activation, generator=generator,
                               dtype=dtype, device=device)


def n_parameters(p: int, width: int, n_vars: int = 2) -> int:
    """Parameters in the model: ``W1`` is ``N x S p`` and ``W2`` is ``p x N``."""
    return width * n_vars * p + p * width


def flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    """Every parameter as one detached vector, in ``parameters()`` order.

    The trajectory sketch needs one vector and the order has to be stable across the
    run, which ``parameters()`` gives for a module whose parameters are declared once.
    """
    return torch.cat([q.detach().reshape(-1) for q in model.parameters()])


# -- the spectra appendix M reads ----------------------------------------------

def fourier_ipr(block: np.ndarray) -> float:
    """Mean inverse participation ratio of the Fourier power of each row.

    ``IPR = sum_j P_j^2`` on a normalised power spectrum. It is 1.0 when one frequency
    carries a whole row, which is what the closed-form solution of appendix M looks
    like, and about ``1/p`` when the row is spectrally flat, which is what random
    initialisation looks like. This is the order parameter of Doshi et al., Eq. (3),
    and the cheapest signal that a periodic representation has formed.

    It is only interpretable against the reference of the task in hand:
    ``actdim.analysis.representation.reference`` computes that, and appendix M shows
    two tasks whose own reference is near the floor.
    """
    power = np.abs(np.fft.rfft(np.asarray(block, dtype=np.float64), axis=-1)) ** 2
    total = power.sum(axis=-1, keepdims=True)
    # A row of zeros has no spectrum to normalise. Dividing anyway would return NaN and
    # poison the mean; leaving the row at zero contributes nothing, which is right.
    total = np.where(total > 0, total, 1.0)
    q = power / total
    return float((q ** 2).sum(axis=-1).mean())


def effective_rank(singular_values: np.ndarray) -> float:
    """Participation ratio of a spectrum: ``(sum s^2)^2 / sum s^4``.

    How evenly the directions of a matrix are used, not how many there are. Appendix M
    makes that distinction the point: the closed-form first layer of a ``p = 97``,
    width-500 run is ``500 x 194`` and full rank, and reads 148.8 here, so the number
    measures the spread over those 194 directions rather than counting them.
    """
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    denom = (s2 ** 2).sum()
    return float(s2.sum() ** 2 / denom) if denom > 0 else 0.0


def weight_spectra(w1: np.ndarray, w2: np.ndarray, p: int,
                   n_vars: int = 2) -> Dict[str, float]:
    """The spectral probes of one weight pair, in float64 NumPy.

    Separate from ``spectra`` so that the closed-form weights of appendix M, which are
    built in NumPy and never trained, are scored by the same code as a trained model.
    """
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)

    out: Dict[str, float] = {}
    for v in range(n_vars):
        out[f"ipr_u{v + 1}"] = fourier_ipr(w1[:, v * p:(v + 1) * p])
    out["ipr_w"] = fourier_ipr(w2.T)  # the readout, per neuron, over the answer index

    sv1 = np.linalg.svd(w1, compute_uv=False)
    sv2 = np.linalg.svd(w2, compute_uv=False)
    out["erank_w1"] = effective_rank(sv1)
    out["erank_w2"] = effective_rank(sv2)
    out["w1_norm"] = float(np.linalg.norm(w1) / math.sqrt(w1.size))
    out["w2_norm"] = float(np.linalg.norm(w2) / math.sqrt(w2.size))
    for i in range(5):
        out[f"sv1_{i}"] = float(sv1[i]) if i < sv1.size else float("nan")
        out[f"sv2_{i}"] = float(sv2[i]) if i < sv2.size else float("nan")
    return out


def spectra(model: QuadraticPerceptron) -> Dict[str, float]:
    """``weight_spectra`` of a live model, off the autograd tape."""
    return weight_spectra(model.W1.detach().to(torch.float64).cpu().numpy(),
                          model.W2.detach().to(torch.float64).cpu().numpy(),
                          model.p, model.n_vars)


def weight_norm(model: QuadraticPerceptron) -> float:
    """The norm appendix O's runs log: root mean square over every parameter.

    Normalised by the parameter count so that it starts at 1.0 under the ``N(0, 1)``
    initialisation this module uses, which is the reading that makes the convention
    visible in a log.
    """
    total = (model.W1.detach() ** 2).sum() + (model.W2.detach() ** 2).sum()
    return float(torch.sqrt(total)) / math.sqrt(model.W1.numel() + model.W2.numel())
