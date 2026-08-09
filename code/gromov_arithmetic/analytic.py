"""The closed-form periodic solution of Sec. 3.1, used as a correctness gate.

This file exists to catch the one class of bug that a training run cannot catch:
a wrong normalisation convention.  If ``forward`` disagrees with Eqs. (1)-(2) by a
constant factor, training still produces *some* curve -- it just produces it at a
different learning rate, and nothing in the log says so.  The analytic solution has
no free scale: Claim I fixes the amplitude, and the network either outputs a clean
one-hot delta or it does not.

    W1[k, n]     = A cos(2 pi k n / p + psi1_k)          (Eq. 6)
    W1[k, p + m] = A cos(2 pi k m / p + psi2_k)
    W2[q, k]     = A cos(-2 pi k q / p - psi1_k - psi2_k) (Eq. 7, Eq. 12)

Substituting into Eq. (4) and using Eq. (13) leaves ``A^3 / (2 D)`` on the diagonal,
so ``A = (2 D)^(1/3)``; for p = 97 that is 7.29, and the resulting weight norm is
about 5.2x the ``N(0,1)`` init -- the same order as the 3.7x that Fig. 0a shows a
trained network reaching.

    python analytic.py
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import tasks
from gromov import Config, GromovMLP, build_dataset


def build(cfg: Config, inner=None, outer=None, seed=0):
    """Claim I / Claim II weights for ``h(f1(n) + f2(m)) mod p``.

    ``inner`` is the pair ``(f1, f2)`` applied to the operands before the cosine, and
    ``outer`` is ``h`` applied to the sum.  Both default to the identity, giving plain
    modular addition.  Claim II says the readout is untouched by ``inner``; a
    non-invertible ``outer`` is handled by writing the readout as a *forward* map --
    row ``h(t)`` accumulates the frequency content of ``t`` -- rather than by
    inverting ``h`` as Eq. (19) does, which is what lets ``sum_sq`` work at all.
    """
    p, n_neurons = cfg.p, cfg.width
    rng = np.random.default_rng(seed)
    d_in = cfg.n_vars * p

    # Frequencies 0 .. (p-1)/2 (Sec. 3.1); cycled when N exceeds that range, with
    # independent phases per neuron, which is what suppresses the spurious terms.
    freqs = np.arange(0, p // 2 + 1)
    k = np.resize(freqs, n_neurons)
    psi = rng.uniform(-np.pi, np.pi, size=(cfg.n_vars, n_neurons))

    f = inner or [lambda v: v] * cfg.n_vars
    h = outer or (lambda t: t)

    amp = (2.0 * d_in) ** (1.0 / 3.0)
    w1 = np.zeros((n_neurons, d_in))
    for v in range(cfg.n_vars):
        vals = np.asarray(f[v](np.arange(p))) % p
        w1[:, v * p:(v + 1) * p] = amp * np.cos(
            2 * np.pi * np.outer(k, vals) / p + psi[v][:, None])

    # Readout: row index is h(t), column index is the neuron, phase closes Eq. (12).
    t = np.arange(p)
    rows = np.asarray(h(t)) % p
    contrib = amp * np.cos(-2 * np.pi * np.outer(k, t) / p - psi.sum(axis=0)[:, None])
    w2 = np.zeros((p, n_neurons))
    np.add.at(w2, rows, contrib.T)
    return w1, w2


def evaluate(cfg: Config, fn, w1, w2):
    model = GromovMLP(cfg.p, cfg.width, cfg.n_vars, cfg.activation, dtype=torch.float64)
    with torch.no_grad():
        model.W1.copy_(torch.as_tensor(w1, dtype=torch.float64))
        model.W2.copy_(torch.as_tensor(w2, dtype=torch.float64))
    data = build_dataset(cfg, fn)
    x = torch.as_tensor(np.concatenate([data["x_train"], data["x_val"]]), dtype=torch.float64)
    y = torch.as_tensor(np.concatenate([data["y_train"], data["y_val"]]), dtype=torch.long)
    with torch.no_grad():
        out = model(x)
        target = torch.zeros_like(out)
        target[torch.arange(y.shape[0]), y] = 1.0
        mse = float(((out - target) ** 2).mean())
        acc = float((out.argmax(1) == y).double().mean())
        peak = float(out[torch.arange(y.shape[0]), y].mean())
    norm = math.sqrt((w1 ** 2).sum() + (w2 ** 2).sum()) / math.sqrt(w1.size + w2.size)
    return dict(acc=acc, mse=mse, mean_peak=peak, weight_norm=norm)


CASES = {
    #  task      inner (f1, f2)                     outer h
    "add":    (None, None),
    "sub":    ([lambda v: v, lambda v: -v], None),
    "sq_sum": ([lambda v: v ** 2, lambda v: v ** 2], None),
    "sum_sq": (None, lambda t: t ** 2),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--widths", type=int, nargs="+", default=[50, 100, 200, 500])
    args = ap.parse_args()

    print(f"analytic solution, p={args.p}, amplitude A=(2D)^(1/3)="
          f"{(4 * args.p) ** (1 / 3):.3f}\n")
    header = f"{'task':<9}{'N':>6}{'acc':>9}{'MSE':>12}{'peak':>9}{'|W|':>8}"
    print(header)
    print("-" * len(header))
    for task, (inner, outer) in CASES.items():
        for n in args.widths:
            cfg = Config(p=args.p, width=n, task=task)
            w1, w2 = build(cfg, inner, outer)
            r = evaluate(cfg, tasks.get(task), w1, w2)
            print(f"{task:<9}{n:>6}{r['acc']:>8.2%}{r['mse']:>12.3e}"
                  f"{r['mean_peak']:>9.3f}{r['weight_norm']:>8.3f}")
        print()
    print("expected: every case -> ~100% at N >= ~200 (Fig. 3b: analytic needs N~90-100),")
    print("          mean peak   -> ~1.0, which is what fixes A and hence the convention.")
    print()
    print("Note sum_sq reaches 100%, not the ~51% of Sec. 3.2. That figure comes from")
    print("Eq. (19), which builds the readout from F^-1 and so picks one branch of the")
    print("square root. `build` writes it as a forward map instead -- row F(t) accumulates")
    print("the frequency content of t -- so every preimage of an output index contributes")
    print("to it constructively. The architecture can represent (n+m)^2 mod p exactly.")


if __name__ == "__main__":
    main()
