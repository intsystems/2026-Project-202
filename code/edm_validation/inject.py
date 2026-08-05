"""Inject a known driver into grokking training, to test driver recovery on our own task.

Phase 3 showed cross mapping recovers an injected driver from a 1-D training-loss log on
ResNet-18 / CIFAR-10 with no false positives. That is one architecture, one dataset, and
somebody else's training script. This reproduces the experimental design on the
1-layer transformer and the algebraic tasks of this repo, so the claim rests on two
independent settings rather than one.

The design copies `../poisoned_batch/batch_poisoning.py` deliberately, including the part
that makes it trustworthy:

``LogisticDriver``   deterministic chaos -- the strongest positive, and the case Stark's
                     theorem for forced systems actually covers
``SinusoidDriver``   slow periodic forcing -- the case that differencing destroyed in
                     Phase 3, kept here to check the raw-embedding fix generalises
``IIDDriver``        i.i.d. forcing, genuinely applied -- coupling without determinism
``GhostDriver``      **logged but never applied**: the labels are returned untouched

The ghost is the control that carries the argument. It produces a run with the same
architecture, the same schedule, the same loss trajectory and a driver column that looks
exactly like the others -- differing only in that nothing was ever coupled. If the
detector fires on it, the positives mean nothing.

RNG discipline: the driver and the choice of which examples to corrupt come from a
dedicated NumPy generator. ``grok.tasks`` seeds one global torch stream that the split,
the initialisation and the batch order all share, so drawing from it here would confound
"the driver changed the run" with "the run was a different run".
"""

import numpy as np

__all__ = ["LogisticDriver", "SinusoidDriver", "IIDDriver", "GhostDriver", "DRIVERS",
           "LabelPoisoner"]


class _Driver:
    """Base: produces a scalar in [0, 1] per step, from its own RNG."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def value(self, step):
        raise NotImplementedError


class LogisticDriver(_Driver):
    """x <- r x (1 - x), r = 3.9: deterministic chaos on a 1-D attractor."""

    def __init__(self, seed=0, r=3.9, x0=0.4, low=0.0, high=0.5):
        super().__init__(seed)
        self.r, self.x, self.low, self.high = r, x0, low, high

    def value(self, step):
        self.x = self.r * self.x * (1 - self.x)
        return self.low + (self.high - self.low) * self.x


class SinusoidDriver(_Driver):
    """Slow periodic forcing -- the regime Phase 3's differencing erased."""

    def __init__(self, seed=0, period=400.0, low=0.0, high=0.5):
        super().__init__(seed)
        self.period, self.low, self.high = period, low, high

    def value(self, step):
        phase = 0.5 * (1 + np.sin(2 * np.pi * step / self.period))
        return self.low + (self.high - self.low) * phase


class IIDDriver(_Driver):
    """Applied, but with no temporal structure: coupling without determinism."""

    def __init__(self, seed=0, low=0.0, high=0.5):
        super().__init__(seed)
        self.low, self.high = low, high

    def value(self, step):
        return float(self.rng.uniform(self.low, self.high))


class GhostDriver(IIDDriver):
    """Logged like a driver, never applied. The negative control."""


DRIVERS = {
    "logistic": LogisticDriver,
    "sinusoid": SinusoidDriver,
    "iid": IIDDriver,
    "ghost": GhostDriver,
}


class LabelPoisoner:
    """``batch_hook`` for :func:`grok.loop.train`: corrupt a driven fraction of labels.

    Records the *realised* fraction each step -- the number of labels actually changed
    over the batch size, not the requested one -- because that is the quantity a
    monitoring method could in principle recover, and because quantization to
    ``1/batch_size`` is part of the real problem (it was what made the poisoned_batch
    sinusoid a nonlinear series in Phase 2).

    A :class:`GhostDriver` is logged identically but the batch is returned untouched.
    """

    def __init__(self, driver, num_classes, seed=0):
        self.driver = driver
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed + 9973)   # never the global torch stream
        self.is_ghost = isinstance(driver, GhostDriver)
        self.values = []          # requested fraction
        self.realised = []        # fraction of labels actually corrupted

    def __call__(self, step, batch_x, batch_y):
        requested = float(self.driver.value(step))
        self.values.append(requested)

        n = len(batch_y)
        count = int(round(requested * n))
        if self.is_ghost or count <= 0:
            self.realised.append(0.0 if self.is_ghost else 0.0)
            return batch_x, batch_y

        index = self.rng.choice(n, size=min(count, n), replace=False)
        corrupted = batch_y.clone()
        new_labels = self.rng.integers(0, self.num_classes, size=len(index))
        corrupted[index] = corrupted.new_tensor(new_labels)
        self.realised.append(len(index) / n)
        return batch_x, corrupted

    def frame(self, steps=None):
        import pandas as pd
        data = {"step": np.arange(len(self.values)) if steps is None else steps,
                "driver": self.values, "realised_fraction": self.realised}
        return pd.DataFrame(data)
