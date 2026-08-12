"""Surrogate data for nonlinear time-series claims (Theiler et al. 1992).

The standing rule of this campaign: no claim without a surrogate test. The reason is the
failure mode the audit found twice. A statistic computed on a smooth, strongly
autocorrelated series will happily return a stable, plausible number that reflects only
the series' *linear* structure -- its autocorrelation and amplitude distribution -- and
nothing about nonlinear dynamics. Surrogates are the standard defence: build an ensemble
that preserves exactly those linear properties while destroying any nonlinear
determinism, and require the statistic on the real data to lie outside the ensemble.

Two nulls, in increasing strictness:

``phase_randomised`` (FT surrogate)
    Randomise the Fourier phases, keep the amplitude spectrum. Preserves the
    autocorrelation exactly; the result is the "best linear Gaussian process with this
    spectrum". Fails to preserve the amplitude *distribution*, so a non-Gaussian series
    can be rejected for its marginal alone -- which is not the hypothesis we care about.

``iaaft`` (Schreiber & Schmitz 1996)
    Iteratively enforces both the amplitude spectrum and the exact rank distribution of
    the data. This is the null to quote: "linearly filtered noise, monotonically
    rescaled". Rejecting it is evidence of genuine nonlinear structure.

The p-value convention is the rank test of Theiler: with ``n`` surrogates, a two-sided
test at level ``alpha`` needs ``n >= 2/alpha - 1`` (39 for 0.05). We report the rank
directly so no distributional assumption sneaks in.
"""

import numpy as np

__all__ = ["phase_randomised", "iaaft", "surrogate_test", "SurrogateResult"]


def phase_randomised(series, rng):
    """One FT surrogate: same power spectrum, randomised phases."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    spectrum = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    phases[0] = 0.0                      # keep the mean real
    if n % 2 == 0:
        phases[-1] = 0.0                 # Nyquist term must stay real
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=n)


def iaaft(series, rng, max_iter=200, tol=1e-8):
    """One IAAFT surrogate: matches the spectrum and the exact value distribution.

    Alternates between imposing the target amplitude spectrum (in Fourier space) and the
    target rank distribution (in signal space) until the spectrum stops changing.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    target_amplitude = np.abs(np.fft.rfft(x))
    sorted_values = np.sort(x)

    current = rng.permutation(x)
    previous_error = np.inf
    for _ in range(max_iter):
        spectrum = np.fft.rfft(current)
        phases = np.angle(spectrum)
        current = np.fft.irfft(target_amplitude * np.exp(1j * phases), n=n)

        ranks = np.argsort(np.argsort(current))
        current = sorted_values[ranks]

        error = np.linalg.norm(np.abs(np.fft.rfft(current)) - target_amplitude)
        if abs(previous_error - error) < tol * max(target_amplitude.sum(), 1e-12):
            break
        previous_error = error
    return current


def circular_shift(series, rng, min_fraction=0.05):
    """One shift surrogate: the same series, rotated in time.

    The right null for a *periodic* driver in a cross-mapping test. IAAFT preserves the
    spectrum, so for a square wave or a sinusoid the surrogate is essentially the same
    waveform with a different phase -- and if the response merely reflects the driver's
    periodicity, cross-mapping succeeds on the surrogate too and nothing is rejected.
    (That is exactly what happened to the `Discrete` and `Sinusoidal` runs.)

    A circular shift keeps the waveform, the spectrum and every value identical, and
    changes only the *alignment* with the response. Rejecting it therefore means the log
    encodes this particular realization in time, which is what coupling means. The shift
    is kept away from 0 and N so the surrogate is never near-identical to the original.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    low = max(1, int(min_fraction * n))
    shift = int(rng.integers(low, n - low)) if n > 2 * low + 1 else 1
    return np.roll(x, shift)


GENERATORS = {"iaaft": iaaft, "ft": phase_randomised, "shift": circular_shift}


class SurrogateResult:
    """Outcome of a rank test against a surrogate ensemble."""

    def __init__(self, statistic, values, kind, larger_is_structured):
        self.statistic = float(statistic)
        self.values = np.asarray(values, dtype=float)
        self.kind = kind
        self.larger_is_structured = larger_is_structured

    @property
    def rank(self):
        """How many surrogates the data beats, in the 'structured' direction."""
        if self.larger_is_structured:
            return int(np.sum(self.statistic > self.values))
        return int(np.sum(self.statistic < self.values))

    @property
    def p_value(self):
        """One-sided rank p-value: (n_not_beaten + 1) / (n + 1)."""
        return (len(self.values) - self.rank + 1) / (len(self.values) + 1)

    @property
    def z_score(self):
        spread = self.values.std()
        if spread == 0:
            return float("nan")
        return (self.statistic - self.values.mean()) / spread

    def __repr__(self):
        return (f"stat={self.statistic:.4f} surrogate={self.values.mean():.4f}"
                f"+-{self.values.std():.4f} z={self.z_score:+.2f} "
                f"p={self.p_value:.3f} ({self.kind}, n={len(self.values)})")


def surrogate_test(series, statistic_fn, n_surrogates=39, kind="iaaft",
                   larger_is_structured=True, seed=0):
    """Compare ``statistic_fn(series)`` against an ensemble of surrogates.

    ``n_surrogates=39`` is the smallest ensemble supporting a one-sided rank test at
    p = 0.025 / two-sided 0.05, which is the convention in Theiler et al.
    """
    if kind not in GENERATORS:
        raise ValueError(f"unknown surrogate kind '{kind}'. Known: {sorted(GENERATORS)}")
    generate = GENERATORS[kind]
    rng = np.random.default_rng(seed)

    observed = statistic_fn(np.asarray(series, dtype=float))
    values = [statistic_fn(generate(series, rng)) for _ in range(n_surrogates)]
    return SurrogateResult(observed, values, kind, larger_is_structured)
