"""The quasiperiodic drive, in one place, with the seed defect fixed.

Every system in section 5 is driven by the same thing: ``r`` sinusoids whose frequencies
are rationally independent, so that the closure of the sampled orbit is an ``r``-torus and
the active dimension is ``r`` by construction. If ``1, f_1, ..., f_r`` admit an integer
relation the closure is a lower-dimensional subtorus and the constructed truth is simply
wrong, so :func:`resonance_margin` measures how far a set is from that failure and is
reported with every system.

The archived tree built this drive three times.

``dimension_recovery/systems.py``
    20 primes; frequencies placed *linearly* in a band ``centre * [1 - 0.3, 1 + 0.3]`` at
    ``2 frac(sqrt p) - 1``; ``resonance_margin`` exact by Cartesian enumeration for k <= 4
    and a bounded Monte-Carlo search above it.

``active_dimension/system.py``
    the same 20 primes; frequencies placed *geometrically* over one octave at
    ``f0 * band ** a``, with ``a`` the fractional parts rescaled so that the lowest sits at
    0.03 and the highest at 0.97 of the band; ``resonance_margin`` exact over integer
    vectors of L1 norm at most 3, memoised.

``active_dimension/generators.py``
    the same construction as ``system.py`` with the prime table truncated at 53, which
    would have raised ``IndexError`` above r = 16, and ``resonance_margin`` at order 4 by
    full Cartesian enumeration, which is exponential in r and unusable past r = 8.

What was kept, and why.

*The geometric layout of* ``system.py``. It is the only one of the three whose realised
bandwidth does not depend on the rank: rescaling the fractional parts pins the lowest and
highest frequency to the band edges at every rank, so no statistic of the frequency support
can order the rank. The linear layout takes the first ``k`` fractional parts as they come,
so its realised band widens from nothing at k = 1 to 77 per cent of the nominal band at
k = 20 -- the confound requirement 6 exists to forbid, present in the very construction
that was meant to remove it. The padding to 0.03 and 0.97 rather than 0 and 1 is load
bearing: at the exact edges the two extreme modes stand at a ratio of exactly ``band``, an
exact resonance, which is one phase and not two.

*One octave everywhere* (``band = 2.0``), which is what appendix F states. The linear
band of 0.3 spanned a ratio of 1.86 between its edges against 1.92 here, so the five
systems that used it are barely moved by the change of layout.

*The exact L1-bounded* ``resonance_margin`` *of* ``system.py``, vectorised: the relation
vectors of a given rank and order are enumerated once and cached, so the margin costs one
matrix product rather than a Python loop over ``(2r)^order`` tuples. This replaces both an
approximation (the Monte-Carlo branch, which could miss the binding relation) and an
enumeration that could not run at r = 20. The default order is 3, as in both files that
used the L1 definition; ``generators.py``'s order of 4 is available as an argument.

*The 20-prime table, extended to 60*, for the reason the next section gives.

The seed defect
---------------

``dimension_recovery/systems.py:frequencies`` accepted a ``seed`` and used it only in the
``widening`` branch. The ``matched`` branch, which every experiment ran, was
``centre * (1 + band * (2 * frac - 1))`` with no randomness at all, and
``active_dimension/system.py:frequencies`` took no seed in the first place. So in both
clusters the held-out seeds reused the calibration seed's frequency geometry exactly: what
was withheld was the phases, the amplitudes, the offsets and the noise, and never the
geometry the estimator is asked to recover. Every held-out claim in section 5 rests on
that, which is why this port exists.

The corrected ``matched`` varies the geometry with the seed in two ways that leave the
properties the construction needs intact:

* the ``k`` primes are **drawn** from a table of 60 rather than taken in order, so a
  different seed gives a different set of fractional parts and therefore a different
  spacing inside the band. Square roots of distinct primes are linearly independent over
  the rationals whichever subset is drawn, so rational independence is preserved exactly
  rather than by luck. The draw is repeated until the resonance margin reaches
  :data:`MARGIN_TARGET` times the drive centre -- the worst case the archived construction
  reached over k = 1..20 -- and otherwise keeps the best of :data:`PRIME_TRIES` candidates,
  so the margin is a property enforced by the construction rather than one reported after
  the fact.
* the two band edges are drawn per seed, near but not at 0.03 and 0.97, and are then held
  fixed across every rank. Bandwidth stays matched across ranks, which is what requirement
  6 asks; it varies by a few per cent across seeds, which is what a held-out seed should
  mean. Without this the geometry at k = 2 could not vary at all, both frequencies being
  pinned to the edges.

At k = 1 the single frequency sits at the centre of that seed's band, so it too moves with
the seed and is neither systematically faster nor slower than the sets above it.

The old behaviour is reachable as ``band_mode="matched_fixed"``, which reproduces the
archived linear layout exactly, seed-ignoring included, for diagnosing what the defect cost.
It is never the default.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from ..runtime.determinism import rng as stream_rng

#: Sixty primes. Twenty were enough to place twenty frequencies in order; drawing twenty
#: of sixty is what lets the set differ between seeds at the top of the published range.
#: The archived ``generators.py`` copy stopped at 53 and would have raised above r = 16.
PRIMES: Tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281,
)

#: Ratio of the highest drive frequency to the lowest. One octave, as in appendix F.
DEFAULT_BAND = 2.0

#: The half-width of the archived linear band, used only by ``matched_fixed``.
LINEAR_BAND = 0.3

#: Where the extreme modes sit inside the band, as a fraction of it. Drawn per seed from
#: these ranges and then held fixed across ranks. The archived geometric layout used 0.03
#: and 0.97 exactly; the point of the padding is that the edges themselves would put the
#: two extreme modes at an exact ``band``-to-one ratio.
EDGE_LOW = (0.02, 0.10)
EDGE_HIGH = (0.90, 0.98)

#: The realised band must stay close to a full octave, or "matched bandwidth" is matched
#: to something narrower than it claims.
MIN_SPAN = 0.86

#: Resonance margin, as a fraction of the drive centre, that a drawn frequency set has to
#: reach before the draw stops. 0.006 is the worst case of the archived linear-band
#: construction over k = 1..20, so a set accepted here is at least as far from degeneracy
#: as the sets the published results used.
MARGIN_TARGET = 0.006

#: How many candidates to draw before settling for the best seen. Bounded so that a rank
#: at which the target is unreachable costs a fixed amount of work rather than looping.
PRIME_TRIES = 64
EDGE_TRIES = 48

#: Order of the integer relations the margin searches over.
DEFAULT_ORDER = 3


# ----------------------------------------------------------------- resonance margin

@lru_cache(maxsize=32)
def _relation_vectors(r: int, order: int) -> np.ndarray:
    """Every nonzero integer vector of length ``r`` with L1 norm at most ``order``.

    Enumerated once per ``(r, order)`` and reused. A vector has at most ``order`` nonzero
    entries, so the enumeration is over supports and over the small set of nonzero
    patterns, not over the full ``(2 order + 1) ** r`` grid the archived code walked.
    """
    blocks = []
    for size in range(1, min(order, r) + 1):
        patterns = [p for p in itertools.product(range(-order, order + 1), repeat=size)
                    if all(v != 0 for v in p) and sum(abs(v) for v in p) <= order]
        if not patterns:
            continue
        pattern = np.asarray(patterns, dtype=np.int8)
        supports = np.asarray(list(itertools.combinations(range(r), size)), dtype=np.intp)
        block = np.zeros((len(supports), len(pattern), r), dtype=np.int8)
        rows = np.arange(len(supports))[:, None, None]
        block[rows, np.arange(len(pattern))[None, :, None], supports[:, None, :]] = pattern
        blocks.append(block.reshape(-1, r))
    return np.concatenate(blocks, axis=0)


def resonance_margin(freqs, order: int = DEFAULT_ORDER) -> float:
    """Smallest distance to an integer of ``m . f`` over nonzero ``m`` with ``|m|_1 <= order``.

    Zero would mean the orbit closes onto a subtorus and the constructed dimension is not
    what it says. It is never exactly zero here -- the frequencies are irrational by
    construction -- so what this measures is a finite-record property: the reciprocal is
    roughly how many samples the orbit needs before it stops looking lower-dimensional than
    it is. That is why it is reported beside every estimate instead of checked once.
    """
    f = np.asarray(freqs, dtype=float).ravel()
    if f.size == 0:
        return float("nan")
    value = _relation_vectors(int(f.size), int(order)).astype(float) @ f
    return float(np.min(np.abs(value - np.round(value))))


# ----------------------------------------------------------------- frequency sets

def centre_for_window(cycles_per_window: float, window: int) -> float:
    """Drive centre in cycles per sample, from how much of the torus one window sees.

    ``cycles_per_window`` is the variable the calibration showed decides whether any of
    this works, so the five synthetic systems are parameterised by it and it is swept.
    """
    return float(cycles_per_window) / float(window)


def centre_for_octave(f0: float, band: float = DEFAULT_BAND) -> float:
    """Drive centre from the slowest frequency, the image-data system's parameterisation.

    ``f0 * band ** a`` for ``a`` in ``[0, 1]`` is the same set as ``centre * band ** (a -
    1/2)`` with ``centre = f0 sqrt(band)``, so the two parameterisations are one
    construction and the archived ``f0``/``band`` pairs keep their meaning.
    """
    return float(f0) * float(np.sqrt(band))


def _band_edges(centre: float, band: float, seed: int) -> Tuple[float, float]:
    """Where the extreme modes sit, for this seed, at every rank.

    Chosen to maximise the two-mode resonance margin among a bounded number of candidates.
    That criterion mentions no rank, so the realised bandwidth is identical for every rank,
    which is what requirement 6 demands; it does mention the seed, which is what makes a
    held-out seed a different geometry even at k = 2, where both modes are pinned.
    """
    generator = stream_rng(seed, "drive_band")
    best, best_margin = (EDGE_LOW[0], EDGE_HIGH[1]), -np.inf
    for _ in range(EDGE_TRIES):
        low = float(generator.uniform(*EDGE_LOW))
        high = float(generator.uniform(*EDGE_HIGH))
        if high - low < MIN_SPAN:
            continue
        margin = resonance_margin(centre * band ** (np.array([low, high]) - 0.5))
        if margin > best_margin:
            best, best_margin = (low, high), margin
    return best


def _positions(fractions: np.ndarray, low: float, high: float) -> np.ndarray:
    """Fractional parts rescaled so the extremes land on the band edges."""
    span = float(fractions.max() - fractions.min())
    if span <= 0.0:
        return np.full_like(fractions, 0.5 * (low + high))
    return low + (high - low) * (fractions - fractions.min()) / span


@lru_cache(maxsize=512)
def _matched(k: int, centre: float, band: float, seed: int) -> Tuple[float, ...]:
    """The corrected construction. Cached: the rejection loop is not free at k = 20."""
    low, high = _band_edges(centre, band, seed)
    if k == 1:
        return (float(centre * band ** (0.5 * (low + high) - 0.5)),)
    if k == 2:
        return tuple(centre * band ** (np.array([low, high]) - 0.5))

    generator = stream_rng(seed, "drive_primes")
    target = MARGIN_TARGET * centre
    best, best_margin = None, -np.inf
    for _ in range(PRIME_TRIES):
        drawn = np.sort(generator.choice(len(PRIMES), size=k, replace=False))
        fractions = np.sqrt(np.asarray(PRIMES, dtype=float)[drawn]) % 1.0
        freqs = centre * band ** (_positions(fractions, low, high) - 0.5)
        margin = resonance_margin(freqs)
        if margin > best_margin:
            best, best_margin = freqs, margin
        if best_margin >= target:
            break
    return tuple(float(v) for v in best)


def frequencies(k: int, centre: float, band: float = DEFAULT_BAND, seed: int = 0,
                band_mode: str = "matched") -> np.ndarray:
    """``k`` rationally independent frequencies in cycles per sample.

    ``matched``
        the corrected construction described in the module docstring: one octave centred
        on ``centre``, the same realised band at every rank, a different set for every
        seed.
    ``matched_fixed``
        the archived linear layout, ``centre * (1 + 0.3 (2 frac(sqrt p) - 1))`` over the
        first ``k`` primes, which ignores the seed. Kept only so an experiment can measure
        what the defect cost.
    ``widening``
        ``f`` proportional to ``sqrt(p)``, the obvious construction, in which each added
        oscillator also adds a higher frequency. The one-line roughness of the observable
        then orders the rank as well as any dimension estimator, because it is reading
        bandwidth. This is the control that shows why requirement 6 is needed, never a
        setting a result is taken at.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    if band_mode == "matched":
        return np.asarray(_matched(k, float(centre), float(band), int(seed)), dtype=float)
    if band_mode == "matched_fixed":
        fractions = np.sqrt(np.asarray(PRIMES[:k], dtype=float)) % 1.0
        return centre * (1.0 + LINEAR_BAND * (2.0 * fractions - 1.0))
    if band_mode == "widening":
        ratios = np.sqrt(np.asarray(PRIMES[:k], dtype=float))
        jitter = 1.0 + 0.05 * stream_rng(seed, "drive_frequencies").standard_normal(k)
        return centre * (ratios / ratios[0]) * jitter
    raise ValueError(f"unknown band_mode {band_mode!r}")


def realised_band(freqs) -> float:
    """Ratio of the highest frequency to the lowest: the bandwidth requirement 6 fixes."""
    f = np.abs(np.asarray(freqs, dtype=float))
    return float(f.max() / f.min())


# ----------------------------------------------------------------- the driver block

@dataclass(frozen=True)
class DriveConfig:
    """The k-oscillator driver, whose constants differed in each of five copies.

    ``amp_scale``, ``amp_low`` and ``amp_span`` give the per-coordinate amplitude
    ``amp_scale * (amp_low + amp_span * u)``; ``offset_scale`` gives the fixed offset each
    oscillator swings about. The archived experiments spelled the same three lines with
    ``0.10 * (0.5 + 0.5 u)``, ``1.2 * (0.7 + 0.3 u)``, ``0.75 * (0.7 + 0.3 u)``,
    ``0.65 * (0.7 + 0.3 u)`` and ``0.14 * (0.8 + 0.2 u)``; those are now five configurations
    of one construction.
    """

    cycles_per_window: float = 1000.0
    window: int = 4000
    band: float = DEFAULT_BAND
    band_mode: str = "matched"
    amp_scale: float = 0.1
    amp_low: float = 0.5
    amp_span: float = 0.5
    offset_scale: float = 0.0

    @property
    def centre(self) -> float:
        return centre_for_window(self.cycles_per_window, self.window)


@dataclass(frozen=True)
class Drive:
    """One realised drive: what it oscillates at, from where, and how far from resonance."""

    frequencies: np.ndarray
    phases: np.ndarray
    amplitudes: np.ndarray
    offsets: np.ndarray
    margin: float
    band: float

    @property
    def k(self) -> int:
        return int(self.frequencies.size)

    def waves(self, n: int) -> np.ndarray:
        """``sin(2 pi f t + phi)``, shape ``(n, k)``: the phases, unscaled."""
        t = np.arange(int(n), dtype=float)
        return np.sin(2.0 * np.pi * np.outer(t, self.frequencies) + self.phases)

    def series(self, n: int) -> np.ndarray:
        """The driven coordinates, ``offset + amplitude * sin(...)``, shape ``(n, k)``."""
        return self.offsets + self.amplitudes * self.waves(n)

    def report(self) -> Dict[str, float]:
        """What every system records about its own drive."""
        return {
            "resonance_margin": float(self.margin),
            "realised_band": realised_band(self.frequencies),
            "frequency_min": float(self.frequencies.min()),
            "frequency_max": float(self.frequencies.max()),
            "samples_per_cycle": float(1.0 / self.frequencies.max()),
        }


def build_drive(config: DriveConfig, k: int, seed: int,
                centre: float = None) -> Drive:
    """Frequencies, phases, amplitudes and offsets for ``k`` oscillators.

    Each draw comes from its own named stream, so adding an amplitude to a system does not
    move its phases, and the phase stream keeps the offset the published runs used.
    """
    freqs = frequencies(k, config.centre if centre is None else centre,
                        band=config.band, seed=seed, band_mode=config.band_mode)
    phases = stream_rng(seed, "drive_phases").uniform(0.0, 2.0 * np.pi, k)
    amplitudes = config.amp_scale * (
        config.amp_low + config.amp_span * stream_rng(seed, "drive_amplitudes").random(k))
    offsets = config.offset_scale * stream_rng(seed, "drive_offsets").standard_normal(k)
    return Drive(frequencies=freqs, phases=phases, amplitudes=amplitudes,
                 offsets=offsets, margin=resonance_margin(freqs), band=config.band)
