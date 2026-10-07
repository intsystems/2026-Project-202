"""Image data, a frozen nonlinear backbone, and a head confined to ``k`` parameter directions.

The seventh rung, and the system the article leans on hardest: rows six and seven of
table 3, the whole of section 5.4, the silence control of section 5.3, and every panel of
section 6 that is not synthetic. It exists to keep three numbers apart, so this module
keeps them apart too.

``available``   how many directions the optimiser is *allowed* to move in. Fixed by
                construction: ``theta = theta_0 + V^T c`` with ``V`` a fixed orthonormal
                frame of that many rows.
``functional``  the rank of the Jacobian of the model's outputs with respect to those
                directions, on held-out data. Measured, because two allowed directions
                can produce the same change in the function.
``active``      how many directions the optimiser actually excites over the analysis
                window. Measured from the trajectory and update covariances, and it is
                the quantity every estimate here is scored against.

Why the third is not the first, even when the excitation is confined to ``r`` directions:
near a minimum ``c_{t+1} = c_t - eta (H c_t + xi_t)``, and the stationary covariance is
``sum_j A^j eta^2 Sigma A^{jT}`` with ``A = I - eta H``, whose range is the smallest
``A``-invariant subspace containing ``range(Sigma)``. For a generic ``H`` that Krylov space
is the whole of ``R^available`` however small ``rank(Sigma)`` is. Only after
preconditioning by ``H^{-1}``, which makes the linearised dynamics isotropic, does rank-``r``
forcing give a rank-``r`` trajectory. Both are run, because the difference is the point.

That argument decides which ground-truth check this rung can pass. Its trajectory is
generically full rank in the available directions at *any* numerical threshold -- the
archived run records a hard rank of 9 at a nominal rank of 2 -- so a hard-rank check would
fail a construction that is doing exactly what it was built to do. What the construction
fixes, and what :func:`equalise_gains` enforces, is the **effective** rank. The hard rank
is measured and reported; the effective rank is checked.

The excitation modes are the four things an optimiser near a solution can be doing, and
they have different true delay-embedding dimensions:

``qp``          ``r`` data groups whose loss weights are modulated by ``r`` rationally
                independent sinusoids. Deterministic, recurrent, an ``r``-torus. The only
                mode in which an ``r``-dimensional set exists for a delay embedding to find.
``noise``       rank-``r`` Gaussian noise added to the update. A stationary stochastic
                fluctuation: the delay vector depends on the state *and* on the last
                ``E - 1`` innovations, so no ``r``-manifold exists.
``batch``       ordinary mini-batch descent. Same class, with the covariance the data
                produces rather than one imposed by hand -- and its rank is then whatever
                the data says, not a parameter of the experiment.
``batch_proj``  that same mini-batch noise projected onto ``r`` directions, so the rank is
                controlled and the amplitude profile stays the data's own.
``gd``          full-batch descent from a start displaced inside the ``r``-dimensional
                subspace. A deterministic transient: a one-dimensional curve for every
                ``r``.
``mixed``       ``qp`` and ``noise`` together, with the noise confined to the torus's own
                ``r`` directions, so that the arm is an ``r``-torus in noise and not an
                ``r``-torus plus an independent rank-``r`` diffusion.

Two properties are load bearing and neither is assumed.

*The drive is equalised.* Modulating one data group tilts the gradient along a direction
``phi_j``; the ``phi_j`` are neither orthogonal nor of equal effect, because random data
groups have correlated gradients. Unmixed forcing gives a trajectory whose effective rank
sits far below ``r``. :func:`equalise_gains` measures the ``phi_j`` by central differences
and orthogonalises them, then divides out the per-mode scalar response gain. What is
achieved is reported with every run rather than assumed.

*The drive comes from* :mod:`actdim.systems.drive`. The archived module built its own
frequency set, without a seed, so every held-out seed reused the calibration seed's
geometry; that is errata item 1 and it is why the numbers here move. The parameterisation
is preserved: ``f0 * band ** a`` for ``a`` in the band is the same set as
``centre * band ** (a - 1/2)`` with ``centre = f0 sqrt(band)``, so the archived ``f0`` and
``band`` keep their meaning.

The series are recorded raw -- not centred, not scaled, not dithered. The estimator
standardises each window itself, and two things depend on the raw log being raw: the
example traces appendix P plots, and the fact that the probe accuracy is *seen* to be a
quantised observer with no estimate rather than dithered into a plausible number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..linalg import TRAJECTORY_RANK_TOL, participation_ratio, rank_report
from ..observers import PAPER_TWELVE, Observation, Probe, directions, select
from ..runtime.determinism import rng as stream_rng
from ..runtime.determinism import stream_seed
from .drive import DEFAULT_BAND, DriveConfig, build_drive, centre_for_octave
from .spec import GroundTruth, Simulation, excitation, register

#: The two drive rates of section 5.4, in cycles per sample. The delay window spans
#: ``(E - 1) tau`` samples and only unfolds a torus when that covers a real fraction of the
#: oscillation period, so ``fast`` is the regime where the estimator is accurate and
#: ``slow`` is the regime a training log is actually in. They sit on the two sides of that
#: line deliberately.
F_FAST = 1.0 / 16.0
F_SLOW = 1.0 / 400.0

#: Relative threshold for the rank of the map from the adapter coordinates to the probe
#: logits. The archived experiment measured the functional rank here and the trajectory
#: rank at :data:`actdim.linalg.TRAJECTORY_RANK_TOL`; a rank is a step function of its
#: threshold, so both are named rather than shared.
FUNCTIONAL_RANK_TOL = 1e-8

#: Floor on the Hessian eigenvalues before inversion, as a fraction of the largest. The
#: preconditioner is ``H^{-1}`` and a flat direction would otherwise be amplified without
#: bound.
CURVATURE_FLOOR = 1e-6


@dataclass(frozen=True)
class _Mode:
    """What one excitation mode claims, and what may therefore be checked on it."""

    dimension_of: Any     # (k) -> the active dimension the construction fixes
    checked: bool         # whether the r-direction claim is checkable at all
    equalised: bool       # whether the r directions are claimed comparably excited
    what: str

    def dimension(self, k: int) -> float:
        return float(self.dimension_of(k))


#: Only the modes that force ``r`` directions deterministically or isotropically claim an
#: active dimension of ``r``. ``batch_proj`` forces ``r`` directions with the data's own
#: amplitude profile, which is not flat, so its rank is checked and its evenness is not.
#: ``batch`` and ``gd`` claim nothing: the first's rank is a property of the data and the
#: second traces a curve whatever ``r`` is. Asserting otherwise would be asserting
#: something false, so both are measured and reported instead.
MODES: Dict[str, _Mode] = {
    "qp": _Mode(lambda k: float(k), True, True, "r sinusoids modulating r data groups"),
    "noise": _Mode(lambda k: float(k), True, True, "injected rank-r gradient noise"),
    "mixed": _Mode(lambda k: float(k), True, True, "qp plus noise in the same directions"),
    "batch_proj": _Mode(lambda k: float(k), True, False,
                        "real mini-batch noise projected to rank r"),
    "batch": _Mode(lambda k: float("nan"), False, False, "plain mini-batch descent"),
    "gd": _Mode(lambda k: 1.0, False, False, "a decaying transient"),
}

MODE_NAMES: Tuple[str, ...] = tuple(MODES)


@dataclass(frozen=True)
class Schedules:
    """Per-step multipliers, for the experiments that vary one thing along a run.

    Held apart from the configuration because they are arrays: a frozen dataclass of
    scalars can be compared, hashed and printed, and one carrying a 30,000-sample ramp
    cannot. Section 6.3's nuisance controls and section 6's change-detection experiment
    are the only callers.
    """

    learning_rate: Optional[np.ndarray] = None   # multiplies eta, per step
    amplitude: Optional[np.ndarray] = None       # multiplies the drive amplitude
    noise: Optional[np.ndarray] = None           # multiplies the injected noise amplitude
    rank: Optional[np.ndarray] = None            # how many directions are excited, per step
    observer_gain: Optional[np.ndarray] = None   # multiplies the observers' fluctuation

    def at(self, name: str, step: int, default: float = 1.0) -> float:
        values = getattr(self, name)
        return default if values is None else float(values[step])


EMPTY = Schedules()


@dataclass(frozen=True)
class DigitsParameterConfig:
    """One run of the constrained head: which directions exist, and how they are excited.

    ``k`` is the *active* rank the construction fixes, named ``k`` for the same reason
    every other system names it that: it is the number the estimator is asked to recover.
    ``available`` is the number of directions the optimiser may move in, which is held
    fixed while ``k`` is swept. The archived ``Spec`` called them ``r`` and ``k``, and the
    result files keep those column names.
    """

    k: int = 3
    available: int = 10
    window: int = 3000
    burn: int = 800
    eta: float = 0.15
    precondition: bool = True
    mode: str = "qp"
    drive_amp: float = 0.8
    f0: float = F_FAST
    band: float = DEFAULT_BAND
    noise_amp: float = 0.0
    noise_rank: int = 0            # 0 -> use k
    batch: int = 0                 # 0 -> full batch
    displacement: float = 0.8      # the gd start, inside the k-dimensional subspace
    groups: int = 12               # fixed for every k; the drive modulates k of them
    drive_space: str = "data"      # 'data' -> loss-weight modulation | 'param' -> direct
    eta_zero: bool = False         # freeze the parameters and keep the drive
    rotate: bool = False           # a fixed orthogonal rotation of the coordinates
    #: the backbone and the data it is trained on
    train_examples: int = 512
    probe_examples: int = 256
    hidden: Tuple[int, ...] = (64, 64)
    backbone_steps: int = 1000
    backbone_batch: int = 256
    backbone_eta: float = 0.08
    solve_steps: int = 4000
    solve_eta: float = 0.5
    classes: int = 10
    observers: Tuple[str, ...] = PAPER_TWELVE

    @property
    def active_dimension(self) -> float:
        """The dimension of the set the trajectory fills -- not always ``k``.

        With the learning rate at zero the optimiser state does not move at all, whatever
        the nominal rank, so the active dimension is one and an observer that still tracks
        ``k`` there is reading the drive. That is requirement 4, stated as a number.
        """
        if self.eta_zero:
            return 1.0
        return MODES[self.mode].dimension(self.k)

    @property
    def length(self) -> int:
        return self.window + self.burn

    @property
    def parameters(self) -> int:
        return self.classes * (self.hidden[-1] + 1)


# ----------------------------------------------------------------- data and backbone

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def digits_split(seed: int, train: int, probe: int):
    """The handwritten digits, column-standardised, split into a train and a probe set.

    The probe is held out and drawn once, so a change in a function-space observer is a
    change in the function and not a change in where it was measured.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y,
        random_state=stream_seed(seed, "digits_split") % (2 ** 31))
    generator = stream_rng(seed, "digits_subset")
    take_train = generator.choice(len(X_train), min(train, len(X_train)), replace=False)
    take_probe = generator.choice(len(X_test), min(probe, len(X_test)), replace=False)
    return (X_train[take_train], y_train[take_train],
            X_test[take_probe], y_test[take_probe])


def frozen_backbone(X: np.ndarray, y: np.ndarray, seed: int, hidden: Tuple[int, ...],
                    steps: int, batch: int, eta: float, classes: int):
    """A small tanh network trained briefly on the data and then frozen. Returns ``phi``.

    Trained rather than random, so the features are of the data; briefly, so the head
    still has work to do. NumPy rather than a framework, because at this size a framework
    costs more than it saves and this runs inside a process pool.
    """
    generator = stream_rng(seed, "backbone")
    sizes = (X.shape[1],) + tuple(hidden)
    W = [generator.standard_normal((sizes[i + 1], sizes[i])) / np.sqrt(sizes[i])
         for i in range(len(hidden))]
    b = [np.zeros(sizes[i + 1]) for i in range(len(hidden))]
    W_out = generator.standard_normal((classes, hidden[-1])) / np.sqrt(hidden[-1])
    b_out = np.zeros(classes)
    onehot = np.eye(classes)[y]

    for _ in range(int(steps)):
        index = generator.choice(len(X), min(batch, len(X)), replace=False)
        activation, history = X[index], [X[index]]
        for weight, bias in zip(W, b):
            activation = np.tanh(activation @ weight.T + bias)
            history.append(activation)
        error = (_softmax(activation @ W_out.T + b_out) - onehot[index]) / len(index)
        g_out, gb_out, back = error.T @ activation, error.sum(axis=0), error @ W_out
        for i in range(len(W) - 1, -1, -1):
            delta = back * (1.0 - history[i + 1] ** 2)
            W[i] -= eta * (delta.T @ history[i])
            b[i] -= eta * delta.sum(axis=0)
            back = delta @ W[i]
        W_out -= eta * g_out
        b_out -= eta * gb_out

    def phi(Z: np.ndarray) -> np.ndarray:
        a = Z
        for weight, bias in zip(W, b):
            a = np.tanh(a @ weight.T + bias)
        return np.hstack([a, np.ones((len(a), 1))])

    return phi


# ----------------------------------------------------------------- the constrained head

@dataclass
class Head:
    """A linear head on frozen nonlinear features, confined to ``k`` parameter directions.

    ``logits(c) = L0 + sum_j c_j M_j`` with ``M_j = Phi V_j^T`` precomputed, so a step
    costs two contractions and a long run costs seconds. The loss is softmax cross-entropy,
    so ``c -> loss`` is nonlinear and the curvature ``H(c)`` is a real, data-dependent,
    anisotropic object -- which is what makes the Krylov argument in the module docstring
    bite rather than being a formality.
    """

    features: np.ndarray        # (n, H)
    labels: np.ndarray          # (n,)
    probe_features: np.ndarray  # (m, H)
    probe_labels: np.ndarray    # (m,)
    available: int
    seed: int = 0
    classes: int = 10
    frame: np.ndarray = field(default=None, repr=False)   # (available, classes * H)

    def __post_init__(self) -> None:
        generator = stream_rng(self.seed, "adapter")
        n, width = self.features.shape
        self.W0 = generator.standard_normal((self.classes, width)) / np.sqrt(width)
        if self.frame is None:
            self.frame = np.linalg.qr(generator.standard_normal(
                (self.classes * width, self.available)))[0].T
        self.L0 = self.features @ self.W0.T
        self.L0_probe = self.probe_features @ self.W0.T
        folded = self.frame.reshape(self.available, self.classes, width)
        self.M = np.einsum("nh,jch->jnc", self.features, folded)
        self.M_probe = np.einsum("mh,jch->jmc", self.probe_features, folded)
        self.onehot = np.eye(self.classes)[self.labels]
        self.n = n
        self.head_norm = float(np.linalg.norm(self.W0))

    # -- forward and backward ---------------------------------------------------

    def logits(self, c: np.ndarray) -> np.ndarray:
        return self.L0 + np.tensordot(c, self.M, axes=(0, 0))

    def probe_logits(self, c: np.ndarray) -> np.ndarray:
        return self.L0_probe + np.tensordot(c, self.M_probe, axes=(0, 0))

    def loss_gradient(self, c: np.ndarray, index=None, weights=None):
        """``(loss, gradient)``. ``index`` selects a mini-batch, ``weights`` reweights it."""
        if index is None:
            logits, onehot, M = self.logits(c), self.onehot, self.M
            labels, size = self.labels, self.n
        else:
            logits = self.L0[index] + np.tensordot(c, self.M[:, index], axes=(0, 0))
            onehot, M = self.onehot[index], self.M[:, index]
            labels, size = self.labels[index], len(index)
            weights = None if weights is None else weights[index]
        p = _softmax(logits)
        per_example = -np.log(np.clip(p[np.arange(size), labels], 1e-12, None))
        if weights is None:
            loss, residual = float(per_example.mean()), (p - onehot) / size
        else:
            total = float(weights.sum())
            loss = float(weights @ per_example / total)
            residual = (weights[:, None] * (p - onehot)) / total
        return loss, np.tensordot(M, residual, axes=([1, 2], [0, 1]))

    def probe_loss(self, c: np.ndarray) -> float:
        p = _softmax(self.probe_logits(c))
        index = np.arange(len(self.probe_labels))
        return float(-np.log(np.clip(p[index, self.probe_labels], 1e-12, None)).mean())

    # -- curvature and the three dimensions -------------------------------------

    def hessian(self, c: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """The Hessian of the full-batch loss, by central differences of the gradient."""
        out = np.zeros((self.available, self.available))
        step = np.zeros(self.available)
        for j in range(self.available):
            step[:] = 0.0
            step[j] = eps
            out[:, j] = (self.loss_gradient(c + step)[1]
                         - self.loss_gradient(c - step)[1]) / (2.0 * eps)
        return 0.5 * (out + out.T)

    def functional_rank(self, tol: float = FUNCTIONAL_RANK_TOL) -> Tuple[int, float]:
        """Rank and effective rank of ``d(probe logits)/dc``.

        The logits are affine in ``c``, so the Jacobian is the constant matrix
        ``M_probe``; measuring it rather than asserting it is what separates the available
        directions from the functional ones.
        """
        report = rank_report(self.M_probe.reshape(self.available, -1).T,
                             center=False, tol=tol)
        return report.rank, report.effective_rank

    def solve(self, steps: int, eta: float) -> np.ndarray:
        """Full-batch descent to the interior minimum used as the operating point."""
        c = np.zeros(self.available)
        for _ in range(int(steps)):
            c = c - eta * self.loss_gradient(c)[1]
        return c


@dataclass(frozen=True)
class Prepared:
    """Everything about a seed that does not depend on how the run is excited.

    Building the backbone and descending to the operating point is the only slow part of
    this system and neither depends on the rank, the mode or the drive, so both are done
    once per seed and reused. The archived tree cached the same thing in an unbounded
    module-level dictionary.
    """

    head: Head
    operating_point: np.ndarray
    curvature: np.ndarray          # eigenvalues, floored
    curvature_basis: np.ndarray
    group: np.ndarray              # which data group each training example is in
    subspace: np.ndarray           # (available, available) orthonormal, drawn once

    def preconditioner(self, on: bool) -> np.ndarray:
        if not on:
            return np.eye(len(self.curvature))
        return (self.curvature_basis / self.curvature) @ self.curvature_basis.T

    @property
    def condition(self) -> float:
        low = float(self.curvature.min())
        return float(self.curvature.max() / low) if low > 0.0 else float("nan")


@lru_cache(maxsize=4)
def _prepare(seed: int, available: int, train: int, probe: int, hidden: Tuple[int, ...],
             backbone_steps: int, backbone_batch: int, backbone_eta: float,
             solve_steps: int, solve_eta: float, classes: int, groups: int) -> Prepared:
    X_train, y_train, X_probe, y_probe = digits_split(seed, train, probe)
    phi = frozen_backbone(X_train, y_train, seed, hidden, backbone_steps,
                          backbone_batch, backbone_eta, classes)
    head = Head(features=phi(X_train), labels=y_train,
                probe_features=phi(X_probe), probe_labels=y_probe,
                available=available, seed=seed, classes=classes)
    c_star = head.solve(solve_steps, solve_eta)
    hessian = head.hessian(c_star)
    values, basis = np.linalg.eigh(hessian)
    values = np.maximum(values, CURVATURE_FLOOR * max(float(values.max()), 1e-30))

    # The partition and the parameter frame are drawn from one stream that knows nothing
    # about the rank or the mode, so that every arm at a seed shares them and the only
    # thing an arm changes is the excitation.
    generator = stream_rng(seed, "drive_groups")
    group = generator.integers(0, groups, head.n)
    subspace = np.linalg.qr(generator.standard_normal((available, available)))[0]
    return Prepared(head=head, operating_point=c_star, curvature=values,
                    curvature_basis=basis, group=group, subspace=subspace)


def prepare(config: DigitsParameterConfig, seed: int) -> Prepared:
    """The backbone, the head, the operating point and the curvature for one seed."""
    return _prepare(int(seed), int(config.available), int(config.train_examples),
                    int(config.probe_examples), tuple(config.hidden),
                    int(config.backbone_steps), int(config.backbone_batch),
                    float(config.backbone_eta), int(config.solve_steps),
                    float(config.solve_eta), int(config.classes), int(config.groups))


# ----------------------------------------------------------------- the drive

def _masks(prepared: Prepared, k: int) -> np.ndarray:
    return np.stack([(prepared.group == j).astype(float) for j in range(k)])


def forcing_frame(prepared: Prepared, config: DigitsParameterConfig,
                  eps: float = 1e-3) -> np.ndarray:
    """The ``k`` directions the data-group drive actually pushes the coordinate along.

    Measured by central differences of the gradient with respect to each group's loss
    weight, not recovered from a probe run: an SVD of a probe trajectory identifies each
    direction only up to sign, and an unknown sign per column destroys the
    orthogonalisation the mixing matrix is for.
    """
    k = max(1, int(config.k))
    masks = _masks(prepared, k)
    P = prepared.preconditioner(config.precondition)
    head, c = prepared.head, prepared.operating_point
    Phi = np.zeros((config.available, k))
    for j in range(k):
        up = np.clip(1.0 + eps * masks[j], 0.05, None)
        down = np.clip(1.0 - eps * masks[j], 0.05, None)
        Phi[:, j] = P @ ((head.loss_gradient(c, weights=up)[1]
                          - head.loss_gradient(c, weights=down)[1]) / (2.0 * eps))
    return Phi


def equalise_gains(prepared: Prepared, config: DigitsParameterConfig,
                   freqs: np.ndarray, Phi: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, float]:
    """The mixing that makes the ``k`` effective forcing directions orthonormal and equal.

    With ``mix = pinv(Phi) Q`` and ``Q`` an orthonormal basis of ``range(Phi)``, mode ``l``
    drives along ``Q[:, l]``. Under preconditioning the linearised dynamics is isotropic,
    so the response direction equals the forcing direction and the only per-mode difference
    left is the scalar gain ``eta / |e^{i omega} - (1 - eta)|``, which is divided out here.

    Returns ``(mix, condition)``. A large ``cond(Phi)`` means equalisation would need
    modulations big enough to leave the linear-response regime, so it is reported with
    every result rather than checked once.
    """
    Phi = forcing_frame(prepared, config) if Phi is None else Phi
    singular = np.linalg.svd(Phi, compute_uv=False)
    condition = float(singular.max() / max(float(singular.min()), 1e-30))
    Q = np.linalg.qr(Phi)[0][:, :Phi.shape[1]]
    mix = np.linalg.pinv(Phi) @ Q

    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    gain = config.eta / np.abs(np.exp(1j * omega) - (1.0 - config.eta))
    mix = mix / np.maximum(gain, 1e-30)[None, :]
    # Keep the largest possible modulation at or below one, so that the loss weights stay
    # positive and the linear-response argument above stays true.
    return mix / max(float(np.abs(mix).sum(axis=1).max()), 1e-30), condition


# ----------------------------------------------------------------- the trajectory

def trajectory(config: DigitsParameterConfig, seed: int,
               schedules: Schedules = EMPTY):
    """Run one trajectory and record the twelve observers along it.

    Returns the post-burn coordinate history, the update history, the observer logs and
    everything measured about the run.
    """
    if config.mode not in MODES:
        raise ValueError(f"unknown mode {config.mode!r}. Known: {', '.join(MODE_NAMES)}")
    if config.drive_space not in ("data", "param"):
        raise ValueError(f"unknown drive_space {config.drive_space!r}")

    n, k = int(config.length), max(1, int(config.k))
    available = int(config.available)
    prepared = prepare(config, seed)
    head = prepared.head

    P = prepared.preconditioner(config.precondition)
    drive = build_drive(
        DriveConfig(band=config.band, amp_scale=1.0, amp_low=1.0, amp_span=0.0,
                    offset_scale=0.0),
        k, seed, centre=centre_for_octave(config.f0, config.band))
    waves = drive.waves(n)                                     # (n, k)

    Phi = (forcing_frame(prepared, config)
           if config.mode in ("qp", "mixed") or config.drive_space == "param" else None)
    if Phi is not None:
        mixing, condition = equalise_gains(prepared, config, drive.frequencies, Phi)
    else:
        mixing, condition = np.eye(k), float("nan")

    # A stream per arm: two arms of one seed must not share their noise draws, or the
    # comparison between them carries a common random number nobody declared.
    arm = f"digits_dynamics:{config.mode}:k{k}:noise{config.noise_amp:g}"
    generator = stream_rng(seed, arm)

    masks = _masks(prepared, k)
    if config.mode == "mixed":
        # The noise must excite the same k directions the torus does, or the arm is an
        # r-torus plus an independent rank-r diffusion, with an active dimension up to 2r,
        # and the signal-to-noise reading is about the wrong system.
        noise_frame = np.linalg.qr(Phi)[0][:, :k]
    else:
        noise_frame = np.linalg.qr(generator.standard_normal((available, available)))[0]
        noise_frame = noise_frame[:, :(int(config.noise_rank) or k)]

    panel = select(tuple(config.observers))
    probe = Probe(labels=head.probe_labels, classes=config.classes)
    frame = directions(available, len(head.probe_labels) * config.classes, seed,
                       rotate=config.rotate)

    logs = {observer.name: np.empty(n) for observer in panel}
    coordinates = np.empty((n, available))
    updates = np.empty((n, available))
    schedule = None if schedules.rank is None else np.asarray(schedules.rank, dtype=int)

    c = prepared.operating_point.copy()
    if config.mode == "gd":
        c = c + config.displacement * (
            prepared.subspace[:, :k] @ generator.standard_normal(k))

    for t in range(n):
        eta = 0.0 if config.eta_zero else config.eta * schedules.at("learning_rate", t)
        amplitude = config.drive_amp * schedules.at("amplitude", t)
        active = k if schedule is None else int(schedule[t])

        weights = None
        modulation = None
        if config.mode in ("qp", "mixed") and amplitude and active:
            modulation = waves[t, :active] @ mixing[:active, :active].T
            if config.drive_space == "data":
                weights = np.clip(1.0 + amplitude * (modulation @ masks[:active]),
                                  0.05, None)

        noise_columns = (min(noise_frame.shape[1], active) if schedule is not None
                         else noise_frame.shape[1])
        batch_noise = None
        if config.mode == "batch_proj":
            step_loss, gradient = head.loss_gradient(c, weights=weights)
            sample = generator.integers(0, head.n, config.batch or 64)
            batch_noise = head.loss_gradient(c, index=sample, weights=weights)[1] - gradient
        elif config.batch:
            sample = generator.integers(0, head.n, config.batch)
            step_loss, gradient = head.loss_gradient(c, index=sample, weights=weights)
        else:
            step_loss, gradient = head.loss_gradient(c, weights=weights)

        if modulation is not None and config.drive_space == "param":
            gradient = gradient + amplitude * (prepared.subspace[:, :active] @ modulation)

        step = eta * (P @ gradient)
        noise_amp = config.noise_amp * schedules.at("noise", t)
        if noise_amp and config.mode in ("noise", "mixed", "batch_proj"):
            # Injected *after* the preconditioner. Passing it through P^-1 = H would
            # spread the k directions' variances by the Hessian's eigenvalue range, and
            # the trajectory's effective rank would come out below k for a reason that has
            # nothing to do with the construction.
            columns = noise_frame[:, :noise_columns]
            if config.mode == "batch_proj":
                step = step + eta * noise_amp * (columns @ (columns.T @ batch_noise))
            else:
                step = step + eta * noise_amp * (
                    columns @ generator.standard_normal(noise_columns))
        c = c - step

        probe_logits = head.probe_logits(c)
        full_loss, full_gradient = head.loss_gradient(c)
        observation = Observation(
            step_loss=step_loss, full_loss=full_loss,
            probe_loss=head.probe_loss(c), coordinate=c, gradient=full_gradient,
            logits=probe_logits, head_norm=head.head_norm, probe=probe, directions=frame)
        for observer in panel:
            logs[observer.name][t] = observer.fn(observation)
        coordinates[t] = c
        updates[t] = -step

    keep = slice(int(config.burn), None)
    series = {name: values[keep] for name, values in logs.items()}
    if schedules.observer_gain is not None:
        # A gain on the *fluctuation*. Scaling the raw series scales its mean too, and for
        # the loss, the parameter norm and the accuracy the mean is orders of magnitude
        # larger than the fluctuation, so a ramp would inject a dominant trend and be
        # misread as the estimate failing to be scale invariant.
        gain = np.asarray(schedules.observer_gain, dtype=float)[keep]
        series = {name: values.mean() + (values - values.mean()) * gain
                  for name, values in series.items()}

    return (series, coordinates[keep], updates[keep], drive, condition, prepared)


# ----------------------------------------------------------------- the ground truth

def _spectrum_string(block: np.ndarray) -> str:
    """The trajectory's normalised covariance spectrum, as one field of a result row."""
    centred = np.asarray(block, dtype=float)
    centred = centred - centred.mean(axis=0, keepdims=True)
    values = np.linalg.svd(centred, compute_uv=False) ** 2
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        return ""
    return ";".join(f"{v:.3e}" for v in values / total)


def ground_truth(config: DigitsParameterConfig, coordinates: np.ndarray,
                 updates: np.ndarray, drive, condition: float,
                 prepared: Prepared) -> GroundTruth:
    """Measure what the run achieved, and check only what the construction claims.

    The hard rank of the trajectory covariance is reported and not checked: the Krylov
    argument in the module docstring says it is generically the number of *available*
    directions however few are forced, and the archived run confirms it. What the
    construction fixes is the effective rank, and that is what ``equalise_gains`` exists
    to deliver, so that is what is checked.
    """
    mode = MODES[config.mode]
    k = max(1, int(config.k))
    measured, checks = excitation(coordinates, k, equalised=mode.equalised,
                                  tol=TRAJECTORY_RANK_TOL)
    reported_rank = measured.pop("covariance_rank")
    checks.pop("covariance_rank", None)
    measured["trajectory_rank"] = reported_rank
    measured["trajectory_effective_rank"] = measured["effective_rank"]

    update = rank_report(updates, center=True, tol=TRAJECTORY_RANK_TOL)
    measured["update_rank"] = float(update.rank)
    measured["update_effective_rank"] = update.effective_rank

    functional_rank, functional_ratio = prepared.head.functional_rank()
    measured["functional_rank"] = float(functional_rank)
    measured["functional_effective_rank"] = functional_ratio
    measured["resonance_margin"] = drive.margin
    measured["drive_condition"] = condition
    measured["curvature_condition"] = prepared.condition

    if not mode.checked or config.eta_zero:
        # A mode that claims no rank has nothing to confirm, and neither has a run whose
        # optimiser never moved. Both are measured and reported; asserting a rank here
        # would be asserting something the construction does not say.
        checks = {}
    else:
        checks["functional_rank"] = functional_rank == int(config.available)
    return GroundTruth(active_dimension=config.active_dimension,
                       measured=measured, checks=checks)


@register("digits_parameter", "Image data, a head confined to k parameter directions",
          DigitsParameterConfig, paper="sec:digits")
def simulate(config: DigitsParameterConfig, seed: int = 0,
             schedules: Schedules = EMPTY) -> Simulation:
    """Excite the head as the mode says, read the twelve observers, and measure the rank."""
    series, coordinates, updates, drive, condition, prepared = trajectory(
        config, seed, schedules)
    truth = ground_truth(config, coordinates, updates, drive, condition, prepared)

    spread = coordinates.std(axis=0)
    info = {
        "k": config.available, "r": int(config.k), "mode": config.mode,
        "available": config.available, "eta": config.eta, "window": config.window,
        "burn": config.burn, "precondition": config.precondition,
        "eta_zero": config.eta_zero, "rotate": config.rotate,
        "drive_amp": config.drive_amp, "noise_amp": config.noise_amp,
        "batch": config.batch, "f0": config.f0, "band": config.band,
        "n_groups": config.groups, "drive_space": config.drive_space,
        "traj_rank": truth.measured["trajectory_rank"],
        "traj_PR": truth.measured["trajectory_effective_rank"],
        "upd_rank": truth.measured["update_rank"],
        "upd_PR": truth.measured["update_effective_rank"],
        "func_rank": truth.measured["functional_rank"],
        "func_PR": truth.measured["functional_effective_rank"],
        "traj_spec": _spectrum_string(coordinates),
        "margin_res": drive.margin,
        "cycles_slow": float(len(coordinates) * float(drive.frequencies.min())),
        "samples_per_cycle": float(1.0 / float(drive.frequencies.max())),
        "hess_cond": prepared.condition,
        "drive_cond": condition,
        "excursion": float(np.linalg.norm(spread)),
        "drift": float(np.linalg.norm(coordinates[-1] - coordinates[0])),
        "state_effective_rank": participation_ratio(spread ** 2),
        **drive.report(),
    }
    return Simulation(series=series, truth=truth, info=info)


#: The two published configurations of this system. Section 5.4 and section 6 run at ten
#: available directions; appendix F's twenty-direction arm and appendix N's exclusion
#: measurement run at twenty, on a shorter record and a smaller backbone. Both are named
#: here rather than rebuilt in each experiment, because three archived scripts each spelled
#: one of them out and two of them disagreed about the backbone.

def ten_direction(**overrides: Any) -> DigitsParameterConfig:
    """Section 5.4's system: ten available directions, 30,000 steps, 4,000 burnt."""
    settings: Dict[str, Any] = dict(
        available=10, window=26_000, burn=4_000, groups=12,
        train_examples=1024, probe_examples=384, hidden=(96, 96), backbone_steps=2000)
    settings.update(overrides)
    return DigitsParameterConfig(**settings)


def twenty_direction(**overrides: Any) -> DigitsParameterConfig:
    """Appendix F's wider arm: twenty available directions on a shorter record."""
    settings: Dict[str, Any] = dict(
        available=20, window=10_000, burn=2_000, groups=24,
        train_examples=512, probe_examples=256, hidden=(64, 64), backbone_steps=1000)
    settings.update(overrides)
    return DigitsParameterConfig(**settings)
