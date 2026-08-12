"""Image data, with the adapter built in function space.

The sixth rung, and the first with real inputs. A tanh network is trained on the
scikit-learn handwritten digits and frozen. Around it an adapter of ``r`` parameter
directions is built, and a quasiperiodic teacher moves in all ``r`` of them while a student
tracks it by exact backpropagation through both layers.

What makes this different from the parameter subspace of :mod:`actdim.systems.subspace` is
where the frame comes from. Drawing ``r`` orthonormal directions in parameter space says
nothing about what they do to the function: some barely move the logits, others move them a
great deal, so the trajectory ends up with a few strongly excited directions and a tail of
weak ones. Here the function-space Jacobian ``F`` of a fixed probe set is computed by
central differences and QR-decomposed, and the frame is ``V R^-1 sqrt(m)``. Then
``(J U)' (J U) / m = I`` locally: the first ``r`` columns have rank ``r`` at equal local
scale, so every adapter direction is functionally visible and equally so.

Three ranks are therefore verified rather than assumed, as the archived experiment verified
them: the functional rank of the adapter measured at three points along the trajectory, the
rank of the trajectory covariance, and the rank of the update covariance. Appendix F states
that all three equal ``r``.

Only version three of the archived experiment is ported. Versions one and two inserted
labels into the logits and held the target fixed, which made their recoveries meaningless;
their own successor's docstring calls them invalid and their results have been deleted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from ..linalg import rank_report
from ..runtime.determinism import rng as stream_rng
from ..runtime.determinism import stream_seed
from .drive import DriveConfig, build_drive
from .spec import GroundTruth, Simulation, excitation, register, standardise
from .subspace import Perceptron

#: The archived experiment measured the trajectory and update ranks at this relative
#: threshold. A rank is a step function of its threshold, so it is named rather than
#: inherited from whatever the shared default happens to be.
RANK_TOL = 1e-7


@dataclass(frozen=True)
class DigitsFunctionConfig:
    """One whitened function-space adapter around a trained digits classifier."""

    k: int = 4
    k_max: int = 8
    network: Perceptron = Perceptron(64, 16, 10)
    probe_per_class: int = 10
    backbone_iterations: int = 500
    window: int = 5000
    burn: int = 1500
    eta: float = 0.12
    jitter: float = 1e-6
    drive: DriveConfig = field(default_factory=lambda: DriveConfig(
        cycles_per_window=1200.0, window=5000, amp_scale=0.14,
        amp_low=0.8, amp_span=0.2, offset_scale=0.02))

    @property
    def active_dimension(self) -> float:
        return float(self.k)

    @property
    def length(self) -> int:
        return self.window + self.burn


@dataclass(frozen=True)
class Adapter:
    """The frozen classifier, the probe it is read on, and the whitened frame."""

    theta0: np.ndarray
    frame: np.ndarray      # (P, k_max), whitened: J U has orthonormal columns at scale 1
    probe: np.ndarray
    labels: np.ndarray
    functional_rank: int
    jacobian_ratio: float
    base_accuracy: float


def _balanced_probe(X: np.ndarray, y: np.ndarray, per_class: int, classes: int,
                    seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """A held-out set with the same number of examples in every class.

    Balanced because the margin and the accuracy observers are averages over it, and an
    unbalanced probe would let one class's difficulty dominate both.
    """
    generator = stream_rng(seed, "probe")
    index = np.concatenate([generator.choice(np.flatnonzero(y == c), per_class,
                                             replace=False) for c in range(classes)])
    generator.shuffle(index)
    return X[index], y[index]


@lru_cache(maxsize=8)
def _prepare(seed: int, k_max: int, inputs: int, hidden: int, outputs: int,
             probe_per_class: int, iterations: int) -> Adapter:
    """Train the classifier, then build and whiten the adapter frame.

    Cached in the process: the classifier is the same for every rank at a given seed, and
    training it is the only slow part of this system. The cache is bounded, unlike the
    archived module-level dictionary it replaces.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    network = Perceptron(inputs, hidden, outputs)
    data = load_digits()
    X = data.data.astype(np.float64) / 16.0
    y = data.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=stream_seed(seed, "digits_split") % (2 ** 31),
        stratify=y)
    classifier = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh",
                               solver="lbfgs", alpha=1e-4, max_iter=iterations,
                               random_state=stream_seed(seed, "backbone") % (2 ** 31))
    classifier.fit(X_train, y_train)
    theta0 = np.concatenate([
        classifier.coefs_[0].T.ravel(), classifier.intercepts_[0],
        classifier.coefs_[1].T.ravel(), classifier.intercepts_[1]]).astype(np.float64)

    probe, labels = _balanced_probe(X_test, y_test, probe_per_class, outputs, seed)
    accuracy = float((network.forward(theta0, probe)[0].argmax(1) == labels).mean())

    # Candidate directions, orthonormal in parameter space and therefore arbitrary in
    # function space; the whitening below is what makes them comparable.
    V = np.linalg.qr(stream_rng(seed, "adapter_frame").standard_normal(
        (network.size, k_max)))[0]
    eps = 1e-5
    jacobian = np.empty((len(probe) * outputs, k_max))
    for j in range(k_max):
        plus = network.forward(theta0 + eps * V[:, j], probe)[0]
        minus = network.forward(theta0 - eps * V[:, j], probe)[0]
        jacobian[:, j] = ((plus - minus) / (2.0 * eps)).ravel()

    R = np.linalg.qr(jacobian, mode="reduced")[1]
    scale = np.sqrt(len(probe))
    inverse = np.linalg.inv(R)
    frame = V @ inverse * scale
    whitened = rank_report(jacobian @ inverse * scale, center=False)
    return Adapter(theta0=theta0, frame=frame, probe=probe, labels=labels,
                   functional_rank=whitened.rank, jacobian_ratio=whitened.singular_ratio,
                   base_accuracy=accuracy)


def prepare(config: DigitsFunctionConfig, seed: int) -> Adapter:
    """The trained classifier and whitened adapter for one seed."""
    return _prepare(int(seed), int(config.k_max), config.network.inputs,
                    config.network.hidden, config.network.outputs,
                    int(config.probe_per_class), int(config.backbone_iterations))


def trajectory(config: DigitsFunctionConfig, seed: int):
    n, k = config.length, int(config.k)
    network = config.network
    adapter = prepare(config, seed)
    U = adapter.frame[:, :k]
    drive = build_drive(config.drive, k, seed)
    z_star = drive.series(n)

    latent = np.empty((n, k))
    updates = np.empty((n, k))
    thetas = np.empty((n, network.size))
    gradients = np.empty((n, network.size))
    outputs = np.empty((n, len(adapter.probe) * network.outputs))
    losses = np.empty(n)
    accuracy = np.empty(n)
    z = drive.offsets.copy()
    for step in range(n):
        theta = adapter.theta0 + U @ z
        teacher = network.forward(adapter.theta0 + U @ z_star[step], adapter.probe)[0]
        loss, gradient, output = network.loss_gradient(theta, adapter.probe, teacher)
        step_z = -config.eta * (U.T @ gradient)
        latent[step], updates[step] = z, step_z
        thetas[step], gradients[step] = theta, gradient
        outputs[step], losses[step] = output.ravel(), loss
        accuracy[step] = float((output.argmax(1) == adapter.labels).mean())
        z = z + step_z

    keep = slice(config.burn, None)
    return (latent[keep], updates[keep], thetas[keep], gradients[keep], outputs[keep],
            losses[keep], accuracy[keep], adapter, drive)


def observe(latent: np.ndarray, thetas: np.ndarray, gradients: np.ndarray,
            outputs: np.ndarray, losses: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """The nine scalars the archived experiment recorded, with unit-length directions."""
    parameters, k = thetas.shape[1], latent.shape[1]
    generator = stream_rng(seed, "observer_directions")
    theta_direction = generator.standard_normal(parameters)
    theta_direction /= np.linalg.norm(theta_direction)
    gradient_direction = generator.standard_normal(parameters)
    gradient_direction /= np.linalg.norm(gradient_direction)
    latent_direction = generator.standard_normal(k)
    latent_direction /= np.linalg.norm(latent_direction)
    output_direction = generator.standard_normal(outputs.shape[1])
    output_direction /= np.linalg.norm(output_direction)
    return {
        "parameter_fro": np.linalg.norm(thetas, axis=1),
        "parameter_projection": thetas @ theta_direction,
        "latent_fro": np.linalg.norm(latent, axis=1),
        "latent_projection": latent @ latent_direction,
        "gradient_fro": np.linalg.norm(gradients, axis=1),
        "gradient_projection": gradients @ gradient_direction,
        "output_fro": np.linalg.norm(outputs, axis=1),
        "output_projection": outputs @ output_direction,
        "loss": losses,
    }


@register("digits_function", "A whitened function-space adapter on image data",
          DigitsFunctionConfig, paper="sec:ladder")
def simulate(config: DigitsFunctionConfig, seed: int = 0) -> Simulation:
    """Run the chase, read nine observers, and verify all three ranks."""
    (latent, updates, thetas, gradients, outputs, losses, accuracy,
     adapter, drive) = trajectory(config, seed)
    series = standardise(observe(latent, thetas, gradients, outputs, losses, seed),
                         seed, config.jitter)

    measured, checks = excitation(latent, config.k, equalised=True, tol=RANK_TOL)
    update = excitation(updates, config.k, equalised=True, tol=RANK_TOL)
    measured["update_rank"] = update[0]["covariance_rank"]
    measured["update_effective_rank"] = update[0]["effective_rank"]
    checks["update_rank"] = update[1]["covariance_rank"]

    # The functional rank is measured at three points, not only at the origin: a rank that
    # holds at the operating point and collapses along the orbit is still a broken
    # construction.
    ranks = [config.network.functional_rank(adapter.theta0, adapter.frame, adapter.probe,
                                            z)
             for z in (np.zeros(config.k), latent[len(latent) // 2], latent[-1])]
    measured["functional_rank"] = float(min(rank for rank, _ in ranks))
    measured["jacobian_ratio"] = float(min(ratio for _, ratio in ranks))
    measured["resonance_margin"] = drive.margin
    measured["mean_accuracy"] = float(accuracy.mean())
    measured["min_accuracy"] = float(accuracy.min())
    checks["functional_rank"] = measured["functional_rank"] == config.k

    info = {"k": config.k, "k_max": config.k_max, "parameters": config.network.size,
            "probe": len(adapter.probe), "eta": config.eta, "window": config.window,
            "burn": config.burn, "base_accuracy": adapter.base_accuracy,
            "whitened_rank": adapter.functional_rank,
            "whitened_ratio": adapter.jacobian_ratio, **drive.report()}
    return Simulation(series=series,
                      truth=GroundTruth(active_dimension=config.active_dimension,
                                        measured=measured, checks=checks),
                      info=info)
