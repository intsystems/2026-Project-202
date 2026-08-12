"""A whole network trained inside a known low-dimensional parameter subspace.

The fifth rung. Every parameter of a small tanh network is written
``theta = theta_0 + U z`` with ``U`` a fixed ``P x r`` orthonormal frame, and only ``z`` is
optimised. The targets on a fixed dataset are produced by the same network at a
quasiperiodically driven ``z*(t)``, so the optimiser is a real one solving a real
regression problem, and the whole of its motion lies in ``r`` directions of parameter space
by construction.

This is the first rung where "available" and "functional" can come apart. The optimiser is
allowed ``r`` directions; whether the network's outputs actually change in ``r``
independent ways is a fact about the frame and the network, not a definition. The archived
experiment measured it -- ``functional_rank() == k`` -- and that check is kept here and
reported, because a construction whose truth is never measured is requirement 1 unmet.

What the archived experiment did *not* measure is how comparably those directions are
excited, and the answer is: not very. The response gain of mode ``j`` goes as
``1 / |eta + i omega_j|`` and the ``r`` forcing directions are not orthogonal in the
network's own metric, so the trajectory's effective rank reaches 3.1 at r = 10 and 4.7 at
r = 20. That is measured here and reported beside the rank. The image-data system of
:mod:`actdim.systems.digits_parameter` exists partly because it removes exactly this
defect, by equalising the gains.

The system also fails requirement 4: the drive moves the targets, so a zero learning rate
leaves the loss still varying through the residual.

The two-layer tanh perceptron below is shared with :mod:`actdim.systems.digits_function`,
which builds its adapter around the same architecture at a different size. The archived
tree had the layout, forward pass and backward pass written out in both files, differing
only in the names of the constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..linalg import orthonormal, rank_report
from ..runtime.determinism import rng as stream_rng
from .drive import DriveConfig, build_drive
from .spec import GroundTruth, Simulation, excitation, register, standardise


@dataclass(frozen=True)
class Perceptron:
    """A two-layer tanh network held as one flat parameter vector.

    Flat because the systems here move the parameters along a fixed frame, and a frame is
    a matrix in ``R^P``: keeping the network in that shape means the subspace arithmetic
    never has to know the layer structure.
    """

    inputs: int
    hidden: int
    outputs: int

    @property
    def size(self) -> int:
        return (self.hidden * self.inputs + self.hidden
                + self.outputs * self.hidden + self.outputs)

    def split(self, theta: np.ndarray):
        at = 0
        W1 = theta[at:at + self.hidden * self.inputs].reshape(self.hidden, self.inputs)
        at += self.hidden * self.inputs
        b1 = theta[at:at + self.hidden]
        at += self.hidden
        W2 = theta[at:at + self.outputs * self.hidden].reshape(self.outputs, self.hidden)
        at += self.outputs * self.hidden
        return W1, b1, W2, theta[at:at + self.outputs]

    def forward(self, theta: np.ndarray, X: np.ndarray):
        W1, b1, W2, b2 = self.split(theta)
        hidden = np.tanh(X @ W1.T + b1)
        return hidden @ W2.T + b2, hidden

    def loss_gradient(self, theta: np.ndarray, X: np.ndarray, target: np.ndarray):
        """Mean squared error and its exact gradient in the full parameter vector."""
        _, _, W2, _ = self.split(theta)
        output, hidden = self.forward(theta, X)
        residual = output - target
        d_output = residual / len(X)
        gW2 = d_output.T @ hidden
        gb2 = d_output.sum(axis=0)
        d_hidden = (d_output @ W2) * (1.0 - hidden * hidden)
        gradient = np.concatenate([(d_hidden.T @ X).ravel(), d_hidden.sum(axis=0),
                                   gW2.ravel(), gb2])
        return 0.5 * float(np.sum(residual * residual)) / len(X), gradient, output

    def functional_rank(self, theta0: np.ndarray, U: np.ndarray, X: np.ndarray,
                        z: np.ndarray, eps: float = 1e-5) -> Tuple[int, float]:
        """Numerical rank of ``z -> flattened predictions``, by central differences.

        Two allowed directions can produce the same change in the function. If they do, the
        available dimension overstates the functional one and the rung's claim is wrong;
        this is the measurement that would show it.
        """
        k = len(z)
        eye = np.eye(k)
        columns = []
        for j in range(k):
            plus = self.forward(theta0 + U[:, :k] @ (z + eps * eye[j]), X)[0]
            minus = self.forward(theta0 + U[:, :k] @ (z - eps * eye[j]), X)[0]
            columns.append(((plus - minus) / (2.0 * eps)).ravel())
        report = rank_report(np.column_stack(columns), center=False)
        return report.rank, report.singular_ratio


@dataclass(frozen=True)
class SubspaceConfig:
    """One network, one ``k``-dimensional slice of its parameter space."""

    k: int = 4
    network: Perceptron = Perceptron(8, 20, 3)
    examples: int = 40
    window: int = 4000
    burn: int = 1500
    eta: float = 0.15
    jitter: float = 1e-6
    drive: DriveConfig = field(default_factory=lambda: DriveConfig(
        cycles_per_window=1000.0, window=4000, amp_scale=0.65,
        amp_low=0.7, amp_span=0.3, offset_scale=0.15))

    @property
    def active_dimension(self) -> float:
        return float(self.k)

    @property
    def length(self) -> int:
        return self.window + self.burn

    @property
    def parameters(self) -> int:
        return self.network.size


def setup(config: SubspaceConfig, seed: int):
    """The fixed dataset, the initial parameters and a generic frame of ``k`` directions."""
    generator = stream_rng(seed, f"subspace_setup:{config.k}")
    X = generator.standard_normal((config.examples, config.network.inputs))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    theta0 = 0.18 * generator.standard_normal(config.parameters)
    return X, theta0, orthonormal((config.parameters, config.k), generator)


def trajectory(config: SubspaceConfig, seed: int):
    n, k = config.length, int(config.k)
    network = config.network
    X, theta0, U = setup(config, seed)
    drive = build_drive(config.drive, k, seed)
    z_star = drive.series(n)

    targets = np.empty((n, config.examples, network.outputs))
    for step in range(n):
        targets[step] = network.forward(theta0 + U @ z_star[step], X)[0]

    latent = np.empty((n, k))
    thetas = np.empty((n, config.parameters))
    gradients = np.empty((n, config.parameters))
    outputs = np.empty((n, config.examples * network.outputs))
    losses = np.empty(n)
    z = drive.offsets.copy()
    for step in range(n):
        theta = theta0 + U @ z
        loss, gradient, output = network.loss_gradient(theta, X, targets[step])
        latent[step], thetas[step], gradients[step] = z, theta, gradient
        outputs[step], losses[step] = output.ravel(), loss
        z = z - config.eta * (U.T @ gradient)

    keep = slice(config.burn, None)
    return (latent[keep], thetas[keep], gradients[keep], outputs[keep], losses[keep],
            X, theta0, U, drive)


def observe(latent: np.ndarray, thetas: np.ndarray, gradients: np.ndarray,
            outputs: np.ndarray, losses: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    parameters, k = thetas.shape[1], latent.shape[1]
    theta_direction = stream_rng(seed, "observer_parameter").standard_normal(parameters)
    latent_direction = stream_rng(seed, "observer_latent").standard_normal(k)
    return {
        "parameter_fro": np.linalg.norm(thetas, axis=1),
        "parameter_projection": thetas @ theta_direction,
        "latent_fro": np.linalg.norm(latent, axis=1),
        "latent_projection": latent @ latent_direction,
        "gradient_fro": np.linalg.norm(gradients, axis=1),
        "output_fro": np.linalg.norm(outputs, axis=1),
        "loss": losses,
    }


@register("subspace", "A network in a k-dimensional parameter subspace", SubspaceConfig,
          paper="sec:ladder")
def simulate(config: SubspaceConfig, seed: int = 0) -> Simulation:
    """Train in the subspace, read seven observers, and measure the functional rank."""
    latent, thetas, gradients, outputs, losses, X, theta0, U, drive = trajectory(
        config, seed)
    series = standardise(observe(latent, thetas, gradients, outputs, losses, seed),
                         seed, config.jitter)

    measured, checks = excitation(latent, config.k)
    rank, ratio = config.network.functional_rank(theta0, U, X, drive.offsets)
    measured["functional_rank"] = float(rank)
    measured["jacobian_ratio"] = ratio
    measured["resonance_margin"] = drive.margin
    checks["functional_rank"] = rank == config.k

    info = {"k": config.k, "parameters": config.parameters, "examples": config.examples,
            "eta": config.eta, "window": config.window, "burn": config.burn,
            **drive.report()}
    return Simulation(series=series,
                      truth=GroundTruth(active_dimension=config.active_dimension,
                                        measured=measured, checks=checks),
                      info=info)
