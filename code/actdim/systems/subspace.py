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

The system fails requirement 4 for the same reason the decoder does: the drive moves the
targets, so a zero learning rate leaves the loss still varying through the residual.
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
class SubspaceConfig:
    """One network, one ``k``-dimensional slice of its parameter space."""

    k: int = 4
    inputs: int = 8
    hidden: int = 20
    outputs: int = 3
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
        return (self.hidden * self.inputs + self.hidden
                + self.outputs * self.hidden + self.outputs)


def _layout(theta: np.ndarray, config: SubspaceConfig):
    at = 0
    W1 = theta[at:at + config.hidden * config.inputs].reshape(config.hidden, config.inputs)
    at += config.hidden * config.inputs
    b1 = theta[at:at + config.hidden]
    at += config.hidden
    W2 = theta[at:at + config.outputs * config.hidden].reshape(config.outputs, config.hidden)
    at += config.outputs * config.hidden
    return W1, b1, W2, theta[at:at + config.outputs]


def forward(theta: np.ndarray, X: np.ndarray, config: SubspaceConfig):
    W1, b1, W2, b2 = _layout(theta, config)
    hidden = np.tanh(X @ W1.T + b1)
    return hidden @ W2.T + b2, hidden


def loss_gradient(theta: np.ndarray, X: np.ndarray, target: np.ndarray,
                  config: SubspaceConfig):
    """Mean squared error and its exact gradient in the full parameter vector."""
    W1, b1, W2, b2 = _layout(theta, config)
    output, hidden = forward(theta, X, config)
    residual = output - target
    d_output = residual / len(X)
    gW2 = d_output.T @ hidden
    gb2 = d_output.sum(axis=0)
    d_hidden = (d_output @ W2) * (1.0 - hidden * hidden)
    gW1 = d_hidden.T @ X
    gb1 = d_hidden.sum(axis=0)
    gradient = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
    return 0.5 * float(np.sum(residual * residual)) / len(X), gradient, output


def setup(config: SubspaceConfig, seed: int):
    """The fixed dataset, the initial parameters and a generic frame of ``k`` directions."""
    generator = stream_rng(seed, f"subspace_setup:{config.k}")
    X = generator.standard_normal((config.examples, config.inputs))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    theta0 = 0.18 * generator.standard_normal(config.parameters)
    U = orthonormal((config.parameters, config.k), generator)
    return X, theta0, U


def functional_rank(config: SubspaceConfig, X: np.ndarray, theta0: np.ndarray,
                    U: np.ndarray, z: np.ndarray, eps: float = 1e-5) -> Tuple[int, float]:
    """Numerical rank of ``z -> flattened predictions``, by central differences.

    Two allowed directions can produce the same change in the function. If they do, the
    available dimension overstates the functional one and the rung's claim is wrong; this
    is the measurement that would show it.
    """
    k = len(z)
    eye = np.eye(k)
    columns = []
    for j in range(k):
        plus = forward(theta0 + U @ (z + eps * eye[j]), X, config)[0]
        minus = forward(theta0 + U @ (z - eps * eye[j]), X, config)[0]
        columns.append(((plus - minus) / (2.0 * eps)).ravel())
    report = rank_report(np.column_stack(columns), center=False)
    return report.rank, report.singular_ratio


def trajectory(config: SubspaceConfig, seed: int):
    n, k = config.length, int(config.k)
    X, theta0, U = setup(config, seed)
    drive = build_drive(config.drive, k, seed)
    z_star = drive.series(n)

    targets = np.empty((n, config.examples, config.outputs))
    for step in range(n):
        targets[step] = forward(theta0 + U @ z_star[step], X, config)[0]

    latent = np.empty((n, k))
    thetas = np.empty((n, config.parameters))
    gradients = np.empty((n, config.parameters))
    outputs = np.empty((n, config.examples * config.outputs))
    losses = np.empty(n)
    z = drive.offsets.copy()
    for step in range(n):
        theta = theta0 + U @ z
        loss, gradient, output = loss_gradient(theta, X, targets[step], config)
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
    rank, ratio = functional_rank(config, X, theta0, U, drive.offsets)
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
