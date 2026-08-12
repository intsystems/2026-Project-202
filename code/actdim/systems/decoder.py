"""A latent vector optimised through a frozen nonlinear decoder.

The fourth rung. A latent ``z`` in ``R^r`` is optimised by exact backpropagation through a
fixed nonlinear map ``F``, against the target ``F(z*(t))`` where ``z*`` moves
quasiperiodically in all ``r`` of its coordinates. The optimiser therefore has exactly
``r`` coordinates to move in and the target it chases traces an ``r``-torus.

``F(z) = A z + s C tanh(B z + b)`` with ``A`` orthonormal. The linear skip is not
decoration: it holds the Jacobian ``J = A + s (C diag(1 - h^2)) B`` away from rank
deficiency, so that no latent direction can be invisible in the output and the *functional*
dimension stays equal to the available one. The tanh branch supplies the curvature that
makes the problem nonlinear. Both are checked rather than assumed: the reported functional
rank is the numerical rank of ``J`` measured at three points on the trajectory.

Unlike regression on one-hot inputs, the latent coordinates interact -- the gradient is
``J^T (F(z) - F(z*))`` and ``J^T J`` is not diagonal -- so the trajectory covariance is not
diagonal either and only the weaker excitation check applies.

This system fails requirement 4: the drive acts on the target, so the loss keeps moving
through the residual when the learning rate is set to zero. Table 3 marks it, and it is
kept because a rung that fails a named requirement in a stated way is more use than one
quietly dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..linalg import rank_report
from ..runtime.determinism import rng as stream_rng
from .drive import DriveConfig, build_drive
from .spec import GroundTruth, Simulation, excitation, register, standardise


@dataclass(frozen=True)
class DecoderConfig:
    """One frozen decoder and the latent chase through it."""

    k: int = 4
    hidden: int = 64
    outputs: int = 64
    nonlinear_scale: float = 0.45
    window: int = 4000
    burn: int = 1500
    eta: float = 0.08
    jitter: float = 1e-6
    drive: DriveConfig = field(default_factory=lambda: DriveConfig(
        cycles_per_window=1000.0, window=4000, amp_scale=0.75,
        amp_low=0.7, amp_span=0.3, offset_scale=0.25))

    @property
    def active_dimension(self) -> float:
        return float(self.k)

    @property
    def length(self) -> int:
        return self.window + self.burn


@dataclass(frozen=True)
class Decoder:
    """The frozen map and its exact Jacobian."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    bias: np.ndarray
    scale: float

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(self.B @ z + self.bias)
        y = self.A @ z + self.scale * (self.C @ h)
        jacobian = self.A + self.scale * ((self.C * (1.0 - h * h)[None, :]) @ self.B)
        return y, jacobian


def decoder(config: DecoderConfig, seed: int) -> Decoder:
    """Draw the frozen parameters. ``A`` has orthonormal columns by construction."""
    generator = stream_rng(seed, f"decoder_parameters:{config.k}")
    A = np.linalg.qr(generator.standard_normal((config.outputs, config.k)))[0]
    B = generator.standard_normal((config.hidden, config.k)) / np.sqrt(config.k)
    C = generator.standard_normal((config.outputs, config.hidden)) / np.sqrt(config.hidden)
    return Decoder(A=A, B=B, C=C, bias=0.35 * generator.standard_normal(config.hidden),
                   scale=config.nonlinear_scale)


def functional_rank(model: Decoder, z: np.ndarray) -> Tuple[int, float]:
    """Numerical rank and conditioning of ``dF/dz`` at one point.

    A latent direction the decoder cannot express is available to the optimiser but not
    visible in the output, and the constructed dimension would then be an overstatement.
    """
    report = rank_report(model.forward(z)[1], center=False)
    return report.rank, report.singular_ratio


def trajectory(config: DecoderConfig, seed: int):
    """Chase the moving target, returning the latent, gradient and output histories."""
    n, k = config.length, int(config.k)
    model = decoder(config, seed)
    drive = build_drive(config.drive, k, seed)
    z_star = drive.series(n)

    targets = np.empty((n, config.outputs))
    for step in range(n):
        targets[step] = model.forward(z_star[step])[0]

    latent = np.empty((n, k))
    gradient = np.empty((n, k))
    outputs = np.empty((n, config.outputs))
    losses = np.empty(n)
    z = drive.offsets.copy()
    for step in range(n):
        y, jacobian = model.forward(z)
        residual = y - targets[step]
        g = jacobian.T @ residual                      # exact backpropagation through F
        latent[step], gradient[step], outputs[step] = z, g, y
        losses[step] = 0.5 * float(residual @ residual)
        z = z - config.eta * g

    keep = slice(config.burn, None)
    return (latent[keep], gradient[keep], outputs[keep], losses[keep], model, drive)


def observe(latent: np.ndarray, gradient: np.ndarray, outputs: np.ndarray,
            losses: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    k, outputs_dim = latent.shape[1], outputs.shape[1]
    latent_direction = stream_rng(seed, "observer_latent").standard_normal(k)
    gradient_direction = stream_rng(seed, "observer_gradient").standard_normal(k)
    output_direction = stream_rng(seed, "observer_output").standard_normal(outputs_dim)
    return {
        "latent_fro": np.linalg.norm(latent, axis=1),
        "latent_projection": latent @ latent_direction,
        "gradient_fro": np.linalg.norm(gradient, axis=1),
        "gradient_projection": gradient @ gradient_direction,
        "output_fro": np.linalg.norm(outputs, axis=1),
        "output_projection": outputs @ output_direction,
        "loss": losses,
    }


@register("decoder", "A frozen nonlinear decoder", DecoderConfig, paper="sec:ladder")
def simulate(config: DecoderConfig, seed: int = 0) -> Simulation:
    """Run the chase, read seven observers, and check both ranks."""
    latent, gradient, outputs, losses, model, drive = trajectory(config, seed)
    series = standardise(observe(latent, gradient, outputs, losses, seed), seed,
                         config.jitter)

    measured, checks = excitation(latent, config.k)
    ranks = [functional_rank(model, z) for z in
             (drive.offsets, latent[len(latent) // 2], latent[-1])]
    measured["functional_rank"] = float(min(r for r, _ in ranks))
    measured["jacobian_ratio"] = float(min(ratio for _, ratio in ranks))
    measured["resonance_margin"] = drive.margin
    checks["functional_rank"] = measured["functional_rank"] == config.k

    info = {"k": config.k, "hidden": config.hidden, "outputs": config.outputs,
            "eta": config.eta, "window": config.window, "burn": config.burn,
            **drive.report()}
    return Simulation(series=series,
                      truth=GroundTruth(active_dimension=config.active_dimension,
                                        measured=measured, checks=checks),
                      info=info)
