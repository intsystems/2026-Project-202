"""Driving the adapter with a controlled *active* rank, and logging what an observer sees.

Four excitation modes, chosen because they are the four things an optimiser near a solution
can actually be doing, and they have **different true delay-embedding dimensions**:

``qp``     r data groups whose loss weights are modulated by r incommensurate sinusoids.
           Deterministic, recurrent, an r-torus.  The only mode in which an intrinsic
           dimension of r exists for a delay embedding to find (Takens).
``noise``  rank-r Gaussian noise added to the gradient -- the experiment as posed in the
           brief.  Stationary stochastic fluctuation: the delay vector depends on the state
           *and* on the last E-1 innovations, so no r-manifold exists.
``batch``  ordinary mini-batch SGD.  Same class as ``noise``, with the covariance the data
           actually produces instead of one imposed by hand.
``gd``     full-batch descent from a start displaced inside the r-dimensional subspace.  A
           deterministic transient: a 1-D curve in phase space for every r, which is why
           ``exp15`` v1 measured 1.33 at every k.

``mixed`` runs ``qp`` and ``noise`` together, which is what makes the signal-to-noise sweep
possible -- the question that decides whether any of this is usable on a real training log.

Two design decisions are forced by the audits of the earlier suite:

* **``eta_zero``.**  Re-running ``exp14`` with the learning rate multiplied by zero -- no
  training at all -- reproduces its headline (MAE 2.06 -> 1.87, rho unchanged, series
  correlation 0.986-0.998), because its observers read the exogenous drive through the
  residual rather than the optimiser state.  Every mode here is therefore also run with
  ``eta_zero=True``, and any observer whose MG survives that is disqualified.  All observers
  below except ``loss_step`` are functions of the optimiser state alone.
* **the drive is whitened.**  ``exp14``'s trajectory had participation ratio 2.9-4.3 while
  its nominal k was 10-20, because the response amplitudes across modes differ by
  ``1/|eta + i omega_j|`` and the forcing directions are not orthogonal.  Here the r
  effective forcing directions are made orthonormal and equal-amplitude by a mixing matrix
  measured in a pre-run (:func:`equalise_gains`), and the achieved participation ratio is
  reported with every result rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from system import Adapter, frequencies, resonance_margin, rank_pr


@dataclass
class Spec:
    seed: int = 0
    k: int = 10
    r: int = 3
    T: int = 40_000
    burn: int = 4_000
    eta: float = 0.15
    precondition: bool = True     # step with H^{-1}: rank-r forcing then gives rank-r motion
    mode: str = "qp"              # 'qp' | 'noise' | 'batch' | 'gd' | 'mixed'
    drive_amp: float = 0.5
    f0: float = 1 / 41.0          # slowest drive frequency, cycles per step
    band: float = 2.0             # one octave for every r: no bandwidth confound
    noise_amp: float = 0.0
    noise_rank: int = 0           # 0 -> use r
    batch: int = 0                # 0 -> full batch
    gd_disp: float = 0.8
    n_groups: int = 12            # FIXED for every r; the drive modulates r of them
    drive_space: str = "data"     # 'data' -> loss-weight modulation | 'param' -> direct
    eta_zero: bool = False        # freeze the parameters, keep the drive: the exp14 control
    rotate: bool = False          # fixed orthogonal rotation of the k coordinates (E4)
    lr_scale: np.ndarray = field(default=None, repr=False)
    amp_scale: np.ndarray = field(default=None, repr=False)
    noise_scale: np.ndarray = field(default=None, repr=False)
    r_schedule: np.ndarray = field(default=None, repr=False)
    obs_scale: np.ndarray = field(default=None, repr=False)   # multiply all observers (E4)

    def key(self):
        return {a: b for a, b in asdict(self).items()
                if not isinstance(b, np.ndarray) and b is not None}


#: The delay window spans ``(max_E-1)*tau`` samples.  A torus is only unfolded if that span
#: covers a real fraction of the oscillation period: exp15 v3, the one earlier experiment
#: that recovered k up to 8 (MAE 0.50), sampled ~2.5 points per cycle and so spanned 6-15
#: cycles; E0's torus arm, at 20-400 points per cycle, spans 0.05-0.95 of a cycle and
#: saturates at MG ~3.  F_FAST and F_SLOW sit on the two sides of that line deliberately --
#: a real training log is always on the F_SLOW side.
F_FAST, F_SLOW = 1 / 16.0, 1 / 400.0

OBSERVERS = ("loss_step", "loss_full", "loss_probe", "w_fro", "w_fro_sq", "c_norm",
             "g_fro", "g_proj", "c_proj1", "c_proj2", "c_proj3",
             "fn_fro", "fn_proj1", "fn_proj2", "margin", "acc_probe")

OBSERVER_FAMILY = {
    "loss_step": "loss*", "loss_full": "loss", "loss_probe": "loss",
    "w_fro": "norm", "w_fro_sq": "norm", "c_norm": "norm", "fn_fro": "norm",
    "g_fro": "gradient", "g_proj": "gradient",
    "c_proj1": "projection", "c_proj2": "projection", "c_proj3": "projection",
    "fn_proj1": "function", "fn_proj2": "function", "margin": "function",
    "acc_probe": "function",
}
#: ``loss_step`` is the only observer that is not a function of the optimiser state alone --
#: it contains the instantaneous loss weights, so it survives ``eta_zero``.  Kept in the
#: sweep precisely so the contamination is visible rather than assumed away.
STATE_ONLY = tuple(o for o in OBSERVERS if o != "loss_step")


def _drive_setup(A, spec, rng):
    """Fixed data partition, fixed frequency set, fixed parameter frame."""
    grng = np.random.default_rng(7717 + spec.seed)      # independent of r and of mode
    grp = grng.integers(0, spec.n_groups, A.n)
    r = max(1, spec.r)
    masks = np.stack([(grp == j).astype(float) for j in range(r)])
    Ud = np.linalg.qr(grng.standard_normal((spec.k, spec.k)))[0][:, :r]
    f = frequencies(r, spec.f0, spec.band)
    ph = np.random.default_rng(31 + spec.seed).uniform(0, 2 * np.pi, r)
    return masks, Ud, f, ph


def _forcing_frame(A, spec, c0, P, eps=1e-3):
    """The r directions the data-group drive actually pushes along (see equalise_gains)."""
    masks, _, _, _ = _drive_setup(A, spec, None)
    r = max(1, spec.r)
    Phi = np.zeros((spec.k, r))
    for j in range(r):
        wp = np.clip(1.0 + eps * masks[j], 0.05, None)
        wm = np.clip(1.0 - eps * masks[j], 0.05, None)
        Phi[:, j] = P @ ((A.loss_grad(c0, w=wp)[1] - A.loss_grad(c0, w=wm)[1]) / (2 * eps))
    return Phi


def simulate(A: Adapter, spec: Spec, c_star=None, mix=None):
    """Run one trajectory.  ``mix`` is the (r, r) drive-whitening matrix.

    Returns ``(logs, C, D, info)`` with ``logs`` cut to the post-burn-in segment.
    """
    rng = np.random.default_rng(1_000_003 * spec.seed + 7919 * spec.r
                                + sum(map(ord, spec.mode)) * 101 + int(spec.noise_amp * 1e4))
    k, T, r = spec.k, spec.T, max(1, spec.r)
    c0 = A.solve() if c_star is None else c_star.copy()

    Hm = A.hessian(c0)
    ev, evec = np.linalg.eigh(Hm)
    ev = np.maximum(ev, 1e-6 * ev.max())
    P = (evec / ev) @ evec.T if spec.precondition else np.eye(k)
    R = np.linalg.qr(np.random.default_rng(4242 + spec.seed).standard_normal((k, k)))[0] \
        if spec.rotate else np.eye(k)

    masks, Ud, f, ph = _drive_setup(A, spec, rng)
    B = np.eye(r) if mix is None else np.asarray(mix, float)
    if spec.mode == "mixed":
        # the noise must excite the SAME r directions the torus does, or the arm is "an
        # r-torus plus an independent rank-r diffusion" with active dimension up to 2r,
        # and the signal-to-noise reading is wrong.  Q is the whitened drive frame.
        Un = np.linalg.qr(_forcing_frame(A, spec, c0, P))[0][:, :max(1, r)]
    else:
        Un = np.linalg.qr(rng.standard_normal((k, k)))[0][:, :(spec.noise_rank or r)]

    logs = {o: np.empty(T) for o in OBSERVERS}
    C = np.empty((T, k)); D = np.empty((T, k))
    orng = np.random.default_rng(555 + spec.seed)
    u_g = orng.standard_normal(k); u_g /= np.linalg.norm(u_g)
    u_c = np.linalg.qr(orng.standard_normal((k, 3)))[0]
    a_fn = orng.standard_normal((2, A.Phi_p.shape[0] * 10))
    a_fn /= np.linalg.norm(a_fn, axis=1, keepdims=True)
    W0f = float(np.linalg.norm(A.W0))
    ip = np.arange(A.Phi_p.shape[0])
    notY = np.eye(10)[A.Yp] > 0

    c = c0.copy()
    if spec.mode == "gd":
        c = c0 + spec.gd_disp * (Ud[:, :r] @ rng.standard_normal(r))

    S = np.sin(2 * np.pi * np.outer(np.arange(T), f) + ph)      # (T, r)
    sched = None if spec.r_schedule is None else np.asarray(spec.r_schedule, int)

    for t in range(T):
        eta = 0.0 if spec.eta_zero else spec.eta * (1.0 if spec.lr_scale is None else spec.lr_scale[t])
        amp = spec.drive_amp * (1.0 if spec.amp_scale is None else spec.amp_scale[t])
        na = r if sched is None else int(sched[t])

        w = None
        s_eff = None
        if spec.mode in ("qp", "mixed") and amp and na:
            s_eff = S[t, :na] @ B[:na, :na].T                    # whitened modulation
            if spec.drive_space == "data":
                w = np.clip(1.0 + amp * (s_eff @ masks[:na]), 0.05, None)

        nr = min(Un.shape[1], na) if sched is not None else Un.shape[1]
        dnoise = None
        if spec.mode == "batch_proj":
            # real mini-batch gradient noise, projected onto r directions.  This is the
            # brief's "gradient noise of rank r" with the covariance the data produces
            # rather than an imposed Gaussian -- and unlike plain mini-batch SGD, whose
            # noise rank is whatever the data says it is (measured PR ~5.9 for every r).
            loss, g = A.loss_grad(c, w=w)
            _, gb = A.loss_grad(c, idx=rng.integers(0, A.n, spec.batch or 64), w=w)
            dnoise = gb - g
        elif spec.batch:
            loss, g = A.loss_grad(c, idx=rng.integers(0, A.n, spec.batch), w=w)
        else:
            loss, g = A.loss_grad(c, w=w)

        if s_eff is not None and spec.drive_space == "param":
            g = g + amp * (Ud[:, :na] @ s_eff)

        step = eta * (P @ g)
        na_t = spec.noise_amp * (1.0 if spec.noise_scale is None else spec.noise_scale[t])
        if na_t and spec.mode in ("noise", "mixed", "batch_proj"):
            # injected *after* the preconditioner.  Passing it through P^{-1}=H would give
            # the r directions variances spread by the Hessian's eigenvalue range (4.3x here,
            # 18x in variance), so the trajectory participation ratio would come out at 4.2
            # for r=6 instead of 6 -- measured.  Injecting after P makes the stationary
            # covariance isotropic on the r-dimensional subspace, which is the point of the
            # controlled arm.  `batch_proj` keeps the data's own amplitude profile.
            if spec.mode == "batch_proj":
                step = step + eta * na_t * (Un[:, :nr] @ (Un[:, :nr].T @ dnoise))
            else:
                step = step + eta * na_t * (Un[:, :nr] @ rng.standard_normal(nr))
        c = c - step

        cr = R @ c
        Lp = A.logits_probe(c)
        lf, gf = A.loss_grad(c)
        tn = np.sqrt(W0f ** 2 + float(c @ c))
        logs["loss_step"][t] = loss
        logs["loss_full"][t] = lf
        logs["loss_probe"][t] = A.loss_probe(c)
        logs["w_fro"][t] = tn
        logs["w_fro_sq"][t] = tn * tn
        logs["c_norm"][t] = float(np.linalg.norm(c))
        logs["g_fro"][t] = float(np.linalg.norm(gf))
        logs["g_proj"][t] = float(u_g @ gf)
        logs["c_proj1"][t] = float(u_c[:, 0] @ cr)
        logs["c_proj2"][t] = float(u_c[:, 1] @ cr)
        logs["c_proj3"][t] = float(u_c[:, 2] @ cr)
        logs["fn_fro"][t] = float(np.linalg.norm(Lp))
        fl = Lp.ravel()
        logs["fn_proj1"][t] = float(a_fn[0] @ fl)
        logs["fn_proj2"][t] = float(a_fn[1] @ fl)
        logs["margin"][t] = float((Lp[ip, A.Yp] - np.max(np.where(notY, -np.inf, Lp), 1)).mean())
        logs["acc_probe"][t] = float((Lp.argmax(1) == A.Yp).mean())
        C[t] = c; D[t] = -step

    b = spec.burn
    out = {o: v[b:] for o, v in logs.items()}
    if spec.obs_scale is not None:
        # a pure gain on the FLUCTUATION.  Scaling the raw series scales its mean too, and
        # for loss_full / w_fro / acc_probe the mean is orders of magnitude larger than the
        # fluctuation, so a 10x ramp would inject a dominant trend and be misread as "MG is
        # not scale-invariant".
        sc = np.asarray(spec.obs_scale)[b:]
        out = {o: v.mean() + (v - v.mean()) * sc for o, v in out.items()}

    Cw, Dw = C[b:], D[b:]
    tr_rank, tr_pr = rank_pr(Cw)
    up_rank, up_pr = rank_pr(Dw)
    fn_rank, fn_pr = A.functional_dim()
    sv = np.linalg.svd(Cw - Cw.mean(0), compute_uv=False)
    info = dict(traj_rank=tr_rank, traj_PR=tr_pr, upd_rank=up_rank, upd_PR=up_pr,
                func_rank=fn_rank, func_PR=fn_pr, available=k,
                traj_spec=";".join(f"{x:.3e}" for x in (sv ** 2 / (sv ** 2).sum())),
                margin_res=float(resonance_margin(f)),
                cycles_slow=float((T - b) * f.min()), samples_per_cycle=float(1 / f.max()),
                hess_cond=float(ev.max() / ev.min()),
                excursion=float(np.linalg.norm(Cw.std(0))),
                drift=float(np.linalg.norm(Cw[-1] - Cw[0])))
    return out, Cw, Dw, info


def equalise_gains(A: Adapter, spec: Spec, c_star, eps=1e-3):
    """The (r, r) mixing that makes the r effective forcing directions orthonormal.

    Modulating data group j alone tilts the gradient along a direction ``phi_j``.  The
    ``phi_j`` are neither orthogonal (random data groups have correlated gradients) nor of
    equal effect, so unmixed forcing gives a trajectory whose participation ratio is far
    below r -- the defect the exp14 audit measured directly (PR 2.9-4.3 at nominal k=10-20).
    With ``mix = pinv(Phi) Q``, ``Q`` an orthonormal basis of ``range(Phi)``, mode l drives
    along ``Q[:, l]``.

    ``phi_j`` is measured by central differences of the gradient with respect to group j's
    loss weight, not by a probe run: an SVD of a probe trajectory recovers each direction
    only up to sign, and an unknown sign per column destroys the orthogonalisation (measured:
    achieved PR 1.8 instead of 6).

    Under preconditioning the linearised dynamics is isotropic, so the response direction
    equals the forcing direction and the only remaining per-mode difference is the scalar
    gain ``|G(omega)| = eta / |e^{i omega} - (1 - eta)|``, which is divided out here.

    Returns ``(mix, cond)``.  A large ``cond(Phi)`` means equalisation needs modulations big
    enough to leave the linear-response regime, so it is reported with every result.
    """
    r = max(1, spec.r)
    masks, _, f, _ = _drive_setup(A, spec, np.random.default_rng(0))
    Hm = A.hessian(c_star)
    ev, evec = np.linalg.eigh(Hm)
    P = (evec / np.maximum(ev, 1e-6 * ev.max())) @ evec.T if spec.precondition else np.eye(spec.k)

    Phi = np.zeros((spec.k, r))
    for j in range(r):
        wp = np.clip(1.0 + eps * masks[j], 0.05, None)
        wm = np.clip(1.0 - eps * masks[j], 0.05, None)
        Phi[:, j] = P @ ((A.loss_grad(c_star, w=wp)[1] - A.loss_grad(c_star, w=wm)[1]) / (2 * eps))

    s = np.linalg.svd(Phi, compute_uv=False)
    cond = float(s.max() / max(s.min(), 1e-30))
    Q = np.linalg.qr(Phi)[0][:, :r]
    mix = np.linalg.pinv(Phi) @ Q
    w = 2 * np.pi * f
    gain = spec.eta / np.abs(np.exp(1j * w) - (1 - spec.eta))     # |G| per mode
    mix = mix / gain[None, :]                                      # equalise the response
    mix = mix / max(np.abs(mix).sum(1).max(), 1e-30)               # keep |modulation| <= 1
    return mix, cond
