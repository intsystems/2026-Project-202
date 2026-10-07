"""Sharpness by Hessian-vector power iteration, and the campaign of appendix Q.

Appendix Q asks whether anything undriven ever occupies the admissible regime. Every
calibration system in the article is driven, so evaluation on them establishes a
conditional; the one candidate the literature offers for an undriven system that is
neither stochastic nor a plain transient is the edge of stability (Cohen et al.,
arXiv:2103.00065). Full-batch gradient descent at a fixed rate sharpens until the top
Hessian eigenvalue reaches ``2/eta`` and then hovers there, and while it hovers the loss
is non-monotone on short scales and descending on long ones. That is the pair of
conditions the diagnostics ask for, so the campaign trains the quadratic perceptron at a
range of rates and writes a log the estimator can read.

Two settings are deliberate and both are options of the one training loop rather than a
second copy of it.

*The log is written at every optimiser step.* The oscillation is the two-cycle of the
unstable mode and its period is a few steps, while every other full-batch log in this
study is strided at ten or more. A stride of ten does not blur that oscillation, it
aliases it away, and a non-monotonicity count computed on such a log counts what
survived the aliasing.

*The sharpness is the largest algebraic eigenvalue*, not the largest by magnitude. The
condition gradient descent must satisfy is ``eta < 2 / lambda_max`` with ``lambda_max``
the largest algebraic eigenvalue, and this Hessian is indefinite early in training, when
the most negative direction is the steeper one. Plain power iteration returns that
direction and reports a negative sharpness and a nonsensical stability ratio.

The eigenvalue is taken on the Hessian of the full-batch training loss, the same
objective the optimiser descends, so ``eta lambda_max / 2`` is the stability ratio of
the linearised dynamics rather than an approximation to it. Power iteration on
Hessian-vector products needs no explicit Hessian: at 145,500 parameters that matrix
would have 2.1e10 entries.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..runtime.determinism import stream_seed
from .perceptron import PerceptronConfig, TrainingRun, train

SHARP_COLUMNS: Tuple[str, ...] = (
    "step", "lam_max", "eta_lam_over_2", "power_iters", "power_rel_change")


# -- the measurement ------------------------------------------------------------

def _flat(vs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in vs])


def _power(grads, params, shift: float, iters: int, tol: float,
           generator: Optional[Any]) -> Tuple[float, int, float]:
    """Dominant eigenvalue of ``H + shift I`` by power iteration, signed.

    Returns ``(rayleigh, iterations_used, last_relative_change)``. The Rayleigh quotient
    is reported rather than the norm ratio, because the norm ratio loses the sign and
    the sign is the whole reason for the shift below.
    """
    v = [torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator)
         for p in params]
    n = torch.linalg.vector_norm(_flat(v))
    v = [vi / n for vi in v]

    lam, prev, used, rel = float("nan"), None, 0, float("nan")
    for it in range(1, iters + 1):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        if shift:
            hv = [h + shift * vi for h, vi in zip(hv, v)]
        flat_hv, flat_v = _flat(hv), _flat(v)
        lam = float(torch.dot(flat_hv, flat_v))
        nrm = float(torch.linalg.vector_norm(flat_hv))
        used = it
        if nrm <= 0 or not math.isfinite(nrm):
            break
        v = [h / nrm for h in hv]
        if prev is not None and abs(lam) > 0:
            rel = abs(lam - prev) / abs(lam)
            if rel < tol:
                break
        prev = lam
    return lam, used, rel


def sharpness(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor,
              iters: int = 30, tol: float = 1e-4,
              generator: Optional[Any] = None) -> Tuple[float, int, float]:
    """Largest algebraic Hessian eigenvalue of the full-batch MSE, by Hessian-vector
    products.

    Returns ``(lam_max, iterations_used, last_relative_change)``. The last two belong in
    the log because an unconverged power iteration *underestimates* the eigenvalue, and
    an underestimate is exactly the direction that would manufacture a false reading of
    "below the stability threshold". Reporting them lets that be checked rather than
    trusted.

    When the dominant eigenvalue comes back negative the iteration is repeated on
    ``H + |lambda| I``, whose spectrum is non-negative and whose dominant eigenvalue is
    therefore the top of the original spectrum; the shift is removed afterwards.

    ``generator`` should be a torch generator that nothing else draws from, so that
    measuring the sharpness cannot move the run being measured. ``run_config`` below
    derives one from the ``sharpness_start`` stream for exactly that reason.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    loss = ((model(x) - target) ** 2).mean()
    grads = torch.autograd.grad(loss, params, create_graph=True)

    lam, used, rel = _power(grads, params, 0.0, iters, tol, generator)
    if math.isfinite(lam) and lam < 0:
        shift = abs(lam)
        lam2, used2, rel2 = _power(grads, params, shift, iters, tol, generator)
        lam, used, rel = lam2 - shift, used + used2, rel2
    return lam, used, rel


# -- the campaign ---------------------------------------------------------------

DEFAULT_LRS: Tuple[float, ...] = (1e5, 3e5, 1e6, 1.5e6, 2e6, 2.5e6, 2.8e6, 3e6)
"""The eight rates of appendix Q's table, spanning monotone descent to divergence."""

DEFAULT_SEEDS: Tuple[int, ...] = (1, 2)
DEFAULT_STEPS = 30_000
DEFAULT_SHARP_EVERY = 100
DEFAULT_SHARP_ITERS = 30

EDGE_THRESHOLD = 0.9
"""Where ``eta lambda_max / 2`` over the second half of a run counts as pinned.

Appendix Q's table separates by an interval, not a point: the monotone runs sit at 0.23
to 0.75 and those at the edge at 0.956 to 0.989, so any threshold in between reproduces
its outcome column and this one is not fitted to either group.
"""


def run_config(lr: float, seed: int, steps: int = DEFAULT_STEPS, p: int = 97,
               width: int = 500, fraction: float = 0.5, task: str = "add",
               sharp_every: int = DEFAULT_SHARP_EVERY,
               sharp_iters: int = DEFAULT_SHARP_ITERS, dtype: str = "float32",
               device: str = "auto", key: Optional[str] = None) -> PerceptronConfig:
    """One rate and one seed of the campaign, as a config of the shared loop.

    Full-batch descent, no weight decay, logged at every step. Snapshots and the
    spectral probes are off: at 30,000 rows an SVD per row would cost more than the
    campaign, and appendix Q reads only the loss and the sharpness.
    """
    key = key or f"eos_lr{lr:g}_s{seed}" + ("_f64" if dtype == "float64" else "")
    return PerceptronConfig(
        key=key, description=f"full-batch descent at eta={lr:g}, seed {seed}",
        task=task, p=p, width=width, fraction=fraction, optimizer="gd", lr=lr,
        weight_decay=0.0, init_seed=seed, max_steps=steps, batch_size=None,
        log_every=1, obs_every=0, sharpness_every=sharp_every,
        sharpness_iters=sharp_iters, n_snapshots=0, progress_every=0,
        dtype=dtype, device=device)


def summarise(run: TrainingRun, cfg: PerceptronConfig) -> Dict[str, Any]:
    """The record of one edge-of-stability run.

    Carries the configuration, the stability ratio over the run and over its tail, the
    divergence step, and the milestones -- which are ``None`` whenever the run diverged,
    because a run that blew up has no generalisation step to report. See
    ``analysable`` for what that guard is protecting.
    """
    ratios = [r["eta_lam_over_2"] for r in run.sharp_rows
              if math.isfinite(r["eta_lam_over_2"])]
    tail = ratios[len(ratios) // 2:] if ratios else []
    record: Dict[str, Any] = dict(
        key=cfg.key, lr=cfg.lr, seed=cfg.init_seed, max_steps=cfg.max_steps,
        p=cfg.p, width=cfg.width, fraction=cfg.fraction, task=cfg.task,
        dtype=cfg.dtype, device=run.device, optimizer=cfg.optimizer,
        weight_decay=cfg.weight_decay, batch=cfg.batch, log_every=cfg.log_every,
        sharp_every=cfg.sharpness_every, sharp_iters=cfg.sharpness_iters,
        n_rows=len(run.train_rows), n_sharp_rows=len(run.sharp_rows),
        seconds=run.seconds,
        eta_lam_over_2_max=max(ratios) if ratios else None,
        eta_lam_over_2_median_tail=float(np.median(tail)) if tail else None,
    )
    record.update(run.summary)
    record["outcome"] = outcome(record)
    record["analysable"] = analysable(record)
    return record


def outcome(record: Dict[str, Any]) -> str:
    """Appendix Q's outcome column: diverges, at the edge, or monotone."""
    if record.get("diverged_at") is not None:
        return "diverges"
    tail = record.get("eta_lam_over_2_median_tail")
    if tail is None:
        return "unmeasured"
    return "at the edge" if tail >= EDGE_THRESHOLD else "monotone"


def analysable(record: Dict[str, Any]) -> bool:
    """Whether a trajectory statistic may be computed on this run.

    A diverged run is not a trajectory. The archived analysis had two paths over these
    logs, one of which required ``diverged_at is None`` and one of which did not, and
    the second sliced post-transition segments out of a run that lasted 567 steps in
    total. Every consumer of these records asks this rather than deciding again.
    """
    return record.get("diverged_at") is None and record.get("n_rows", 0) > 1


#: The settings a resumed record has to agree with the request on. Not the whole config:
#: the record also holds what the run *did* -- how many steps it survived, where it
#: diverged -- and a diverging run legitimately stops short of its budget.
_RESUME_FIELDS = ("max_steps", "sharp_every", "sharp_iters", "p", "width", "fraction",
                  "task", "dtype", "lr", "seed")


def _disagreements(record: Dict[str, Any], cfg: Any) -> str:
    """Where a stored record and the requested configuration differ, as one phrase."""
    wanted = {"max_steps": cfg.max_steps, "sharp_every": cfg.sharpness_every,
              "sharp_iters": cfg.sharpness_iters, "p": cfg.p, "width": cfg.width,
              "fraction": cfg.fraction, "task": cfg.task, "dtype": cfg.dtype,
              "lr": cfg.lr, "seed": cfg.init_seed}
    out = []
    for field in _RESUME_FIELDS:
        if field not in record:
            continue
        was, now = record[field], wanted[field]
        if isinstance(now, float) or isinstance(was, float):
            if was is None or abs(float(was) - float(now)) > 1e-9 * max(1.0, abs(float(now))):
                out.append(f"{field} {was} -> {now}")
        elif was != now:
            out.append(f"{field} {was} -> {now}")
    return ", ".join(out)


def campaign(ctx: Any, lrs: Sequence[float] = DEFAULT_LRS,
             seeds: Sequence[int] = DEFAULT_SEEDS, steps: int = DEFAULT_STEPS,
             sharp_every: int = DEFAULT_SHARP_EVERY,
             sharp_iters: int = DEFAULT_SHARP_ITERS, p: int = 97, width: int = 500,
             fraction: float = 0.5, task: str = "add", dtype: str = "float32",
             resume: bool = True, verbose: bool = True) -> List[Dict[str, Any]]:
    """Every rate by every seed, writing through the caller's store.

    Per run: ``<key>_train.csv`` at every step, ``<key>_sharp.csv`` at the sharpness
    stride, and ``<key>_meta.json``, the run's own record. Returns the records in the
    order the runs were launched, so the campaign table is reproducible rather than
    written in completion order.

    ``resume`` skips a run whose record already exists and reads that record back. The
    campaign is hours long on a machine that can be reclaimed, and re-running the
    finished half of it to recover the summary would be the more expensive mistake. The
    record read back is the one the run wrote, never a rebuild from the current code.

    A record is only reused when it was made at the settings being asked for. The run key
    is the rate and the seed, so a ``--fast`` pass and the full campaign write the same
    two keys; without this check the full campaign skipped them and the committed table
    carried two 200-step rows among fourteen 30,000-step ones, reporting a rate at the
    edge of stability as monotone because nothing had happened yet at step 200.
    """
    import json

    import pandas as pd

    ctx.config(lrs=list(lrs), seeds=list(seeds), steps=steps, sharp_every=sharp_every,
               sharp_iters=sharp_iters, p=p, width=width, fraction=fraction, task=task,
               dtype=dtype, batch="full batch", optimizer="gd", weight_decay=0.0,
               log_every=1, sharpness_seed_role="sharpness_start")

    records: List[Dict[str, Any]] = []
    for lr in lrs:
        for seed in seeds:
            cfg = replace(run_config(lr, seed, steps, p=p, width=width,
                                     fraction=fraction, task=task,
                                     sharp_every=sharp_every, sharp_iters=sharp_iters,
                                     dtype=dtype), device=ctx.device)
            existing = ctx.store.existing(f"{cfg.key}_meta.json")
            if resume and existing is not None:
                record = json.loads(existing.read_text(encoding="utf-8"))
                stale = _disagreements(record, cfg)
                if not stale:
                    if verbose:
                        print(f"[skip] {cfg.key} (already recorded)", flush=True)
                    records.append(record)
                    continue
                if verbose:
                    print(f"[redo] {cfg.key} ({stale})", flush=True)

            if verbose:
                print(f"==> {cfg.key}: {cfg.summary()}", flush=True)
            started = time.time()
            run = train(cfg, verbose=False)
            record = summarise(run, cfg)

            ctx.store.table(f"{cfg.key}_train.csv", pd.DataFrame(run.train_rows))
            ctx.store.table(f"{cfg.key}_sharp.csv",
                            pd.DataFrame(run.sharp_rows, columns=list(SHARP_COLUMNS)))
            ctx.store.json(f"{cfg.key}_meta.json", record)
            records.append(record)
            if verbose:
                tail = record["eta_lam_over_2_median_tail"]
                print(f"    {record['outcome']:<12} eta*lam/2 tail "
                      f"{'n/a' if tail is None else format(tail, '.3f')}  "
                      f"generalised at {record['t_grok']}  "
                      f"{time.time() - started:.0f}s", flush=True)
    return records


def campaign_table(records: Sequence[Dict[str, Any]]):
    """The campaign as one frame, columns in a stable order.

    Written from the records the runs produced, in launch order, so that a re-run can be
    diffed against it line by line.
    """
    import pandas as pd

    frame = pd.DataFrame(list(records))
    lead = ["key", "lr", "seed", "outcome", "analysable", "diverged_at",
            "eta_lam_over_2_median_tail", "eta_lam_over_2_max", "t_memorise", "t_grok",
            "final_train_acc", "final_val_acc", "n_rows", "seconds"]
    columns = [c for c in lead if c in frame.columns]
    columns += [c for c in sorted(frame.columns) if c not in columns]
    return frame[columns]


def sharpness_seed(init_seed: int) -> int:
    """The seed of the power-iteration start vector for a run at ``init_seed``.

    Exposed so a test can assert that the stream is separate from the initialisation
    and the batch order. The offset is the one the archived campaign used, carried in
    ``LEGACY_OFFSETS``, so a re-run reproduces its series.
    """
    return stream_seed(init_seed, "sharpness_start")
