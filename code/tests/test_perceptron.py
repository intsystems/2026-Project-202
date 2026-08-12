"""The perceptron: the registry against appendix O, the loop, and the sharpness.

Everything here runs on a CPU in seconds. The modulus is small, the width is narrow and
the budgets are a few hundred steps, which is enough to exercise every branch: the
article's own settings are three orders of magnitude larger in cost and nothing here
depends on them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from actdim.analysis import representation
from actdim.models.perceptron import QuadraticPerceptron, n_parameters, weight_norm
from actdim.runtime import build as build_context
from actdim.runtime.determinism import LEGACY_OFFSETS, stream_seed
from actdim.training import eos, runs_perceptron as reg
from actdim.training.perceptron import (PerceptronConfig, SketchRecorder, build_dataset,
                                        grok_summary, label_function, sketch_cost,
                                        train, train_registered)

# -- the registry against appendix O -------------------------------------------

# task, p, width, weight decay, training fraction, optimiser, learning rate, budget.
# Transcribed from the table, not read back from the registry, so that an edit to the
# registry has something to fail against.
APPENDIX_O = {
    "a_add": ("add", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "a_mul": ("mul", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "a_sub": ("sub", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "a_sq_sum": ("sq_sum", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    # The budget column of the table says 100k for this row. The run stopped at 46,000
    # steps, and the registry records what ran; see the separate test below.
    "a_sum_sq": ("sum_sq", 97, 500, 0.0, 0.5, "gd", 1e5, 46_000),
    "x_mix_quad": ("mix_quad", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "x_no_grok": ("no_grok", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "g_p1": ("p1", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "g_p1x": ("p1x", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "g_p2": ("p2", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "g_p2x": ("p2x", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "g_p3": ("p3", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
    "g_p3x": ("p3x", 97, 500, 0.0, 0.5, "gd", 1e5, 100_000),
}


def test_inventory_is_the_thirteen_rows():
    assert set(reg.INVENTORY) == set(APPENDIX_O)
    assert len(reg.INVENTORY) == 13
    assert set(reg.PAPER_MILESTONES) == set(reg.INVENTORY)


@pytest.mark.parametrize("key", sorted(APPENDIX_O))
def test_registry_matches_appendix_o(key):
    task, p, width, wd, alpha, optimizer, lr, budget = APPENDIX_O[key]
    cfg = reg.get(key)
    assert (cfg.task, cfg.p, cfg.width) == (task, p, width)
    assert cfg.weight_decay == wd
    assert cfg.fraction == alpha
    assert cfg.optimizer == optimizer
    assert cfg.lr == pytest.approx(lr)
    assert cfg.max_steps == budget
    assert cfg.batch_size is None            # every row is full batch
    assert cfg.batch == "full batch"         # and says so without a null


def test_a_sum_sq_records_the_budget_that_ran():
    """The article's inventory is wrong on this row and the registry is not."""
    assert reg.get("a_sum_sq").max_steps == 46_000
    assert reg.get("a_add").max_steps == 100_000
    # It generalised at 8,150, well inside the shorter budget, so no claim moves.
    assert reg.PAPER_MILESTONES["a_sum_sq"]["generalise"] < 46_000


def test_sketched_runs_are_not_the_tables_first_four_rows():
    assert reg.SKETCHED == ("a_add", "x_no_grok", "g_p1", "g_p1x")
    assert reg.SKETCHED != reg.INVENTORY[:4]
    assert set(reg.SKETCHED) <= set(reg.RUNS)


def test_broken_runs_carry_a_reason_and_no_group_can_reach_them():
    assert "r_add_adamw" in reg.BROKEN
    assert {f"f_{n}" for n in ("p1", "p1x", "p2", "p2x", "p3", "p3x")} <= set(reg.BROKEN)
    for key, reason in reg.BROKEN.items():
        assert key in reg.RUNS, "a broken run is documentation and must still resolve"
        assert len(reason) > 80, f"{key} needs the reason, not a label"
    for group, keys in reg.GROUPS.items():
        assert not set(keys) & set(reg.BROKEN), group


def test_expand_refuses_a_broken_run_by_name():
    with pytest.raises(KeyError) as excinfo:
        reg.expand(["r_add_adamw"])
    assert "broken" in str(excinfo.value)
    assert reg.expand(["r_add_adamw"], allow_broken=True) == ["r_add_adamw"]
    assert reg.why_broken("a_add") is None


def test_expand_resolves_groups_in_order_without_repeats():
    assert reg.expand(["sketched"]) == list(reg.SKETCHED)
    assert reg.expand(["a_add", "sketched"]) == ["a_add", "x_no_grok", "g_p1", "g_p1x"]
    with pytest.raises(KeyError):
        reg.expand(["no_such_group"])


def test_polynomial_rows_follow_their_modulus():
    """The p = 23 arm is a different training fraction and a different rate."""
    assert reg.get("g_p1").p == 97 and reg.get("g_p1").fraction == 0.5
    assert reg.get("g_p1_p23").p == 23 and reg.get("g_p1_p23").fraction == 0.8
    assert reg.get("g_p1_p23").lr == pytest.approx(1e5 * (23 / 97) ** 3)


# -- the model and its normalisation convention ---------------------------------

def test_initial_loss_is_one_over_p_and_the_norm_is_one():
    """The reading that says the mean-field convention was not mixed with the other.

    Under ``N(0, 1)`` weights with the ``1/(D N)`` prefactor in the forward pass the
    output at step zero is about zero, so the mean-reduced MSE starts at ``1/p`` and the
    normalised weight norm starts at 1. Folding the normalisation into the
    initialisation instead moves both, and moves the usable learning rate with them.
    """
    cfg = PerceptronConfig(key="t", task="add", p=23, width=64, max_steps=1,
                           log_every=1, obs_every=0, n_snapshots=0, progress_every=0)
    run = train(cfg, verbose=False)
    first = run.train_rows[0]
    assert first["train_loss"] == pytest.approx(1.0 / cfg.p, rel=0.05)
    assert first["weight_norm"] == pytest.approx(1.0, rel=0.05)
    assert run.config["n_params"] == n_parameters(23, 64)


def test_a_narrow_run_learns_the_training_set():
    cfg = PerceptronConfig(key="t", task="add", p=13, width=64, lr=3e3, max_steps=400,
                           log_every=20, obs_every=100, n_snapshots=0, progress_every=0)
    run = train(cfg, verbose=False)
    assert run.diverged_at is None
    assert run.train_rows[-1]["train_acc"] > run.train_rows[0]["train_acc"]
    assert run.train_rows[-1]["train_loss"] < run.train_rows[0]["train_loss"]
    assert [r["step"] for r in run.obs_rows] == [0, 100, 200, 300, 400]
    assert run.obs_rows[0]["ipr_u1"] < 0.5     # random init is spectrally flat


def test_two_runs_of_one_config_agree_to_the_last_bit():
    cfg = PerceptronConfig(key="t", task="add", p=13, width=32, lr=3e3, max_steps=120,
                           log_every=10, obs_every=0, n_snapshots=0, progress_every=0)
    a = [r["train_loss"] for r in train(cfg, verbose=False).train_rows]
    b = [r["train_loss"] for r in train(cfg, verbose=False).train_rows]
    assert a == b


def test_the_split_depends_on_its_own_seed_and_nothing_else():
    cfg = PerceptronConfig(key="t", task="add", p=13, width=32)
    other = PerceptronConfig(key="t", task="add", p=13, width=32, init_seed=99)
    assert np.array_equal(build_dataset(cfg)["y_val"], build_dataset(other)["y_val"])


def test_the_device_is_resolved_and_never_recorded_as_auto():
    cfg = PerceptronConfig(key="t", task="add", p=11, width=16, max_steps=1,
                           log_every=1, obs_every=0, n_snapshots=0, progress_every=0)
    assert cfg.device == "auto"
    run = train(cfg, verbose=False)
    assert run.device != "auto"
    assert run.config["device"] == run.device


def test_a_stride_that_misses_the_logged_steps_is_refused():
    for kw in ({"obs_every": 15}, {"sharpness_every": 15}):
        with pytest.raises(ValueError):
            PerceptronConfig(key="t", log_every=10, **kw)


def test_label_function_follows_an_override_of_the_modulus():
    """A polynomial evaluator closes over p, so it has to be rebuilt after an override."""
    cfg = reg.get("g_p1")
    at23 = cfg.with_overrides({"p": 23})
    n = np.arange(5)
    assert not np.array_equal(label_function(cfg)(n, n), label_function(at23)(n, n))
    assert label_function(at23)(n, n).max() < 23


# -- one loop, with the per-step and sharpness options ---------------------------

def _eos_style(**kw):
    base = dict(key="t", task="add", p=11, width=24, lr=1e3, max_steps=40, log_every=1,
                obs_every=0, n_snapshots=0, progress_every=0)
    base.update(kw)
    return PerceptronConfig(**base)


def test_per_step_logging_writes_every_step():
    run = train(_eos_style(), verbose=False)
    assert [r["step"] for r in run.train_rows] == list(range(41))
    assert run.obs_rows == []       # the SVD probes are off at this stride


def test_measuring_the_sharpness_does_not_move_the_trajectory():
    """The whole reason the power iteration draws from its own stream."""
    plain = train(_eos_style(), verbose=False)
    measured = train(_eos_style(sharpness_every=10), verbose=False)
    assert ([r["train_loss"] for r in plain.train_rows]
            == [r["train_loss"] for r in measured.train_rows])
    assert [r["step"] for r in measured.sharp_rows] == [0, 10, 20, 30, 40]
    assert all(math.isfinite(r["lam_max"]) for r in measured.sharp_rows)


def test_the_sharpness_stream_is_the_one_the_archived_campaign_used():
    assert LEGACY_OFFSETS["sharpness_start"] == 7_000_000
    assert eos.sharpness_seed(1) == 7_000_001
    cfg = _eos_style(sharpness_every=10, init_seed=1)
    assert cfg.resolved()["sharpness_seed"] == stream_seed(1, "sharpness_start")
    assert cfg.resolved()["sharpness_seed"] not in (cfg.init_seed, cfg.batch_seed,
                                                    cfg.split_seed)


def _exact_hessian(model, x, target):
    params = list(model.parameters())
    loss = ((model(x) - target) ** 2).mean()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    n = sum(p.numel() for p in params)
    H = np.zeros((n, n))
    for i in range(n):
        seed = torch.zeros(n, dtype=torch.float64)
        seed[i] = 1.0
        chunks, offset = [], 0
        for p in params:
            chunks.append(seed[offset:offset + p.numel()].reshape(p.shape))
            offset += p.numel()
        hv = torch.autograd.grad(grads, params, grad_outputs=chunks, retain_graph=True)
        H[:, i] = torch.cat([v.reshape(-1) for v in hv]).detach().numpy()
    return (H + H.T) / 2


def test_sharpness_agrees_with_an_exact_eigendecomposition():
    """Appendix Q's check, at its own size: p = 5 and width 8 is 120 parameters."""
    cfg = PerceptronConfig(key="t", task="add", p=5, width=8, max_steps=1, log_every=1,
                           obs_every=0, n_snapshots=0, progress_every=0)
    assert cfg.n_params == 120
    data = build_dataset(cfg)
    x = torch.as_tensor(data["x_train"], dtype=torch.float64)
    y = torch.as_tensor(data["y_train"], dtype=torch.long)
    target = torch.zeros(x.shape[0], cfg.p, dtype=torch.float64)
    target[torch.arange(y.shape[0]), y] = 1.0
    model = QuadraticPerceptron(cfg.p, cfg.width, 2, "quadratic", dtype=torch.float64,
                                generator=torch.Generator().manual_seed(cfg.init_seed))

    exact = float(np.linalg.eigvalsh(_exact_hessian(model, x, target)).max())
    generator = torch.Generator().manual_seed(eos.sharpness_seed(cfg.init_seed))
    lam, used, _ = eos.sharpness(model, x, target, iters=200, tol=1e-12,
                                 generator=generator)
    assert lam == pytest.approx(exact, rel=1e-6)
    assert used <= 200

    # At the campaign's budget the iteration is not run to convergence, and what it
    # returns is then a Rayleigh quotient of a unit vector, which cannot exceed the top
    # eigenvalue. An unconverged reading underestimates the sharpness, which is the
    # direction that would make a run look stable rather than unstable.
    coarse, _, _ = eos.sharpness(model, x, target, iters=30, tol=1e-4,
                                 generator=torch.Generator().manual_seed(1))
    assert coarse <= exact * (1 + 1e-9)
    assert coarse == pytest.approx(exact, rel=1e-2)


class _KnownCurvature(torch.nn.Module):
    """A model whose loss Hessian is diagonal and chosen, to check the sign handling.

    With ``f(w) = w * w`` and the mean squared error against ``t``, the Hessian is
    ``diag(4 (3 w_i^2 - t_i) / n)``. One coordinate is given positive curvature and the
    rest a larger negative one, so the eigenvalue of largest magnitude is negative while
    the largest algebraic eigenvalue is positive. Plain power iteration returns the
    first; the stability condition needs the second.
    """

    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w, dtype=torch.float64))

    def forward(self, x):
        return self.w * self.w


def test_sharpness_takes_the_largest_algebraic_eigenvalue():
    w = [1.0, 0.1, 0.1, 0.1]
    target = torch.tensor([0.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    model = _KnownCurvature(w)
    diagonal = 4 * (3 * np.asarray(w) ** 2 - target.numpy()) / len(w)
    assert diagonal.max() > 0 > diagonal.min()
    assert abs(diagonal.min()) > diagonal.max(), "the negative direction must dominate"

    lam, _, _ = eos.sharpness(model, torch.zeros(1), target, iters=200, tol=1e-12,
                              generator=torch.Generator().manual_seed(3))
    assert lam == pytest.approx(diagonal.max(), rel=1e-6)


# -- divergence -----------------------------------------------------------------

def test_a_diverged_run_reports_no_milestones():
    """A run that blew up has no generalisation step, whatever it crossed on the way.

    The archived campaign let one report a generalisation step of 463 on a run that
    diverged at 567, and an analysis then went looking for post-transition structure in
    a 567-step record.
    """
    rows = [dict(step=s, train_loss=0.1, val_loss=0.1, train_acc=1.0,
                 val_acc=1.0 if s >= 40 else 0.0, weight_norm=1.0)
            for s in range(0, 70, 10)]
    clean = grok_summary(rows)
    assert clean["t_grok"] == 40 and clean["diverged_at"] is None

    censored = grok_summary(rows, diverged_at=60)
    assert censored["t_grok"] is None
    assert censored["t_memorise"] is None
    assert censored["t_grok_before_divergence"] == 40
    assert censored["diverged_at"] == 60
    assert not eos.analysable(dict(censored, n_rows=len(rows)))
    assert eos.analysable(dict(clean, n_rows=len(rows)))
    assert eos.outcome(censored) == "diverges"


def test_divergence_is_recorded_not_only_printed():
    run = train(_eos_style(lr=1e9, max_steps=200), verbose=False)
    assert run.diverged_at is not None
    assert run.summary["t_grok"] is None
    assert len(run.train_rows) < 201


# -- the entry point -------------------------------------------------------------

def _context(tmp_path, name="test.perceptron", seed=0):
    return build_context(name, device="cpu", jobs=1, seed=seed, root=tmp_path)


def test_train_registered_writes_its_logs_and_records_its_configuration(tmp_path):
    ctx = _context(tmp_path)
    record = train_registered(
        ctx, "a_add", overrides={"p": 11, "width": 24, "lr": 200.0, "max_steps": 60,
                                 "log_every": 10, "obs_every": 30, "n_snapshots": 3},
        verbose=False)
    ctx.store.close("ok")

    for name in ("a_add_train.csv", "a_add_obs.csv", "a_add_snapshots.npz",
                 "provenance.json"):
        assert (ctx.store.dir / name).exists(), name

    provenance = ctx.store.provenance.to_dict()
    # The run records itself. The archived campaign rebuilt this from whatever the
    # registry said later, so editing an entry rewrote the recorded configuration of
    # logs made under the old one, and lost every wall-clock time.
    assert provenance["config"]["a_add"]["p"] == 11
    assert provenance["config"]["a_add"]["device"] == "cpu"
    assert provenance["wall_seconds"] >= 0
    assert "a_add_train.csv" in provenance["outputs"]

    # The fields that tell one campaign from another, none of them null.
    assert record["batch"] == "full batch"
    assert record["batch_size"] is None
    assert record["task"] == "add" and record["width"] == 24
    assert record["rows"] == 7 and record["seconds"] > 0


def test_a_minibatch_run_is_distinguishable_from_a_full_batch_one(tmp_path):
    """The only difference between the two sketch campaigns, and it was null in both."""
    ctx = _context(tmp_path)
    common = {"p": 11, "width": 24, "lr": 200.0, "max_steps": 40, "log_every": 20,
              "obs_every": 0, "n_snapshots": 0}
    full = train_registered(ctx, "a_add", overrides=common, verbose=False)
    mini = train_registered(ctx, "x_no_grok",
                            overrides=dict(common, batch_size=16), verbose=False)
    assert full["batch"] == "full batch"
    assert mini["batch"] == "minibatch 16"
    assert full["batch"] != mini["batch"]


def test_the_edge_of_stability_campaign_runs_and_resumes(tmp_path):
    ctx = _context(tmp_path)
    kw = dict(lrs=(50.0, 5e8), seeds=(1,), steps=20, sharp_every=10, sharp_iters=5,
              p=11, width=24, verbose=False)
    records = eos.campaign(ctx, **kw)
    assert [r["key"] for r in records] == ["eos_lr50_s1", "eos_lr5e+08_s1"]
    for record in records:
        for name in ("_train.csv", "_sharp.csv", "_meta.json"):
            assert (ctx.store.dir / f"{record['key']}{name}").exists()

    slow, fast = records
    assert slow["outcome"] == "monotone" and slow["analysable"]
    assert slow["eta_lam_over_2_median_tail"] < eos.EDGE_THRESHOLD
    assert fast["diverged_at"] is not None
    assert fast["outcome"] == "diverges" and not fast["analysable"]
    assert fast["t_grok"] is None

    frame = eos.campaign_table(records)
    assert list(frame.columns)[:4] == ["key", "lr", "seed", "outcome"]
    assert len(frame) == 2

    again = eos.campaign(ctx, **kw)
    assert [r["key"] for r in again] == [r["key"] for r in records]
    assert again[0]["seconds"] == records[0]["seconds"], "a resumed run is not re-run"


def test_sketch_cost_records_the_resolved_device():
    """The archived measurement wrote ``auto`` and so could not say where it ran."""
    pytest.importorskip("actdim.sketch.probe")
    cfg = PerceptronConfig(key="sketch_cost", task="add", p=11, width=24, lr=200.0,
                           max_steps=40, log_every=10, obs_every=0, n_snapshots=0,
                           progress_every=0)
    result = sketch_cost(cfg, repeats=1, dim=64, n_probe=16, device="auto")
    assert result["device"] != "auto"
    assert result["device"] == "cpu"      # this machine has no GPU
    assert result["storage_ratio"] > 1
    assert result["model"] == "perceptron"


def test_the_sketch_observer_records_one_row_per_logged_step():
    pytest.importorskip("actdim.sketch.probe")
    cfg = PerceptronConfig(key="t", task="add", p=11, width=24, lr=200.0, max_steps=40,
                           log_every=10, obs_every=0, n_snapshots=0, progress_every=0)
    recorder = SketchRecorder(dim=64, n_sketch=2, n_probe=16, seed=0)
    probed = train(cfg, observer=recorder, verbose=False)
    plain = train(cfg, verbose=False)

    # Nothing in the loop reads the observer back, so attaching it cannot move the run.
    assert ([r["train_loss"] for r in probed.train_rows]
            == [r["train_loss"] for r in plain.train_rows])

    arrays = recorder.arrays()
    assert arrays["step"].tolist() == [0, 10, 20, 30, 40]
    assert arrays["z"].shape == (5, 2, 64)
    assert arrays["zf"].shape == (5, 2, 64)
    assert math.isnan(arrays["param_step"][0])
    assert recorder.metadata()["n_params"] == cfg.n_params


def test_the_model_and_the_closed_form_share_one_forward_pass():
    """A trained model and the analytic weights are scored by the same code.

    Setting the closed-form weights into a model must reproduce the score the analysis
    reports for them, or the two halves of appendix M are measuring different networks.
    """
    p, width = 11, 128
    w1, w2 = representation.closed_form("add", p, width)
    scored = representation.score_weights(w1, w2, p, "add")
    model = QuadraticPerceptron(p, width, 2, "quadratic", dtype=torch.float64)
    with torch.no_grad():
        model.W1.copy_(torch.as_tensor(w1))
        model.W2.copy_(torch.as_tensor(w2))
    assert scored["acc"] == pytest.approx(1.0)
    assert weight_norm(model) == pytest.approx(scored["weight_norm"], rel=1e-9)
