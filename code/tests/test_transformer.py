"""The transformer stack: the architecture, the tasks, the loop and the registry.

Everything here runs on the CPU in seconds. The training runs use a small modulus and a few
hundred steps, which is enough to exercise every branch of the loop -- the gradient probe,
the double step, the validation stride -- without training anything to convergence.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from actdim.models.transformer import NandaTransformer, parameter_count  # noqa: E402
from actdim.runtime import build as build_context                        # noqa: E402
from actdim.tasks.groups import SymmetricGroup, symmetric_group          # noqa: E402
from actdim.tasks.modular import modular_addition                        # noqa: E402
from actdim.training import runs, transformer as trainer                 # noqa: E402


def tiny(key: str = "mod_wd1", **overrides):
    """A registry run shrunk to test size, with nothing about its shape changed."""
    settings = {"p": 17, "max_steps": 60, "log_every": 10, "device": "cpu"}
    settings.update(overrides)
    return runs.get(key).replace(**settings)


# -- the architecture ----------------------------------------------------------


def test_parameter_count_is_the_one_appendix_o_states():
    """226,816 at p = 113 and 228,608 on S_5, by arithmetic and by construction.

    The cheapest check that the architecture has not drifted: any added norm layer, bias or
    head changes one of these two numbers.
    """
    assert parameter_count(d_vocab=114) == 226_816
    assert parameter_count(d_vocab=121) == 228_608

    built = sum(p.numel() for p in NandaTransformer(d_vocab=114).parameters())
    assert built == 226_816
    built_s5 = sum(p.numel() for p in NandaTransformer(d_vocab=121).parameters())
    assert built_s5 == 228_608


def test_the_model_has_no_normalisation_layer():
    """No layer normalisation, which is what makes the parameter norm informative."""
    model = NandaTransformer(d_vocab=20)
    assert not any(isinstance(m, torch.nn.modules.normalization.LayerNorm)
                   for m in model.modules())


def test_forward_returns_the_last_position_only():
    model = NandaTransformer(d_vocab=20)
    x = torch.randint(0, 20, (7, 3))
    assert model(x).shape == (7, 20)


# -- the tasks -----------------------------------------------------------------


def test_vocabularies_are_the_symbol_set_plus_equals():
    task = modular_addition(p=113, fraction=0.3, seed=42)
    assert task.vocab_size == 114 and task.n_ctx == 3
    assert len(task.X_train) + len(task.X_val) == 113 * 113

    s5 = symmetric_group(n=5, fraction=0.5, seed=42)
    assert s5.vocab_size == 121
    assert len(s5.X_train) + len(s5.X_val) == 120 * 120


def test_modular_labels_are_correct():
    task = modular_addition(p=17, fraction=0.3, seed=0)
    a, b = task.X_train[:, 0], task.X_train[:, 1]
    assert torch.equal((a + b) % 17, task.Y_train)
    assert torch.all(task.X_train[:, 2] == 17)      # the '=' token is the last symbol


def test_group_composition_applies_the_right_factor_first():
    group = SymmetricGroup(4)
    table = group.table()
    assert table.shape == (24, 24)
    assert np.array_equal(table[0], np.arange(24))   # the identity is rank 0
    assert np.array_equal(table[:, 0], np.arange(24))


def test_the_split_is_drawn_from_one_seeded_stream():
    """Same seed, same split. This is what a stray draw in an observer would break."""
    first = modular_addition(p=17, fraction=0.3, seed=7)
    second = modular_addition(p=17, fraction=0.3, seed=7)
    assert torch.equal(first.X_train, second.X_train)
    assert not torch.equal(first.X_train, modular_addition(p=17, fraction=0.3, seed=8).X_train)


# -- the registry --------------------------------------------------------------

APPENDIX_O = {
    # key: (task, weight decay, training fraction, budget, logging stride)
    "mod_wd1":      ("modular_addition", 1.0, 0.30, 20_000, 10),
    "mod_wd1_s43":  ("modular_addition", 1.0, 0.30, 20_000, 10),
    "mod_wd1_s44":  ("modular_addition", 1.0, 0.30, 20_000, 10),
    "s5_wd1":       ("symmetric_group",  0.2, 0.50, 15_000, 5),
    "mod_wd0":      ("modular_addition", 0.0, 0.30, 20_000, 10),
    "s5_wd0":       ("symmetric_group",  0.0, 0.50, 15_000, 5),
    "grokpos_s0":   ("modular_addition", 1.0, 0.30, 120_000, 10),
    "lowdata15_s0": ("modular_addition", 1.0, 0.15, 120_000, 10),
    "lowdata15_s1": ("modular_addition", 1.0, 0.15, 120_000, 10),
    "lowdata15_s2": ("modular_addition", 1.0, 0.15, 120_000, 10),
    "lowdata20_s0": ("modular_addition", 1.0, 0.20, 120_000, 10),
    "wd0_s0":       ("modular_addition", 0.0, 0.30, 120_000, 10),
    "wd0_s1":       ("modular_addition", 0.0, 0.30, 120_000, 10),
    "p211_wd0":     ("modular_addition", 0.0, 0.16, 200_000, 50),
}


def test_every_transformer_row_of_appendix_o_has_an_entry():
    assert set(runs.TRANSFORMER_RUNS) == set(APPENDIX_O)
    assert len(runs.TRANSFORMER_RUNS) == 14


@pytest.mark.parametrize("key", sorted(APPENDIX_O))
def test_entries_match_the_row_they_stand_for(key):
    task, wd, fraction, budget, stride = APPENDIX_O[key]
    config = runs.get(key)
    assert config.key == key
    assert config.task == task
    assert config.weight_decay == pytest.approx(wd)
    assert config.fraction == pytest.approx(fraction, abs=5e-3)
    assert config.max_steps == budget
    assert config.log_every == stride
    assert config.provenance, "every row must say what produced it"


def test_the_shared_settings_of_appendix_o():
    for key in runs.TRANSFORMER_RUNS:
        config = runs.get(key)
        assert (config.d_model, config.num_heads, config.d_head, config.d_mlp) == (128, 4, 32, 512)
        assert config.optimizer == "adamw" and config.lr == 1e-3
        assert tuple(config.betas) == (0.9, 0.98)
        assert config.val_batch_size == 512      # appendix O's stated protocol
        # p211_wd0 came from a notebook that trained in float32 with batches of 512; every
        # row the trainer produced is float64 at 256.
        expected = ("float32", 512) if key == "p211_wd0" else ("float64", 256)
        assert (config.dtype, config.batch_size) == expected


def test_the_double_step_is_set_on_the_two_s5_runs_and_nowhere_else():
    """Appendix O states it and the regime those two runs are assigned to depends on it."""
    stepped = {k for k, c in runs.TRANSFORMER_RUNS.items() if c.double_step}
    assert stepped == {"s5_wd1", "s5_wd0"}


def test_the_s5_runs_log_the_gradient_diagnostics():
    for key in ("s5_wd1", "s5_wd0"):
        config = runs.get(key)
        assert config.needs_grad_probe
        assert set(trainer.GRAD_OBSERVABLES) <= set(config.columns)
    assert not runs.get("mod_wd1").needs_grad_probe


def test_the_registry_leaves_room_for_the_other_architecture():
    """The perceptron entries land in the same registry, under their own name."""
    assert runs.PERCEPTRON_RUNS == {} or set(runs.PERCEPTRON_RUNS) <= set(runs.RUNS)
    assert set(runs.TRANSFORMER_RUNS) <= set(runs.RUNS)


def test_an_unknown_run_names_what_is_available():
    with pytest.raises(KeyError, match="mod_wd1"):
        runs.get("no_such_run")


def test_a_config_rejects_a_column_the_loop_cannot_produce():
    with pytest.raises(ValueError, match="unknown log column"):
        runs.get("mod_wd1").replace(columns=("step", "train_acc", "val_acc", "nonsense"))


# -- the loop ------------------------------------------------------------------


def test_a_run_is_reproducible_step_for_step():
    first = trainer.train(tiny())
    second = trainer.train(tiny())
    for column in first.log.columns:
        assert np.array_equal(first.log[column].to_numpy(), second.log[column].to_numpy())


def test_the_log_has_one_row_per_logged_step():
    config = tiny(max_steps=60, log_every=10)
    result = trainer.train(config)
    assert len(result.log) == config.expected_rows == 6
    assert list(result.log.step) == [0, 10, 20, 30, 40, 50]
    assert list(result.log.columns) == list(config.columns)


def test_the_double_step_changes_the_trajectory():
    """A flag that reproduces a bug has to be a flag that does something."""
    plain = trainer.train(tiny(double_step=False))
    doubled = trainer.train(tiny(double_step=True))
    assert not np.allclose(plain.log.weight_norm.to_numpy(),
                           doubled.log.weight_norm.to_numpy())


def test_the_gradient_probe_is_read_between_the_two_steps(monkeypatch):
    """Where the published S_5 gradient columns were read, and the flag's effect after it.

    One step of budget, so the two runs differ by exactly the duplicated update. The probe
    sees the same weights in both, because it is read after the first step; the logged
    parameter norm differs, because the second step happens after the probe.
    """
    seen = []

    class Spy(trainer.GradientProbe):
        def update(self, model):
            seen.append(float(model.embed.W_E[0, 0].detach()))
            return super().update(model)

    monkeypatch.setattr(trainer, "GradientProbe", Spy)
    base = runs.get("s5_wd1").replace(n=4, max_steps=1, log_every=1, device="cpu",
                                      val_batch_size=32)
    single = trainer.train(base.replace(double_step=False))
    doubled = trainer.train(base)

    assert seen[0] == seen[1]
    assert single.log.weight_norm.iloc[0] != doubled.log.weight_norm.iloc[0]


def test_the_s5_path_logs_the_gradient_columns():
    """Gradient columns, double step and a stride of 5, together."""
    config = runs.get("s5_wd1").replace(n=4, max_steps=30, log_every=5, device="cpu",
                                        val_batch_size=64)
    result = trainer.train(config)
    assert result.log.grad_norm.iloc[1:].gt(0).all()
    assert result.log.embed_grad_norm.iloc[1:].gt(0).all()
    assert result.log.grad_cosine.iloc[0] == 0.0     # no previous gradient on the first step
    assert result.log.grad_cosine.abs().le(1.0 + 1e-9).all()


def test_init_seed_separates_the_initialisation_from_the_split():
    """Same split, different weights. The archived runs never used this axis."""
    same_split = tiny(seed=3, init_seed=11)
    other_init = tiny(seed=3, init_seed=12)
    assert torch.equal(modular_addition(p=17, fraction=0.3, seed=3).X_train,
                       modular_addition(p=17, fraction=0.3, seed=3).X_train)
    assert not np.allclose(trainer.train(same_split).log.weight_norm.to_numpy(),
                           trainer.train(other_init).log.weight_norm.to_numpy())


def test_validation_can_be_evaluated_less_often_than_the_log():
    result = trainer.train(tiny(max_steps=60, log_every=10, val_every=20))
    assert result.log.val_acc.notna().sum() == 3
    assert result.log.train_loss.notna().all()


def test_the_milestones_are_the_first_crossing_of_each_accuracy():
    result = trainer.train(tiny(max_steps=200, log_every=10))
    t_mem, t_gen = result.milestones()
    assert t_mem is not None and t_mem % 10 == 0
    hit = result.log[result.log.train_acc >= 0.95].step.iloc[0]
    assert t_mem == hit
    assert t_gen is None or t_gen >= 0


def test_the_loop_writes_only_where_the_caller_says(tmp_path):
    config = tiny()
    result = trainer.train(config, outdir=tmp_path)
    assert result.paths["log"] == tmp_path / "mod_wd1_train.csv"
    assert result.paths["log"].exists()
    with pytest.raises(FileExistsError):
        trainer.train(config, outdir=tmp_path)


def test_the_resolved_config_reaches_the_provenance_record(tmp_path):
    """A stored sketch that cannot name the overrides that produced it is unusable.

    The archived tree never wrote its configuration beside its outputs; this is the check
    that the port does.
    """
    ctx = build_context("train.smoke", device="cpu", root=tmp_path)
    result = trainer.run(ctx, "mod_wd1", overrides={"p": 17, "max_steps": 30,
                                                    "log_every": 10})
    recorded = ctx.store.provenance.config
    assert recorded["run.key"] == "mod_wd1"
    assert recorded["run.p"] == 17
    assert recorded["run.weight_decay"] == 1.0
    assert recorded["run.double_step"] is False
    assert recorded["run.device"] == "cpu"           # resolved, never the string 'auto'
    assert result.device == "cpu"
    assert set(ctx.store.provenance.notes["milestones.mod_wd1"]) == {"t_mem", "t_gen"}
