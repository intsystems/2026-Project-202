"""Sanity checks for the training package and the run registry.

Run with ``pytest test_train.py`` or directly: ``python test_train.py``.

The group algebra and the registry checks run without torch; the end-to-end
training checks skip themselves if torch is not installed.
"""

import csv
import itertools
import math

import numpy as np

import runs
from grok.config import GRAD_OBSERVABLES, OBSERVABLES, RunConfig, coerce
from grok.groups import SymmetricGroup, minimal_faithful_dimension, permutations, rank

try:
    import torch
except ImportError:                                 # pragma: no cover - environment dependent
    torch = None


class Skipped(Exception):
    """Raised instead of failing when an optional dependency is missing."""


def _require_torch():
    if torch is None:
        raise Skipped("torch is not installed")


# ---------------------------------------------------------------------------
# Group algebra
# ---------------------------------------------------------------------------

def _reference_table(n):
    """The original ``get_s5_composition_data`` table construction, generalised to n."""
    perms = list(itertools.permutations(range(n)))
    perm2id = {p: i for i, p in enumerate(perms)}
    table = np.zeros((len(perms), len(perms)), dtype=np.int64)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            table[i, j] = perm2id[tuple(a[b[k]] for k in range(n))]
    return table


def test_cayley_table_matches_the_original_construction():
    """The vectorised table must equal the notebooks' nested-loop one, element for element."""
    for n in (3, 4, 5):
        np.testing.assert_array_equal(SymmetricGroup(n).table(), _reference_table(n))


def test_rank_inverts_permutations():
    for n in (2, 4, 6):
        perms = permutations(n)
        assert len(perms) == math.factorial(n)
        np.testing.assert_array_equal(rank(perms), np.arange(len(perms)))


def test_group_axioms_hold_on_s5():
    group = SymmetricGroup(5)
    table = group.table()
    ids = np.arange(group.order)

    # identity
    np.testing.assert_array_equal(table[group.identity, :], ids)
    np.testing.assert_array_equal(table[:, group.identity], ids)

    # Latin square: every row and column is a permutation of the elements
    for row in (table[3], table[:, 7], table[100]):
        np.testing.assert_array_equal(np.sort(row), ids)

    # associativity on a random sample of triples
    rng = np.random.default_rng(0)
    a, b, c = (rng.integers(0, group.order, size=500) for _ in range(3))
    np.testing.assert_array_equal(
        group.compose(group.compose(a, b), c),
        group.compose(a, group.compose(b, c)),
    )


def test_s5_is_non_abelian_but_s2_is_abelian():
    group = SymmetricGroup(5)
    table = group.table()
    assert not np.array_equal(table, table.T), "S_5 must not commute"
    assert np.array_equal(SymmetricGroup(2).table(), SymmetricGroup(2).table().T)


def test_composition_is_chunk_invariant():
    """``compose`` blocks internally; the blocking must not change the answer."""
    group = SymmetricGroup(4)
    ids = np.arange(group.order)
    a = np.repeat(ids, group.order)
    b = np.tile(ids, group.order)
    np.testing.assert_array_equal(
        group.compose(a, b).reshape(group.order, group.order), group.table()
    )


def test_minimal_faithful_dimension_matches_the_article():
    """Sec. 4.2 lists the irreps of S_5 as 1,1,4,4,5,5,6 -> the algebraic floor is 4."""
    assert minimal_faithful_dimension(5) == 4
    assert minimal_faithful_dimension(6) == 5


def test_large_group_refuses_to_materialise_the_full_product_set():
    try:
        SymmetricGroup(7).table()
    except ValueError as exc:
        assert "max_pairs" in str(exc)
        return
    raise AssertionError("expected S_7 to refuse a 25M-pair Cayley table")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_config_rejects_logs_the_analysis_cannot_read():
    """``edm.load_logs`` requires step / train_acc / val_acc, so RunConfig does too."""
    for bad in (("train_acc", "val_acc"), ("step", "val_acc"), ("step", "train_acc")):
        try:
            RunConfig(columns=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected {bad} to be rejected")


def test_config_rejects_unknown_columns():
    try:
        RunConfig(columns=("step", "train_acc", "val_acc", "hessian_trace"))
    except ValueError as exc:
        assert "hessian_trace" in str(exc)
        return
    raise AssertionError("expected an unknown column to be rejected")


def test_overrides_are_coerced_to_the_declared_types():
    assert coerce("n", "6") == 6
    assert coerce("lr", "1e-3") == 1e-3
    assert coerce("batch_size", "none") is None
    assert coerce("double_step", "false") is False
    assert coerce("columns", "step,train_acc,val_acc") == ("step", "train_acc", "val_acc")
    assert coerce("betas", "0.9,0.98") == (0.9, 0.98)


def test_csv_name_templates_resolve():
    sn = runs.get("sn").with_overrides({"n": "6"})
    assert sn.csv_name == "grokking_s6_logs.csv"
    assert runs.get("s5_wd1").csv_name.endswith(".csv")


def test_expected_rows_matches_the_published_logs():
    """Row count is ``ceil(max_steps / log_every)`` -- the analysis windows depend on it."""
    assert runs.get("s5_wd1").expected_rows == 3000        # 15000 / 5
    assert runs.get("mod_wd1").expected_rows == 2000       # 20000 / 10
    assert runs.get("full_batch").expected_rows == 15000   # every step


def test_grad_probe_is_requested_exactly_when_its_columns_are():
    assert runs.get("s5_wd1").needs_grad_probe
    assert not runs.get("mod_wd1").needs_grad_probe
    assert set(GRAD_OBSERVABLES) <= set(OBSERVABLES)


# ---------------------------------------------------------------------------
# The registry against the logs the analysis package actually consumes
# ---------------------------------------------------------------------------

def test_registry_reproduces_the_headers_of_the_published_logs():
    """Column names *and order* must match the CSVs in ../grokking_analysis."""
    checked = 0
    for key in runs.ARTICLE_RUNS:
        config = runs.get(key)
        path = runs.ANALYSIS_LOG_DIR / config.csv_name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert tuple(header) == tuple(config.columns), f"{key}: {header} != {config.columns}"
        checked += 1
    assert checked, f"no published logs found under {runs.ANALYSIS_LOG_DIR}"


def test_registry_row_counts_match_the_published_logs():
    checked = 0
    for key in runs.ARTICLE_RUNS:
        config = runs.get(key)
        path = runs.ANALYSIS_LOG_DIR / config.csv_name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        assert rows == config.expected_rows, f"{key}: {rows} rows != {config.expected_rows}"
        checked += 1
    assert checked, f"no published logs found under {runs.ANALYSIS_LOG_DIR}"


def test_article_runs_pair_up_with_the_analysis_figure_registry():
    """Every figure key on the analysis side must be produced by a run here."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "experiments", runs.ANALYSIS_LOG_DIR.parent / "experiments.py"
    )
    if spec is None or not (runs.ANALYSIS_LOG_DIR.parent / "experiments.py").exists():
        raise Skipped("grokking_analysis/experiments.py not found")
    experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiments)

    produced = {runs.get(key).csv_name for key in runs.ARTICLE_RUNS}
    consumed = {fig.csv for fig in experiments.FIGURES.values()}
    assert consumed <= produced, f"no run produces {sorted(consumed - produced)}"


# ---------------------------------------------------------------------------
# End to end (needs torch)
# ---------------------------------------------------------------------------

def test_modular_addition_task_is_correct():
    _require_torch()
    from grok import tasks

    task = tasks.modular_addition(p=11, fraction=0.5, seed=0)
    assert task.num_total == 121 and task.vocab_size == 12
    assert len(task.X_train) == 60 and len(task.X_val) == 61

    for X, Y in ((task.X_train, task.Y_train), (task.X_val, task.Y_val)):
        assert (X[:, 2] == 11).all(), "the '=' token must be the last position"
        np.testing.assert_array_equal(
            ((X[:, 0] + X[:, 1]) % 11).numpy(), Y.numpy()
        )


def test_symmetric_group_task_answers_are_group_products():
    _require_torch()
    from grok import tasks

    task = tasks.symmetric_group(n=4, fraction=0.5, seed=0)
    group = SymmetricGroup(4)
    assert task.num_classes == 24 and task.vocab_size == 25
    np.testing.assert_array_equal(
        group.compose(task.X_train[:, 0].numpy(), task.X_train[:, 1].numpy()),
        task.Y_train.numpy(),
    )


def test_subsampling_keeps_the_products_correct():
    """`max_pairs` is what makes S_7+ representable -- it must not corrupt the labels."""
    _require_torch()
    from grok import tasks

    task = tasks.symmetric_group(n=5, fraction=0.5, seed=0, max_pairs=2000)
    assert task.num_total == 2000
    np.testing.assert_array_equal(
        SymmetricGroup(5).compose(task.X_val[:, 0].numpy(), task.X_val[:, 1].numpy()),
        task.Y_val.numpy(),
    )


def test_task_split_is_deterministic_given_the_seed():
    _require_torch()
    from grok import tasks

    a = tasks.modular_addition(p=13, fraction=0.4, seed=7)
    b = tasks.modular_addition(p=13, fraction=0.4, seed=7)
    np.testing.assert_array_equal(a.X_train.numpy(), b.X_train.numpy())


def test_models_emit_one_logit_row_per_prompt():
    _require_torch()
    from grok import models
    from grok.config import RunConfig

    prompts = torch.zeros((8, 3), dtype=torch.long)
    for name in ("omnigrok", "encoder"):
        config = RunConfig(model=name, d_model=32, d_mlp=64, d_head=8, num_heads=2)
        logits = models.build(config, vocab_size=20)(prompts)
        assert logits.shape == (8, 20), f"{name}: {logits.shape}"


def test_attention_head_merge_matches_einops():
    """``permute().reshape()`` replaces the original ``einops.rearrange``."""
    _require_torch()
    try:
        import einops
    except ImportError:
        raise Skipped("einops is not installed")

    z = torch.randn(4, 3, 5, 7)
    np.testing.assert_allclose(
        z.permute(0, 2, 1, 3).reshape(z.shape[0], z.shape[2], -1).numpy(),
        einops.rearrange(z, "b i q h -> b q (i h)").numpy(),
    )


def test_short_run_produces_a_log_the_analysis_package_can_read():
    _require_torch()
    from grok import train

    config = runs.get("s5_wd1").with_overrides(
        {"n": "3", "max_steps": "40", "log_every": "4", "batch_size": "16",
         "val_batch_size": "16", "d_model": "16", "d_mlp": "32", "d_head": "4",
         "num_heads": "2", "device": "cpu", "double_step": "false"}
    )
    df, path = train(config, outdir=None, progress=False)

    assert path is None
    assert list(df.columns) == list(config.columns)
    assert len(df) == config.expected_rows == 10
    np.testing.assert_array_equal(df["step"].to_numpy(), np.arange(0, 40, 4))
    assert df.notna().all().all(), "no column may contain NaN"
    assert (df["weight_norm"] > 0).all()
    assert df["train_loss"].iloc[0] > 0
    assert df["grad_cosine"].iloc[0] == 0.0, "no previous gradient on the first step"
    assert (df["grad_cosine"].abs() <= 1.0 + 1e-9).all()


def test_double_step_actually_changes_the_trajectory():
    """The published S_5 logs depend on the double-step bug; the flag must be real."""
    _require_torch()
    from grok import train

    base = runs.get("sn").with_overrides(
        {"n": "3", "max_steps": "20", "log_every": "5", "batch_size": "16",
         "val_batch_size": "16", "d_model": "16", "d_mlp": "32", "d_head": "4",
         "num_heads": "2", "device": "cpu", "weight_decay": "1.0"}
    )
    single, _ = train(base.with_overrides({"double_step": "false"}), progress=False)
    double, _ = train(base.with_overrides({"double_step": "true"}), progress=False)
    assert single["weight_norm"].iloc[-1] != double["weight_norm"].iloc[-1]


def test_full_batch_configuration_runs():
    """`batch_size=None` must take the whole training split every step."""
    _require_torch()
    from grok import train

    config = runs.get("full_batch").with_overrides(
        {"p": "7", "max_steps": "5", "d_model": "16", "num_heads": "2",
         "device": "cpu", "seed": "0"}
    )
    df, _ = train(config, progress=False)
    assert len(df) == 5
    assert list(df.columns) == list(config.columns)


def test_a_fresh_log_round_trips_through_the_analysis_package():
    """The point of the whole module: train -> CSV -> ``edm.sliding_dimension``."""
    _require_torch()
    import sys
    import tempfile

    analysis_dir = runs.ANALYSIS_LOG_DIR.parent
    if not (analysis_dir / "edm").is_dir():
        raise Skipped(f"{analysis_dir} not found")
    sys.path.insert(0, str(analysis_dir))
    try:
        from edm import load_logs, sliding_dimension
    except ImportError as exc:                      # scikit-learn is an edm dependency
        raise Skipped(f"analysis package unavailable: {exc}")

    from grok import train

    config = runs.get("sn").with_overrides(
        {"n": "3", "max_steps": "400", "log_every": "1", "batch_size": "16",
         "val_batch_size": "16", "d_model": "16", "d_mlp": "32", "d_head": "4",
         "num_heads": "2", "device": "cpu"}
    )
    with tempfile.TemporaryDirectory() as tmp:
        _, path = train(config, outdir=tmp, progress=False)
        reloaded = load_logs(path)                  # enforces step / train_acc / val_acc
        trace = sliding_dimension(reloaded, target_metric="weight_norm", window_size=300,
                                  step_size=50, seed=0, progress=False)

    assert len(reloaded) == config.expected_rows
    assert len(trace.dimension) > 0 and np.isfinite(trace.dimension).all()


def test_training_is_reproducible_on_cpu():
    _require_torch()
    from grok import train

    config = runs.get("sn").with_overrides(
        {"n": "3", "max_steps": "20", "log_every": "5", "batch_size": "16",
         "val_batch_size": "16", "d_model": "16", "d_mlp": "32", "d_head": "4",
         "num_heads": "2", "device": "cpu", "seed": "123"}
    )
    first, _ = train(config, progress=False)
    second, _ = train(config, progress=False)
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Skipped as exc:
            print(f"SKIP  {name}: {exc}")
        except Exception as exc:                    # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
