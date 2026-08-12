"""The trajectory sketch: the compression, the observer, and the analysis over windows.

The load-bearing test here is :func:`test_the_probe_leaves_the_run_bit_identical`. The task
module seeds one global torch stream that the split, the initial weights and the mini-batch
order all continue, so an observer that draws from it changes the run it claims to be
watching. The archived check ran on the CPU only; this one also runs the CUDA branch of the
RNG save and restore when a GPU is present, and skips it otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from actdim.models.transformer import parameter_count                        # noqa: E402
from actdim.sketch.analysis import (ARTICLE, COARSE, FINE, WindowGeometry,   # noqa: E402
                                    block_mean, collapse, collapse_controls,
                                    collapse_controls_aligned, detrend, milestones,
                                    output_paths, pr, sliding, summarise)
from actdim.sketch.countsketch import CountSketch, hash_family                # noqa: E402
from actdim.sketch.probe import (TrajectoryRecorder, TrajectorySketch,        # noqa: E402
                                 preserve_torch_rng)
from actdim.training import runs, transformer as trainer                      # noqa: E402

CUDA = torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not CUDA, reason="no GPU"))]


def tiny(**overrides):
    settings = {"p": 17, "max_steps": 200, "log_every": 10, "device": "cpu"}
    settings.update(overrides)
    return runs.get("mod_wd1").replace(**settings)


# -- the compression -----------------------------------------------------------


def test_the_hashes_are_a_function_of_the_seed_alone():
    idx, sign = hash_family(500, 32, seed=11)
    again, again_sign = hash_family(500, 32, seed=11)
    assert np.array_equal(idx, again) and np.array_equal(sign, again_sign)
    assert idx.min() >= 0 and idx.max() < 32
    assert set(np.unique(sign)) <= {-1.0, 1.0}
    assert not np.array_equal(idx, hash_family(500, 32, seed=12)[0])


def test_the_sketch_preserves_inner_products():
    """The property everything downstream rests on: geometry survives the compression."""
    rng = np.random.default_rng(0)
    sketch = CountSketch(n_in=4000, dim=1024, n_sketch=4, seed=3)
    u, v = rng.normal(size=4000), rng.normal(size=4000)
    exact = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    su, sv = sketch.apply_numpy(u), sketch.apply_numpy(v)
    estimates = [float(su[s] @ sv[s]) / (np.linalg.norm(su[s]) * np.linalg.norm(sv[s]))
                 for s in range(4)]
    assert abs(float(np.mean(estimates)) - exact) < 0.1


def test_the_torch_and_numpy_paths_agree():
    sketch = CountSketch(n_in=257, dim=16, n_sketch=2, seed=5)
    vec = np.random.default_rng(1).normal(size=257)
    assert np.allclose(sketch.apply(torch.from_numpy(vec)).numpy(), sketch.apply_numpy(vec))


def test_a_sketch_refuses_a_vector_of_the_wrong_length():
    sketch = CountSketch(n_in=10, dim=4, n_sketch=1, seed=0)
    with pytest.raises(ValueError, match="must not change size"):
        sketch.apply(torch.zeros(11))


# -- the interface the perceptron trainer also imports --------------------------


def test_trajectory_sketch_shapes_and_metadata():
    sketch = TrajectorySketch(n_params=1000, dim=64, n_sketch=2, seed=0)
    z = sketch.sketch_parameters(torch.randn(1000, dtype=torch.float64))
    assert isinstance(z, np.ndarray) and z.shape == (2, 64)

    zf = sketch.sketch_function(torch.randn(8, 5, dtype=torch.float64))
    assert zf.shape == (2, 64)
    assert sketch.metadata() == {"n_params": 1000, "dim": 64, "n_sketch": 2, "seed": 0}


def test_the_function_sketch_is_blind_to_offset_and_scale():
    """Raw logit scale tracks weight decay, so the sketch must not see it."""
    sketch = TrajectorySketch(n_params=10, dim=32, n_sketch=2, seed=1)
    logits = torch.randn(6, 4, dtype=torch.float64)
    base = sketch.sketch_function(logits)
    shifted = sketch.sketch_function(logits * 7.5 + 3.0)
    assert np.allclose(base, shifted, atol=1e-9)


def test_two_hash_families_disagree_but_not_by_much():
    sketch = TrajectorySketch(n_params=5000, dim=512, n_sketch=2, seed=2)
    flat = torch.randn(5000, dtype=torch.float64)
    z = sketch.sketch_parameters(flat)
    norms = np.linalg.norm(z, axis=1)
    exact = float(torch.linalg.norm(flat))
    assert abs(norms[0] - norms[1]) / exact < 0.2
    assert np.allclose(norms / exact, 1.0, atol=0.2)


@pytest.mark.parametrize("device", DEVICES)
def test_preserve_torch_rng_restores_the_stream(device):
    """Including the CUDA stream, which the archived check never exercised."""
    torch.manual_seed(0)
    expected = torch.randn(4, device=device)

    torch.manual_seed(0)
    with preserve_torch_rng(device):
        torch.randn(64, device=device)      # the draw an observer must not be able to make
    assert torch.equal(torch.randn(4, device=device), expected)


# -- the observer --------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_the_probe_leaves_the_run_bit_identical(device):
    """Every logged column, bit for bit, with the probe attached and without it.

    A single stray draw from the global torch stream changes the initial weights, and the
    initial weights decide whether these runs generalise at all.
    """
    config = tiny(device=device)
    plain = trainer.train(config)
    probed = trainer.train(config, observer=TrajectoryRecorder(dim=64, n_sketch=2, n_probe=16))

    assert list(plain.log.columns) == list(probed.log.columns)
    assert len(plain.log) == len(probed.log)
    for column in plain.log.columns:
        assert np.array_equal(plain.log[column].to_numpy(), probed.log[column].to_numpy()), (
            f"{column} moved when the probe was attached")


def test_the_recorder_stores_both_spaces_and_its_own_metadata():
    recorder = TrajectoryRecorder(dim=32, n_sketch=2, n_probe=8, seed=4)
    result = trainer.train(tiny(max_steps=60), observer=recorder)
    arrays = recorder.arrays()

    rows = len(result.log)
    assert arrays["z"].shape == (rows, 2, 32)
    assert arrays["zf"].shape == (rows, 2, 32)
    assert np.array_equal(arrays["step"], result.log.step.to_numpy())
    assert np.isnan(arrays["param_step"][0]) and np.isfinite(arrays["param_step"][1:]).all()
    assert np.isfinite(arrays["param_norm"]).all()
    assert int(arrays["n_params"]) == parameter_count(d_vocab=18)   # p = 17, plus '='
    assert recorder.metadata()["source"] == "train"
    assert recorder.metadata()["n_probe"] == 8


def test_the_sketch_is_written_as_one_compressed_file(tmp_path):
    recorder = TrajectoryRecorder(dim=32, n_sketch=2, n_probe=8)
    result = trainer.train(tiny(max_steps=60), outdir=tmp_path, observer=recorder)
    path = result.paths["sketch"]
    assert path.exists() and path.suffix == ".npz"

    stored = np.load(path)
    assert stored["z"].shape == recorder.arrays()["z"].shape
    assert int(stored["dim"]) == 32 and int(stored["n_sketch"]) == 2
    assert str(stored["source"]) == "train"
    assert int(stored["n_params"]) == recorder.n_params


# -- the analysis --------------------------------------------------------------


def test_the_window_geometry_is_configuration_not_prose():
    """The pass the article reports was recorded only in a report file, in prose."""
    assert (FINE.window, FINE.stride) == (60, 10)
    assert (COARSE.window, COARSE.stride) == (200, 25)
    assert ARTICLE is FINE


def test_a_block_size_that_cannot_fit_the_window_is_dropped():
    """Eight all-NaN columns in the published table came from emitting them anyway."""
    assert FINE.usable_smoothing() == (5,)
    assert FINE.dropped_smoothing() == (20,)
    assert COARSE.usable_smoothing() == (5, 20)
    assert "PR_step20" not in FINE.columns()
    assert "PR_step5" in FINE.columns() and "PR_step1" not in FINE.columns()


def test_output_paths_follow_the_results_directory(tmp_path):
    """The archived collapse script pinned its figure directory beside the code."""
    fine = output_paths(tmp_path / "runA", FINE)
    coarse = output_paths(tmp_path / "runA", COARSE)
    other = output_paths(tmp_path / "runB", FINE)

    assert set(fine.values()).isdisjoint(set(coarse.values()))
    assert set(fine.values()).isdisjoint(set(other.values()))
    for path in fine.values():
        assert (tmp_path / "runA") in path.parents


def test_the_participation_ratio_counts_directions():
    rng = np.random.default_rng(0)
    for rank in (1, 3, 8):
        basis = rng.normal(size=(rank, 200))
        weights = rng.normal(size=(400, rank))
        assert pr(weights @ basis) == pytest.approx(rank, rel=0.2)
    assert np.isnan(pr(np.zeros((10, 4))))
    assert np.isnan(pr(np.ones((2, 4))))          # too few rows to have a shape


def test_detrending_removes_a_straight_drift():
    t = np.arange(60.0)[:, None]
    drift = t @ np.ones((1, 5))
    assert np.isnan(pr(detrend(drift))) or pr(detrend(drift)) < 1.5
    assert block_mean(np.arange(20.0).reshape(-1, 1), 5).shape == (4, 1)


def synthetic_sketch(rows=200, dim=16, n_sketch=2, seed=0, stride=10):
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 1, rows)[:, None] * rng.normal(size=(1, dim))
    z = np.stack([trend + 0.1 * rng.normal(size=(rows, dim)) for _ in range(n_sketch)], axis=1)
    zf = np.stack([0.5 * trend + 0.1 * rng.normal(size=(rows, dim)) for _ in range(n_sketch)],
                  axis=1)
    step = np.arange(rows) * stride
    move = np.concatenate([[np.nan], np.linalg.norm(np.diff(z[:, 0, :], axis=0), axis=1)])
    return {"step": step, "z": z, "zf": zf, "param_step": move,
            "param_norm": np.linalg.norm(z[:, 0, :], axis=1),
            "n_params": np.asarray(1000), "dim": np.asarray(dim),
            "n_sketch": np.asarray(n_sketch)}


def test_no_window_column_is_silently_all_nan():
    """The archived fine pass emitted eight, with a warning per window.

    ``pytest`` here turns a RuntimeWarning into an error, so the warnings are caught by
    this test running at all; the assertion catches the columns.
    """
    frame = sliding(synthetic_sketch(), FINE, run="synthetic")
    assert len(frame) == (200 - 60) // 10 + 1
    empty = [c for c in frame.columns if frame[c].isna().all()]
    assert empty == []
    assert list(frame.columns) == FINE.columns()
    assert (frame.centre == (frame.left_step + frame.right_step) / 2).all()


def test_a_window_is_labelled_by_its_centre_not_its_right_edge():
    frame = sliding(synthetic_sketch(stride=10), FINE, run="synthetic")
    first = frame.iloc[0]
    assert first.left_step == 0 and first.right_step == 590 and first.centre == 295.0


def test_the_summary_records_the_geometry_it_was_computed_at():
    sketch = synthetic_sketch()
    log = pd.DataFrame({"step": np.arange(0, 2000, 10),
                        "train_acc": np.linspace(0, 1, 200),
                        "val_acc": np.linspace(0, 1, 200)})
    row = summarise("synthetic", log, sketch, sliding(sketch, FINE, run="synthetic"), FINE)
    assert row["geometry"] == "fine" and row["window"] == 60 and row["stride"] == 10
    assert row["n_params"] == 1000 and row["t_mem"] is not None

    # A recorder that keeps its metadata in the provenance rather than in the arrays is
    # summarised all the same, with the missing fields reported as absent.
    lean = {k: v for k, v in sketch.items() if k not in ("n_params", "dim", "n_sketch")}
    assert summarise("synthetic", log, lean, windows=pd.DataFrame(), geometry=FINE)[
        "n_params"] is None


def test_the_two_milestone_rules():
    log = pd.DataFrame({
        "step": np.arange(12) * 10,
        "train_acc": [0.1] + [1.0] * 11,
        "val_acc": [0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.9, 0.95, 0.99, 1.0, 1.0, 1.0],
    })
    # The plain rule takes the first crossing, spike or not; the sustained rule the
    # extended reruns use waits for the block around it to hold.
    assert milestones(log) == (10, 20)
    assert milestones(log, sustain=5) == (30, 80)
    assert milestones(log.assign(val_acc=0.0))[1] is None


def planted_windows():
    """Two runs: one with a dip that recovers at step 1000, one control that only decays."""
    centre = np.arange(0, 8001, 100, dtype=float)
    dip = 4.0 - 2.5 * np.exp(-((centre - 1000) ** 2) / (2 * 150.0 ** 2))
    decay = 4.0 * np.exp(-centre / 900.0)
    frames = []
    for run, values in (("grokker", dip), ("control", decay)):
        frame = pd.DataFrame({"run": run, "centre": centre})
        for stat in ("fn_PR_pos_det", "fn_PR_step5", "PR_pos_det", "PR_step5", "move"):
            frame[stat] = values
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_the_collapse_table_locates_the_dip_and_its_recovery():
    windows = planted_windows()
    marks = {"grokker": (100, 1000), "control": (100, None)}

    table = collapse(windows, marks)
    assert set(table.run) == {"grokker"}
    row = table[table.stat == "PR_pos_det"].iloc[0]
    assert row["offset"] == pytest.approx(0.0, abs=100)
    assert row["dip"] < row["plateau"] and row["recovered"] > row["dip"]
    assert row["depth"] > 1.5

    controls = collapse_controls(windows, marks)
    assert set(controls.run) == {"control"}
    assert controls[controls.stat == "PR_pos_det"].iloc[0]["end"] < 1.0

    aligned = collapse_controls_aligned(windows, marks, reference={"control": "grokker"})
    assert set(aligned.run) == {"control"}
    assert aligned.reference.eq("grokker").all()


def test_the_whole_chain_runs_on_a_real_run():
    """Train, sketch, slide, summarise -- the path an experiment module will take."""
    recorder = TrajectoryRecorder(dim=32, n_sketch=2, n_probe=8)
    result = trainer.train(tiny(max_steps=800, log_every=10), observer=recorder)
    geometry = WindowGeometry(name="test", window=20, stride=10, smooth=(5,))

    windows = sliding(recorder.arrays(), geometry, run="mod_wd1")
    assert len(windows) == (len(result.log) - 20) // 10 + 1
    assert windows.PR_pos_det.between(1.0, 20.0).all()
    assert windows.notna().all().all()

    row = summarise("mod_wd1", result.log, recorder.arrays(), windows, geometry)
    assert row["n_params"] == recorder.n_params and row["n_windows"] == len(windows)
