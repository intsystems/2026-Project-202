"""The runtime, and one failure it now cannot have.

A ``--fast`` run computes the smallest grid that exercises every branch. It writes the
right files with the right columns and the wrong numbers, and it writes a provenance
record like any other run. Three parts of the system have to know the difference, and each
of them was a real failure before it did:

* ``data/`` must not receive it, or a plumbing check ends up in the article;
* a downstream experiment must not read it as an input, or a real result is computed from
  check values with nothing in the output saying so;
* it must not count as "already run", or a smoke test at teatime quietly removes an
  experiment from the overnight campaign.

The sharpest instance was the calibration. A twenty-five-second ``--fast`` run of
``calib.e8`` selects a frozen configuration, writes it to the file the real calibration
writes, and every downstream experiment and the whole test suite then read the estimator
at ``max_E = 10`` instead of 20 -- silently, because a configuration file that exists is a
configuration file that loads.
"""
from __future__ import annotations

import json

import pytest

from actdim import frozen
from actdim.runtime import store as store_mod
from actdim.runtime.context import build


def _write_run(root, experiment, name, payload, fast):
    directory = root / experiment
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    (directory / "provenance.json").write_text(
        json.dumps({"experiment": experiment, "status": "ok", "fast": fast}),
        encoding="utf-8")
    return directory


def test_a_fast_run_is_recognised(tmp_path):
    fast = _write_run(tmp_path, "calib.e8", "frozen_config.json", {}, fast=True)
    real = _write_run(tmp_path, "calib.e20", "frozen_k20.json", {}, fast=False)
    assert store_mod.is_plumbing_check(fast)
    assert not store_mod.is_plumbing_check(real)


def test_a_directory_without_provenance_is_not_a_fast_run(tmp_path):
    directory = tmp_path / "somewhere"
    directory.mkdir()
    assert not store_mod.is_plumbing_check(directory)


def test_a_fast_calibration_does_not_shadow_the_frozen_configuration(tmp_path, monkeypatch):
    """The failure this guard exists for, reproduced and then prevented.

    The real configuration lives in ``data/``; a ``--fast`` calibration writes a different
    one into ``runs/``, which is searched first. Without the guard the estimator silently
    changes for every experiment in the tree.
    """
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    monkeypatch.setattr(store_mod, "data_root", lambda: data)
    monkeypatch.setattr(store_mod, "runs_root", lambda: runs)
    monkeypatch.setattr(frozen, "data_root", lambda: data)
    monkeypatch.setattr(frozen, "runs_root", lambda: runs)

    real = {"config": {"max_E": 20, "tau": 4, "k_neighbors": 20, "theiler": "autocorr",
                       "window": 8000, "stride": 2000, "dither": 1e-9}}
    check = {"config": dict(real["config"], max_E=10, theiler="embedding", window=400)}
    _write_run(data, "calib.e8", "frozen_config.json", real, fast=False)

    assert frozen.eight_direction().max_E == 20

    _write_run(runs, "calib.e8", "frozen_config.json", check, fast=True)
    config = frozen.eight_direction()
    assert config.max_E == 20, (
        "a --fast calibration in runs/ shadowed the frozen configuration; every "
        "downstream experiment would run at the plumbing-check estimator")
    assert frozen.frozen_path(frozen.EIGHT_DIRECTION).parent.parent == data

    # A real calibration in runs/ is picked up, which is the behaviour the guard must
    # not have broken: a fresh calibration should be usable before it is promoted.
    _write_run(runs, "calib.e8", "frozen_config.json", check, fast=False)
    assert frozen.eight_direction().max_E == 10


def test_a_fast_run_does_not_satisfy_a_downstream_input(tmp_path, monkeypatch):
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    monkeypatch.setattr(store_mod, "data_root", lambda: data)
    monkeypatch.setattr(store_mod, "runs_root", lambda: runs)
    import actdim.runtime.context as context_mod

    monkeypatch.setattr(context_mod, "data_root", lambda: data)
    monkeypatch.setattr(context_mod, "runs_root", lambda: runs)

    _write_run(data, "upstream", "table.csv", {"real": True}, fast=False)
    _write_run(runs, "upstream", "table.csv", {"real": False}, fast=True)

    ctx = build("downstream", device="cpu", root=runs)
    resolved = ctx.input("upstream", "table.csv")
    assert resolved.parent.parent == data, (
        "a --fast upstream run satisfied a downstream input; the result would be "
        "computed from plumbing-check values")


def test_a_missing_input_names_the_command_that_produces_it(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(store_mod, "runs_root", lambda: tmp_path / "runs")
    import actdim.runtime.context as context_mod

    monkeypatch.setattr(context_mod, "data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(context_mod, "runs_root", lambda: tmp_path / "runs")

    ctx = build("downstream", device="cpu", root=tmp_path / "runs")
    with pytest.raises(FileNotFoundError, match="python -m actdim run upstream"):
        ctx.input("upstream", "missing.csv")


def test_the_seed_derivation_is_stable_across_processes():
    """Never ``hash()``: Python salts string hashing per interpreter, so a stream seeded
    that way is reproducible inside one process and nowhere else."""
    import subprocess
    import sys

    from actdim.runtime.determinism import stream_seed

    here = stream_seed(0, "drive_phases"), stream_seed(7, "observation_noise:loss")
    out = subprocess.run(
        [sys.executable, "-c",
         "from actdim.runtime.determinism import stream_seed;"
         "print(stream_seed(0, 'drive_phases'), stream_seed(7, 'observation_noise:loss'))"],
        capture_output=True, text=True, cwd=str(store_mod.repo_root()))
    assert out.returncode == 0, out.stderr
    assert tuple(int(v) for v in out.stdout.split()) == here
