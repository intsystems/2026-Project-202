"""The figures build, and they build at the width LaTeX expects.

Two things are checked here that a reader of the output cannot check by eye. The first
is that every figure is exactly 5.5 in wide: a figure that is not gets scaled by LaTeX
and its 8 pt type shrinks with it, which is why ``bbox_inches="tight"`` is forbidden and
why the assertion is on the figure object rather than on the file. The second is that
the archive fallback stays loud -- a figure built from the archived tree must say so in
the record, since a figure silently built from stale data is exactly what the release
check exists to catch.

The build here passes ``allow_archive=True`` so that it does not depend on how far the
migration has got: whatever has not been promoted into ``data/`` yet is drawn from the
archived tree instead. The tests that check the fallback itself point ``data_root`` at an
empty directory rather than reading the state of the tree, so they say the same thing
before and after a promotion.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless, and the same backend on every machine

import pytest

from actdim.figures import panels, sources
from actdim.figures.style import WIDTH

pytestmark = pytest.mark.skipif(not sources.archive_root().is_dir(),
                                reason="the archived tree is not present")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """All seventeen, drawn once into a temporary directory."""
    outdir = tmp_path_factory.mktemp("figures")
    return panels.build(outdir, allow_archive=True)


def test_seventeen_figures():
    assert len(panels.NAMES) == 17
    assert set(panels.NAMES) == set(panels.PANELS)


def test_every_figure_builds(built):
    assert sorted(built["figures"]) == sorted(panels.NAMES)


@pytest.mark.parametrize("name", panels.NAMES)
def test_output_files_are_written_and_not_empty(built, name):
    from pathlib import Path

    files = built["figures"][name]["files"]
    assert [Path(f).suffix for f in files] == [".pdf", ".png"]
    for path in map(Path, files):
        assert path.is_file(), path
        assert path.stat().st_size > 0, path


@pytest.mark.parametrize("name", panels.NAMES)
def test_figure_is_the_text_width(name):
    """5.5 in exactly, measured on the figure before anything saves it."""
    import matplotlib.pyplot as plt

    fig = panels.draw(name, sources.Reader(allow_archive=True))
    try:
        assert fig.get_size_inches()[0] == pytest.approx(WIDTH, abs=1e-9)
    finally:
        plt.close(fig)


def test_save_refuses_a_figure_of_the_wrong_width(tmp_path):
    import matplotlib.pyplot as plt

    from actdim.figures import style

    fig = plt.figure(figsize=(6.0, 2.0))
    try:
        with pytest.raises(ValueError, match="5.5 in wide"):
            style.save(fig, "fig_wrong", tmp_path)
    finally:
        plt.close(fig)


# -- the sources table ---------------------------------------------------------

#: Sources with no counterpart in the archived tree, and why. A figure that reads one of
#: these cannot be drawn from the archive at all, which is a fact about the figure and has
#: to be listed rather than discovered as a KeyError three frames down.
NO_ARCHIVE = {
    "controls_scored": "the nuisance sweep was rerun for this port; the archived tree "
                       "kept only the summary the article printed",
    "curve_series": "valid.curves is new: the archived tree drew no figure from a log",
    "curve_windows": "valid.curves is new: the archived tree drew no figure from a log",
    "curve_shapes": "valid.curves is new: the archived tree drew no reconstruction",
    "geometry_switch": "the switch trace was written by the archived run but never read; "
                       "it reached data/ only when a figure was drawn from it",
    "theiler_sweep": "the per-window sweep was reduced to a table in the archived tree",
    "surrogate_depths": "the archived tree kept the surrogate summary the article printed "
                        "and not the draws behind it",
}


def test_every_source_has_an_archived_counterpart():
    assert sorted(sources.SOURCES) == sorted(set(sources.ARCHIVE) | set(NO_ARCHIVE))
    assert not set(sources.ARCHIVE) & set(NO_ARCHIVE)
    for name, relative in sources.ARCHIVE.items():
        assert (sources.archive_root() / relative).is_file(), name


def test_the_archive_is_opt_in(monkeypatch, tmp_path):
    """With nothing promoted, an absent file names the experiment that makes it."""
    monkeypatch.setattr(sources, "data_root", lambda: tmp_path)
    for name, (experiment, _) in sources.SOURCES.items():
        with pytest.raises(FileNotFoundError, match=experiment):
            sources.resolve(name)
        if name in NO_ARCHIVE:
            # Asking for the archive says so rather than pretending one exists.
            with pytest.raises(FileNotFoundError, match="no archived equivalent"):
                sources.resolve(name, allow_archive=True)
            continue
        found = sources.resolve(name, allow_archive=True)
        assert found.archived and found.path.is_file(), name


def test_data_wins_over_the_archive(monkeypatch, tmp_path):
    """A promoted file is preferred, and is not reported as archived."""
    promoted = tmp_path / "sys.digits.parameter" / "sweep_raw.csv"
    promoted.parent.mkdir(parents=True)
    promoted.write_text("arm,r\n", encoding="utf-8")
    monkeypatch.setattr(sources, "data_root", lambda: tmp_path)

    found = sources.resolve("sweep_raw", allow_archive=True)
    assert found.path == promoted
    assert not found.archived


def test_a_missing_input_stops_the_build(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "data_root", lambda: tmp_path / "nothing_promoted")
    with pytest.raises(FileNotFoundError, match="valid.theiler.contrast"):
        panels.build(tmp_path / "out", names=["fig_traces"])


def test_the_per_run_resolvers_find_their_logs():
    for kind, run in [("eos_sharp", "eos_lr2e+06_s1"),
                      ("eos_train", "eos_lr2e+06_s1"),
                      ("poly_train", "g_p2_p97")]:
        source = sources.resolve_run(kind, run, allow_archive=True)
        assert source.path.is_file()
        assert source.experiment in ("train.perceptron.eos", "train.perceptron.poly")


def test_every_promoted_name_is_one_the_experiment_can_write():
    """A promotes entry naming a file the trainer never writes kills a whole campaign.

    ``train.perceptron.poly`` declared ``g_p2_p97_train.csv`` while its trainer keys runs
    as appendix O prints them and wrote ``g_p2_train.csv``. Every run in the group had
    already finished when promotion raised on the filename.
    """
    from actdim.runtime import registry as reg
    from actdim.training import runs_perceptron

    poly = reg.load()["train.perceptron.poly"]
    written = {f"{key}_train.csv" for key in runs_perceptron.GROUPS["poly"]}
    for name in poly.promotes:
        if name.endswith("_train.csv"):
            assert name in written, (
                f"train.perceptron.poly promotes {name!r}, which no run in its group "
                f"writes; the group writes {sorted(written)}")


def test_unknown_names_are_refused():
    with pytest.raises(KeyError):
        sources.resolve("no_such_dataset", allow_archive=True)
    with pytest.raises(KeyError):
        panels.build("nowhere", names=["fig_nothing"])


# -- the record ----------------------------------------------------------------

def test_the_record_says_where_every_input_came_from(built):
    for name, entry in built["figures"].items():
        assert entry["sources"], name
        assert set(entry["archived"]) <= set(entry["sources"]), name
        assert bool(entry["archived"]) == (name in built["archived_figures"]), name

    stale = {s for e in built["figures"].values() for s in e["archived"]}
    assert stale == set(built["archived_sources"])


def test_the_summary_is_loud_about_the_archive(tmp_path):
    """A build with anything archived says so, names it, and says not to publish it."""
    record = panels.build(tmp_path, names=["fig_pairs"], allow_archive=True)
    text = panels.summary(record)

    if record["archived_figures"]:
        assert "WARNING" in text
        assert "must not be published" in text
        for name in record["archived_figures"]:
            assert name in text
        for source in record["archived_sources"]:
            assert source in text
    else:  # everything fig_pairs reads has been promoted
        assert "every input came from data/" in text


def test_the_summary_is_quiet_when_nothing_is_archived():
    record = {"outdir": "figures", "figures": {"fig_traces": {}},
              "archived_figures": [], "archived_sources": []}
    assert panels.summary(record).endswith("every input came from data/")
