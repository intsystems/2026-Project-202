"""The table auditor: the LaTeX reader, the rounding rule, and the defects it must catch.

This repository has twice committed a result file its own script could no longer reproduce.
``actdim.tables`` is the mechanical guard against a third, and these tests are the guard on
the guard. Two kinds of assertion are made here.

The first kind is on the reader and the comparison, which are pure and are checked against
strings written out in the test. The second kind is on the audit of the real article: the
defects ``docs/errata.md`` already records must appear in the report, because an auditor
that reports nothing is indistinguishable from one that checks nothing. Those tests name the
errata item they stand for, so that a change which silences one says which claim it is
withdrawing.

The audit is run once for the module. It reads ``icomp_v2/report.tex`` and the tracked half
of ``data/``, and writes nothing.
"""
from __future__ import annotations

import pytest

from actdim import tables
from actdim.tables import (ARCHIVED, CLAIM, MISMATCH, OK, REGENERATED, ROUNDING, Data,
                           ParseError, as_number, audit, compare, decimals, format_report,
                           read_tables)

pytestmark = pytest.mark.skipif(not tables.article_path().is_file(),
                                reason="../icomp_v2/report.tex is not present")


@pytest.fixture(scope="module")
def report():
    return audit()


@pytest.fixture(scope="module")
def parsed():
    return read_tables()


def bad(report, label):
    """The disagreeing findings of one table, as ``(row, column)`` pairs and their text."""
    return report.by_label(label).mismatches


# -- the LaTeX reader ----------------------------------------------------------


def test_every_labelled_table_in_the_article_is_read(parsed):
    """Twenty-nine ``tabular`` blocks, and none of them silently dropped."""
    assert len(parsed) == 29
    assert all(label.startswith("tab:") for label in parsed)


def test_the_formatting_macros_are_stripped(parsed):
    """A cell has to compare as a number, so nothing that only affects type may survive."""
    runs = parsed["tab:runs"]
    row = runs.find("mod_wd1_s43")
    assert runs.printed(row, 0) == "mod_wd1_s43"      # \texttt{} and the escaped underscore
    assert as_number(runs.printed(row, 6)) == 1580.0  # the \, thin space inside the number
    assert runs.printed(runs.find("mod_wd0"), 7) is None   # the --- of a censored run


def test_a_multicolumn_heading_spans_its_columns(parsed):
    """The block headings of ``tab:runs`` span all ten, and must not shift the rows below."""
    runs = parsed["tab:runs"]
    heading = [row for row in runs.rows if row.width >= 10 and len(row.cells) == 1]
    assert len(heading) == 3
    assert all(row.cells[0].span == 10 for row in heading)
    assert all(runs.rows[i].width == 10 for i in runs.body())


def test_a_row_is_found_by_its_label_and_not_by_its_number(parsed):
    """Checks name rows, so that inserting one above does not silently move every check."""
    diagnostics = parsed["tab:grok-diagnostics"]
    add = diagnostics.find("perceptron, n+m")
    multiply = diagnostics.find("perceptron, n*m")
    assert add != multiply
    with pytest.raises(ParseError):
        diagnostics.find("a row this table does not have")


def test_a_table_with_no_rows_is_refused():
    """A block this reader cannot read is refused rather than guessed at: a mis-parsed row
    would put a computed value beside the wrong printed one."""
    with pytest.raises(ParseError):
        tables._parse_block("tab:empty", [(1, "\\begin{tabular}{ll}"),
                                          (2, "\\end{tabular}")], 1)


# -- reading a printed cell ----------------------------------------------------


@pytest.mark.parametrize("text,value", [
    ("1.62", 1.62), ("1\\,470", None), ("100k", 100000.0), ("20k", 20000.0),
    ("-0.57", -0.57), ("+712", 712.0), ("fails 2, 5", None), ("", None),
])
def test_as_number_reads_what_the_article_prints(text, value):
    # The thin space is stripped by the cleaner, not by this function, so the raw form of
    # 1,470 is not a number here.
    assert as_number(text) == value


def test_the_comparison_is_at_the_precision_printed():
    """The rule the whole check rests on: 1.62 is 1.6239 and is not 1.554."""
    assert compare("1.62", 1.6239)[0] == OK
    assert compare("1.62", 1.554)[0] == MISMATCH
    assert decimals("1.62") == 2 and decimals("100k") == 0


def test_a_cell_out_by_one_unit_in_the_last_place_is_reported_as_rounding():
    """The measurement agrees and the digits do not, which is a different fix."""
    assert compare("5.47", 5.464913)[0] == ROUNDING
    assert compare("5.47", 5.9)[0] == MISMATCH


def test_a_cell_printed_as_a_bound_is_compared_as_one():
    """`<0.001` states an inequality, and the inequality is what has to hold. Reading it as
    unreadable left the two smallest roughness values of tab:grok-diagnostics unchecked."""
    assert compare("<0.001", 0.0002)[0] == OK
    assert compare("<0.001", 0.0021)[0] == MISMATCH
    assert compare(">0.9", 0.95)[0] == OK
    assert compare(">0.9", 0.5)[0] == MISMATCH


def test_a_row_is_reported_at_the_line_its_content_starts_on():
    """A rule on its own line is buffered with the row below it. Taking the line the buffer
    opened on pointed every such row at the rule above, which nothing downstream notices --
    the row still parses, and only an edit made at the reported line lands in the wrong
    row."""
    source = ("\\begin{tabular}{lr}\n"
              "\\toprule\n"
              "name & value \\\\\n"
              "\\midrule\n"
              "first & 1.00 \\\\\n"
              "\\addlinespace\n"
              "second & 2.00 \\\\\n"
              "\\bottomrule\n"
              "\\end{tabular}\n")
    lines = source.split("\n")
    table = tables._parse_block("tab:x", list(enumerate(lines, start=1)), 1)
    for row in table.rows:
        assert row.label() in lines[row.line - 1], (
            f"row {row.label()!r} is reported at line {row.line}, "
            f"which holds {lines[row.line - 1]!r}")


def test_a_dash_and_a_missing_value_agree_only_with_each_other():
    assert compare(None, None)[0] == OK
    assert compare(None, 12.0)[0] == MISMATCH
    assert compare("13\\,700", None)[0] == MISMATCH


def test_a_per_seed_list_is_compared_elementwise_and_in_order():
    """``tab:ladder`` row 1 prints two seeds per cell; the order is the defect."""
    assert compare("0.99 / 1.00", [0.99, 1.0])[0] == OK
    assert compare("0.99 / 1.00", [1.0, 0.9879])[0] == MISMATCH


def test_a_tolerance_is_only_for_a_step_between_two_logged_ones():
    """The S_5 runs log every five steps, so a window centre lands on 712.5."""
    assert compare("+712", 712.5)[0] == ROUNDING
    assert compare("+712", 712.5, tolerance=tables.HALF_STEP)[0] == OK


# -- the audit of the real article ---------------------------------------------


def test_the_audit_covers_every_table_in_the_article(report, parsed):
    """Every label is registered, one way or the other. A table nobody decided about is
    the state this module exists to make impossible."""
    registered = {entry.label for entry in tables.REGISTRY}
    assert registered == set(parsed)
    assert {r.label for r in report.results} == registered


def test_every_skipped_table_says_why(report):
    for result in report.results:
        if result.state == "skipped":
            assert result.reason, f"{result.label} was skipped without a reason"


def test_the_checked_tables_actually_compared_something(report):
    for result in report.results:
        if result.state == "checked":
            assert result.checked > 0, f"{result.label} is registered but compared nothing"


def test_the_report_says_which_inputs_are_archived(report):
    """A table checked against an archived file has been checked against the numbers the
    article was written from, not against a regeneration, and that has to be visible."""
    assert report.inputs
    assert report.archived_inputs, "every promoted file is archived today"
    assert all(report.inputs[name] == ARCHIVED for name in report.archived_inputs)


def test_a_finding_carries_the_provenance_of_its_own_source(report):
    """One table checked against a regeneration and one against the archive, so that both
    provenances are exercised. tab:k20 moved from the second group to the first when the
    twenty-direction calibration was rerun; tab:ladder's matrix row has not been rerun."""
    regenerated = next(f for f in report.by_label("tab:k20").findings)
    assert regenerated.source.startswith("data/")
    assert regenerated.provenance == REGENERATED

    archived = next(f for f in report.by_label("tab:ladder").findings
                    if "stationary_validation" in f.source)
    assert archived.provenance == ARCHIVED


def test_the_rows_carry_what_the_caller_needs(report):
    """``check.tables`` writes these as a CSV and counts the mismatches in them."""
    rows = report.rows()
    assert rows and all({"table", "status", "row", "column"} <= set(row) for row in rows)
    assert {row["status"] for row in rows} <= {OK, ROUNDING, MISMATCH, "unreadable",
                                               "skipped"}
    disagreeing = [row for row in rows if row["status"] in (MISMATCH, ROUNDING)]
    assert len(disagreeing) == len(report.mismatches)
    assert sum(1 for row in rows if row["status"] == MISMATCH) == len(KNOWN)


def test_the_computed_value_is_reported_at_full_precision(report):
    """Rounding it here would hide exactly the disagreement being reported.

    The cell agrees now that the slope row has been corrected, but the report still has to
    carry the unrounded value: a reader checking a cell that passes needs to see what it
    was compared against, and the next edit to this row will be judged against it.
    """
    row = next(r for r in report.rows()
               if r["table"] == "tab:ceiling" and r["column"] == "max_E = 28"
               and r["row"] == "slope")
    assert row["printed"] == "0.29"
    assert row["computed"].startswith("0.2872")


def test_the_report_formats_and_names_the_disagreements(report):
    text = format_report(report)
    assert "table(s) checked" in text
    for finding in report.mismatches:
        assert finding.label in text


# -- the defects docs/errata.md already records --------------------------------


def test_errata_3_the_caption_of_tab_alts_now_states_the_asymmetry(report):
    """Closed in the article, not in the data: the two halves are still aggregated by
    different rules, and the caption now says so. The claim reads the caption, so a caption
    that stopped saying it would fail here."""
    claim = next(f for f in report.by_label("tab:alts").findings if f.kind == CLAIM)
    assert claim.status == OK
    assert "all twenty" in claim.computed and "withheld ranks" in claim.computed


def test_errata_5_the_a_sum_sq_budget_now_prints_the_46000_that_ran(report):
    """Corrected in the article. The run was stopped at 46,000 steps and the budget column
    said 100k; it now says what ran, so the cell agrees and no finding is raised."""
    assert not [f for f in bad(report, "tab:runs")
                if f.row == "a_sum_sq" and f.column == "budget"]
    row = next(r for r in report.rows()
               if r["table"] == "tab:runs" and r["row"] == "a_sum_sq"
               and r["column"] == "budget")
    assert row["status"] == OK and row["printed"] == "46k"


def test_errata_6_tab_ladder_now_prints_its_correlations_in_the_seed_order(report):
    """Corrected in the article: the pair was printed as 0.99 / 1.00 and the seeds give
    1.00 / 0.99."""
    assert not [f for f in bad(report, "tab:ladder")
                if f.row == "oscillating matrix" and f.kind != CLAIM]


def test_errata_9_every_constructed_row_of_tab_ladder_now_slides(report):
    """Closed by the regeneration. The archived files held one estimate per (rank, seed,
    observer) and no window index, so the caption's sliding-window recipe was false of every
    constructed row; the rerun records seven windows behind each."""
    claim = next(f for f in report.by_label("tab:ladder").findings
                 if f.row == "one window or several")
    assert claim.status == OK
    assert "every constructed row slides" in claim.computed
    for system in ("sys.linear", "sys.logistic", "sys.decoder", "sys.subspace"):
        assert system in claim.computed


def test_errata_11_the_sentence_now_names_the_four_that_were_sketched(report):
    """Closed in the article. The claim reads the run names out of the sentence below the
    table rather than assuming which rows it means, so either side changing is reported."""
    claim = next(f for f in report.by_label("tab:runs").findings if f.kind == CLAIM)
    assert claim.status == OK
    assert "a_add, x_no_grok, g_p1, g_p1x" in claim.computed
    assert "a_mul" not in claim.printed


def test_errata_12_is_named_as_unchecked_rather_than_passed(report):
    """The sketched campaigns' milestones were never promoted, so the size of the
    disagreement cannot be measured here. Silence would read as agreement."""
    notes = " ".join(report.by_label("tab:runs").notes)
    assert "one or two logged steps" in notes and "unchecked" in notes


def test_errata_24_the_rescaling_control_reads_zero(report):
    """Fixed at source before the port; the article's README is stale on it. The claim
    passes, which is what makes the check worth keeping."""
    claim = next(f for f in report.by_label("tab:controls").findings
                 if f.row == "constant rescaling")
    assert claim.status == OK


# -- what the auditor found that the errata does not list ----------------------


def test_the_edge_of_stability_campaign_is_whole(report):
    """Two of the sixteen runs were 200-step records a `--fast` pass left behind, which
    the full campaign then skipped: the run key is the rate and the seed, so both passes
    write the same name. The check reports the budget whether or not it holds, because a
    check that is silent when it passes cannot be told from one that was never made."""
    claim = next(f for f in report.by_label("tab:eos").findings
                 if f.row == "the campaign budget")
    assert claim.status == OK
    assert "30,000 steps" in claim.printed
    assert "all 16 runs are" in claim.computed


def test_the_ceiling_scans_now_agree_on_the_cell_they_share(report):
    """E_max = 20 at N = 8000 is one cell printed twice, once in each scan. It read 0.27
    against 0.24, and the file gives 0.2682; both now print the same value."""
    assert not bad(report, "tab:ceiling")


def test_the_theiler_error_row_is_the_mean_under_its_own_label(report):
    """The label decides which summary is compared. It reads "mean absolute error", the row
    is compared as a mean, and the median the other reading would give is reported beside
    it, so relabelling the row without recomputing it would fail here."""
    claim = next(f for f in report.by_label("tab:theiler").findings if f.kind == CLAIM)
    assert claim.status == OK
    assert "mean absolute error" in claim.printed
    assert "the median over the 7 ranks would be" in claim.computed


def test_the_ground_truth_cell_is_rounded_the_way_the_file_reads(report):
    """5.4649 was printed as 5.47 and now prints 5.46."""
    assert not bad(report, "tab:gt")


# -- the whole report ----------------------------------------------------------


#: Every disagreement the audit reports today, as (table, row, column). Written out rather
#: than counted, so that a new defect names itself in the failure and a defect that stops
#: being reported does too. Editing this set is the correct fix once the change has been
#: recorded somewhere a reader will find it.
#:
#: Three hundred and fifty-one cells were rewritten from the regenerated data and five
#: claims were closed, three of them in the article and two by re-running. What remains is
#: one, and it is not an editorial matter: the function-subspace system excites its three
#: highest ranks to about 0.86 r against the 0.9 r requirement 1 asks. That is the drive
#: regression of errata item 31, and re-running reproduces it exactly, so the row is
#: marked as failing requirement 1 until the construction is decided.
KNOWN = {
    ("tab:ladder", "image data, function subspace", "claim"),
}


def test_the_known_disagreements_and_no_others(report):
    seen = {(f.label, f.row, f.column) for f in report.mismatches}
    assert len(seen) == len(report.mismatches), "two findings share a cell"
    assert seen == KNOWN
    assert sum(1 for f in report.mismatches if f.status == ROUNDING) == 0


def test_the_command_line_exits_non_zero_while_a_cell_disagrees(capsys):
    assert tables.main([]) == 1
    assert "disagree" in capsys.readouterr().out


def test_an_empty_data_tree_skips_every_table_and_reports_no_agreement(tmp_path):
    """A missing input must not read as a pass. This is the failure mode the release check
    exists for: a checker that finds nothing to check and says nothing is wrong."""
    empty = audit(root=tmp_path)
    assert all(result.state == "skipped" for result in empty.results)
    assert empty.ok            # nothing disagreed, because nothing was compared
    assert all("is missing" in r.reason or r.reason for r in empty.results)


def test_the_data_reader_names_the_file_it_cannot_find(tmp_path):
    with pytest.raises(FileNotFoundError, match="actdim bootstrap"):
        Data(tmp_path).frame("nowhere/nothing.csv")
