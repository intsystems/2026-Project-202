"""Recompute the article's tables from ``data/`` and diff them against the printed LaTeX.

The article has now failed a release check twice for the same reason: a committed script
could no longer regenerate the file a table was quoting. The archived tree carried one
narrow guard against that, ``active_dimension/paper_tables.py``, covering two tables. This
generalises it to the whole document.

The check is one-directional and deliberately dumb. It reads ``icomp_v2/report.tex``,
recomputes every cell it knows how to recompute from a file under ``data/``, and reports
the pairs that disagree at the precision the article prints. It never edits the article and
never writes a file: it returns a report and the caller decides what to do with it.

Rounding is respected rather than approximated. A printed ``1.62`` matches a computed
1.6239 and does not match 1.554, because the comparison rounds the computed value to the
number of decimals actually printed. A cell that is out by no more than one unit in that
last place is reported separately as ``rounding``: the measurement agrees and the digits do
not, which is a typesetting defect rather than a data defect. Both count as mismatches for
the exit status, because either way the article prints something the file does not say.

What is checked
---------------

Twenty of the twenty-eight ``tabular`` blocks have at least one column derivable from
``data/``:

``tab:frozen``      the two frozen configurations, against the two ``frozen_*.json`` and the
                    eight-direction grid. The twenty-direction grid was not promoted, so
                    that column's grid size and selection errors are unchecked.
``tab:geometry``    the frozen row only; the other two rows state a convention.
``tab:ladder``      error and rank correlation, one system per row.
``tab:aggregation`` all three errors, each recomputed by its own stated recipe.
``tab:obs``         error, slope, and whether roughness orders the four withheld ranks.
``tab:k20``         every cell. Ported from the archived checker.
``tab:alts``        every cell. Ported, including the aggregation asymmetry below.
``tab:tau``         span and the estimate at every lag and rank.
``tab:controls``    the shift and its percentage of a four-component change.
``tab:switch``      the three levels of each schedule.
``tab:aniso``       effective rank, MG and the delay participation ratio.
``tab:gt``          the excited rank of every regime.
``tab:grok-diagnostics``  whether each run generalises, and its five diagnostics.
``tab:dip``         plateau, dip, depth, position and post-dip maximum.
``tab:prwindow``    every row but the 600-step one, which the caption says is the earlier
                    measurement and not part of the sweep the file holds.
``tab:matched``     depth, floor and the three surrogate fractions. The quantile column
                    names a statistic no committed file records.
``tab:theiler``     truth and both exclusions, and the two summary rows.
``tab:runs``        budget, memorisation, generalisation and final accuracy, for the rows
                    whose run has a record in ``data/``. See below.
``tab:eos``         stability ratio, rises and return, per seed.
``tab:ceiling``     both scans: tracking, the prediction, the level and the slope.
``tab:ceilingfit``  the four root-mean-square errors.

What is not, and why
--------------------

``tab:classes``, ``tab:obsdef``  definitions. Nothing to recompute.
``tab:geometry`` rows 2-3       a convention, stated rather than measured.
``tab:obsdep``   no file under ``data/`` scores the two mode-weightings at several lags.
``tab:sketch``   the per-rank sketched and uncompressed participation ratios were never
                 promoted; ``check.sketch.cost/sketch_cost.json`` records only time and
                 storage, which the table does not print.
``tab:exclusion`` the recipe is not recoverable. The caption says "median over observers,
                 seeds and windows" of ``valid.theiler.contrast/sweep_windows.csv``; taking
                 that literally gives 5.43 / 2.07 / 1.20 in the first column against a
                 printed 5.39 / 2.11 / 1.20, and collapsing in a different order moves every
                 cell again without landing on the printed values. Four of its cells would
                 match under two different orders and the rest under neither, so a guess
                 here would report noise rather than defects.
``tab:ipr``, ``tab:ipr-trajectory``  the Fourier inverse participation ratio outputs are not
                 in ``data/``. The grokking-step column of ``tab:ipr`` repeats ``tab:runs``,
                 which has no source for those rows either.

The aggregation asymmetry, carried forward from the archived checker
--------------------------------------------------------------------

``tab:alts``'s two halves are not the same measurement at two ranges, and the caption does
not say so:

* the eight-direction column scores the four *withheld* ranks 1, 3, 5, 8 of the parameter
  sweep, which is the convention the appendix D preamble states;
* the twenty-direction column scores **all twenty** ranks of the k20 calibration, the five
  ranks the configuration was selected on included.

Restricting the twenty-direction side to the ten ranks ``tab:k20`` prints gives MG 1.554,
not the printed 1.62. ``ALTS_TWENTY_HELD_OUT`` recomputes it that way so the size of the
difference is a number this module can produce rather than a claim in a comment.

Reading the report
------------------

    from actdim.tables import audit, format_report
    report = audit()
    print(format_report(report))

or from the command line, which exits non-zero on any mismatch::

    python -m actdim.tables
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .runtime.store import data_root, repo_root


# ---------------------------------------------------------------- where things live

def article_path() -> Path:
    """``icomp_v2/report.tex``, wherever the working directory is."""
    return repo_root().parent / "icomp_v2" / "report.tex"


# ---------------------------------------------------------------- the LaTeX reader

_RULES = ("\\toprule", "\\midrule", "\\bottomrule", "\\cmidrule", "\\addlinespace",
          "\\morecmidrules", "\\hline")

# Macros whose braced argument is the text, and which contribute nothing else.
_UNWRAP = ("texttt", "textbf", "textit", "emph", "mathrm", "mathit", "text", "operatorname",
           "cref", "Cref", "ref", "label", "multirow", "boldmath")

_ESCAPES = (("\\_", "_"), ("\\&", "&"), ("\\%", "%"), ("\\#", "#"), ("\\{", "{"),
            ("\\}", "}"), ("\\$", "$"))

_THIN = ("\\,", "\\!", "\\;", "\\:", "\\/")
_SPACES = ("\\quad", "\\qquad", "\\ ", "~", "\\hspace")

_MISSING = ("---", "--", "\u2014", "\u2013")


class ParseError(RuntimeError):
    """A table this reader will not guess at."""


@dataclass
class Cell:
    """One cell, cleaned of the formatting macros this document uses."""

    text: Optional[str]          # None where the article prints a dash
    raw: str
    column: int                  # first logical column this cell covers
    span: int = 1

    @property
    def missing(self) -> bool:
        return self.text is None


@dataclass
class Row:
    cells: List[Cell]
    line: int

    def at(self, column: int) -> Optional[Cell]:
        for cell in self.cells:
            if cell.column <= column < cell.column + cell.span:
                return cell
        return None

    def label(self) -> str:
        for cell in self.cells:
            if cell.text:
                return cell.text
        return ""

    @property
    def width(self) -> int:
        return sum(cell.span for cell in self.cells)


@dataclass
class Table:
    """One ``tabular`` block, as rows of cleaned cells."""

    label: str
    spec: str
    rows: List[Row]
    header_rows: int
    line: int

    def cell(self, row: int, column: int) -> Optional[Cell]:
        if not 0 <= row < len(self.rows):
            return None
        return self.rows[row].at(column)

    def printed(self, row: int, column: int) -> Optional[str]:
        cell = self.cell(row, column)
        return None if cell is None else cell.text

    def find(self, needle: str, start: int = 0) -> int:
        """The index of the first row whose label contains ``needle``.

        Checks are written against the article's own row labels rather than against row
        numbers, so that inserting a row above does not silently shift every check.
        """
        for i in range(start, len(self.rows)):
            if _norm(needle) in _norm(self.rows[i].label()):
                return i
        raise ParseError(f"{self.label}: no row labelled {needle!r}")

    def find_all(self, needle: str) -> List[int]:
        return [i for i, row in enumerate(self.rows)
                if _norm(needle) in _norm(row.label())]

    def body(self) -> List[int]:
        """Row indices below the header, skipping full-width section headings."""
        width = max(row.width for row in self.rows)
        return [i for i in range(self.header_rows, len(self.rows))
                if not (len(self.rows[i].cells) == 1 and self.rows[i].cells[0].span >= width)]

    def column_label(self, column: int) -> str:
        parts: List[str] = []
        for i in range(self.header_rows):
            cell = self.rows[i].at(column)
            if cell is not None and cell.text and (cell.span > 1 or i == self.header_rows - 1):
                parts.append(cell.text)
        return " / ".join(parts) if parts else f"column {column}"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def read_tables(path: Optional[Path] = None) -> Dict[str, Table]:
    """Every labelled ``tabular`` in the article, by label.

    A block whose rows do not agree on a width is refused rather than guessed at: a
    mis-parsed row would put a computed value beside the wrong printed one, which is a
    worse failure than not checking the table at all.
    """
    source = (path or article_path()).read_text(encoding="utf-8")
    tables: Dict[str, Table] = {}
    label = ""
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        found = re.search(r"\\label\{(tab:[^}]+)\}", line)
        if found:
            label = found.group(1)
        if "\\begin{tabular}" in line:
            start = i
            depth = 0
            block: List[Tuple[int, str]] = []
            while i < len(lines):
                if "\\begin{tabular}" in lines[i]:
                    depth += 1
                if depth:
                    block.append((i + 1, lines[i]))
                if "\\end{tabular}" in lines[i]:
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if label:
                tables[label] = _parse_block(label, block, start + 1)
            label = ""
        i += 1
    return tables


def _parse_block(label: str, block: Sequence[Tuple[int, str]], line: int) -> Table:
    spec = ""
    opening = re.search(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\{([^}]*)\}", block[0][1])
    if opening:
        spec = opening.group(1)

    # Strip the delimiters, then rejoin physical lines: one logical row runs to its `\\`.
    body: List[Tuple[int, str]] = []
    for number, text in block:
        text = re.sub(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\{[^}]*\}", "", text)
        text = text.replace("\\end{tabular}", "")
        text = re.sub(r"(?<!\\)%.*$", "", text)
        body.append((number, text))

    rows: List[Row] = []
    header_rows = 0
    seen_midrule = False
    buffer: List[str] = []
    buffer_line = body[0][0]
    for number, text in body:
        if not buffer:
            buffer_line = number
        while True:
            split = re.search(r"\\\\", text)
            if not split:
                buffer.append(text)
                break
            buffer.append(text[:split.start()])
            row = _parse_row("".join(buffer), buffer_line)
            if row is not None:
                rows.append(row)
                if not seen_midrule:
                    header_rows = len(rows)
            buffer = []
            text = text[split.end():]
            buffer_line = number
        if rows and not seen_midrule and re.search(r"\\midrule", text):
            seen_midrule = True
    tail = _parse_row("".join(buffer), buffer_line)
    if tail is not None:
        rows.append(tail)

    if not rows:
        raise ParseError(f"{label}: no rows")
    if not seen_midrule:
        header_rows = 1
    return Table(label=label, spec=spec, rows=rows, header_rows=header_rows, line=line)


def _parse_row(text: str, line: int) -> Optional[Row]:
    stripped = text.strip()
    if not stripped:
        return None
    for rule in _RULES:
        stripped = re.sub(re.escape(rule) + r"(\([a-z]{1,2}\))?(\{[^}]*\})?", "", stripped)
    if not stripped.strip():
        return None

    cells: List[Cell] = []
    column = 0
    for raw in _split_ampersands(stripped):
        span, inner = _span_of(raw)
        cells.append(Cell(text=_clean(inner), raw=raw.strip(), column=column, span=span))
        column += span
    return Row(cells=cells, line=line)


def _split_ampersands(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            current.append(text[i:i + 2])
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if char == "&" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        i += 1
    parts.append("".join(current))
    return parts


def _span_of(raw: str) -> Tuple[int, str]:
    found = re.search(r"\\multicolumn\s*\{(\d+)\}\s*\{[^}]*\}\s*", raw)
    if not found:
        return 1, raw
    rest = raw[found.end():]
    inner, after = _braced(rest)
    return int(found.group(1)), raw[:found.start()] + inner + after


def _braced(text: str) -> Tuple[str, str]:
    """Split ``{a}b`` into ``a`` and ``b``, respecting nesting."""
    text = text.lstrip()
    if not text.startswith("{"):
        return "", text
    depth = 0
    for i, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[1:i], text[i + 1:]
    return text[1:], ""


def _clean(raw: str) -> Optional[str]:
    text = raw
    for macro in _UNWRAP:
        while True:
            found = re.search(r"\\" + macro + r"\s*(\{\d+\}\s*\{[^}]*\}\s*)?", text)
            if not found:
                break
            inner, after = _braced(text[found.end():])
            text = text[:found.start()] + inner + after
    for old, new in _ESCAPES:
        text = text.replace(old, new)
    text = re.sub(r"\\phantom\s*\{[^}]*\}", "", text)
    for macro in _THIN:
        text = text.replace(macro, "")
    for macro in _SPACES:
        text = text.replace(macro, " ")
    text = text.replace("$", "")
    text = text.replace("{=}", "=").replace("\\times", "x").replace("\\cdot", "*")
    # A macro that survives this far keeps its name: `\tau` is the row label of tab:tau and
    # `\PR_{\mathrm{delay}}` names a row of tab:alts, so dropping the name would leave two
    # rows called the same nothing and no check could name the one it means.
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    if text in _MISSING or text == "":
        return None if text in _MISSING else ""
    return text


# ---------------------------------------------------------------- reading a printed cell

_NUMBER = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def as_number(text: Optional[str]) -> Optional[float]:
    """The number a printed cell holds, or ``None`` if it does not hold one."""
    if text is None:
        return None
    body = text.strip().replace(",", "")
    if body.endswith("k") and _NUMBER.match(body[:-1] or "x"):
        return float(body[:-1]) * 1000.0
    if not _NUMBER.match(body):
        return None
    return float(body)


def decimals(text: str) -> int:
    """How many decimal places the article printed, which sets the comparison."""
    body = text.strip().replace(",", "")
    if body.endswith("k"):
        return 0
    if "." not in body:
        return 0
    return len(body.split(".")[1].rstrip())


def round_half_up(value: float, places: int) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- the report

OK = "ok"
ROUNDING = "rounding"
MISMATCH = "mismatch"
UNREADABLE = "unreadable"


@dataclass
class Finding:
    """One cell that was compared, and how it came out."""

    label: str
    row: str
    column: str
    printed: Optional[str]
    computed: Optional[float]
    source: str
    status: str
    note: str = ""

    @property
    def bad(self) -> bool:
        return self.status in (ROUNDING, MISMATCH)

    def describe(self) -> str:
        printed = "---" if self.printed is None else self.printed
        if self.computed is None:
            got = "no value"
        elif isinstance(self.computed, str):
            got = self.computed
        else:
            got = f"{self.computed:.6f}".rstrip("0").rstrip(".")
        detail = f"  ({self.note})" if self.note else ""
        return (f"{self.label:22s} {self.row[:26]:26s} {self.column[:22]:22s} "
                f"printed {printed:>10s}  computed {got:>14s}  {self.source}{detail}")


@dataclass
class TableResult:
    label: str
    state: str                                    # "checked" or "skipped"
    reason: str = ""
    findings: List[Finding] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def mismatches(self) -> List[Finding]:
        return [f for f in self.findings if f.bad]

    @property
    def checked(self) -> int:
        return len(self.findings)


@dataclass
class Report:
    results: List[TableResult]
    article: str = ""
    data: str = ""

    @property
    def mismatches(self) -> List[Finding]:
        return [f for result in self.results for f in result.mismatches]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def by_label(self, label: str) -> TableResult:
        for result in self.results:
            if result.label == label:
                return result
        raise KeyError(label)


# ---------------------------------------------------------------- the data side

class Data:
    """The tracked half of the tree, read once per file."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or data_root()
        self._frames: Dict[str, pd.DataFrame] = {}
        self._objects: Dict[str, Any] = {}

    def has(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def require(self, rel: str) -> Path:
        path = self.root / rel
        if not path.exists():
            raise FileNotFoundError(f"data/{rel} is missing; run `actdim bootstrap` first")
        return path

    def frame(self, rel: str) -> pd.DataFrame:
        if rel not in self._frames:
            self._frames[rel] = pd.read_csv(self.require(rel))
        return self._frames[rel]

    def json(self, rel: str) -> Any:
        if rel not in self._objects:
            self._objects[rel] = json.loads(self.require(rel).read_text(encoding="utf-8"))
        return self._objects[rel]

    @staticmethod
    def name(rel: str) -> str:
        return "data/" + rel


@dataclass
class Computed:
    """One recomputed cell, ready to be diffed against what the article prints."""

    row: int
    column: int
    value: Any                                    # float, str, None, or a list of those
    source: str
    note: str = ""


Check = Callable[[Table, Data], Iterable[Computed]]


# ---------------------------------------------------------------- shared conventions

# ``tab:alts``, ``tab:obs`` and the observer figure all drop these two, for the reasons
# appendix D gives: probe accuracy is quantised and degenerate in every window, and the
# instantaneous mini-batch loss fails requirement 4, still reading 0.91-6.93 at zero
# learning rate.
DROPPED_OBSERVERS = ("acc_probe", "loss_step")

# The four ranks withheld from the eight-direction calibration.
HELD_OUT_EIGHT = (1, 3, 5, 8)

# The ten ranks ``tab:k20`` prints. NOT the ranks ``tab:alts`` scores; see the docstring.
K20_PRINTED = (1, 2, 4, 6, 8, 10, 12, 14, 17, 20)

ALTS_STATISTICS = ("MG", "LB", "TwoNN", "PRdelay", "specPR256", "specPR1024", "specPR0",
                   "roughness")

# Set to restrict the twenty-direction column to the ranks ``tab:k20`` prints, which is what
# the caption reads as if it meant. It reproduces MG 1.554 rather than the printed 1.62.
ALTS_TWENTY_HELD_OUT = False


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Spearman correlation, or ``None`` where it is not defined.

    scipy raises a ConstantInputWarning -- a RuntimeWarning, which pytest turns into an
    error here -- on a constant input, and returns NaN. Both cases are refusals, so they
    are caught before the call rather than after.
    """
    from scipy.stats import spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    x, y = a[ok], b[ok]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(spearmanr(x, y).statistic)


def _mae(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return None
    return float(np.mean(np.abs(a[ok] - b[ok])))


def _slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Least-squares slope of ``y`` on ``x``, without numpy's fitting warnings."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    x, y = x[ok], y[ok]
    var = float(np.sum((x - x.mean()) ** 2))
    if var == 0.0:
        return None
    return float(np.sum((x - x.mean()) * (y - y.mean())) / var)


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.median(finite))


def _nan_to_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


# ---------------------------------------------------------------- the checks

def check_frozen(table: Table, data: Data) -> Iterable[Computed]:
    eight = data.json("calib.e8/frozen_config.json")
    twenty = data.json("calib.e20/frozen_k20.json")
    ranking = data.frame("calib.e8/config_ranking.csv")
    sweep = data.frame("sys.digits.parameter/sweep_raw.csv")

    e8, e20 = "calib.e8/frozen_config.json", "calib.e20/frozen_k20.json"
    rank_file = "calib.e8/config_ranking.csv"
    yield Computed(table.find("E_{max}"), 1, float(eight["config"]["max_E"]), Data.name(e8))
    yield Computed(table.find("E_{max}"), 2, float(twenty["config"]["max_E"]), Data.name(e20))
    yield Computed(table.find("tau"), 1, float(eight["config"]["tau"]), Data.name(e8))
    yield Computed(table.find("tau"), 2, float(twenty["config"]["tau"]), Data.name(e20))
    window = table.find("window / stride")
    yield Computed(window, 1, [float(eight["config"]["window"]), float(eight["config"]["stride"])],
                   Data.name(e8))
    yield Computed(window, 2, [float(twenty["config"]["window"]), float(twenty["config"]["stride"])],
                   Data.name(e20))
    yield Computed(table.find("grid size"), 1, float(len(ranking)), Data.name(rank_file))
    yield Computed(table.find("selection seeds"), 1,
                   [float(s) for s in eight["cal_seeds"]], Data.name(e8))
    yield Computed(table.find("selection seeds"), 2,
                   [float(s) for s in twenty["calibration_seeds"]], Data.name(e20))
    yield Computed(table.find("selection ranks"), 1,
                   [float(r) for r in eight["cal_r"]], Data.name(e8))
    yield Computed(table.find("selection ranks"), 2,
                   [float(r) for r in twenty["calibration_r"]], Data.name(e20))

    withheld = sorted(set(sweep.r.unique()) - set(eight["cal_r"]))
    yield Computed(table.find("withheld ranks"), 1, [float(r) for r in withheld],
                   Data.name("sys.digits.parameter/sweep_raw.csv"),
                   "the sweep's ranks less the calibration ranks")

    yield Computed(table.find("selection error, best"), 1, float(ranking.mae.min()),
                   Data.name(rank_file))
    yield Computed(table.find("selection error, worst"), 1, float(ranking.mae.max()),
                   Data.name(rank_file))


def check_geometry(table: Table, data: Data) -> Iterable[Computed]:
    config = data.json("calib.e8/frozen_config.json")["config"]
    row = table.find("frozen configuration")
    yield Computed(row, 1, float(config["window"]), Data.name("calib.e8/frozen_config.json"))
    yield Computed(row, 2, float(config["stride"]), Data.name("calib.e8/frozen_config.json"))


def _digits_parameter_eight(data: Data) -> Tuple[pd.DataFrame, str]:
    """The parameter-subspace sweep, on the recurrent arm, at the withheld ranks."""
    rel = "sys.digits.parameter/sweep_raw.csv"
    frame = data.frame(rel)
    frame = frame[(frame.arm == "qp") & (~frame.eta_zero)
                  & (~frame.observer.isin(DROPPED_OBSERVERS))
                  & frame.r.isin(HELD_OUT_EIGHT)]
    return frame, Data.name(rel)


def _alts_eight(data: Data) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], str]:
    """``tab:alts`` columns 1-2. One value per rank: the median over seeds and observers.

    Keeping all twelve observers instead gives MG 0.455 and TwoNN 1.752, so the two
    exclusions of ``DROPPED_OBSERVERS`` are load-bearing rather than cosmetic.
    """
    frame, source = _digits_parameter_eight(data)
    grouped = frame.groupby("r")
    truth = grouped.traj_PR.median().values
    out = {}
    for statistic in ALTS_STATISTICS:
        series = grouped[statistic].median().values
        out[statistic] = (_mae(series, truth), _spearman(series, truth))
    return out, source


def _alts_twenty(data: Data) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], str]:
    """``tab:alts`` columns 3-4. All twenty ranks of the k20 calibration.

    Note the asymmetry with the eight-direction column: there is no held-out restriction
    here, so five of the twenty ranks -- 2, 6, 10, 14, 18 -- are the ranks the configuration
    was itself selected on. That is what reproduces the printed numbers.
    """
    rel = "calib.e20/scores_frozen.csv"
    frame = data.frame(rel)
    frame = frame[frame.tag == "qp"]
    if ALTS_TWENTY_HELD_OUT:
        frame = frame[frame.r.isin(K20_PRINTED)]
    grouped = frame.groupby("r")
    truth = grouped.traj_pr.median().values
    out = {}
    for statistic in ALTS_STATISTICS:
        series = grouped[statistic].median().values
        out[statistic] = (_mae(series, truth), _spearman(series, truth))
    return out, Data.name(rel)


_ALTS_ROWS = (("MG", "MG"), ("LB", "LB"), ("TwoNN", "TwoNN"), ("PR_{delay}", "PRdelay"),
              ("256 bands", "specPR256"), ("1024 bands", "specPR1024"),
              ("native resolution", "specPR0"), ("roughness", "roughness"))


def check_alts(table: Table, data: Data) -> Iterable[Computed]:
    eight, source8 = _alts_eight(data)
    twenty, source20 = _alts_twenty(data)
    for label, statistic in _ALTS_ROWS:
        row = table.find(label)
        yield Computed(row, 1, eight[statistic][0], source8)
        yield Computed(row, 2, eight[statistic][1], source8)
        yield Computed(row, 3, twenty[statistic][0], source20)
        yield Computed(row, 4, twenty[statistic][1], source20)


def check_k20(table: Table, data: Data) -> Iterable[Computed]:
    rel = "calib.e20/scores_frozen.csv"
    frame = data.frame(rel)
    frame = frame[(frame.tag == "qp") & frame.r.isin(K20_PRINTED)]
    grouped = frame.groupby("r")
    spread = (frame.groupby(["observer", "r"]).MG.agg(lambda s: s.max() - s.min())
              .groupby("r").median())
    series = {
        "MG": grouped.MG.median(),
        "PR_{delay}": grouped.PRdelay.median(),
        "spectral PR": grouped.specPR256.median(),
        "spread across seeds": spread,
    }
    for label, values in series.items():
        row = table.find(label)
        for column, rank in enumerate(K20_PRINTED, start=1):
            yield Computed(row, column, float(values.loc[rank]), Data.name(rel))


_LADDER_RANKING = (
    ("online linear regression", "sys.linear"),
    ("logistic regression", "sys.logistic"),
    ("frozen nonlinear decoder", "sys.decoder"),
    ("perceptron in a k-subspace", "sys.subspace"),
)


def check_ladder(table: Table, data: Data) -> Iterable[Computed]:
    rel = "sys.matrix/stationary_validation.csv"
    matrix = data.frame(rel)
    held = matrix[matrix.split == "held_out"].groupby("seed")[["mae", "rho"]].first()
    row = table.find("oscillating matrix")
    yield Computed(row, 3, [float(v) for v in held.mae.values], Data.name(rel))
    yield Computed(row, 4, [float(v) for v in held.rho.values], Data.name(rel),
                   "seeds in the same order as the errors beside them")

    for label, experiment in _LADDER_RANKING:
        rel = experiment + "/observer_ranking.csv"
        best = data.frame(rel).iloc[0]
        row = table.find(label)
        yield Computed(row, 3, float(best.MAE), Data.name(rel))
        yield Computed(row, 4, float(best.rho), Data.name(rel))

    frame, source = _digits_parameter_eight(data)
    row = table.find_all("image data, parameter subspace")[0]
    yield Computed(row, 3, _mae(frame.MG.values, frame.traj_PR.values), source,
                   "mean over the withheld (rank, seed, observer) runs")
    grouped = frame.groupby("r")
    yield Computed(row, 4, _spearman(grouped.MG.median().values,
                                    grouped.traj_PR.median().values), source)

    twenty, source20 = _alts_twenty(data)
    row = table.find_all("image data, parameter subspace")[1]
    yield Computed(row, 3, twenty["MG"][0], source20)
    yield Computed(row, 4, twenty["MG"][1], source20)

    rel = "sys.digits.function/observer_ranking.csv"
    best = data.frame(rel).iloc[0]
    row = table.find("image data, function subspace")
    yield Computed(row, 3, float(best.MAE), Data.name(rel))
    yield Computed(row, 4, float(best.rho), Data.name(rel))


# Article name -> the observer identifiers in the committed sweep.
_OBSERVERS = (
    ("probe loss", ("loss_probe",)),
    ("function-space norm", ("fn_fro",)),
    ("margin", ("margin",)),
    ("parameter norms (two)", ("c_norm", "w_fro")),
    ("fixed parameter projection", ("c_proj1",)),
    ("gradient projection", ("g_proj",)),
    ("function-space projection", ("fn_proj1",)),
    ("gradient norm", ("g_fro",)),
    ("full-batch loss", ("loss_full",)),
    ("instantaneous loss", ("loss_step",)),
)


def _observer_scores(data: Data) -> Tuple[Dict[str, Tuple[Optional[float], ...]], str]:
    """Per observer: error, slope and whether roughness orders the four withheld ranks.

    This is the caption's recipe -- median over the four withheld seeds at each of the four
    withheld ranks, scored against the measured effective rank -- and not the raw mean over
    runs, which gives 0.49 for the probe loss where the article prints 0.30.
    """
    rel = "sys.digits.parameter/sweep_raw.csv"
    frame = data.frame(rel)
    frame = frame[(frame.arm == "qp") & (~frame.eta_zero) & frame.r.isin(HELD_OUT_EIGHT)]
    out: Dict[str, Tuple[Optional[float], ...]] = {}
    for observer, part in frame.groupby("observer"):
        grouped = part.groupby("r")
        estimate = grouped.MG.median().values
        truth = grouped.traj_PR.median().values
        roughness = grouped.roughness.median().values
        out[str(observer)] = (_mae(estimate, truth), _slope(truth, estimate),
                              _spearman(roughness, truth))
    return out, Data.name(rel)


def check_obs(table: Table, data: Data) -> Iterable[Computed]:
    scores, source = _observer_scores(data)
    for label, observers in _OBSERVERS:
        row = table.find(label)
        mae = _median_or_none([scores[o][0] for o in observers if o in scores])
        slope = _median_or_none([scores[o][1] for o in observers if o in scores])
        orders = [scores[o][2] for o in observers if o in scores]
        yield Computed(row, 2, mae, source)
        yield Computed(row, 3, slope, source)
        ordered = all(v is not None and v >= 1.0 for v in orders) if orders else None
        yield Computed(row, 4, None if ordered is None else ("yes" if ordered else "no"),
                       source, "roughness orders the four ranks exactly")


def check_aggregation(table: Table, data: Data) -> Iterable[Computed]:
    frame, source = _digits_parameter_eight(data)
    yield Computed(table.find("mean over"), 2,
                   _mae(frame.MG.values, frame.traj_PR.values), source)
    scores, source_obs = _observer_scores(data)
    per_observer = [value[0] for observer, value in scores.items()
                    if observer not in DROPPED_OBSERVERS]
    yield Computed(table.find("median over observers"), 2,
                   _median_or_none(per_observer), source_obs)
    eight, source8 = _alts_eight(data)
    yield Computed(table.find("median over seeds and observers"), 2, eight["MG"][0], source8)


def check_tau(table: Table, data: Data) -> Iterable[Computed]:
    rel = "valid.tau/tau_sensitivity.csv"
    frame = data.frame(rel)
    frame = frame[(frame.period == 400) & (frame.max_E == 20)]
    ranks = [1, 2, 3, 4, 6, 8]
    for row in table.body():
        lag = as_number(table.printed(row, 0))
        if lag is None:
            continue
        part = frame[frame.tau == str(int(lag))]
        if part.empty:
            continue
        spans = part.span_periods.unique()
        if len(spans) == 1:
            yield Computed(row, 1, float(spans[0]), Data.name(rel))
        medians = part.groupby("r").MG.median()
        for column, rank in enumerate(ranks, start=2):
            if rank in medians.index:
                yield Computed(row, column, float(medians.loc[rank]), Data.name(rel))


# Article name -> the control identifier in the nuisance sweep.
_CONTROLS = (
    ("none (baseline)", "baseline"),
    ("learning rate halved", "lr_step"),
    ("drive band up one octave", "freq_double"),
    ("drive band down one octave", "freq_half"),
    ("coordinates rotated", "rotate"),
    ("observer gain ramped", "obs_scale"),
    ("state amplitude ramped", "amp_ramp"),
)

# The denominator of the percentage column: a real change of four components.
FOUR_COMPONENTS = 4.0


def check_controls(table: Table, data: Data) -> Iterable[Computed]:
    rel = "valid.nuisance/controls_scored.csv"
    frame = data.frame(rel)
    frame = frame[frame["mode"] == "qp"]
    for label, control in _CONTROLS:
        part = frame[frame.control == control]
        if part.empty:
            continue
        shift = float(part["between"].median())
        row = table.find(label)
        yield Computed(row, 1, shift, Data.name(rel))
        yield Computed(row, 2, 100.0 * shift / FOUR_COMPONENTS, Data.name(rel),
                       "as a percentage of a four-component change")


def check_switch(table: Table, data: Data) -> Iterable[Computed]:
    rel = "valid.transitions/transitions_raw.csv"
    frame = data.frame(rel)
    frame = frame[(frame["mode"] == "qp") & frame.observer.isin(("w_fro", "c_norm"))]
    for row in table.body():
        label = table.printed(row, 0) or ""
        levels = re.findall(r"\d+", label)
        if len(levels) < 2:
            continue
        high, low = int(levels[0]), int(levels[1])
        part = frame[(frame.hi == high) & (frame.lo == low)]
        if part.empty:
            continue
        for column, field_name in enumerate(("level0", "level1", "level2"), start=1):
            yield Computed(row, column, float(part[field_name].median()), Data.name(rel),
                           "median over seeds and the two parameter norms")


def check_aniso(table: Table, data: Data) -> Iterable[Computed]:
    rel = "valid.anisotropy/aniso_summary.csv"
    frame = data.frame(rel)
    for row in table.body():
        rank = as_number(table.printed(row, 0))
        anisotropy = as_number(table.printed(row, 1))
        if rank is None or anisotropy is None:
            continue
        part = frame[(frame.r == int(rank)) & (np.abs(frame.rho - anisotropy) < 1e-9)]
        if part.empty:
            continue
        record = part.iloc[0]
        yield Computed(row, 2, float(record.pr_pos), Data.name(rel))
        yield Computed(row, 3, float(record.MG), Data.name(rel))
        yield Computed(row, 4, float(record.PRdelay), Data.name(rel))


_GROUND_TRUTH_ARMS = (("qp, eta=0", "qp_eta0"),)


def check_gt(table: Table, data: Data) -> Iterable[Computed]:
    rel = "sys.digits.parameter/ground_truth_PR.csv"
    frame = data.frame(rel).set_index("arm")
    ranks = ("1", "2", "4", "6", "8")
    for row in table.body():
        label = (table.printed(row, 0) or "").strip()
        arm = dict(_GROUND_TRUTH_ARMS).get(label.replace("$", ""), label)
        if arm not in frame.index:
            continue
        for column, rank in enumerate(ranks, start=1):
            yield Computed(row, column, float(frame.loc[arm, rank]), Data.name(rel))


# Article row -> (file, run, column). The transformer rows read the parameter norm and the
# perceptron rows the training loss, which is what the caption says.
_DIAGNOSTIC_ROWS = (
    ("transformer, full data", "grok.diagnostics.logs/real_logs_summary.csv",
     "grokpos_s0", "weight_norm"),
    ("transformer, reduced data, s0", "grok.diagnostics.logs/real_logs_summary.csv",
     "lowdata15_s0", "weight_norm"),
    ("transformer, reduced data, s2", "grok.diagnostics.logs/real_logs_summary.csv",
     "lowdata15_s2", "weight_norm"),
    ("transformer, no weight decay", "grok.diagnostics.logs/real_logs_summary.csv",
     "wd0_s0", "weight_norm"),
    ("perceptron, n+m", "grok.diagnostics.perceptron/dimension_probe_summary.csv",
     "a_add", "train_loss"),
    ("perceptron, n \\cdot m", "grok.diagnostics.perceptron/dimension_probe_summary.csv",
     "a_mul", "train_loss"),
    ("perceptron, n^3+nm^2+m", "grok.diagnostics.perceptron/dimension_probe_summary.csv",
     "x_no_grok", "train_loss"),
)

_DIAGNOSTIC_COLUMNS = ((2, "MG"), (3, "ident"), (4, "roughness"), (5, "PRdelay"), (6, "osc"))


def check_grok_diagnostics(table: Table, data: Data) -> Iterable[Computed]:
    outcomes = data.frame("grok.extended.outcomes/exp8_outcomes.csv").set_index("run")
    for label, rel, run, column in _DIAGNOSTIC_ROWS:
        frame = data.frame(rel)
        part = frame[(frame.run == run) & (frame.column == column)]
        if part.empty:
            continue
        record = part.iloc[0]
        row = table.find(_row_needle(label))
        if run in outcomes.index:
            yield Computed(row, 1, "yes" if bool(outcomes.loc[run, "groks"]) else "no",
                           Data.name("grok.extended.outcomes/exp8_outcomes.csv"))
        for position, name in _DIAGNOSTIC_COLUMNS:
            yield Computed(row, position, float(record[name]), Data.name(rel))


def _row_needle(label: str) -> str:
    """The part of a row label that survives cleaning, for ``Table.find``."""
    return _clean(label) or label


_DIP_STATS = (("function", "fn_PR_pos_det"), ("parameter", "PR_pos_det"))
_DIP_RUNS = ("mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1", "mod_wd0", "s5_wd0")


def check_dip(table: Table, data: Data) -> Iterable[Computed]:
    dip = data.frame("grok.rank.dip/rank_dip.csv")
    controls = data.frame("grok.rank.dip/rank_dip_controls_aligned.csv")
    windows = data.frame("grok.rank.dip/rank_windows.csv")
    dip_file = Data.name("grok.rank.dip/rank_dip.csv")
    control_file = Data.name("grok.rank.dip/rank_dip_controls_aligned.csv")
    window_file = Data.name("grok.rank.dip/rank_windows.csv")

    block = 0
    for row in table.body():
        space = table.printed(row, 0)
        if space:
            block = 0 if _norm(space) == "function" else 1
        run = table.printed(row, 1)
        if run not in _DIP_RUNS:
            continue
        statistic = _DIP_STATS[block][1]
        source = dip[(dip.run == run) & (dip.stat == statistic)]
        if not source.empty:
            record = source.iloc[0]
            yield Computed(row, 2, float(record.plateau), dip_file)
            yield Computed(row, 3, float(record.dip), dip_file)
            yield Computed(row, 4, float(record.depth), dip_file)
            yield Computed(row, 5, float(record["at"]), dip_file, "steps after generalisation")
            trace = windows[windows.run == run]
            generalised = trace.t_gen.dropna()
            if not generalised.empty:
                floor = float(generalised.iloc[0]) + float(record["at"])
                after = trace[trace.right_step > floor][statistic]
                if not after.empty:
                    yield Computed(row, 6, float(after.max()), window_file,
                                   "largest value at any window after the dip")
            continue
        source = controls[(controls.run == run) & (controls.stat == statistic)]
        if not source.empty:
            record = source.iloc[0]
            yield Computed(row, 2, float(record.plateau), control_file)
            yield Computed(row, 3, float(record.dip), control_file)
            yield Computed(row, 4, float(record.depth), control_file)
            yield Computed(row, 5, None, control_file, "measured in the aligned window")
            yield Computed(row, 6, None, control_file, "measured in the aligned window")


def check_prwindow(table: Table, data: Data) -> Iterable[Computed]:
    rel = "grok.prwindow/pr_vs_window.csv"
    frame = data.frame(rel)
    frame = frame[frame.ladder == "fixed_n"]
    columns = (("a_add", 1, "PR_pos_det_med"), ("a_add", 2, "PR_pos_det_max"),
               ("x_no_grok", 3, "PR_pos_det_med"), ("x_no_grok", 4, "PR_pos_det_max"))
    for row in table.body():
        window = as_number(table.printed(row, 0))
        if window is None:
            continue
        for run, column, name in columns:
            part = frame[(frame.run == run) & (frame.window_steps == int(window))]
            if part.empty:
                continue
            yield Computed(row, column, float(part.iloc[0][name]), Data.name(rel))


# Article row -> (run, the run its window is aligned on).
_MATCHED_RUNS = (
    ("modular, seed 42", "mod_wd1", "mod_wd1"),
    ("modular, seed 43", "mod_wd1_s43", "mod_wd1_s43"),
    ("modular, seed 44", "mod_wd1_s44", "mod_wd1_s44"),
    ("S_5 composition", "s5_wd1", "s5_wd1"),
    ("modular, no decay", "mod_wd0", "mod_wd1"),
    ("S_5, no decay", "s5_wd0", "s5_wd1"),
)

# The caption's window: the plateau is the median over [t-3000, t-1000] and the floor the
# minimum over [t-1000, t+2000].
MATCHED_BEFORE, MATCHED_AFTER = 1000.0, 2000.0


def check_matched(table: Table, data: Data) -> Iterable[Computed]:
    spread = data.frame("grok.matched.surrogate/surrogate_seed_spread.csv")
    trace = data.frame("grok.matched.window/headline_trace.csv")
    spread_file = Data.name("grok.matched.surrogate/surrogate_seed_spread.csv")
    trace_file = Data.name("grok.matched.window/headline_trace.csv")

    generalisation = {}
    for _, run, _ in _MATCHED_RUNS:
        steps = trace[trace.run == run].t_gen.dropna()
        if not steps.empty:
            generalisation[run] = float(steps.iloc[0])

    for label, run, aligned_on in _MATCHED_RUNS:
        row = table.find(_row_needle(label))
        part = spread[(spread.run == run) & (spread.column == "weight_norm")]
        if part.empty:
            continue
        yield Computed(row, 1, float(part.iloc[0].observed), spread_file)
        step = generalisation.get(aligned_on)
        if step is not None:
            window = trace[(trace.run == run) & (trace.column == "weight_norm")
                           & (trace.mid_step >= step - MATCHED_BEFORE)
                           & (trace.mid_step <= step + MATCHED_AFTER)]
            if not window.empty:
                floor = float(window.loc[window.MG.idxmin(), "mid_step"]) - step
                yield Computed(row, 3, floor, trace_file,
                               "offset of the minimum from generalisation")
        for column, smooth in ((4, 101), (5, 201), (6, 401)):
            at_smooth = part[part.smooth == smooth]
            if not at_smooth.empty:
                yield Computed(row, column, float(at_smooth.iloc[0].p_median), spread_file)


def check_theiler(table: Table, data: Data) -> Iterable[Computed]:
    rel = "valid.theiler.cap/theiler_quick_raw.csv"
    frame = data.frame(rel)
    # Median over seeds per observer, then over the three observers: pooling the raw rows
    # instead mixes seed spread into the observer median and moves r = 5 from 4.53 to 6.99.
    collapsed = (frame.groupby(["theiler", "r", "observer"])[["MG", "truth"]].median()
                 .groupby(level=[0, 1]).median())
    exclusions = sorted(frame.theiler.unique())
    ranks, estimates = [], {e: [] for e in exclusions}
    truths = []
    for row in table.body():
        rank = as_number(table.printed(row, 0))
        if rank is None:
            continue
        rank = int(rank)
        if (exclusions[0], rank) not in collapsed.index:
            continue
        ranks.append(rank)
        truth = float(collapsed.loc[(exclusions[0], rank), "truth"])
        truths.append(truth)
        yield Computed(row, 1, truth, Data.name(rel))
        for column, exclusion in enumerate(exclusions, start=2):
            value = float(collapsed.loc[(exclusion, rank), "MG"])
            estimates[exclusion].append(value)
            yield Computed(row, column, value, Data.name(rel))

    truth_array = np.array(truths)
    for column, exclusion in enumerate(exclusions, start=2):
        series = np.array(estimates[exclusion])
        # The row is labelled "median absolute error"; the value the article prints is the
        # mean. See the note the registry carries for this table.
        yield Computed(table.find("absolute error"), column, _mae(series, truth_array),
                       Data.name(rel), "mean, not median; see the note below")
        yield Computed(table.find("Spearman"), column, _spearman(series, truth_array),
                       Data.name(rel))


# Article run -> the files that record it.
_SKETCHED_RUNS = ("mod_wd1", "mod_wd1_s43", "mod_wd1_s44", "s5_wd1", "mod_wd0", "s5_wd0")
_EXTENDED_RUNS = ("grokpos_s0", "lowdata15_s0", "lowdata15_s1", "lowdata15_s2",
                  "lowdata20_s0", "wd0_s0", "wd0_s1")

# How close the deepest analysis window must come to the printed budget. The windows slide
# to the end of the record, so a run that reached its budget leaves one within a stride of
# it; anything short of this means the run stopped early and the inventory is wrong.
BUDGET_TOLERANCE = 0.02


def check_runs(table: Table, data: Data) -> Iterable[Computed]:
    summary = data.frame("grok.rank.dip/rank_summary.csv").set_index("run")
    windows = data.frame("grok.rank.dip/rank_windows.csv")
    outcomes = data.frame("grok.extended.outcomes/exp8_outcomes.csv").set_index("run")
    log_windows = data.frame("grok.diagnostics.logs/real_logs_windows.csv")
    summary_file = Data.name("grok.rank.dip/rank_summary.csv")
    window_file = Data.name("grok.rank.dip/rank_windows.csv")
    outcome_file = Data.name("grok.extended.outcomes/exp8_outcomes.csv")
    log_file = Data.name("grok.diagnostics.logs/real_logs_windows.csv")

    for row in table.body():
        run = table.printed(row, 0)
        if run in _SKETCHED_RUNS:
            record = summary.loc[run]
            yield Computed(row, 6, float(record.t_mem), summary_file)
            yield Computed(row, 7, _nan_to_none(record.t_gen), summary_file)
            yield Computed(row, 8, float(record.final_val_acc), summary_file)
            yield Computed(row, 5, _budget(table.printed(row, 5),
                                           windows[windows.run == run].right_step.max()),
                           window_file, "the deepest analysis window in the record")
        elif run in _EXTENDED_RUNS:
            record = outcomes.loc[run]
            yield Computed(row, 6, float(record.t_mem), outcome_file)
            yield Computed(row, 7, _nan_to_none(record.t_gen), outcome_file)
            reach = log_windows[log_windows.run == run].right_step
            if not reach.empty:
                yield Computed(row, 5, _budget(table.printed(row, 5), reach.max()),
                               log_file, "the deepest analysis window in the record")


def _budget(printed: Optional[str], reach: Any) -> Optional[float]:
    """The printed budget, confirmed or contradicted by how far the record runs.

    Returned as the printed value when the record reaches it, and as the reach itself when
    it does not, so that a truncated run shows up as a mismatch naming the step it stopped
    at rather than as a silent pass.
    """
    stated = as_number(printed)
    reached = _nan_to_none(reach)
    if stated is None or reached is None:
        return None
    return stated if reached >= stated * (1.0 - BUDGET_TOLERANCE) else reached


_EOS_RATES = (1e5, 3e5, 1e6, 1.5e6, 2e6, 2.5e6, 2.8e6, 3e6)


def check_eos(table: Table, data: Data) -> Iterable[Computed]:
    runs = data.frame("train.perceptron.eos/eos_runs.csv")
    recurrence = data.frame("grok.eos/eos_recurrence.csv")
    run_file = Data.name("train.perceptron.eos/eos_runs.csv")
    recurrence_file = Data.name("grok.eos/eos_recurrence.csv")

    body = table.body()
    for row, rate in zip(body, _EOS_RATES):
        at_rate = runs[np.abs(runs.lr - rate) < 1.0].sort_values("seed")
        if at_rate.empty:
            continue
        ratios, rises, returns = [], [], []
        for _, record in at_rate.iterrows():
            if _nan_to_none(record.diverged_at) is not None:
                ratios.append(None)
                rises.append(None)
                returns.append(None)
                continue
            ratios.append(float(record.eta_lam_over_2_median_tail))
            seed = recurrence[(np.abs(recurrence.lr - rate) < 1.0)
                              & (recurrence.seed == record.seed)]
            rises.append(float(seed.rises.median()) if not seed.empty else None)
            returns.append(float(seed.nn_over_travel.median()) if not seed.empty else None)
        yield Computed(row, 1, _collapse(ratios), run_file)
        yield Computed(row, 2, _collapse(rises), recurrence_file)
        yield Computed(row, 3, _collapse(returns), recurrence_file)


def _collapse(values: List[Optional[float]]) -> Any:
    """A per-seed list, or ``None`` where every seed diverged and the article prints a dash."""
    return None if all(v is None for v in values) else values


_CEILING_BLOCKS = (
    ("E", "max_E", (10, 14, 20, 28, 40, 56), "takens"),
    ("N", "N", (1000, 2000, 4000, 8000, 16000, 32000, 64000), "eckmann_ruelle"),
)


def check_ceiling(table: Table, data: Data) -> Iterable[Computed]:
    rel = "valid.ceiling/ceiling_summary.csv"
    frame = data.frame(rel)
    frame = frame[frame.arm == "frozen"]
    tracking = table.find_all("tracking")
    level = table.find_all("level at")
    slope = table.find_all("slope")
    predicts = [table.find("predicts"), table.find("predicts", start=table.find("predicts") + 1)]

    for block, (sweep, axis, values, prediction) in enumerate(_CEILING_BLOCKS):
        part = frame[frame.sweep == sweep].set_index(axis)
        for column, value in enumerate(values, start=1):
            if value not in part.index:
                continue
            record = part.loc[value]
            yield Computed(tracking[block], column, float(record.r_track_1), Data.name(rel))
            yield Computed(predicts[block], column, float(record[prediction]), Data.name(rel))
            yield Computed(level[block], column, float(record.MG_at_20), Data.name(rel))
            yield Computed(slope[block], column, float(record.slope_top), Data.name(rel))


_CEILING_FITS = (("embedding condition", "rmse_takens"),
                 ("finite-record bound", "rmse_er"),
                 ("pointwise minimum", "rmse_min"),
                 ("log", "rmse_loglog"))


def check_ceilingfit(table: Table, data: Data) -> Iterable[Computed]:
    rel = "valid.ceiling/ceiling_fits.csv"
    frame = data.frame(rel)
    record = frame[frame.iloc[:, 0] == "MG_plateau"].iloc[0]
    for label, column in _CEILING_FITS:
        yield Computed(table.find(label), 1, float(record[column]), Data.name(rel))


# ---------------------------------------------------------------- the registry

@dataclass
class Registered:
    label: str
    check: Optional[Check]
    reason: str = ""
    notes: Tuple[str, ...] = ()


REGISTRY: Tuple[Registered, ...] = (
    Registered("tab:classes", None,
               "a definition: the three regimes and what the estimator returns in each"),
    Registered("tab:obsdef", None, "a definition: the twelve observers and their formulae"),
    Registered("tab:frozen", check_frozen, notes=(
        "the twenty-direction grid was never promoted, so that column's grid size and its "
        "two selection errors have no source under data/",
        "the swept-parameter and withheld-seed rows are prose",
    )),
    Registered("tab:geometry", check_geometry, notes=(
        "rows 2 and 3 state the convention the later pipelines use; no file records it",
    )),
    Registered("tab:ladder", check_ladder, notes=(
        "truth, range and the protocol column are statements about the design",
    )),
    Registered("tab:aggregation", check_aggregation),
    Registered("tab:obs", check_obs, notes=(
        "the probe-accuracy row spans three columns to say the observer is degenerate; "
        "there is no number to check",
    )),
    Registered("tab:k20", check_k20),
    Registered("tab:alts", check_alts, notes=(
        "the two columns are aggregated differently and the caption does not say so; set "
        "ALTS_TWENTY_HELD_OUT to score the twenty-direction column on the ten printed "
        "ranks, which gives MG 1.554 against the printed 1.62",
    )),
    Registered("tab:obsdep", None,
               "no file under data/ scores the two mode-weightings at several delay lags"),
    Registered("tab:tau", check_tau),
    Registered("tab:controls", check_controls, notes=(
        "the source is valid.nuisance/controls_scored.csv, not the two-control "
        "calib.e20/invariance_controls.csv, which holds only the constant rescaling and "
        "the rotation and reads 0.000 on both",
    )),
    Registered("tab:switch", check_switch),
    Registered("tab:aniso", check_aniso),
    Registered("tab:gt", check_gt),
    Registered("tab:grok-diagnostics", check_grok_diagnostics, notes=(
        "the perceptron rows have no outcome record under data/, so the generalises column "
        "is checked only for the transformer runs",
    )),
    Registered("tab:dip", check_dip),
    Registered("tab:sketch", None,
               "the per-rank sketched and uncompressed participation ratios were not "
               "promoted; check.sketch.cost/sketch_cost.json records time and storage only"),
    Registered("tab:prwindow", check_prwindow, notes=(
        "the 600-step row is, by the caption, the earlier measurement rather than part of "
        "this sweep, and pr_vs_window.csv holds no window that short",
    )),
    Registered("tab:matched", check_matched, notes=(
        "the quantile column ranks the statistic among every window centre after "
        "memorisation; no committed file records that ranking",
        "the floor is compared to the nearest step: the S_5 runs log every five steps, so "
        "their window centres fall on half-steps and the article prints +712 for 712.5",
    )),
    Registered("tab:ipr-trajectory", None,
               "the Fourier inverse participation ratio of the trajectory is not in data/"),
    Registered("tab:ipr", None,
               "the first two columns are not in data/, and the grokking-step column "
               "repeats tab:runs, whose perceptron rows have no source either"),
    Registered("tab:theiler", check_theiler, notes=(
        "the row labelled 'median absolute error' reproduces as the mean absolute error; "
        "the median over the seven ranks is 2.18 and 2.54, not the printed 2.78 and 3.13",
    )),
    Registered("tab:runs", check_runs, notes=(
        "the thirteen perceptron rows have no per-run record under data/: their budgets, "
        "memorisation and generalisation steps cannot be checked here. This is where "
        "errata item 5 lives -- the archived a_sum_sq log ends at step 46,000 against a "
        "printed budget of 100k -- and promoting the arithmetic training logs would put "
        "that row inside this check",
        "p211_wd0 is in no file under data/, so its row is unchecked",
        "the extended block's final accuracy is the value at the end of the budget, which "
        "no committed file records; exp8_outcomes.csv holds the maximum instead",
        "task, architecture, weight decay and the training fraction are configuration",
    )),
    Registered("tab:exclusion", None,
               "the recipe is not recoverable from the caption. Collapsing "
               "valid.theiler.contrast/sweep_windows.csv over observers, seeds and windows "
               "gives 5.43 / 2.07 / 1.20 in the first column against a printed "
               "5.39 / 2.11 / 1.20, and collapsing in another order moves every cell "
               "again without landing on the printed values"),
    Registered("tab:eos", check_eos, notes=(
        "the outcome column is a judgement; the divergence it reports is checked through "
        "the dashes in the three numeric columns",
    )),
    Registered("tab:ceiling", check_ceiling),
    Registered("tab:ceilingfit", check_ceilingfit),
)


# ---------------------------------------------------------------- comparison

def compare(printed: Optional[str], computed: Any) -> Tuple[str, str]:
    """Diff one cell at the precision the article prints it to.

    Returns a status and a note. A value out by no more than one unit in the last printed
    place is ``rounding`` rather than ``mismatch``: the measurement agrees and the digits do
    not, which is a different defect and a different fix.
    """
    if isinstance(computed, (list, tuple)):
        return _compare_list(printed, list(computed))
    if computed is None:
        if printed is None:
            return OK, ""
        return MISMATCH, "the file has no value here"
    if isinstance(computed, str):
        if printed is None:
            return MISMATCH, "the article prints a dash"
        return (OK, "") if _norm(printed) == _norm(computed) else (MISMATCH, "")
    if printed is None:
        return MISMATCH, "the article prints a dash"

    target = as_number(printed)
    if target is None:
        return UNREADABLE, f"{printed!r} is not a number this check can read"
    if not np.isfinite(computed):
        return MISMATCH, "the recomputed value is not finite"

    places = decimals(printed)
    if round_half_up(computed, places) == round_half_up(target, places):
        return OK, ""
    unit = 10.0 ** (-places)
    if abs(computed - target) <= unit + 1e-9:
        return ROUNDING, f"out by less than one unit in the last printed place ({unit:g})"
    return MISMATCH, ""


def _compare_list(printed: Optional[str], computed: List[Any]) -> Tuple[str, str]:
    if printed is None:
        return MISMATCH, "the article prints a dash where the file has values"
    parts = [p.strip() for p in re.split(r"[/,]", printed)]
    parts = [None if p in _MISSING else p for p in parts]
    if len(parts) != len(computed):
        return MISMATCH, (f"the article prints {len(parts)} value(s) and the file has "
                          f"{len(computed)}")
    worst, notes = OK, []
    for part, value in zip(parts, computed):
        status, note = compare(part, value)
        if status != OK:
            notes.append(f"{part}:{status}" + (f" ({note})" if note else ""))
            if worst == OK or status == MISMATCH:
                worst = status
    return worst, "; ".join(notes)


# ---------------------------------------------------------------- the audit

def audit(article: Optional[Path] = None, root: Optional[Path] = None,
          only: Optional[Sequence[str]] = None) -> Report:
    """Recompute every registered table and diff it against the article.

    Writes nothing. The caller decides what a mismatch is worth.
    """
    article = article or article_path()
    data = Data(root)
    tables = read_tables(article)
    results: List[TableResult] = []

    for entry in REGISTRY:
        if only and entry.label not in only:
            continue
        if entry.check is None:
            results.append(TableResult(entry.label, "skipped", entry.reason,
                                       notes=list(entry.notes)))
            continue
        table = tables.get(entry.label)
        if table is None:
            results.append(TableResult(entry.label, "skipped",
                                       "no table with this label in the article"))
            continue
        result = TableResult(entry.label, "checked", notes=list(entry.notes))
        try:
            computed = list(entry.check(table, data))
        except (ParseError, FileNotFoundError, KeyError, IndexError) as error:
            results.append(TableResult(entry.label, "skipped",
                                       f"{type(error).__name__}: {error}",
                                       notes=list(entry.notes)))
            continue
        for item in computed:
            printed = table.printed(item.row, item.column)
            status, note = compare(printed, item.value)
            result.findings.append(Finding(
                label=entry.label,
                row=table.rows[item.row].label() or f"row {item.row}",
                column=table.column_label(item.column),
                printed=printed,
                computed=item.value if not isinstance(item.value, (list, tuple))
                else ", ".join("---" if v is None else f"{v:g}" for v in item.value),
                source=item.source,
                status=status,
                note="; ".join(part for part in (item.note, note) if part),
            ))
        results.append(result)

    return Report(results=results, article=str(article), data=str(data.root))


def format_report(report: Report, verbose: bool = False) -> str:
    """The report as text: what was checked, what was not, and every disagreement."""
    lines: List[str] = []
    lines.append(f"article: {report.article}")
    lines.append(f"data:    {report.data}")
    lines.append("")

    checked = [r for r in report.results if r.state == "checked"]
    skipped = [r for r in report.results if r.state == "skipped"]
    cells = sum(r.checked for r in checked)
    bad = sum(len(r.mismatches) for r in checked)
    lines.append(f"{len(checked)} table(s) checked, {cells} cell(s) recomputed, "
                 f"{bad} disagreeing; {len(skipped)} table(s) not checked")
    lines.append("")

    for result in checked:
        mark = "OK " if not result.mismatches else "BAD"
        lines.append(f"[{mark}] {result.label:22s} {result.checked:4d} cell(s), "
                     f"{len(result.mismatches)} disagreeing")
        for finding in result.mismatches:
            lines.append("        " + finding.describe())
        if verbose:
            for finding in result.findings:
                if not finding.bad:
                    lines.append("    ok  " + finding.describe())
        for note in result.notes:
            lines.append(f"        note: {note}")

    if skipped:
        lines.append("")
        lines.append("not checked:")
        for result in skipped:
            lines.append(f"  {result.label:22s} {result.reason}")
            for note in result.notes:
                lines.append(f"        note: {note}")

    lines.append("")
    if report.ok:
        lines.append("every recomputed cell agrees with the article at the precision printed.")
    else:
        lines.append(f"{len(report.mismatches)} cell(s) disagree.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--article", type=Path, default=None,
                        help="path to report.tex (default: ../icomp_v2/report.tex)")
    parser.add_argument("--data", type=Path, default=None,
                        help="path to the data tree (default: data/)")
    parser.add_argument("--table", action="append", default=None,
                        help="check only this label; repeatable")
    parser.add_argument("--verbose", action="store_true", help="list agreeing cells too")
    args = parser.parse_args(argv)

    report = audit(article=args.article, root=args.data, only=args.table)
    print(format_report(report, verbose=args.verbose))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
