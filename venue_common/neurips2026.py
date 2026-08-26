#!/usr/bin/env python3
r"""Put ``icomp_v2/report.tex`` on the NeurIPS 2026 workshop template.

Two workshops want this paper and both use the same kit: ``article`` plus
``\usepackage[<track>]{neurips_2026}``.  They differ in the workshop name in
the footer, the page limit, and nothing else that reaches the LaTeX.  So the
conversion lives here once and each venue folder supplies a ``VENUE`` dict and
calls :func:`main`.

    newinml_article/make_newinml.py     NewInML,   2--8 pages
    artifacts_article/make_artifacts.py Artifacts, 8--12 pages

The input is one shape: a standalone LaTeX article, ``\documentclass`` through
``\end{document}``.  ``report.tex`` is already that, so it is read where it
lies and never written to.  Everything the ICOMP venue supplied and NeurIPS
does not, or supplies differently, is handled below:

  1.  The class line becomes NeurIPS's, plus ``\workshoptitle`` -- what the
      style file prints in the first-page footer.
  2.  ``icomp2026_conference`` and ``times`` go.  The first is the old venue's
      style file; the second neurips_2026 loads itself, so loading it again
      risks an option clash.  Both share one ``\usepackage`` line in the
      source, which is why package lists are rewritten member by member.
  3.  ``\bibliographystyle{icomp2026_conference}`` goes with it: that ``.bst``
      stays behind, and ``plainnat`` is emitted instead.
  4.  The title loses its manual ``\\``.  They are there because the ICOMP
      style file sets a title in small caps at about forty characters to the
      line and hyphenates it otherwise; NeurIPS centres a title of its own
      width, where the same breaks fall in arbitrary places.
  5.  The author block is withheld under ``anon`` -- the style file prints
      "Anonymous Author(s)" itself -- and taken from the venue's authors file
      under ``final``.
  6.  ``float`` is lifted above ``hyperref``, and float and theorem link
      destinations are made unique.  Both are repairs; see the two long
      comments below for what they repair.

Nothing here writes to ``icomp_v2/``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent      # EDMGrokking
SOURCE_DIR = REPO / "icomp_v2"
SOURCE_MAIN = SOURCE_DIR / "report.tex"

JOBNAME = "main"

# The old venue's style file, and what neurips_2026 loads for itself.  Any
# \usepackage naming one of these is dropped; a line naming some of these and
# some others is rewritten to keep the others.
DROP_PACKAGES = {"icomp2026_conference", "times", "natbib"}

# Packages that have to be loaded before hyperref.  float patches \@caption,
# and doing so after hyperref has patched it too writes every float's
# destination twice, of which the second is dropped: report.tex loads hyperref
# on line 13 and float on line 21, so as it stands every one of its 36 floats
# has a dead link and every \cref to one lands on the float before it.
# Loading float first fixes all of them and changes nothing else.
PRECEDE_HYPERREF = ["float", "subfig", "subfigure"]

# Emitted just before \bibliography.  main.aux then records the page the main
# text ends on, which is the number the page limit is counted against: both
# venues exclude references, and the appendix follows them.
BODY_END_LABEL = "venue:endofmaintext"

_COMMENT = re.compile(r"(?<!\\)%")
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
_GRAPHICS = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_HYPERREF = re.compile(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{[^}]*\bhyperref\b[^}]*\}", re.S)


BANNER = r"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GENERATED FILE -- DO NOT EDIT.
%
%  {workshop} @ NeurIPS 2026 edition of "{title}"
%  track: {trackname} ({pages}), mode: {mode}
%
%  Converted from icomp_v2/report.tex by {script}.
%  Edit report.tex, not this file, and rerun:
%      python {script}{trackarg}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

CLASS_HEADER = r"""\documentclass{{article}}

% {why}
\usepackage[{option}]{{neurips_2026}}
% Printed in the first-page footer.  Required for a workshop paper, and under
% [{option}] it is the only thing that says which workshop this is.
\workshoptitle{{{workshoptitle}}}
"""

LINK_FIXES = r"""%%% ---- unique link destinations (inserted by the converter) ----------------
% hyperref names a destination after the raw counter value, so two objects that
% share a counter value share a destination and the second one is dropped:
% every link to it lands on the first.  This paper does it in the appendix,
% which restarts float numbering, so appendix Table 1 collides with body
% Table 1.
%
% \theH<counter> is the name hyperref actually uses, and it is the printed
% number that has to go into it.  Qualifying by section separates an appendix
% copy from a body one whatever the numbering scheme.  Nothing visible changes.
\makeatletter
\@for\@vnc:={theorem,definition,lemma,corollary,proposition,remark,assumption,%
table,figure,algorithm}\do{%
  \expandafter\providecommand\csname theH\@vnc\endcsname{}%
  \expandafter\edef\csname theH\@vnc\endcsname{%
    \noexpand\thesection.\expandafter\noexpand\csname the\@vnc\endcsname}}
\makeatother
%%% -------------------------------------------------------------------------
"""


# --------------------------------------------------------------------------
# Reading LaTeX.  Everything here reads source with comments stripped, so a
# commented-out \input or \includegraphics is not acted on.
# --------------------------------------------------------------------------

def code_of(line: str) -> str:
    """The part of a line TeX acts on: everything before an unescaped %."""
    hit = _COMMENT.search(line)
    return line if hit is None else line[: hit.start()]


def match_brace(text: str, start: int) -> int:
    """Index just past the '}' matching the '{' at `start`, comments ignored."""
    depth, i = 0, start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "%":
            nl = text.find("\n", i)
            i = len(text) if nl < 0 else nl + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise SystemExit(f"unbalanced braces starting at offset {start}")


def uncommented(text: str, pat: re.Pattern) -> list[int]:
    """Offsets of every match of `pat` that TeX would act on."""
    out = []
    for m in pat.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        if _COMMENT.search(text[line_start: m.start()]):
            continue                                  # inside a comment
        out.append(m.start())
    return out


def find_in_code(text: str, pat: re.Pattern) -> int | None:
    """Offset of the first match outside a comment, or None."""
    hits = uncommented(text, pat)
    return hits[0] if hits else None


def comment_block_above(text: str, idx: int) -> int:
    r"""Start of the contiguous comment lines immediately above the line at `idx`.

    A comment block that sits on top of a command describes that command, so
    when the conversion rewrites the command the block above it stops being
    true and has to go with it.  Both of this paper's front-matter comments do
    exactly that: one explains why the title carries manual \\ breaks, directly
    above a title from which they have just been removed, and the other says
    the style file prints "Anonymous authors / Paper under double-blind review"
    unless \icompfinalcopy is set -- naming a switch that no longer exists and
    a string neurips_2026 never prints.  A comment contradicting the code under
    it is worse than no comment.

    A blank line or any line carrying code ends the block: the run that touches
    the command is the run that describes it.  Returns `idx`'s own line start
    when there is no such run.
    """
    start = text.rfind("\n", 0, idx) + 1               # the command's own line
    while start > 0:
        prev = text.rfind("\n", 0, start - 1) + 1
        if not text[prev: start - 1].strip().startswith("%"):
            break
        start = prev
    return start


def flatten(path: Path, seen: list[str] | None = None) -> str:
    r"""Expand \input and \include recursively, one file per standalone line.

    report.tex pulls in math_commands.tex this way.  Inlining it is what makes
    the build directory self-contained: the zip has to compile in an empty
    directory with only the style file beside it.
    """
    seen = [] if seen is None else seen
    if path.name in seen:
        raise SystemExit(f"input loop: {' -> '.join(seen + [path.name])}")
    if not path.exists():
        raise SystemExit(f"missing source: {path}")

    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        hit = _INPUT.search(code_of(line))
        if hit is None:
            out.append(line)
            continue
        before = code_of(line)[: hit.start()].strip()
        after = code_of(line)[hit.end():].strip()
        if before or after:
            raise SystemExit(
                f"{path.name}:{lineno}: \\input shares a line with other code; "
                "put it on a line of its own so it can be flattened.")
        child = hit.group(1).strip()
        child_path = path.parent / (child if child.endswith(".tex") else child + ".tex")
        out.append(f"%%% >>> begin {child_path.name} "
                   f"(was \\input on {path.name}:{lineno})")
        out.append(flatten(child_path, seen + [path.name]))
        out.append(f"%%% <<< end {child_path.name}")
    return "\n".join(out)


def split_document(text: str) -> tuple[str, str]:
    r"""Preamble and body, split at \begin{document}."""
    lines = text.splitlines()
    for lineno, line in enumerate(lines):
        if r"\begin{document}" in code_of(line):
            return "\n".join(lines[:lineno]), "\n".join(lines[lineno + 1:])
    raise SystemExit(r"no \begin{document} in the source article")


# --------------------------------------------------------------------------
# Preamble surgery.
# --------------------------------------------------------------------------

def convert_preamble(preamble: str) -> tuple[str, dict]:
    r"""Drop the old class line and the packages the new style file supplies."""
    stats = {"dropped": []}
    out = []
    for line in preamble.splitlines():
        code = code_of(line)

        if re.search(r"\\documentclass", code):
            continue                                  # CLASS_HEADER replaces it

        # A \PassOptionsToPackage aimed at a package that is no longer loaded.
        opt = re.search(r"\\PassOptionsToPackage\s*\{[^}]*\}\s*\{([^}]*)\}", code)
        if opt and opt.group(1).strip() in DROP_PACKAGES:
            stats["dropped"].append(line.strip())
            continue

        pkg = re.search(r"\\usepackage\s*(\[[^\]]*\])?\s*\{([^}]*)\}", code)
        if pkg:
            names = [n.strip() for n in pkg.group(2).split(",")]
            keep = [n for n in names if n not in DROP_PACKAGES]
            if not keep:
                stats["dropped"].append(line.strip())
                continue
            if keep != names:                         # a mixed list: rewrite it
                stats["dropped"].append(
                    line.strip() + "   [kept " + ", ".join(keep) + "]")
                out.append("\\usepackage" + (pkg.group(1) or "")
                           + "{" + ", ".join(keep) + "}")
                continue

        out.append(line)
    return "\n".join(out), stats


def hyperref_line(lines: list[str]) -> int | None:
    r"""Index of the line that opens the \usepackage loading hyperref.

    Found on the joined text, because report.tex spreads that call over two
    lines: its option list does not fit on one.
    """
    text = "\n".join(lines)
    hit = _HYPERREF.search(text)
    return None if hit is None else text[: hit.start()].count("\n")


def reorder_before_hyperref(preamble: str) -> tuple[str, list[str]]:
    """Lift the packages of PRECEDE_HYPERREF above the hyperref call."""
    lines = preamble.splitlines()
    where = hyperref_line(lines)
    if where is None:
        return preamble, []

    moved, kept = [], []
    for i, line in enumerate(lines):
        pkg = re.search(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", code_of(line))
        names = [n.strip() for n in pkg.group(1).split(",")] if pkg else []
        if i > where and any(n in PRECEDE_HYPERREF for n in names):
            moved.append(line.strip())
            continue
        kept.append(line)
    if not moved:
        return preamble, []

    where = hyperref_line(kept)
    note = ["% Lifted above hyperref by the converter: loaded after it, these",
            "% patch \\@caption a second time and every float loses its link."]
    kept[where:where] = note + moved
    return "\n".join(kept), moved


def normalize_title(preamble: str) -> tuple[str, bool]:
    r"""Take the manual line breaks out of the title.

    report.tex carries three \\ and a comment explaining them: the ICOMP style
    file sets the title in small caps at about forty characters to the line and
    hyphenates it otherwise.  NeurIPS sets a title of its own width, where
    those breaks fall in arbitrary places, so they come out.
    """
    hit = find_in_code(preamble, re.compile(r"\\title\s*\{"))
    if hit is None:
        return preamble, False
    open_brace = preamble.index("{", hit)
    end = match_brace(preamble, open_brace)
    inner = preamble[open_brace + 1: end - 1]
    flat = " ".join(inner.replace(r"\\", " ").split())
    if flat == " ".join(inner.split()):
        return preamble, False

    # The comment block above the title argues for the breaks this line has
    # just taken out, so it goes too, and says instead what happened.
    note = ("% The source title carries manual \\\\ breaks, sized for the ICOMP "
            "style file, which\n"
            "% sets a title in small caps at about forty characters to the "
            "line and hyphenates\n"
            "% it otherwise.  neurips_2026 centres a title of its own width, "
            "where those breaks\n"
            "% fall in arbitrary places, so the converter removed them.\n")
    return (preamble[:comment_block_above(preamble, hit)] + note
            + preamble[hit:open_brace + 1] + flat + preamble[end - 1:], True)


def title_of(preamble: str) -> str:
    hit = find_in_code(preamble, re.compile(r"\\title\s*\{"))
    if hit is None:
        return "(untitled)"
    start = preamble.index("{", hit)
    raw = preamble[start + 1: match_brace(preamble, start) - 1]
    return " ".join(raw.replace(r"\\", " ").split())


def convert_authors(preamble: str, mode: str, authors: Path) -> tuple[str, str]:
    r"""Settle the author block, which report.tex keeps in the preamble.

    Brace-matched, because it runs over four lines and dropping only the line
    that opens it would leave the address behind.  Under a double-blind track
    the style file prints "Anonymous Author(s)" itself, so an anonymous build
    keeps no \author at all.

    Every block is located before any is rewritten, and they are rewritten back
    to front.  Substituting one at a time would re-find the block just
    inserted -- the authors file is itself an \author{...} -- and, counting it
    as a second block, delete it again: a camera-ready with no author names on
    it, which is what the check in sanity_check() now refuses to ship.
    """
    replacement = None
    if mode == "final" and authors.exists():
        replacement = authors.read_text(encoding="utf-8").rstrip()

    spans = []
    for start in uncommented(preamble, re.compile(r"\\author\s*\{")):
        end = match_brace(preamble, preamble.index("{", start))
        spans.append((comment_block_above(preamble, start), end))
    seen = len(spans)

    for i, (start, end) in reversed(list(enumerate(spans))):
        if mode == "anon":
            new = ""
        elif replacement is not None:
            new = replacement if i == 0 else ""        # one block, the first
        else:
            new = preamble[start:end]                 # keep the source's own
        preamble = preamble[:start] + new + preamble[end:]

    if mode != "final":
        return preamble, "withheld by the style file"
    if replacement is None:
        return preamble, "the source's own (no authors file)"
    if seen == 0:
        preamble = preamble.rstrip() + "\n\n" + replacement + "\n"
    return preamble, authors.name


# --------------------------------------------------------------------------
# Body surgery.
# --------------------------------------------------------------------------

def convert_body(body: str) -> tuple[str, dict]:
    r"""Drop \bibliographystyle; the preamble emits plainnat instead.

    The ICOMP .bst is not copied into the build, so a surviving
    \bibliographystyle{icomp2026_conference} would stop bibtex dead.
    """
    stats = {"bibstyle": 0}
    out = []
    for line in body.splitlines():
        if re.search(r"\\bibliographystyle\s*\{", code_of(line)):
            stats["bibstyle"] += 1
            continue
        out.append(line)
    return "\n".join(out), stats


def mark_end_of_main_text(body: str) -> str:
    r"""Label the page the main text ends on, for the page-limit report.

    The label goes on the line before \bibliography with no blank line between:
    a comment does not end a paragraph but a blank line does, and a \label
    alone in a paragraph is pushed to the next page, which would overcount.
    """
    marker = ("%%% Where the main text ends: the page limit is counted against\n"
              "%%% this page, references and appendix excluded.\n"
              f"\\label{{{BODY_END_LABEL}}}\n")
    hit = re.search(r"(?m)^\s*\\bibliography\s*\{", body)
    if hit is None:
        raise SystemExit(
            r"no \bibliography in the source: nothing to measure the limit against")
    return body[:hit.start()] + marker + body[hit.start():]


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def build_document(venue: dict, track: dict, mode: str, main: Path,
                   authors: Path, script: str) -> tuple[str, str, dict]:
    text = flatten(main)
    preamble, body = split_document(text)
    preamble, pstats = convert_preamble(preamble)
    preamble, who = convert_authors(preamble, mode, authors)
    body, bstats = convert_body(body)
    body = mark_end_of_main_text(body)

    preamble, moved = reorder_before_hyperref(preamble.strip("\n"))
    preamble, unwrapped = normalize_title(preamble)
    title = title_of(preamble)

    option = track["option"] + (", final" if mode == "final" else "")
    trackarg = "" if track.get("default") else f" --track {track['key']}"
    doc = "\n".join([
        BANNER.format(workshop=venue["workshop"], title=title, mode=mode,
                      trackname=track["name"], script=script,
                      pages=f"{track['min']}--{track['max']} pages",
                      trackarg=trackarg),
        CLASS_HEADER.format(option=option, why=track["why"],
                            workshoptitle=venue["workshoptitle"]),
        preamble,
        "",
        LINK_FIXES,
        r"% The ICOMP .bst stayed behind with its style file.",
        r"\bibliographystyle{plainnat}",
        "",
        r"\begin{document}",
        body.strip("\n"),
        "",
    ])
    stats = {**pstats, **bstats, "author": who, "moved": moved,
             "unwrapped": unwrapped}
    return doc, title, stats


def sanity_check(doc: str, mode: str) -> None:
    """Fail rather than ship a document that will not compile or will mislead."""
    code = "\n".join(code_of(line) for line in doc.splitlines())
    problems = []
    if code.count(r"\documentclass") != 1:
        problems.append(r"expected exactly one \documentclass")
    if code.count(r"\begin{document}") != 1 or code.count(r"\end{document}") != 1:
        problems.append("document environment is not balanced")
    if code.count(r"\maketitle") != 1:
        problems.append(r"expected exactly one \maketitle")
    if code.count(r"\bibliographystyle") != 1:
        problems.append(r"expected exactly one \bibliographystyle")
    if code.count(r"\workshoptitle") != 1:
        problems.append(r"a workshop paper needs exactly one \workshoptitle")
    for banned, why in [
        ("icomp2026_conference", "the ICOMP style file must not be loaded"),
        (r"\usepackage{times}", "neurips_2026 loads times itself"),
        (r"\icompfinalcopy", "an ICOMP-only switch survived"),
    ]:
        if banned in code:
            problems.append(f"{banned!r} still present: {why}")
    # Counted rather than merely looked for: a camera-ready that lost its
    # author block still compiles, and the missing names show up only in the
    # rendered PDF, which is too late.
    blocks = len(uncommented(code, re.compile(r"\\author\s*\{")))
    if mode == "anon" and blocks:
        problems.append(r"\author survived an anonymous build")
    if mode == "final" and blocks != 1:
        problems.append(f"a camera-ready needs exactly one \\author block, "
                        f"found {blocks}")

    # \cref takes a comma-separated list, so each group is split before lookup.
    inside = code.split(r"\begin{document}", 1)[-1]
    defined = set(re.findall(r"\\label\{([^}]+)\}", inside))
    referenced = set()
    for group in (re.findall(r"\\(?:ref|eqref|cref|Cref|autoref)\*?\{([^}]+)\}", inside)
                  + re.findall(r"\\hyperref\[([^\]]+)\]", inside)):
        referenced |= {n.strip() for n in group.split(",") if n.strip()}
    for lost in sorted(referenced - defined):
        problems.append(f"label {lost!r} is referenced and defined nowhere")

    if problems:
        raise SystemExit("converted document failed its checks:\n  - "
                         + "\n  - ".join(problems))


# --------------------------------------------------------------------------
# Assets and compilation.
# --------------------------------------------------------------------------

def copy_assets(doc: str, assets: Path, style: Path,
                build: Path) -> tuple[list[str], list[str]]:
    """Copy the style file, the .bib and exactly the images the document uses."""
    sty = style / "neurips_2026.sty"
    if not sty.exists():
        raise SystemExit(f"missing {sty}\nit ships in the venue's template zip.")
    shutil.copy2(sty, build / sty.name)

    bibs = []
    for m in re.finditer(r"\\bibliography\s*\{([^}]*)\}", doc):
        for name in (n.strip() for n in m.group(1).split(",")):
            src = assets / (name + ".bib")
            if not src.exists():
                raise SystemExit(f"bibliography not found: {src}")
            shutil.copy2(src, build / src.name)
            bibs.append(src.name)

    wanted = sorted({m.group(1).strip()
                     for line in doc.splitlines()
                     for m in _GRAPHICS.finditer(code_of(line))})
    copied, missing = [], []
    for rel in wanted:
        found = None
        for candidate in (assets / rel, *(assets.glob(rel + ".*"))):
            if candidate.exists() and candidate.is_file():
                found = candidate
                break
        if found is None:
            missing.append(rel)
            continue
        target = build / rel
        if not target.suffix:
            target = target.with_suffix(found.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, target)
        copied.append(rel)
    if missing:
        raise SystemExit(f"images referenced but not found under {assets}:\n  - "
                         + "\n  - ".join(missing))
    return copied, bibs


def compile_pdf(build: Path, workdir: Path, out_pdf: Path,
                track: dict, keep_work: bool) -> Path:
    """Compile a throwaway copy, so `build` holds only what the venue receives."""
    if shutil.which("latexmk") is None:
        raise SystemExit("latexmk not on PATH; rerun with --no-compile")
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(build, workdir)

    run = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         f"{JOBNAME}.tex"], cwd=workdir, capture_output=True, text=True)
    log = workdir / f"{JOBNAME}.log"
    if run.returncode != 0:
        tail = (log.read_text(encoding="utf-8", errors="replace").splitlines()
                if log.exists() else [])
        errors = [ln for ln in tail if ln.startswith("!")] or tail[-30:]
        raise SystemExit("latexmk failed:\n  " + "\n  ".join(errors[:20])
                         + f"\n\nfull log: {log}")

    bbl = workdir / f"{JOBNAME}.bbl"
    if not bbl.exists():
        raise SystemExit(f"no {JOBNAME}.bbl was produced; the bibliography did not run")
    shutil.copy2(bbl, build / bbl.name)
    shutil.copy2(workdir / f"{JOBNAME}.pdf", out_pdf)
    report_log(log)
    report_length(workdir / f"{JOBNAME}.aux", track)
    if not keep_work:
        shutil.rmtree(workdir)
    return out_pdf


def report_log(log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on .*?\((\d+) pages", text)
    print(f"  pages          : {pages.group(1) if pages else '?'} (whole document)")
    for label, pattern in [
        ("undefined refs", r"Reference `[^']*' on page \d+ undefined"),
        ("undefined cites", r"Citation `[^']*' on page \d+ undefined"),
        ("duplicate links", r"destination with the same identifier"),
        ("overfull hbox", r"Overfull \\hbox"),
    ]:
        n = len(re.findall(pattern, text))
        flag = "" if n == 0 or label == "overfull hbox" else "   <-- fix this"
        print(f"  {label:15}: {n}{flag}")


def report_length(aux: Path, track: dict) -> int | None:
    """The page the main text ends on, against the track's limit."""
    lo, hi = track["min"], track["max"]
    if not aux.exists():
        return None
    text = aux.read_text(encoding="utf-8", errors="replace")
    hit = re.search(r"\\newlabel\{" + re.escape(BODY_END_LABEL)
                    + r"\}\{\{[^{}]*\}\{(\d+)\}", text)
    if hit is None:
        print(f"  main text      : ? pages (limit {lo}--{hi})")
        return None
    pages = int(hit.group(1))
    if pages > hi:
        note = f"   <-- {pages - hi} over, cut before submitting"
    elif pages < lo:
        note = f"   <-- {lo - pages} short"
    else:
        note = "   <-- fits"
    print(f"  main text      : {pages} pages, limit {lo}--{hi}{note}")
    return pages


def pack_zip(build: Path, archive: Path) -> Path:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build.rglob("*")):
            if path.is_file():
                zf.write(path,
                         arcname=str(path.relative_to(build)).replace("\\", "/"))
    return archive


# --------------------------------------------------------------------------

def main(venue: dict, here: Path, script: str, argv: list[str] | None = None) -> int:
    tracks = venue["tracks"]
    default_track = next(k for k, t in tracks.items() if t.get("default"))

    ap = argparse.ArgumentParser(
        description=f"Build the {venue['workshop']} @ NeurIPS 2026 edition of "
                    "icomp_v2/report.tex.")
    if len(tracks) > 1:
        ap.add_argument("--track", choices=sorted(tracks), default=default_track,
                        help="; ".join(f"{k}: {t['name']}, {t['min']}-{t['max']} pages"
                                       for k, t in sorted(tracks.items()))
                             + f" (default: {default_track})")
    ap.add_argument("--mode", choices=["anon", "final"], default="anon",
                    help="anon: blind submission (default). "
                         "final: camera-ready, author names shown.")
    ap.add_argument("--workshop-title", default=venue["workshoptitle"],
                    help="name printed in the first-page footer "
                         f"(default: {venue['workshoptitle']})")
    ap.add_argument("--source-main", type=Path, default=SOURCE_MAIN,
                    help="the article to convert; assets are looked for beside "
                         f"it (default: {SOURCE_MAIN.relative_to(REPO)})")
    ap.add_argument("-o", "--build", type=Path, default=None,
                    help="output directory (default: build/<track>)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args(argv)

    track = dict(tracks[getattr(args, "track", default_track)])
    track["key"] = getattr(args, "track", default_track)
    venue = {**venue, "workshoptitle": args.workshop_title}

    main_tex = args.source_main.resolve()
    if not main_tex.exists():
        raise SystemExit(f"no such source document: {main_tex}\n"
                         "pass --source-main to point at it if the paper moved.")
    assets = main_tex.parent
    style = here / "style"
    authors = here / venue["authors"]
    slug = venue["slug"] + ("" if len(tracks) == 1 else "_" + track["key"])
    build = (args.build or here / "build" / track["key"]).resolve()

    print(f"reading  {main_tex}")
    doc, title, stats = build_document(venue, track, args.mode, main_tex,
                                       authors, script)
    sanity_check(doc, args.mode)

    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    with open(build / f"{JOBNAME}.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    images, bibs = copy_assets(doc, assets, style, build)

    print(f"writing  {build}")
    print(f"  title          : {title}")
    print(f"  venue          : {venue['workshop']} @ NeurIPS 2026")
    print(f"  track          : {track['name']} "
          f"[{track['option']}{', final' if args.mode == 'final' else ''}]")
    print(f"  {JOBNAME}.tex       : {len(doc.splitlines())} lines")
    print(f"  images         : {len(images)}   bibliography: {', '.join(bibs)}")
    print(f"  authors        : {stats['author']}")
    if stats["unwrapped"]:
        print("  title breaks   : removed (ICOMP set it in small caps; "
              "NeurIPS does not)")
    if stats["moved"]:
        print(f"  reordered      : {len(stats['moved'])} package(s) lifted "
              "above hyperref")

    if not args.no_compile:
        pdf = compile_pdf(build, here / ".build-work",
                          here / f"{slug}.pdf", track, args.keep_work)
        print(f"  preview        : {pdf}")
    if not args.no_zip:
        archive = pack_zip(build, here / f"{slug}.zip")
        print(f"  zip            : {archive} "
              f"({archive.stat().st_size / 1e6:.1f} MB)")

    if stats["dropped"]:
        print("\nDropped from the preamble, the ICOMP style file having "
              "provided them or neurips_2026 providing them again:")
        for line in stats["dropped"]:
            print(f"  {line}")

    if args.mode == "final" and authors.exists():
        todos = [f"{authors.name}:{n}: {l.strip()}"
                 for n, l in enumerate(
                     authors.read_text(encoding="utf-8").splitlines(), 1)
                 if "TODO" in code_of(l)]
        if todos:
            print("\nTODO left in the author block -- edit before submitting:")
            for item in todos:
                print(f"  {item}")

    for note in venue.get("notes", []):
        print(f"\n{note}")
    print(f"\nSubmit {here / (slug + '.pdf')} to {venue['submit']}")
    return 0
