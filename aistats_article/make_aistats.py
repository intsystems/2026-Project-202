#!/usr/bin/env python3
"""Build the AISTATS 2027 edition from the shared ICOMP source."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / "icomp_v2" / "report.tex"
STYLE = HERE / "style"

TITLE = "Estimating Active Dimension of Training Dynamics from One Scalar Log, with an Application to Grokking"


def _source_body() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    if "\\begin{document}" not in text or "\\maketitle" not in text:
        raise SystemExit(f"cannot extract body from {SOURCE}")
    body = text.split("\\begin{document}", 1)[1]
    body = body.split("\\maketitle", 1)[1]
    body = body.rsplit("\\end{document}", 1)[0]
    body = body.replace(
        "\\bibliographystyle{icomp2026_conference}",
        "\\bibliographystyle{plainnat}",
    )
    # The shared ICOMP source is single-column and uses \textwidth for figures.
    # AISTATS is two-column, so ordinary figure environments must fit one column.
    body = re.sub(
        r"(\\includegraphics\s*\[[^]]*?width=)\\textwidth",
        r"\1\\columnwidth",
        body,
    )
    # The shared source contains full-width single-column tables.  Scale each
    # tabular independently so the same content remains valid in AISTATS's
    # two-column layout, including the appendix tables.
    body = body.replace(
        r"\begin{tabular}",
        r"\resizebox{\linewidth}{!}{%" + "\n" + r"\begin{tabular}",
    )
    body = body.replace(
        r"\end{tabular}",
        r"\end{tabular}%" + "\n" + r"}",
    )
    return body.strip()


def _document(claude: str) -> str:
    switch = r"\def\claudedraft{}" if claude == "blue" else "% black build"
    return "\n".join([
        "% GENERATED from ../icomp_v2/report.tex; edit the shared source.",
        switch,
        r"\documentclass[twoside]{article}",
        r"\usepackage{aistats2027}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[english]{babel}",
        r"\usepackage{xcolor}",
        r"\usepackage{url}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage{booktabs}",
        r"\usepackage{microtype}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{amsthm}",
        r"\usepackage{float}",
        r"\usepackage{algorithm}",
        r"\usepackage{algpseudocode}",
        r"\usepackage{multirow}",
        r"\usepackage{enumitem}",
        r"\usepackage[round]{natbib}",
        r"\usepackage[capitalize]{cleveref}",
        r"\input{math_commands.tex}",
        r"\setlength{\emergencystretch}{3em}",
        r"\newtheorem{definition}{Definition}",
        r"\newtheorem{theorem}{Theorem}",
        r"\newcommand{\vphi}{{\bm{\phi}}}",
        r"\DeclareMathOperator{\PR}{PR}",
        r"\ifdefined\claudedraft",
        r"  \definecolor{claudecolor}{RGB}{25,90,160}",
        r"\else",
        r"  \definecolor{claudecolor}{RGB}{0,0,0}",
        r"\fi",
        r"\newcommand{\cl}[1]{\textcolor{claudecolor}{#1}}",
        r"\newenvironment{claude}{\color{claudecolor}}{}",
        r"\crefname{section}{section}{sections}",
        r"\Crefname{section}{Section}{Sections}",
        r"\crefname{table}{table}{tables}",
        r"\Crefname{table}{Table}{Tables}",
        r"\crefname{figure}{figure}{figures}",
        r"\Crefname{figure}{Figure}{Figures}",
        r"\crefname{equation}{equation}{equations}",
        r"\crefname{theorem}{theorem}{theorems}",
        r"\Crefname{theorem}{Theorem}{Theorems}",
        r"\begin{document}",
        r"\runningtitle{Active Dimension from One Scalar Log}",
        r"\runningauthor{Karlov and Kravatskiy}",
        r"\twocolumn[",
        rf"\aistatstitle{{{TITLE}}}",
        r"\aistatsauthor{Anonymous}",
        r"\aistatsaddress{Anonymous Institution}",
        r"]",
        _source_body(),
        r"\end{document}",
        "",
    ])


def _graphics(body: str) -> list[str]:
    return sorted(set(re.findall(r"\\includegraphics\s*(?:\[[^]]*\])?\s*\{([^}]+)\}", body)))


def _copy_assets(build: Path, body: str) -> None:
    shutil.copy2(SOURCE.parent / "math_commands.tex", build / "math_commands.tex")
    shutil.copy2(SOURCE.parent / "references.bib", build / "references.bib")
    shutil.copy2(SOURCE.parent / "natbib.sty", build / "natbib.sty")
    shutil.copy2(STYLE / "aistats2027.sty", build / "aistats2027.sty")
    shutil.copy2(STYLE / "fancyhdr.sty", build / "fancyhdr.sty")

    for rel in _graphics(body):
        source = SOURCE.parent / rel
        if not source.exists():
            raise SystemExit(f"AISTATS asset not found: {source}")
        target = build / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build(claude: str) -> Path:
    if claude not in {"black", "blue"}:
        raise ValueError(claude)
    out_name = "aistats2027" + ("_blue" if claude == "blue" else "")
    work = HERE / ".build-work" / claude
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    document = _document(claude)
    (work / "main.tex").write_text(document, encoding="utf-8", newline="\n")
    _copy_assets(work, document)
    run = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-gg", "main.tex"],
        cwd=work, text=True, capture_output=True,
    )
    if run.returncode:
        log = work / "main.log"
        tail = log.read_text(encoding="utf-8", errors="replace")[-8000:] if log.exists() else run.stdout + run.stderr
        raise SystemExit(f"AISTATS {claude} build failed:\n{tail}")
    output = HERE / f"{out_name}.pdf"
    shutil.copy2(work / "main.pdf", output)
    shutil.copy2(work / "main.tex", HERE / f"{out_name}.tex")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claude", choices=["black", "blue"], default="black")
    args = parser.parse_args()
    output = build(args.claude)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

