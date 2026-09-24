#!/usr/bin/env python3
"""Build all synchronized article editions from icomp_v2/report.tex.

One invocation builds six PDFs: black and blue versions of ICOMP, Artifacts,
and AISTATS 2027.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "icomp_v2" / "report.tex"


def run(command: list[str], cwd: Path, log_path: Path) -> None:
    print(f"+ {' '.join(command)}  [{cwd}]")
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        tail = (result.stdout + result.stderr)[-12000:]
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}\n\n{tail}"
        )


def check_pdf(path: Path, started_ns: int) -> None:
    if not path.exists():
        raise RuntimeError(f"expected PDF was not produced: {path}")
    if path.stat().st_mtime_ns < started_ns:
        raise RuntimeError(f"PDF was not rebuilt during this run: {path}")
    if path.stat().st_size < 10_000 or path.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"output is not a valid PDF: {path}")


def copy_source_tree(source_dir: Path, destination: Path) -> None:
    generated = (".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log",
                 ".out", ".pdf", ".synctex.gz")

    def ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != source_dir.resolve():
            return set()
        return {name for name in names if name.endswith(generated)}

    shutil.copytree(source_dir, destination, ignore=ignore)


def build_icomp(mode: str, started_ns: int, log_dir: Path) -> Path:
    source_dir = ROOT / "icomp_v2"
    output = source_dir / ("report_blue.pdf" if mode == "blue" else "report.pdf")
    with tempfile.TemporaryDirectory(prefix=f"icomp_{mode}_") as temp:
        work = Path(temp) / "icomp_v2"
        copy_source_tree(source_dir, work)
        report = work / "report.tex"
        source = report.read_text(encoding="utf-8")
        if mode == "blue":
            source = "\\def\\claudedraft{}\n" + source
        report.write_text(source, encoding="utf-8", newline="\n")
        run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-gg", "report.tex"],
            work, log_dir / f"icomp_{mode}.log",
        )
        shutil.copy2(work / "report.pdf", output)
    check_pdf(output, started_ns)
    return output


def build_artifacts(mode: str, started_ns: int, log_dir: Path) -> Path:
    output = ROOT / "artifacts_article" / (
        "icomp_artifacts_blue.pdf" if mode == "blue" else "icomp_artifacts.pdf"
    )
    run(
        [sys.executable, "make_artifacts.py", "--claude", mode],
        ROOT / "artifacts_article", log_dir / f"artifacts_{mode}.log",
    )
    check_pdf(output, started_ns)
    return output


def build_aistats(mode: str, started_ns: int, log_dir: Path) -> Path:
    output = ROOT / "aistats_article" / (
        "aistats2027_blue.pdf" if mode == "blue" else "aistats2027.pdf"
    )
    run(
        [sys.executable, "make_aistats.py", "--claude", mode],
        ROOT / "aistats_article", log_dir / f"aistats_{mode}.log",
    )
    check_pdf(output, started_ns)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("black", "blue", "all"), default="all")
    parser.add_argument(
        "--edition", choices=("icomp", "artifacts", "aistats", "all"), default="all"
    )
    args = parser.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"shared source does not exist: {SOURCE}")

    modes = ("black", "blue") if args.mode == "all" else (args.mode,)
    editions = ("icomp", "artifacts", "aistats") if args.edition == "all" else (args.edition,)
    builders = {"icomp": build_icomp, "artifacts": build_artifacts, "aistats": build_aistats}
    started_ns = time.time_ns()
    log_dir = ROOT / "build_logs"
    log_dir.mkdir(exist_ok=True)
    outputs: dict[str, str] = {}

    try:
        for mode in modes:
            for edition in editions:
                path = builders[edition](mode, started_ns, log_dir)
                key = f"{edition}_{mode}"
                outputs[key] = str(path)
                print(f"OK  {key}: {path} ({path.stat().st_size:,} bytes)")
    except (RuntimeError, OSError) as error:
        print(f"\nBUILD FAILED: {error}", file=sys.stderr)
        print(f"Detailed logs: {log_dir}", file=sys.stderr)
        return 1

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "modes": list(modes),
        "editions": list(editions),
        "outputs": outputs,
    }
    (ROOT / "article_versions.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nBuilt {len(outputs)} PDF(s) successfully.")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
