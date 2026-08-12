"""Run the standard p=211 EDM pipeline on the WD=0 ablation log."""

from __future__ import annotations

import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "training_log_p_211_wd_0.csv"
OUTPUT = HERE / "edm_report_p211_wd0_tau1"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # The standard report script expects explicit transition flags.  WD=0 did
    # not genuinely grok, so keep the flag absent; the analysis script is
    # patched below to use an accuracy-derived fallback transition boundary.
    # The ablation log is four times longer than the original run.  A larger
    # stride keeps the same W=200 window and all metrics while avoiding an
    # unnecessarily dense set of highly overlapping, non-independent windows.
    cmd = ["python", str(HERE / "analyze_p211_wd05_edm.py"), str(INPUT), str(OUTPUT), "--fast", "--window-size", "200", "--stride", "100"]
    subprocess.run(cmd, check=True)
    # Reuse the report builder with the ablation analysis directory.  The
    # source log is deliberately not copied; all artifacts remain in one
    # clearly named subdirectory.
    subprocess.run([
        "python", str(HERE / "build_p211_edm_pdf_report.py"),
        "--analysis-dir", str(OUTPUT),
        "--output", str(OUTPUT / "p211_wd0_detailed_edm_report_tau1_lb_mg.pdf"),
    ], check=True)
    print(OUTPUT / "p211_wd0_detailed_edm_report_tau1_lb_mg.pdf")


if __name__ == "__main__":
    main()
