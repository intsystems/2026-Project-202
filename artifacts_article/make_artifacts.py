#!/usr/bin/env python3
r"""Build the Neural Network Artifacts edition of icomp_v2/report.tex.

    python make_artifacts.py                # full paper, 8--12 pages
    python make_artifacts.py --mode final   # camera-ready, names shown

The NeurIPS 2026 Workshop on Neural Network Artifacts as a New Data Modality
takes two lengths -- extended abstracts of 4--6 pages and full papers of
8--12.  This paper goes as a full paper, so that is the only track here and
the script takes no --track.  Page limits exclude references and supplementary
material, which is what the converter measures: it labels the page the main
text ends on, and the appendix follows the bibliography.

The conversion itself lives in venue_common/neurips2026.py, shared with the
NewInML edition -- both workshops use the same NeurIPS 2026 kit.  This file is
the venue.  Nothing here writes to icomp_v2/.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "venue_common" / "neurips2026.py"


def _engine():
    if not ENGINE.exists():
        raise SystemExit(
            f"cannot find {ENGINE}.\n"
            "make_artifacts.py is a venue profile; the conversion itself lives "
            "in venue_common/. If that folder moved, point ENGINE at it.")
    spec = importlib.util.spec_from_file_location("neurips2026", ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The venue.  https://artifactsasdata.org/cfp/
#
#   Extended abstracts 4--6 pages, full papers 8--12, both excluding
#   references and supplementary material.  OpenReview, deadline
#   1 September 2026 AoE.
#
# The call says the official policies and anonymity requirements "will be
# linked here when available", so the blind track below is the assumption the
# rest of the call supports -- it names anonymity requirements as a thing that
# will apply, and the template it ships is the standard NeurIPS workshop kit.
# If the workshop turns out to review single-blind, the fix is one word:
# sglblindworkshop.
# --------------------------------------------------------------------------

VENUE = {
    "workshop": "Neural Network Artifacts as a New Data Modality",
    "workshoptitle": "Neural Network Artifacts as a New Data Modality",
    "slug": "icomp_artifacts",
    "authors": "artifacts_authors.tex",
    "submit": "https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/NeuralArtifacts",
    # The extended-abstract track (4--6 pages) is deliberately not here.  The
    # paper goes as a full paper, whose 8--12 the main text already fits, and
    # a track nobody builds is a page limit nobody checks.
    "tracks": {
        "full": {
            "name": "full paper, double-blind",
            "option": "dblblindworkshop",
            "why": ("The call defers its anonymity policy, so this assumes the "
                    "double-blind\n% default it points at; the style file then "
                    "prints \"Anonymous Author(s)\" in\n% place of the author "
                    "block and numbers the lines for the reviewers.\n"
                    "% Single-blind instead: sglblindworkshop."),
            "min": 8,
            "max": 12,
            "default": True,
        },
    },
    "notes": [
        "The call requires the paper to be self-contained: reviewers are not\n"
        "required to consult the appendix.",
    ],
}


if __name__ == "__main__":
    sys.exit(_engine().main(VENUE, HERE, "artifacts_article/make_artifacts.py"))
