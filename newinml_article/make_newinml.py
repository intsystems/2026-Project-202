#!/usr/bin/env python3
r"""Build the NewInML @ NeurIPS 2026 edition of icomp_v2/report.tex.

    python make_newinml.py                 # anonymous submission
    python make_newinml.py --mode final    # camera-ready, names shown

NewInML asks for 2--8 pages excluding references, reviewed double-blind on
OpenReview, on the NeurIPS 2026 workshop template.  report.tex is already a
standalone article, so the whole conversion is a preamble away; it lives in
venue_common/neurips2026.py, which the Artifacts edition shares.  This file is
the venue: the footer name, the page limit, and where it goes.

Nothing here writes to icomp_v2/.
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
            "make_newinml.py is a venue profile; the conversion itself lives "
            "in venue_common/. If that folder moved, point ENGINE at it.")
    spec = importlib.util.spec_from_file_location("neurips2026", ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The venue.  https://newinml.github.io/NewInML2026NeurIPS/
#
#   2--8 pages excluding references, double-blind on OpenReview,
#   non-archival, deadline 29 August 2026 11:59pm AoE.
#
# One track, so make_newinml.py takes no --track.
# --------------------------------------------------------------------------

VENUE = {
    "workshop": "NewInML",
    "workshoptitle": "NewInML",
    "slug": "icomp_newinml",
    "authors": "newinml_authors.tex",
    "submit": "https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/NewInML",
    "tracks": {
        "workshop": {
            "name": "workshop paper, double-blind",
            "option": "dblblindworkshop",
            "why": "NewInML reviews double-blind and asks for full anonymization, "
                   "so the track is\n% dblblindworkshop: the style file then "
                   "prints \"Anonymous Author(s)\" in place of\n% the author "
                   "block and numbers the lines for the reviewers.",
            "min": 2,
            "max": 8,
            "default": True,
        },
    },
    "notes": [
        "NewInML is open to authors who have not yet published at a top ML\n"
        "conference (NeurIPS, ICML, ICLR). Check that before submitting.",
    ],
}


if __name__ == "__main__":
    sys.exit(_engine().main(VENUE, HERE, "newinml_article/make_newinml.py"))
