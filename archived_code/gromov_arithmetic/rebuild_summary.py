"""Rebuild ``summary.json`` from the training logs in a results directory.

``run.py`` writes ``summary.json`` incrementally, merging each finished run into
whatever is already there.  That is right on the VM and wrong on the way back: a
``-Fetch`` unpacks the VM's copy over the local one, and the VM's copy only knows about
the runs *that VM* executed.  A campaign spread over two sessions therefore loses the
first session's entries the moment the second is fetched -- which happened here.

The CSVs are never lost that way, since each run has its own file, so the summary is
better treated as derived data.  This rebuilds it from every ``<key>_train.csv`` present,
taking the milestones from the log and the configuration from whichever registry knows
the key.

    python rebuild_summary.py --results ./results/arith
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_POLY = Path(__file__).resolve().parent.parent / "gromov_polynomials"
if str(_POLY) not in sys.path:
    sys.path.insert(0, str(_POLY))

import runs as arith_registry            # noqa: E402
from gromov import grok_summary          # noqa: E402

try:
    import runs_poly as poly_registry
    import polynomials as P
except ImportError:                        # the arithmetic folder can stand alone
    poly_registry = P = None


def config_for(key):
    try:
        return arith_registry.get(key)
    except KeyError:
        pass
    if poly_registry is not None:
        try:
            return poly_registry.get(key)
        except KeyError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="./results/arith")
    args = ap.parse_args()

    root = Path(args.results)
    paths = sorted(root.glob("*_train.csv"))
    if not paths:
        raise SystemExit(f"no *_train.csv under {root}")

    records = []
    for path in paths:
        key = path.stem.replace("_train", "")
        df = pd.read_csv(path)
        rows = df.to_dict("records")
        rec = grok_summary(rows)
        rec["key"] = key
        cfg = config_for(key)
        if cfg is None:
            print(f"  [{key}] not in any registry -- milestones only")
        else:
            rec.update(task=cfg.task, p=cfg.p, width=cfg.width, fraction=cfg.fraction,
                       optimizer=cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay)
            if P is not None and cfg.task in P.POLYNOMIALS:
                image = P.distinct_outputs(cfg.task, cfg.p)
                rec.update(polynomial=P.EXPRESSIONS[cfg.task],
                           learnable=P.is_learnable(cfg.task),
                           paper_test_acc=P.PAPER_TEST_ACC[(cfg.p, cfg.task)],
                           n_distinct=image["n_distinct"],
                           majority_share=image["majority_share"])
        records.append(rec)
        g = "never" if rec["t_grok"] is None else rec["t_grok"]
        print(f"  {key:<16} steps={rec['steps']:>7} t_mem={rec['t_memorise']} "
              f"t_gen={g} val={rec['final_val_acc']:.2%}")

    out = root / "summary.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {out} ({len(records)} runs)")


if __name__ == "__main__":
    raise SystemExit(main())
