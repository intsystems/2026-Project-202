#!/bin/bash
# fresh-seed confirmation (seeds 20, 21), 8 single-thread workers, then window statistics
cd "$(dirname "$0")"
export NWORK=8
python -u run_grid.py runs_confirm MAIN 20 21 > confirm.log 2>&1
python -u features.py runs_confirm feats_confirm > features_confirm.log 2>&1
echo done >> features_confirm.log
