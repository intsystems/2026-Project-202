#!/bin/bash
# waits for the main grid, then computes window statistics and runs the analysis
cd "$(dirname "$0")"
until [ $(grep -c " s[0-9]* " main.log) -ge 145 ]; do sleep 30; done
python -u features.py runs_main feats_main > features.log 2>&1
python -u analyze.py runs_main feats_main results_main > analyze.log 2>&1
echo done >> analyze.log
