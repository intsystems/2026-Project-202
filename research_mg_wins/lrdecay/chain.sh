#!/bin/bash
cd /c/Users/karlo/notebooks/grokking_prediction_original/2026-Project-202/research_mg_wins/lrdecay
until [ "$(grep -c 'none' pilot.log)" -ge 3 ] || grep -q Traceback pilot.log; do sleep 10; done
date > main_started.txt
python -u main.py all > main.log 2>&1
