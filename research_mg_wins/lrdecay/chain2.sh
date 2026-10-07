#!/bin/bash
cd /c/Users/karlo/notebooks/grokking_prediction_original/2026-Project-202/research_mg_wins/lrdecay
until grep -q "ALL DONE" main.log; do sleep 20; done
date > extra_started.txt
python -u main.py extra 2 > extra.log 2>&1
