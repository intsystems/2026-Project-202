#!/bin/bash
cd /c/Users/karlo/notebooks/grokking_prediction_original/2026-Project-202/research_mg_wins/lrdecay
until grep -q "ALL DONE" anneal_first.log; do sleep 20; done
python -u anneal.py 1 > anneal_b.log 2>&1
