#!/bin/bash
cd /c/Users/karlo/notebooks/grokking_prediction_original/2026-Project-202/research_mg_wins/lrdecay
until grep -q "ALL DONE" main.log && grep -q "ALL DONE" anneal_first.log; do sleep 20; done
date > anneal_started.txt
python -u anneal.py 3 > anneal.log 2>&1
