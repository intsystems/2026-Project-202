until grep -q "^done" main.log; do sleep 20; done
python analyse.py > results/analyse_main.txt 2>&1
python -u diag.py > diag.log 2>&1
