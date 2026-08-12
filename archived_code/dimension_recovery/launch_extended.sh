#!/bin/sh
# Long-budget reruns of the controls, to settle whether they are counterexamples or
# right-censored observations. 120000 steps is 6x the original budget and 10x the
# longest observed memorisation-to-generalisation gap (12070).
#
# Two threads per process so seven fit on this machine without contending; the training
# loop is deterministic given the seed, so thread count does not change the trajectory.
#
#     sh launch_extended.sh            # about 90 minutes wall clock
set -e
OUT="$(dirname "$0")/results/extended"
TRAIN="$(dirname "$0")/../grokking_train/train.py"
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

run() {                       # run <tag> <base> <steps> <seed> [extra --set args]
  tag=$1; base=$2; steps=$3; seed=$4; shift 4
  python "$TRAIN" "$base" --outdir "$OUT" --force --quiet \
      --set max_steps="$steps" --set seed="$seed" --set init_seed="$seed" \
      --set csv="${tag}_train.csv" "$@" > "$OUT/${tag}.log" 2>&1 &
  echo "launched $tag (pid $!)"
}

# the counterexample under scrutiny: three seeds, the longest budget
run lowdata15_s0 mod_wd1 120000 0 --set fraction=0.15
run lowdata15_s1 mod_wd1 120000 1 --set fraction=0.15
run lowdata15_s2 mod_wd1 120000 2 --set fraction=0.15
# the censored one
run lowdata20_s0 mod_wd1 120000 0 --set fraction=0.20
# WD=0: does its dimension fall late, and does it ever generalise?
run wd0_s0 mod_wd0 120000 0
run wd0_s1 mod_wd0 120000 1
# positive control: if this stops grokking, nothing above means anything
run grokpos_s0 mod_wd1 120000 0

wait
echo "all done"
