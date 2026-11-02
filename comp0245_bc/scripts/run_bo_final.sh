#!/bin/bash
# LAST RUN (31 Oct night). BO winner (bo_t03) + Kalman (kfv) + nav, same tests as the R final system.
# Rule fixed BEFORE results: BO replaces R only if fresh >= 86% AND no other test > 2 points below R's final system.
# Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1 NAV_STUCK=0
PY=/tmp/sim/bin/python
P=trained/bo_t03.npz:kfv:nav
N=bo_t03_kfv_nav
E="--n 50 --workers 2 --out results50 --plots 0 --hold 5"
run() { local f=results50/${N}${1:+_$1}_hold5_tier$2.csv; [ -f $f ] || $PY evaluate.py --policy $P --tiers $2 $E $3 | grep -- "->" | sed "s/^/$4 /"; }
echo "== BO final system  $(date +%H:%M)"
[ -f results_fresh/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 --hold 5 \
    --scenes benchmark/scenes_fresh.json --out results_fresh --workers 2 --plots 0 | grep -- "->" | sed "s/^/fresh /"
run "" 1 "" normal
run obs_noise0.01 1 "--set obs_noise=0.01" noise0.01
run obs_noise0.005 1 "--set obs_noise=0.005" noise0.005
run objects3 1 "--set objects=3" box
run "" 5 "" tier5
run "" 7 "" tier7
run objects4 1 "--set objects=4" door
echo "== done  $(date +%H:%M)"
