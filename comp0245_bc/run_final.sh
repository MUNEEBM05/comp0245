#!/bin/bash
# Final system = R + Kalman (kfv) + navigation layer, on everything. Then a 120 s door side check.
# Success = held 5 s within 1 cm, 50 tier-1 scenes. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1 NAV_STUCK=0
PY=/tmp/sim/bin/python
P=trained/vR_ik.npz:kfv:nav
N=vR_ik_kfv_nav
E="--n 50 --workers 2 --out results50 --plots 0 --hold 5"
run() { local f=results50/${N}${1:+_$1}_hold5_tier$2.csv; [ -f $f ] || $PY evaluate.py --policy $P --tiers $2 $E $3 | grep -- "->" | sed "s/^/$4 /"; }
echo "== final system  $(date +%H:%M)"
run objects4 1 "--set objects=4" door
run objects3 1 "--set objects=3" box
run "" 1 "" normal
run obs_noise0.01 1 "--set obs_noise=0.01" noise0.01
run obs_noise0.005 1 "--set obs_noise=0.005" noise0.005
run "" 5 "" tier5
run "" 7 "" tier7
echo "== door 120 s side check  $(date +%H:%M)"
[ -f results50_t120/${N}_objects4_hold5_tier1.csv ] || OBJ_TIME=120 $PY evaluate.py --policy $P --tiers 1 --n 50 --workers 2 --out results50_t120 --plots 0 --hold 5 --set objects=4 | grep -- "->" | sed "s/^/door120 /"
echo "== done  $(date +%H:%M)"
