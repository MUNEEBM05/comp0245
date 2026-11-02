#!/bin/bash
# Map + A* navigation vs the original Bug 2 layer, same model (R + 20 ms filter), same 50 scenes.
# All three tests (door, box, normal) because a fix can break the others. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1 NAV_STUCK=0
PY=/tmp/sim/bin/python
while pgrep -f "run_final[.]sh" > /dev/null; do sleep 30; done
P=trained/vR_ik.npz:filt20noqd:mapnav; N=vR_ik_filt20noqd_mapnav
E="--n 50 --workers 2 --out results50 --plots 0 --hold 5"
echo "== map + A*  $(date +%H:%M)"
[ -f results50/${N}_objects4_hold5_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 $E --set objects=4 | grep -- "->" | sed "s/^/door /"
[ -f results50/${N}_objects3_hold5_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 $E --set objects=3 | grep -- "->" | sed "s/^/box /"
[ -f results50/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 $E | grep -- "->" | sed "s/^/normal /"
echo "== done  $(date +%H:%M)"
