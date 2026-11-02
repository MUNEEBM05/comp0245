#!/bin/bash
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
for P in expert trained/vR_ik.npz trained/vP_hold.npz trained/vK_kick.npz; do
  N=$(basename $P .npz)
  [ -f results_fresh/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 --workers 2 --scenes benchmark/scenes_fresh.json --out results_fresh --plots 0 --hold 5 | grep -- "->"
  [ -f results_fresh/${N}_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 --workers 2 --scenes benchmark/scenes_fresh.json --out results_fresh --plots 0 | grep -- "->"
done
