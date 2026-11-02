#!/bin/bash
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
[ -f trained/vP_hold.npz ] || $PY train.py --name vP_hold --history deltas --goal --separate --reach-weight 3 --integral \
   --extra-train data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme 2>&1 | grep -v "robot_desc\|pybullet build" | tail -3
[ -f results50/vP_hold_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vP_hold.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
[ -f results50/vP_hold_tier1.csv ] || $PY evaluate.py --policy trained/vP_hold.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 | grep -- "->"
