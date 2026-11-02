#!/bin/bash
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
for HW in 3 6; do
  N=vQ_hw$HW
  [ -f trained/$N.npz ] || $PY train.py --name $N --history deltas --goal --separate --reach-weight 3 --integral --hold-weight $HW \
     --extra-train data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme 2>&1 | grep -v "robot_desc\|pybullet build" | tail -2
  [ -f results50/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$N.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
  [ -f results50/${N}_tier1.csv ] || $PY evaluate.py --policy trained/$N.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 | grep -- "->"
done
