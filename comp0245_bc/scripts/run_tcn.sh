#!/bin/bash
# Safe to re-run after a restart: training resumes from its checkpoint, finished steps are skipped.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python

[ -f trained/tcn2.npz ] || $PY train_conv.py --name tcn2 --history deltas --goal --integral \
   --rows-per-epoch 600000 --extra-train data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme 2>&1 | grep -v "robot_desc\|pybullet build"
[ -f results50/tcn2_hold5_tier1.csv ] || $PY evaluate.py --policy trained/tcn2.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
[ -f results50/tcn2_tier1.csv ] || $PY evaluate.py --policy trained/tcn2.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 | grep -- "->"
