#!/bin/bash
# Fair GRU: same data as model P (hold + extreme heights). Resumable.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
[ -f trained/gru_P.npz ] || $PY train_gru.py --name gru_P --history deltas --goal --integral --epochs 60 --windows-per-epoch 10000 \
   --window 200 --burn-in 50 --extra-train $D 2>&1 | grep -v "robot_desc\|pybullet build" | tail -2
[ -f results50/gru_P_hold5_tier1.csv ] || $PY evaluate.py --policy trained/gru_P.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
[ -f results50/gru_P_tier1.csv ] || $PY evaluate.py --policy trained/gru_P.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 | grep -- "->"
