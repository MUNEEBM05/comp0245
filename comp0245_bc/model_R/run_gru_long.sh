#!/bin/bash
# Idempotent: re-run after a restart and it continues (training resumes from its checkpoint).
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
if [ ! -f trained/gru_full_long.npz ]; then
  $PY train_gru.py --name gru_full_long --history deltas --goal --integral --epochs 60 --windows-per-epoch 10000 \
     --window 200 --burn-in 50 --extra-train data/runs_nudged data/runs_reach data/runs_kick 2>&1 | grep -v robot_desc
fi
[ -f results50/gru_full_long_tier1.csv ] || \
  $PY evaluate.py --policy trained/gru_full_long.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 | grep -- "->"
