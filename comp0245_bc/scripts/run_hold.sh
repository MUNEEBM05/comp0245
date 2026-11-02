#!/bin/bash
# Hold-still + extreme-height data, then baselines with the hold rule. Safe to re-run after a restart.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
while pgrep -f "[v]K_clean" > /dev/null; do sleep 20; done
echo "== recording: arrive then hold 5 s"
$PY collect.py --start 60000 --end 60200 --hold 5 --out data/runs_hold --workers 2 | tail -1
echo "== recording: extreme heights, with hold"
$PY collect.py --start 61000 --end 61150 --hold 5 --z-bands 0.15 0.3 0.95 1.1 --out data/runs_extreme --workers 2 | tail -1
echo "== baselines with the hold-5s rule"
[ -f results50/expert_hold5_tier1.csv ] || $PY evaluate.py --policy expert --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
[ -f results50/vK_kick_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vK_kick.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
