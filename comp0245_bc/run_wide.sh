#!/bin/bash
# Objects anywhere. Idempotent: re-run after a restart and it carries on.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
while pgrep -f "[e]valuate.py --policy trained/vR_ik.npz --tiers [23]|[e]valuate.py --policy expert --tiers 2 3" > /dev/null; do sleep 20; done
echo "== record: objects anywhere (300 drives, hold 5 s, calm give-up kept)"
$PY collect.py --start 70000 --end 70300 --wide --hold 5 --out data/runs_wide --workers 2 | tail -1
t6() {  # t6 <policy> <name>
  [ -f results50/$2_hold5_tier6.csv ] || $PY evaluate.py --policy $1 --tiers 6 --n 50 --workers 2 --out results50 --plots 0 --hold 5 --max-time 60 | grep -- "->"
}
t6 expert expert
t6 trained/vR_ik.npz vR_ik
for N in vS_rel vS_wide; do
  EXTRA=""; [ $N = vS_wide ] && EXTRA="data/runs_wide --keep-unreachable"
  [ -f trained/$N.npz ] || $PY train.py --name $N --relative --ik --history deltas --goal --separate --reach-weight 3 --integral \
     --extra-train $D $EXTRA 2>&1 | grep -v "robot_desc\|pybullet build" | tail -1
  [ -f results50/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$N.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
  t6 trained/$N.npz $N
done
echo "== done"
