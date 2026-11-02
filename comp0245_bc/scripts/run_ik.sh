#!/bin/bash
# Model R = model P's recipe + IK-direction inputs. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
ev() {
  [ -f results50/$1_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$1.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
  [ -f results50/$1_mlp_arm_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$1.npz:mlp_arm --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
  [ -f results50/$1_tier1.csv ] || $PY evaluate.py --policy trained/$1.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 | grep -- "->"
}
for S in 0 1 2; do
  N=vR_ik; [ $S -gt 0 ] && N=vR_ik_s$S
  [ -f trained/$N.npz ] || $PY train.py --name $N --seed $S --ik --history deltas --goal --separate --reach-weight 3 --integral \
     --extra-train $D 2>&1 | grep -v "robot_desc\|pybullet build" | tail -2
  echo "== $N"; ev $N
done
