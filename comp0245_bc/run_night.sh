#!/bin/bash
# Overnight, after the noise-cancelling queue: PCA models, then resume the input ablation. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
while pgrep -f "[r]un_cancel.sh|[r]un_robust.sh" > /dev/null; do sleep 30; done
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
echo "== PCA: model R trained on the first 25 / 15 principal components (95% / 80% of input variance)"
for K in 25 15; do
  N=vR_pca$K
  [ -f trained/$N.npz ] || $PY train.py --name $N --pca $K --ik --history deltas --goal --separate --reach-weight 3 --integral \
     --extra-train $D 2>&1 | grep -v "robot_desc\|pybullet build" | tail -2
  [ -f results50/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$N.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->" | sed "s/^/$N: /"
done
echo "== resume input ablation"
./run_refine.sh
