#!/bin/bash
# Tonight's queue (after the fair GRU). Idempotent: re-run after a restart and it carries on.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
while pgrep -f "[r]un_fair_gru.sh" > /dev/null; do sleep 20; done
ev() {  # ev <name>: hold-5 s test and normal test on tier 1
  [ -f results50/$1_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$1.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->"
  [ -f results50/$1_tier1.csv ] || $PY evaluate.py --policy trained/$1.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 | grep -- "->"
}
echo "== attention"
[ -f trained/attn.npz ] || $PY train_conv.py --arch attn --name attn --history deltas --goal --integral --epochs 30 --patience 5 \
   --rows-per-epoch 400000 --extra-train $D 2>&1 | grep -v "robot_desc\|pybullet build" | tail -3
ev attn
echo "== repeat runs: MLP P, seeds 1 and 2"
for S in 1 2; do
  [ -f trained/vP_hold_s$S.npz ] || $PY train.py --name vP_hold_s$S --seed $S --history deltas --goal --separate --reach-weight 3 --integral \
     --extra-train $D 2>&1 | grep -v "robot_desc\|pybullet build" | tail -1
  ev vP_hold_s$S
done
echo "== repeat runs: convolution v2, seeds 1 and 2"
for S in 1 2; do
  [ -f trained/tcn2_s$S.npz ] || $PY train_conv.py --name tcn2_s$S --seed $S --history deltas --goal --integral \
     --rows-per-epoch 600000 --extra-train $D 2>&1 | grep -v "robot_desc\|pybullet build" | tail -1
  ev tcn2_s$S
done
echo "== done"
