#!/bin/bash
# Step 5: new arm models, R's recipe + recordings parked off the spot. Fills the IK cache first; then trains two seeds, one at a time, and tests on the obstacle scenes.
SIM=../RoboEnv/.pixi/envs/default/bin/python
ML=../RoboEnv/.pixi/envs/ml/bin/python
export OMP_NUM_THREADS=1
while pgrep -f run_step5a.sh > /dev/null; do sleep 20; done
D="../data/runs_nudged ../data/runs_reach ../data/runs_kick ../data/runs_hold ../data/runs_extreme ../data/auto_arm ../data/auto_arm_nudged"
for I in 0 1 2; do
  $ML precompute_ik.py --shard $I --shards 3 ../data/runs $D > results/precompute_ik_$I.txt 2>&1 &
done
wait
for S in 0 1; do
  N=vS_arm; [ $S -gt 0 ] && N=vS_arm_s$S
  [ -f trained/$N.npz ] || $ML train.py --name $N --seed $S --verbose --ik --history deltas --goal --separate --reach-weight 3 --integral --extra-train $D > results/train_$N.txt 2>&1
done
for M in vS_arm vS_arm_s1; do
  $SIM evaluate.py --policy planner:trained/$M.npz --scenes scenes/obstacles_val.json --tiers 10 11 12 13 14 --n 15 --workers 4 --max-time 90 --hold 5 --plots 0 --out results/dev > results/dev/planner_$M.txt 2>&1
done
$SIM evaluate.py --policy planner:trained/vR_ik.npz --scenes scenes/obstacles_val.json --tiers 10 11 12 13 14 --n 15 --workers 4 --max-time 90 --hold 5 --plots 0 --out results/dev > results/dev/planner_vR_ik.txt 2>&1
