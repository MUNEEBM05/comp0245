#!/bin/bash
# Steps 3 and 4: test the planner and the learned drivers on the frozen obstacle test scenes, with the 5 s hold.
SIM=../RoboEnv/.pixi/envs/default/bin/python
ML=../RoboEnv/.pixi/envs/ml/bin/python
export OMP_NUM_THREADS=1
test() { $SIM evaluate.py --policy "$1" --scenes scenes/obstacles_test.json --tiers 10 11 12 13 14 --workers 6 --max-time 90 --hold 5 --plots 0 --out "$2" 2>&1; }
OMP_NUM_THREADS=2 $ML train_driver.py --name driver_B --no-waypoint > results/train_driver_B.txt 2>&1 &
TRAIN=$!
test driver:trained/driver_A.npz results/stage3 > results/stage3_A.txt
test driver:trained/driver_A.npz:trained/vR_ik.npz results/stage3 > results/stage3_A_R.txt
wait $TRAIN
OMP_NUM_THREADS=2 $ML train_driver.py --name driver_B_s1 --no-waypoint --seed 1 > results/train_driver_B_s1.txt 2>&1 &
TRAIN=$!
test driver:trained/driver_A_s1.npz results/stage3 > results/stage3_A_s1.txt
test driver:trained/driver_A_s1.npz:trained/vR_ik.npz results/stage3 > results/stage3_A_s1_R.txt
wait $TRAIN
test planner results/stage3 > results/stage3_planner.txt
test planner:trained/vR_ik.npz results/stage3 > results/stage3_planner_R.txt
test driver:trained/driver_B.npz results/stage4 > results/stage4_B.txt
test driver:trained/driver_B_s1.npz results/stage4 > results/stage4_B_s1.txt
test driver:trained/driver_B.npz:trained/vR_ik.npz results/stage4 > results/stage4_B_R.txt
