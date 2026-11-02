#!/bin/bash
# Comparison on the team's benchmark, tiers 1-3 (no obstacles), hold 5 s. Same scenes as the team's results.
SIM=../RoboEnv/.pixi/envs/default/bin/python
export OMP_NUM_THREADS=1
B=../benchmark/scenes.json
W=7
mkdir -p results/tiers
ev() { $SIM evaluate.py --policy "$1" --scenes $B --tiers 1 2 3 --workers $W --max-time 90 --hold 5 --plots 0 --out results/tiers > "results/tiers/$2.txt" 2>&1; }
ev expert expert
ev trained/vR_ik.npz R_alone
ev planner planner
ev planner:trained/vR_ik.npz planner_R
ev driverbrake:trained/driver_A.npz brake_A
ev driverbrake:trained/driver_A_s1.npz brake_A_s1
$SIM eval_selfnav.py --mode cam --scenes $B --tiers 1 2 3 --n 50 --workers $W --out results/tiers > results/tiers/cam_planner.txt 2>&1
