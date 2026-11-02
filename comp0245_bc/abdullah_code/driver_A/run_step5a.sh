#!/bin/bash
# Step 5 data + validation of the parked brake. Waits for the stage 3-4 queue to finish first.
SIM=../RoboEnv/.pixi/envs/default/bin/python
export OMP_NUM_THREADS=1
while pgrep -f run_stage34.sh > /dev/null; do sleep 20; done
(
for B in driver driverbrake; do for M in driver_A driver_A_s1; do
  $SIM evaluate.py --policy $B:trained/$M.npz --scenes scenes/obstacles_val.json --tiers 10 11 12 13 14 --n 15 --workers 4 --max-time 90 --hold 5 --plots 0 --out results/dev > results/dev/${B}_$M.txt 2>&1
done; done
) &
$SIM record_arm.py --out ../data/auto_arm --per-scenario 40 --workers 4 > results/record_arm.txt 2>&1
$SIM record_arm.py --out ../data/auto_arm_nudged --per-scenario 20 --nudge --workers 4 > results/record_arm_nudged.txt 2>&1
$SIM collect.py --start 60000 --end 60200 --hold 5 --out ../data/runs_hold --workers 4 > results/collect_hold.txt 2>&1
$SIM collect.py --start 61000 --end 61150 --hold 5 --z-bands 0.15 0.3 0.95 1.1 --out ../data/runs_extreme --workers 4 > results/collect_extreme.txt 2>&1
wait
