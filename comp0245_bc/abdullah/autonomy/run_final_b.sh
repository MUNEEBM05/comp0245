#!/bin/bash
# Rest of the final tests, replacing the end of run_final.sh. vS was no better than R on validation, so A + vS and vS seed 1 are
# not run on the test scenes; the fully learned setups (A + R, with brake and with the camera) are added.
SIM=../RoboEnv/.pixi/envs/default/bin/python
ML=../RoboEnv/.pixi/envs/ml/bin/python
export OMP_NUM_THREADS=1
T=scenes/obstacles_test.json
W=7
sn() { name=$1; shift; $SIM eval_selfnav.py --scenes $T --out results/stage6 "$@" > results/stage6/test_$name.txt 2>&1; }
ev() { $SIM evaluate.py --policy "$1" --scenes $T --tiers 10 11 12 13 14 --workers $W --max-time 90 --hold 5 --plots 0 --out "$2" > "$3" 2>&1; }
while pgrep -f "evaluate.py --policy planner:trained/vS_arm.npz" > /dev/null; do sleep 20; done
ev driverbrake:trained/driver_A.npz results/stage3 results/stage3_brake_A.txt
ev driverbrake:trained/driver_A_s1.npz results/stage3 results/stage3_brake_A_s1.txt
ev driverbrake:trained/driver_A.npz:trained/vR_ik.npz results/stage5 results/stage5_brake_A_R.txt
ev driverbrake:trained/driver_A_s1.npz:trained/vR_ik.npz results/stage5 results/stage5_brake_A_s1_R.txt
sn cam_A_R --mode cam --drive trained/driver_A.npz --arm trained/vR_ik.npz --brake --workers $W
sn cam_A_s1_R --mode cam --drive trained/driver_A_s1.npz --arm trained/vR_ik.npz --brake --workers $W
sn cam_planner_vS --mode cam --arm trained/vS_arm.npz --workers $W
# Learning curve: how many drives does the driver need? Validation scenes.
for F in 25 50 75; do
  [ -f trained/driver_A_p$F.npz ] || $ML train_driver.py --name driver_A_p$F --fraction 0.$F > results/train_driver_A_p$F.txt 2>&1
  $SIM evaluate.py --policy driverbrake:trained/driver_A_p$F.npz --scenes scenes/obstacles_val.json --tiers 10 11 12 13 14 --n 15 --workers $W --max-time 90 --hold 5 --plots 0 --out results/dev > results/dev/brake_A_p$F.txt 2>&1
done
