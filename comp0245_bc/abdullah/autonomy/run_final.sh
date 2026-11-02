#!/bin/bash
# Final tests on the frozen obstacle test scenes. Part 1 (no true position) runs while the new arm trains; part 2 waits for it.
SIM=../RoboEnv/.pixi/envs/default/bin/python
ML=../RoboEnv/.pixi/envs/ml/bin/python
export OMP_NUM_THREADS=1
T=scenes/obstacles_test.json
sn() { name=$1; shift; $SIM eval_selfnav.py --scenes $T --out results/stage6 "$@" > results/stage6/test_$name.txt 2>&1; }
ev() { $SIM evaluate.py --policy "$1" --scenes $T --tiers 10 11 12 13 14 --workers $W --max-time 90 --hold 5 --plots 0 --out "$2" > "$3" 2>&1; }
$SIM check_camera.py --n 30 --out results/camera.json > results/camera.txt 2>&1
W=4
sn odo_planner --mode odo --workers $W
sn cam_planner --mode cam --workers $W
sn search_planner --mode search --workers $W
sn cam_A --mode cam --drive trained/driver_A.npz --brake --workers $W
sn cam_A_s1 --mode cam --drive trained/driver_A_s1.npz --brake --workers $W
sn cam_planner_R --mode cam --arm trained/vR_ik.npz --workers $W
while pgrep -f run_step5b.sh > /dev/null; do sleep 20; done
W=7
mkdir -p results/stage5
ev planner:trained/vS_arm.npz results/stage5 results/stage5_planner_vS.txt
ev planner:trained/vS_arm_s1.npz results/stage5 results/stage5_planner_vS_s1.txt
ev driverbrake:trained/driver_A.npz results/stage3 results/stage3_brake_A.txt
ev driverbrake:trained/driver_A_s1.npz results/stage3 results/stage3_brake_A_s1.txt
ev driverbrake:trained/driver_A.npz:trained/vS_arm.npz results/stage5 results/stage5_brake_A_vS.txt
ev driverbrake:trained/driver_A_s1.npz:trained/vS_arm.npz results/stage5 results/stage5_brake_A_s1_vS.txt
sn cam_planner_vS --mode cam --arm trained/vS_arm.npz --workers $W
# Learning curve: how many drives does the driver need? Validation scenes.
for F in 25 50 75; do
  [ -f trained/driver_A_p$F.npz ] || OMP_NUM_THREADS=6 $ML train_driver.py --name driver_A_p$F --fraction 0.$F > results/train_driver_A_p$F.txt 2>&1
  $SIM evaluate.py --policy driverbrake:trained/driver_A_p$F.npz --scenes scenes/obstacles_val.json --tiers 10 11 12 13 14 --n 15 --workers $W --max-time 90 --hold 5 --plots 0 --out results/dev > results/dev/brake_A_p$F.txt 2>&1
done
