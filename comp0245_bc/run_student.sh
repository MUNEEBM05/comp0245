#!/bin/bash
# LiDAR student: (1) score the new teacher (expert + navigation layer) on box and door,
# (2) record the teacher on 240 training scenes with obstacles (seeds 90000-90239),
# (3) train model R's recipe + 24 LiDAR inputs (vS_lidar) on R's data + the new recordings,
# (4) test the student on box, door and normal scenes (50 each, held 5 s). Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
E="--tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5"
while pgrep -f "run_nav[.]sh|run_kf[.]sh" > /dev/null; do sleep 30; done
echo "== teacher (expert + nav)  $(date +%H:%M)"
for K in 3 4; do
  [ -f results50/expert_nav_objects${K}_hold5_tier1.csv ] || $PY evaluate.py --policy expert:nav $E --set objects=$K | grep -- "->" | sed "s/^/objects$K expert+nav /"
done
echo "== recording  $(date +%H:%M)"
$PY record_nav.py --start 90000 --end 90240 --out data/runs_navobs --workers 2 2>&1 | grep "^\[" | tail -1
echo "== training  $(date +%H:%M)"
[ -f trained/vS_lidar.npz ] || $PY train.py --name vS_lidar --lidar --ik --history deltas --goal --separate --reach-weight 3 --integral \
    --extra-train data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme data/runs_navobs 2>&1 | grep -v "robot_desc\|pybullet build" | tail -3
echo "== testing student  $(date +%H:%M)"
for K in 3 4; do
  [ -f results50/vS_lidar_filt20noqd_objects${K}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vS_lidar.npz:filt20noqd $E --set objects=$K | grep -- "->" | sed "s/^/objects$K student /"
done
[ -f results50/vS_lidar_filt20noqd_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vS_lidar.npz:filt20noqd $E | grep -- "->" | sed "s/^/normal student /"
for S in obs_noise=0.005 obs_noise=0.01; do
  K=${S%%=*}; X=${S##*=}
  [ -f results50/vS_lidar_filt20noqd_${K}${X}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vS_lidar.npz:filt20noqd $E --set $S | grep -- "->" | sed "s/^/$S student /"
done
echo "== done  $(date +%H:%M)"
