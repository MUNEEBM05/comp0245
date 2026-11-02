#!/bin/bash
# SUPER TEST (1 Nov): every team system on one combined suite, one rule. Idempotent (skips finished parts).
# Rule: success = hand held within 1 cm for 5 s. Limit 40 s, or 90 s with obstacles (OBJ_TIME=90).
# Systems:  S1 expert arm + Abdullah's A* planner        S2 BO arm + Kalman + David's nav (VFH + Bug 2)
#           S3 BO arm + Kalman + Abdullah's planner      S4 BO arm + Kalman + learned Driver A (+ brake)
#           S5 S3 + camera, no true robot position (extension)
# Every navigation layer uses the same LiDAR mounting (0.30 m, see sensors.py).
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1 NAV_STUCK=0 OBJ_TIME=90
PY=/tmp/sim/bin/python
BO=trained/bo_t03.npz:kfv
SYS=("trained/bo_t03.npz:kfv:nav" "planner:$BO" "driverbrake:trained/driver_A.npz:$BO" "planner" "cam:planner:$BO")
E="--workers 2 --plots 0 --hold 5"
ev() {  # ev <spec> <outdir> <expected csv basename-suffix> <label> <args...>
  local spec=$1 out=$2 suf=$3 label=$4; shift 4
  local name; name=$($PY -c "import os,sys;sys.path.insert(0,'.');os.environ['ROBOENV_HEADLESS']='1'
from rollout import make_policy;print(make_policy('$spec').name)" 2>/dev/null | tail -1)
  [ -f $out/${name}${suf}.csv ] && return
  echo "  $(date +%H:%M) start $label  [$name]"
  $PY evaluate.py --policy "$spec" $E --out $out "$@" | grep -- "->" | sed "s/^/  $label /"
}
echo "== super test  $(date +%H:%M)"
for S in "${SYS[@]}"; do
  echo "== $S"
  ev "$S" results_super/fresh "_hold5_tier1"                    "1 fresh"      --scenes benchmark/scenes_fresh.json --tiers 1
  ev "$S" results_super/bench "_hold5_tier2"                    "1 hard-targ"  --tiers 2
  ev "$S" results_super/bench "_hold5_tier3"                    "1 awkward"    --tiers 3
  ev "$S" results_super/fresh "_obs_noise0.01_hold5_tier1"      "2 noise.01"   --scenes benchmark/scenes_fresh.json --tiers 1 --set obs_noise=0.01
  ev "$S" results_super/bench "_hold5_tier4"                    "2 noise+delay" --tiers 4
  for T in 10 11 12 13 14; do
    ev "$S" results_super/obst "_hold5_tier$T"                  "3 obst-t$T"   --scenes scenes_obstacles/obstacles_test.json --tiers $T --max-time 90
  done
  ev "$S" results_super/bench "_objects3_hold5_tier1"           "3 box"        --tiers 1 --set objects=3
  ev "$S" results_super/bench "_objects4_hold5_tier1"           "3 door"       --tiers 1 --set objects=4
done
echo "== done  $(date +%H:%M)"
