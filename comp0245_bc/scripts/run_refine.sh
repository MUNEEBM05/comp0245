#!/bin/bash
# After Stage B: (1) model R repeat runs on the fresh scenes; (2) input ablation of model R. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
while pgrep -f "[r]un_stageB.sh" > /dev/null; do sleep 30; done
echo "== (1) fresh scenes, R seeds 1 and 2"
for N in vR_ik_s1 vR_ik_s2; do
  [ -f results_fresh/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$N.npz --tiers 1 --workers 2 --scenes benchmark/scenes_fresh.json --out results_fresh --plots 0 --hold 5 | grep -- "->" | sed "s/^/$N held: /"
  [ -f results_fresh/${N}_tier1.csv ] || $PY evaluate.py --policy trained/$N.npz --tiers 1 --workers 2 --scenes benchmark/scenes_fresh.json --out results_fresh --plots 0 | grep -- "->" | sed "s/^/$N reached: /"
done
echo "== (2) ablation: model R minus one input group (tier 1, held 5 s)"
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
abl() {  # abl <name> <drop prefixes...>
  N=$1; shift
  [ -f trained/$N.npz ] || $PY train.py --name $N --ik --history deltas --goal --separate --reach-weight 3 --integral \
     --extra-train $D --drop "$@" 2>&1 | grep -v "robot_desc\|pybullet build" | tail -1
  [ -f results50/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$N.npz --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->" | sed "s/^/$N: /"
}
abl abl_no_ikavg qd_ik_avg
abl abl_no_iknow qd_ik_1 qd_ik_2 qd_ik_3 qd_ik_4 qd_ik_5 qd_ik_6 qd_ik_7
abl abl_no_erravg err_avg
abl abl_no_history d_fwd d_left d_yaw dq
abl abl_no_goal goal_ face_err
echo "== done"
