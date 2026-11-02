#!/bin/bash
# Cancelling noise, 3 ways: (1) sensor filter, (2) smoother networks (weight decay / dropout), (3) ensemble.
# Success = reached and held 5 s, tier-1 scenes. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
while pgrep -f "[r]un_robust.sh" > /dev/null; do sleep 30; done
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
ev() {  # ev <policy spec> <name> <batches...>
  P=$1; N=$2; shift 2
  for S in "$@"; do
    if [ $S = clean ]; then F=results50/${N}_hold5_tier1.csv; ARGS=""; else K=${S%%=*}; V=${S##*=}; F=results50/${N}_${K}${V}_hold5_tier1.csv; ARGS="--set $S"; fi
    [ -f $F ] || $PY evaluate.py --policy "$P" --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 $ARGS | grep -- "->" | sed "s/^/$N $S /"
  done
}
echo "== (1) sensor filter on model R"
ev trained/vR_ik.npz:filt10noqd vR_ik_filt10noqd clean obs_noise=0.01 delay_steps=2
ev trained/vR_ik.npz:filt20noqd vR_ik_filt20noqd obs_noise=0.01
ev trained/vR_ik.npz:filt20base vR_ik_filt20base obs_noise=0.01
echo "== (3) ensemble of R's 3 training runs"
E=ens:trained/vR_ik.npz,trained/vR_ik_s1.npz,trained/vR_ik_s2.npz
ev $E ens_vR_ik+vR_ik_s1+vR_ik_s2 clean obs_noise=0.01
ev $E:filt10noqd ens_vR_ik+vR_ik_s1+vR_ik_s2_filt10noqd obs_noise=0.01
echo "== (2) smoother networks"
COMMON="--ik --history deltas --goal --separate --reach-weight 3 --integral --extra-train $D"
[ -f trained/vU_wd.npz ] || $PY train.py --name vU_wd --weight-decay 0.01 $COMMON 2>&1 | grep -v "robot_desc\|pybullet build" | tail -1
[ -f trained/vU_drop.npz ] || $PY train.py --name vU_drop --dropout 0.1 $COMMON 2>&1 | grep -v "robot_desc\|pybullet build" | tail -1
ev trained/vU_wd.npz vU_wd clean obs_noise=0.01
ev trained/vU_drop.npz vU_drop clean obs_noise=0.01
echo "== done"
