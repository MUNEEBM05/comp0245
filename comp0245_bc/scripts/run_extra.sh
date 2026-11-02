#!/bin/bash
# Optional extras (31 Oct). Idempotent.
# 1) Final system (R + Kalman + navigation) on the 50 FRESH test scenes.
# 2) BO winner (trial 3 settings) vs model R, 3 training seeds each, on the 50 benchmark scenes, clean and noise 0.01,
#    each with its own filter (BO: 29 ms, R: 20 ms). Success = held 5 s within 1 cm.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1 NAV_STUCK=0
PY=/tmp/sim/bin/python
E="--n 50 --workers 2 --out results50 --plots 0 --hold 5"
echo "== fresh scenes  $(date +%H:%M)"
[ -f results_fresh/vR_ik_kfv_nav_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vR_ik.npz:kfv:nav --tiers 1 --hold 5 \
    --scenes benchmark/scenes_fresh.json --out results_fresh --workers 2 --plots 0 | grep -- "->" | sed "s/^/fresh final /"
echo "== BO seeds  $(date +%H:%M)"
DATA="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
for S in 1 2; do
  [ -f trained/bo_t03_s$S.npz ] || $PY train.py --name bo_t03_s$S --seed $S --ik --history deltas --goal --separate --integral \
      --reach-weight 2.4729661593621923 --lr 0.0007411498939805824 --batch 2048 --hidden 192 192 192 --extra-train $DATA \
      | tail -2
done
for M in bo_t03 bo_t03_s1 bo_t03_s2 vR_ik_s1 vR_ik_s2 vR_ik; do
  F=$([[ $M == bo* ]] && echo 29 || echo 20)
  [ -f results50/${M}_filt${F}noqd_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$M.npz:filt${F}noqd --tiers 1 $E | grep -- "->" | sed "s/^/$M clean /"
  [ -f results50/${M}_filt${F}noqd_obs_noise0.01_hold5_tier1.csv ] || $PY evaluate.py --policy trained/$M.npz:filt${F}noqd --tiers 1 $E --set obs_noise=0.01 | grep -- "->" | sed "s/^/$M noise0.01 /"
done
echo "== done  $(date +%H:%M)"
