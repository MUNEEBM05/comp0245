#!/bin/bash
# Stage B benchmark: one disturbance at a time (tier-1 scenes), then physics (tier 5) and everything mixed (tier 7).
# Success = reached AND held within 1 cm for 5 s. Idempotent: re-run after a restart and it carries on.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
for P in trained/vR_ik.npz expert; do
  N=$(basename $P .npz)
  for S in obs_noise=0.01 delay_steps=2 torque_noise=2 payload_kg=0.5; do
    K=${S%%=*}; V=${S##*=}
    [ -f results50/${N}_${K}${V}_hold5_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 --set $S | grep -- "->" | sed "s/^/$N $S /"
  done
  for T in 5 7; do
    [ -f results50/${N}_hold5_tier$T.csv ] || $PY evaluate.py --policy $P --tiers $T --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->" | sed "s/^/$N tier$T /"
  done
done
echo "== done"
