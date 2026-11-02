#!/bin/bash
# Kalman filter tests: model R with a Kalman filter on the sensors instead of the 20 ms moving average.
# kf  = filtered positions, raw joint speeds (like filt20noqd); kfv = filtered positions AND joint speeds.
# Both on the two sensor-noise levels; the better one at noise 0.01 then also on clean, tier 5 and tier 7.
# Success = held 5 s within 1 cm, 50 tier-1 scenes. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
E="--n 50 --workers 2 --out results50 --plots 0 --hold 5"
while pgrep -f "run_nav[.]sh" > /dev/null; do sleep 30; done
for V in kf kfv; do
  for S in obs_noise=0.01 obs_noise=0.005; do
    K=${S%%=*}; X=${S##*=}
    [ -f results50/vR_ik_${V}_${K}${X}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vR_ik.npz:$V --tiers 1 $E --set $S | grep -- "->" | sed "s/^/$V $S /"
  done
done
BEST=$(python3 -c "
import json
s={v: json.load(open(f'results50/vR_ik_{v}_obs_noise0.01_hold5_summary.json'))['1']['success_rate'] for v in ('kf','kfv')}
print(max(s, key=s.get))")
echo "best at noise 0.01: $BEST"
[ -f results50/vR_ik_${BEST}_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vR_ik.npz:$BEST --tiers 1 $E | grep -- "->" | sed "s/^/$BEST clean /"
for T in 5 7; do
  [ -f results50/vR_ik_${BEST}_hold5_tier$T.csv ] || $PY evaluate.py --policy trained/vR_ik.npz:$BEST --tiers $T $E | grep -- "->" | sed "s/^/$BEST tier$T /"
done
echo "== done"
