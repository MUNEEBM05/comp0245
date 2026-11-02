#!/bin/bash
# Objects test (Valerio's suggestion): 50 normal tier-1 scenes with (1) an object at the target, (2) clutter that is
# not in the way, (3) one box in the path. Expert vs model R (+20 ms filter). Held 5 s within 1 cm. Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
for K in 1 2 3; do
  for P in expert trained/vR_ik.npz:filt20noqd; do
    N=$( [ $P = expert ] && echo expert || echo vR_ik_filt20noqd )
    [ -f results50/${N}_objects${K}_hold5_tier1.csv ] || $PY evaluate.py --policy $P --tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5 --set objects=$K | grep -- "->" | sed "s/^/objects$K $N /"
  done
done
echo "== done"
