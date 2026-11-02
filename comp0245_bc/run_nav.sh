#!/bin/bash
# Navigation layer test: model R (+20 ms filter) + LiDAR/VFH/Bug 2 layer, 50 tier-1 scenes, held 5 s within 1 cm.
# Box in the path (objects=3), wall with door (objects=4), and normal scenes (does the layer hurt?).
# Plus expert and plain R on the wall-with-door scene (objects 3 for them is in run_objects.sh). Idempotent.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
E="--tiers 1 --n 50 --workers 2 --out results50 --plots 0 --hold 5"
NAV=trained/vR_ik.npz:filt20noqd:nav; N=vR_ik_filt20noqd_nav
for K in 3 4; do
  [ -f results50/${N}_objects${K}_hold5_tier1.csv ] || $PY evaluate.py --policy $NAV $E --set objects=$K | grep -- "->" | sed "s/^/objects$K R+nav /"
done
[ -f results50/expert_objects4_hold5_tier1.csv ] || $PY evaluate.py --policy expert $E --set objects=4 | grep -- "->" | sed "s/^/objects4 expert /"
[ -f results50/vR_ik_filt20noqd_objects4_hold5_tier1.csv ] || $PY evaluate.py --policy trained/vR_ik.npz:filt20noqd $E --set objects=4 | grep -- "->" | sed "s/^/objects4 R /"
[ -f results50/${N}_hold5_tier1.csv ] || $PY evaluate.py --policy $NAV $E | grep -- "->" | sed "s/^/normal R+nav /"
echo "== done"
