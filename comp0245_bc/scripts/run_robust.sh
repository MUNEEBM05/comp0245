#!/bin/bash
# Teach R the messy world: messy expert recordings + input-noise augmentation, then the Stage B table again.
# Idempotent: re-run after a restart and it carries on.
cd /home/claude/comp0245_bc
export ROBOENV_HEADLESS=1
PY=/tmp/sim/bin/python
D="data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme"
echo "== record: expert in a messy world (delay 0-1 ms, motor noise 0-2 Nm, hand payload 0-0.5 kg, hold 5 s)"
$PY collect.py --start 80000 --end 80300 --hold 5 --delay 0 1 --motor-noise 0 1 2 --payload 0 0.25 0.5 \
    --friction 0.8 1.2 --mass 0.9 1.1 --out data/runs_messy --workers 2 | tail -1
COMMON="--ik --history deltas --goal --separate --reach-weight 3 --integral --aug-noise"
[ -f trained/vT_aug.npz ] || $PY train.py --name vT_aug $COMMON --extra-train $D 2>&1 | grep -v "robot_desc\|pybullet build" | tail -2
[ -f trained/vT_robust.npz ] || $PY train.py --name vT_robust $COMMON --extra-train $D data/runs_messy 2>&1 | grep -v "robot_desc\|pybullet build" | tail -2
sb() {  # sb <model> <batches...>: Stage B batches, success = reached and held 5 s
  P=trained/$1.npz; N=$1; shift
  for S in "$@"; do
    case $S in
      clean) F=results50/${N}_hold5_tier1.csv; ARGS="--tiers 1";;
      tier5|tier7) F=results50/${N}_hold5_$S.csv; ARGS="--tiers ${S#tier}";;
      *) K=${S%%=*}; V=${S##*=}; F=results50/${N}_${K}${V}_hold5_tier1.csv; ARGS="--tiers 1 --set $S";;
    esac
    [ -f $F ] || $PY evaluate.py --policy $P $ARGS --n 50 --workers 2 --out results50 --plots 0 --hold 5 | grep -- "->" | sed "s/^/$N $S /"
  done
}
sb vT_robust clean obs_noise=0.01 delay_steps=2 torque_noise=2 payload_kg=0.5 tier5 tier7
sb vT_aug clean obs_noise=0.01
echo "== done"
