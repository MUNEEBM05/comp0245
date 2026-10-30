#!/usr/bin/env bash
# DAgger round 2: re-synced labels on the final approach, biased to low / far targets.
#   D1  vK_C_dagger2         C's recipe            + DAgger runs driven by C
#   D2  vK_C_long_mem_dagger2 C + arm-only long memory + DAgger runs driven by that model
# Each model learns from the states IT gets into (DAgger). Only the rows with the arm reaching and
# the hand within 5 cm are added. Only new folders/names are written. Resumable: rerun to continue.
set -euo pipefail
cd "$(dirname "$0")"
ML=RoboEnv/.pixi/envs/ml/bin/python
PY=RoboEnv/.pixi/envs/default/bin/python
W=3

$PY dagger_collect.py --policy trained_lc/vK_more_hold_arm.npz --start 60000 --end 60150 --resync --scene edge \
    --max-time 30 --workers $W --out data/runs_dagger2_C
$PY dagger_collect.py --policy trained_lc/vK_C_long_mem_arm.npz --start 61000 --end 61150 --resync --scene edge \
    --max-time 30 --workers $W --out data/runs_dagger2_Lmem

K="--history deltas --goal --separate --reach-weight 3 --integral"
EXTRA="data/runs_nudged data/runs_reach data/runs_kick data/runs_more data/runs_nudged_more data/runs_reach_more data/runs_kick_more"

train_eval () {   # name, extra train_dagger options...
    local name=$1; shift
    [ -f trained_lc/$name.npz ] || $ML train_dagger.py --name $name --out-dir trained_lc $K --extra-train $EXTRA \
        --arm-only data/runs_arm --dagger-radius 0.05 "$@"
    [ -f results_lc/${name}_tier1.csv ] || $PY evaluate.py --policy trained_lc/$name.npz --tiers 1 --n 50 --workers $W --plots 0 --out results_lc
    [ -f random_test/$name/${name}_tier1.csv ] || $PY evaluate.py --policy trained_lc/$name.npz --tiers 1 --workers $W --plots 30 \
        --scenes random_test/scenes.json --out random_test/$name
}
train_eval vK_C_dagger2 --dagger data/runs_dagger2_C
train_eval vK_C_long_mem_dagger2 --dagger data/runs_dagger2_Lmem --integral-taus 0.5 2 5 10 --wheel-taus 0.5 2
