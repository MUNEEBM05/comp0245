#!/usr/bin/env bash
# Sensor-noise augmentation: C's data and recipe + one noisy copy of every training run (tier-4 noise
# model, sigma 0.005 / 0.01 / 0.02; labels stay clean). Tested on benchmark tiers 1 and 4 (50 each)
# and on the fresh 4-tier set new_test/ (30 each). Only new names are written. Resumable.
set -euo pipefail
cd "$(dirname "$0")"
ML=../RoboEnv/.pixi/envs/ml/bin/python
PY=../RoboEnv/.pixi/envs/default/bin/python
W=3
N=vK_C_noise
K="--history deltas --goal --separate --reach-weight 3 --integral"
EXTRA="../data/runs_nudged ../data/runs_reach ../data/runs_kick ../data/runs_more ../data/runs_nudged_more ../data/runs_reach_more ../data/runs_kick_more"

[ -f trained_lc/$N.npz ] || $ML train_dagger.py --name $N --out-dir trained_lc $K --extra-train $EXTRA \
    --arm-only ../data/runs_arm --noise-aug 0.005 0.01 0.02
[ -f results_lc/${N}_tier4.csv ] || $PY evaluate.py --policy trained_lc/$N.npz --tiers 1 4 --n 50 --workers $W --plots 0 --out results_lc
[ -f new_test/$N/${N}_tier4.csv ] || $PY evaluate.py --policy trained_lc/$N.npz --tiers 1 2 3 4 --workers $W --plots 30 \
    --scenes new_test/scenes.json --out new_test/$N
