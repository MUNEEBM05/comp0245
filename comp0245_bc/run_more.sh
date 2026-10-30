#!/usr/bin/env bash
# "More data" round: record ~50% more of each set (with a 3 s hold after reaching) plus arm-focused runs,
# then train and benchmark three models, each adding one change:
#   A  vK_lc_p150_s0     old + new runs, hold rows cut off   -> does more data help? (150% learning-curve point)
#   B  vK_more_hold      old + new runs, hold rows kept      -> does data inside 1 cm help?
#   C  vK_more_hold_arm  B + arm-focused runs (arm net only) -> does targeted reach data help?
# Only new folders are written (data/*_more, data/runs_arm, trained_lc/, results_lc/). Resumable: rerun to continue.
set -euo pipefail
cd "$(dirname "$0")"
ML=RoboEnv/.pixi/envs/ml/bin/python
W=3

python collect_more.py --start 300   --end 450   --out data/runs_more        --workers $W
python collect_more.py --start 10150 --end 10225 --noise 0.05 0.1 0.15 --out data/runs_nudged_more --workers $W
python collect_more.py --start 20120 --end 20180 --noise 0.05 0.1 --nudge-when reach --out data/runs_reach_more --workers $W
python collect_more.py --start 30120 --end 30180 --kick 0.02 --out data/runs_kick_more --workers $W
python collect_more.py --start 50000 --end 50200 --scene arm --out data/runs_arm --workers $W

K="--history deltas --goal --separate --reach-weight 3 --integral"
EXTRA="data/runs_nudged data/runs_reach data/runs_kick data/runs_more data/runs_nudged_more data/runs_reach_more data/runs_kick_more"

train_eval () {   # name, extra train_dagger options...
    local name=$1; shift
    [ -f trained_lc/$name.npz ] || $ML train_dagger.py --name $name --out-dir trained_lc $K --extra-train $EXTRA "$@"
    [ -f results_lc/${name}_tier1.csv ] || python evaluate.py --policy trained_lc/$name.npz --tiers 1 --n 50 --workers $W --out results_lc
}
train_eval vK_lc_p150_s0 --cut-hold
train_eval vK_more_hold
train_eval vK_more_hold_arm --arm-only data/runs_arm

python plot_curves.py
