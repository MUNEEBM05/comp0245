# Abdullah's code: Driver A, the A\* planner tests, and model C

This folder holds the scripts Abdullah used to **record data and train** his models, exactly as he wrote
them (only their folder paths were updated when the repo was tidied). Run every command from `comp0245_bc`
(see the main README for setup).

**Testing** his models does not need this folder: the main `evaluation/evaluate.py` gives identical results
(checked scene by scene). His models are in `trained/`, his results in `results/1_main_models/abdullah_model_C/`
and `results/2_obstacles/abdullah_bases/`, his figures in `figures/abdullah_bases/` and `figures/abdullah_model_C/`.

## Why his code is kept separate

His scripts were written against earlier versions of the shared modules, which have changed since:

- **LiDAR height.** His `sensors.py` puts the LiDAR at 1.0 m; the main code uses 0.30 m (changed when his
  navigation was merged). Recording teacher data with the main code would give different data.
- His `train_dagger.py` needs `integral_names()` and a `taus` option of `run_integral` that the main
  `features.py` does not have; his `train_driver.py` uses an older `train_mlp`.
- `collect_more.py`, `dagger_collect.py` and `record_arm.py` use older versions of `run_episode` and `rollout`.

So each subfolder keeps its own copies of the modules it needs, and they are imported from there.

| Folder | What |
|---|---|
| `driver_A/` | obstacles, LiDAR, learned driving, camera (his branch's `autonomy/`; its own README describes every file and his full result tables) |
| `model_C/` | model C, the learning curve and his DAgger (his branch's `baseline/`, only the files his scripts need) |

The `.sh` files in both folders are records of the commands he ran, with the paths of his own folder layout
(`../data`, `../RoboEnv`); the commands below are the up-to-date versions.

## Driver A and the off-spot arm (`driver_A/`)

```bash
# record the teacher (A* planner + expert arm) through obstacle scenes
python3 abdullah_code/driver_A/record_teacher.py --out data/auto_teacher --per-scenario 60 --workers 3
python3 abdullah_code/driver_A/record_teacher.py --out data/auto_teacher_nudged --per-scenario 20 --nudge --workers 3
python3 abdullah_code/driver_A/record_teacher.py --out data/auto_teacher_val --per-scenario 8 --block val --workers 3
# train the wheel networks (needs -e ml); they are saved to trained/
python3 abdullah_code/driver_A/train_driver.py --name my_driver_A                    # Driver A
python3 abdullah_code/driver_A/train_driver.py --name my_driver_B --no-waypoint      # Driver B
python3 abdullah_code/driver_A/train_driver.py --name my_driver_A_p50 --fraction 0.5 # less data
# the off-spot arm (vS_arm): arm recordings from where the robot really parks, then R's recipe
python3 abdullah_code/driver_A/record_arm.py --out data/auto_arm --per-scenario 40 --workers 4
python3 abdullah_code/driver_A/record_arm.py --out data/auto_arm_nudged --per-scenario 20 --nudge --workers 4
# test without true position (camera, odometry, search)
python3 abdullah_code/driver_A/eval_selfnav.py --mode cam --drive trained/driver_A.npz --arm trained/vR_ik.npz --brake \
        --scenes scenes_obstacles/obstacles_test.json --workers 4
# his figures and results summary
python3 abdullah_code/driver_A/make_figures.py
```

## Model C, the learning curve and DAgger (`model_C/`)

```bash
# record ~50% more of each data set (with a 3 s hold after reaching) plus arm-focused runs
python3 abdullah_code/model_C/collect_more.py --start 300 --end 450 --out data/runs_more --workers 3
python3 abdullah_code/model_C/collect_more.py --start 10150 --end 10225 --noise 0.05 0.1 0.15 --out data/runs_nudged_more --workers 3
python3 abdullah_code/model_C/collect_more.py --start 20120 --end 20180 --noise 0.05 0.1 --nudge-when reach --out data/runs_reach_more --workers 3
python3 abdullah_code/model_C/collect_more.py --start 30120 --end 30180 --kick 0.02 --out data/runs_kick_more --workers 3
python3 abdullah_code/model_C/collect_more.py --start 50000 --end 50200 --scene arm --out data/runs_arm --workers 3
# train model C (needs -e ml); saved to trained/
K="--history deltas --goal --separate --reach-weight 3 --integral"
E="data/runs_nudged data/runs_reach data/runs_kick data/runs_more data/runs_nudged_more data/runs_reach_more data/runs_kick_more"
python3 abdullah_code/model_C/train_dagger.py --name my_C $K --extra-train $E --arm-only data/runs_arm
# the learning curve (K's recipe on 25 / 50 / 75% of the runs) and its figure
python3 abdullah_code/model_C/learning_curve.py --fractions 0.25 0.5 0.75 --workers 3
python3 abdullah_code/model_C/plot_curves.py
# DAgger: the model drives, the expert labels every state
python3 abdullah_code/model_C/dagger_collect.py --policy trained/vK_kick.npz --start 40000 --end 40100 --workers 3 --out data/runs_dagger1
# test model C on his 30-scene fresh test (tiers 1-4)
python3 evaluation/evaluate.py --policy trained/vK_more_hold_arm.npz --tiers 1 2 3 4 --scenes benchmark/abdullah_new_test_scenes.json --workers 4
```

`train_dagger.py` never overwrites an existing model: pick a new `--name`.
