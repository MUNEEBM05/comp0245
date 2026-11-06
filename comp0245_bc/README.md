# COMP0245 behaviour cloning: a mobile manipulator learns to reach

A Robotnik four-wheeled base carrying a seven-joint Kinova arm, simulated in PyBullet. The course template
provides an expert controller. We record the expert and train neural networks to copy it: every 1 ms the
learned controller maps the robot's measured state and the target position to 7 arm torques and 4 wheel
speeds. A trial succeeds when the hand comes within **1 cm** of the target and **stays there for 5 s**.

Team: Muneeb Musharaf, David Wild, Abdullah Qureshi.

| System | What is learned | Fresh scenes (held 5 s) | Super test (8 tests, mean) |
|---|---|---|---|
| Expert arm + A\* planner (ceiling) | nothing | 100% | 96.7% |
| **Final system (S3)**: BO network arm + Kalman filter + A\* planner | the arm | **90%** | **81.3%** |
| Model R alone, fully learned (3 seeds) | arm and wheels | 86%, 90%, 84% | not run |

---

## 1. For the examiner: one command

```bash
# once: link the robot models and open RoboEnv's environment (details in section 2)
ln -s /path/to/RoboEnv/models models
pixi shell --manifest-path /path/to/RoboEnv/pixi.toml

bash reproduce.sh check    # ~10 s : setup, every model loads, every report number recounted from the result files
bash reproduce.sh quick    # ~5 min : also reruns K, R, the BO network and the final system on 10 scenes each
bash reproduce.sh full     # ~3 h  : also reruns the final system on the whole super test (500 trials)
```

`check` prints each number the report quotes next to the value counted from its result file, with the
file's path, and rebuilds the super-test table (S1–S5). `quick` and `full` write their reruns to
`results/reproduced/` and never touch the saved results. Use `WORKERS=8 bash reproduce.sh quick` for more cores.
None of this needs the recorded training data (`data/`).

## 2. Setup

The code runs inside the course's **RoboEnv** environment (pixi). Linux, WSL or macOS shown; on Windows
PowerShell use the same commands with `\` paths.

```bash
ln -s /path/to/RoboEnv/models models                                # once: the robot models
pixi shell --manifest-path /path/to/RoboEnv/pixi.toml               # testing and figures
pixi shell --manifest-path /path/to/RoboEnv/pixi.toml -e ml         # the same plus PyTorch, for training
cd /path/to/comp0245_bc
```

Run every command in this README **from this folder** (`comp0245_bc`).

**What is in git.** All code, the 43 trained models the report uses (`trained/`, plain NumPy arrays) and
every result the report quotes (`results/`). The recorded expert drives (`data/`, several GB) are not in
git: they are only needed to **train**. Re-record them (section 6) or download them from the team OneDrive.

## 3. Report map: where every result comes from

A *policy* is a model file, optionally wrapped: `model.npz:kfv` adds the Kalman filter,
`model.npz:filt29noqd` a 29 ms position filter, `model.npz:kfv:nav` the VFH/Bug 2 obstacle layer,
`planner:<arm>` the A\* planner driving the base with `<arm>` on the arm, `driverbrake:<driver>:<arm>` the
learned Driver A (model A) plus brake, `cam:planner:<arm>` the camera version with no true position.
`<arm>` is `expert` or a model. All result folders are under `results/`.

| Report part | Policy | Results |
|---|---|---|
| Expert (the ceiling) | `expert` | `1_main_models/expert/` |
| Choosing a model: linear regression, random forest | `trained/cmp_linear.npz`, `trained/cmp_forest.pkl` | `1_main_models/other_learners/` |
| DAgger on model K | `trained/vL_dagger1.npz` | `1_main_models/side_experiments/DAgger/` |
| Model K, built step by step from v0 | `trained/mlp_v0.npz`, `mlp_v1`, `mlp_v2_hist`, `vA_goal`, `vD_all`, `vE_integral`, then `vK_kick` | `0_old_tests/<step>/`, `1_main_models/K/` |
| Noise and delay: T4, noise only, delay only, K-filtered, K-clean-labels | `trained/vK_kick.npz`, `vM_smooth`, `vM2_noise0` (also L `vL_disturbed`, M2 `vM2_smooth_hist`, N `vN_delay`) | `1_main_models/K/tier4/`, `1_main_models/noise_and_delay/K_diagnosis/` |
| Model P, and the arm swap | `trained/vP_hold.npz`; `vP_hold.npz:mlp_wheels` (expert arm), `vP_hold.npz:mlp_arm` (expert base) | `1_main_models/P/`, `1_main_models/arm_swap/` |
| Model R (3 seeds), ablation | `trained/vR_ik.npz`, `vR_ik_s1`, `vR_ik_s2`; `abl_no_goal`, `abl_no_iknow`, `abl_no_history` | `1_main_models/R/`, `1_main_models/ablations/` |
| Bayesian optimisation (12 trials) and the BO network | `trained/bo_t00.npz` … `bo_t11.npz`; final `bo_t03.npz` | `1_main_models/R-BO/` (trials in `bayes_opt_trials/`) |
| BO network: 29 ms filter vs Kalman filter | `trained/bo_t03.npz:filt29noqd`, `trained/bo_t03.npz:kfv` | `1_main_models/R-BO/` |
| Fresh scenes (K, P, R, expert) | as above, with `--scenes benchmark/scenes_fresh.json` | `3_final_tests/fresh_scenes/` |
| Obstacles: expert alone, A\* planner, box and door | `expert`, `planner:expert` | `2_obstacles/abdullah_bases/before_navigation/`, `.../stage3_planner_and_driver_A/`, `2_obstacles/expert/` |
| Model A (learned driver), with and without brake, 2 seeds | `driverbrake:trained/driver_A.npz:expert`, `driver_A_s1` | `2_obstacles/abdullah_bases/stage3_planner_and_driver_A/` |
| Localisation, camera, search mode | `navigation/eval_selfnav.py` (section 4) | `2_obstacles/abdullah_bases/stage6_camera_and_odometry/` |
| Mixing learned and expert parts | `planner:trained/vR_ik.npz`, `driverbrake:trained/driver_A.npz:trained/vR_ik.npz` | `2_obstacles/abdullah_bases/stage3_…/`, `stage5_learned_arms/` |
| **Super test S1–S5** | S1 `planner:expert`, S2 `trained/bo_t03.npz:kfv:nav`, **S3 `planner:trained/bo_t03.npz:kfv`**, S4 `driverbrake:trained/driver_A.npz:trained/bo_t03.npz:kfv`, S5 `cam:planner:trained/bo_t03.npz:kfv` | `3_final_tests/super_test/` (S2 also `2_obstacles/R-BO+Kalman+VFH_Bug2_nav/`) |
| Final models: no true position, learned driver and arm | `navigation/eval_selfnav.py --mode cam` (section 4) | `2_obstacles/abdullah_bases/stage6_camera_and_odometry/` |

`analysis/report_numbers.py` checks every number in this table against its file (it is step 3 of
`bash reproduce.sh check`). `results/all_results.xlsx` is a fixed summary of all results, including the
pivot table behind the super-test chart; the script that first built it was removed with the experiments
the report does not use, and `evaluation/super_table.py` rebuilds the super-test numbers on its own.

## 4. Test any model

```bash
python3 evaluation/evaluate.py --policy <POLICY> --tiers 1 --n 50 --workers 4 --hold 5
```

Results go to `results/new_runs/` (or `--out <folder>`): one CSV per tier with a row per scene, a summary
JSON, and a one-line score per tier.

| Option | Meaning |
|---|---|
| `--tiers 1 2 3 4` | benchmark tiers, 50 scenes each (`benchmark/scenes.json`): **1** normal, **2** far or high targets, **3** awkward starts, **4** sensor noise 0.005–0.02 and a 1–4 ms command delay |
| `--n 10` | only the first 10 scenes of each tier |
| `--hold 5` | the report's success rule: within 1 cm **and held 5 s**. Without it, success is the first touch within 1 cm ("reached") |
| `--scenes benchmark/scenes_fresh.json --tiers 1` | the 50 fresh scenes, used once for final scores |
| `--scenes scenes_obstacles/obstacles_test.json --tiers 10 11 12 13 14 --max-time 90` | the 150 obstacle scenes (30 each): 10 clear, 11 sparse, 12 cluttered, 13 narrow gap, 14 blocked line |
| `--set obs_noise=0.01` | sensor noise on every scene; `objects=3` a 0.3×0.3×0.5 m box in the path, `objects=4` a 1 m wall with a 1.4 m doorway |
| `--scenes benchmark/tier4_noise_only.json --tiers 4` | tier 4 with only its noise (`tier4_delay_only.json`: only its delay) |
| `--plots 0` | skip the drive plots |

Before any obstacle, box, door or super-test run, `export NAV_STUCK=0 OBJ_TIME=90` (the settings all saved
results used: the VFH/Bug 2 stuck detector off, and 90 s for the box and door scenes).

Without true positions (localisation, camera, search mode):

```bash
python3 navigation/eval_selfnav.py --mode cam --scenes scenes_obstacles/obstacles_test.json --workers 4
#   --mode odo (Kalman filter only) | cam (Kalman filter + camera) | search (target not given)
#   default: the A* planner drives and the expert arm reaches. For the learned parts add
#   --drive trained/driver_A.npz --brake   and/or   --arm trained/vR_ik.npz
```

Compare saved results: `python3 evaluation/compare.py --tier 1`. Why drives failed:
`python3 evaluation/analyse_failures.py --results results --policies vK_kick vR_ik`. Watch a drive:
`python3 abdullah_code/driver_A/watch.py --policy <POLICY> --tier 1 --index 0` (in WSL first
`export LIBGL_ALWAYS_SOFTWARE=1`).

## 5. Train a model

Needs the recordings in `data/` and `pixi shell ... -e ml`. Use a new `--name`, so the committed models are
never overwritten. Every model stores the exact options it was trained with (`meta` inside the `.npz`).

```bash
K="--history deltas --goal --separate --reach-weight 3 --integral"
D="data/runs_nudged data/runs_reach data/runs_kick"
H="data/runs_hold data/runs_extreme"
python3 training/train.py --name my_K $K --extra-train $D                       # model K
python3 training/train.py --name my_P $K --extra-train $D $H                    # model P
python3 training/train.py --name my_R --ik $K --extra-train $D $H               # model R
python3 training/bayes_opt.py                                                   # the 12-trial BO search (hours)
python3 training/train.py --name my_BO --ik --history deltas --goal --separate --integral --reach-weight 2.473 \
        --lr 0.000741 --batch 2048 --hidden 192 192 192 --extra-train $D $H     # the BO network (trial 3)
# noise and delay (section 6 records the disturbed sets)
python3 training/train.py --name my_K_filtered     $K --smooth 0.05 --extra-train $D data/runs_disturbed
python3 training/train.py --name my_K_clean_labels $K --smooth 0.05 --smooth-history-only --extra-train $D data/runs_disturbed_n0
python3 training/dagger.py --help                                               # DAgger on K
python3 abdullah_code/driver_A/train_driver.py --name my_driver_A               # Driver A (model A)
```

Driver A's training data is recorded with `abdullah_code/driver_A/record_teacher.py` (the A\* planner and the
expert arm driving through obstacle scenes; `--nudge` for the nudged drives). That folder keeps its own copies
of the shared modules, written against an earlier version of them (its LiDAR is mounted at 1.0 m, as in the
report's planner section; the super test used 0.30 m, set in `navigation/sensors.py`).

## 6. Record expert data

Recording is seeded, so the same command gives the same drives.

```bash
python3 training/collect.py --start 0 --end 300 --workers 4                                   # data/runs (clean)
python3 training/collect.py --start 10000 --end 10150 --noise 0.05 0.1 0.15 --out data/runs_nudged
python3 training/collect.py --start 20000 --end 20120 --noise 0.05 0.1 --nudge-when reach --out data/runs_reach
python3 training/collect.py --start 30000 --end 30120 --kick 0.02 --out data/runs_kick
python3 training/collect.py --start 60000 --end 60200 --hold 5 --out data/runs_hold
python3 training/collect.py --start 61000 --end 61150 --hold 5 --z-bands 0.15 0.3 0.95 1.1 --out data/runs_extreme
python3 noise_and_delay/collect_noisy.py --start 40000 --end 40160 --obs-noise 0 0.005 0.01 0.02 --delay 0 1 2 3 4 --out data/runs_disturbed
mkdir -p data/runs_disturbed_n0 && cp data/runs_disturbed/run_400{00..156..4}.npz data/runs_disturbed_n0/   # the noise-free ones
python3 training/make_split.py      # train / validation / test split by run (already saved in splits/)
```

The test scenes in `benchmark/` and `scenes_obstacles/` were generated once (`training/make_benchmark.py`,
`navigation/obstacle_scenes.py`) and must never be regenerated or edited.

## 7. Figures and tables

```bash
python3 analysis/report_numbers.py          # every report number from its result file
python3 evaluation/super_table.py           # the super-test table (S1-S5)
python3 analysis/plot_learning_curves.py    # K/P/R/BO learning curves -> figures/learning_curves.pdf
python3 analysis/report_figures.py          # result figures -> figures/report/
python3 analysis/make_figures.py            # model K figures -> figures/
python3 abdullah_code/driver_A/make_figures.py   # obstacle and camera figures -> figures/abdullah_bases/
python3 analysis/data_report.py             # data coverage figures (needs data/)
```

## 8. External code, and what we wrote

| Source | Used for | How |
|---|---|---|
| Course template (RoboEnv coursework code) | the expert controller and the reaching task | `core/robotnik_arm_control.py` is the template's controller file, **unchanged** apart from its import line. `core/expert_episode.py` and `noise_and_delay/expert_episode_noisy.py` are the template's `robotnik_arm_task.py` main loop **turned into a function** with recording added; they reproduce the template's own run exactly. The A\* planner hands over to the template's parking controller |
| `simulation_and_control` (course library, in RoboEnv) | the PyBullet simulation wrapper | used as provided |
| PyBullet | physics simulation | library |
| Pinocchio | robot model: gravity torque and the arm Jacobian for the IK inputs | library |
| PyTorch | training the MLPs | library |
| scikit-learn | the random-forest comparison | library |
| scikit-optimize (`gp_minimize`) | Bayesian optimisation | library, as in the course's Lab 3 |
| NumPy, SciPy, Matplotlib | numerics and figures | libraries |

Everything else was written by the team for this project: recording with disturbances, the input features
(history, parking spot, error averages, IK inputs), training, the MLP running in NumPy, the filters and
Kalman filters, the benchmark and obstacle scenes, the VFH/Bug 2 layer, the A\* planner, Driver A,
localisation, the camera, search mode, and all evaluation and analysis code.

## 9. Where everything is

| Folder | What |
|---|---|
| `core/` | the shared engine: the expert, model inputs (`features.py`, `ik_features.py`), running any policy (`rollout.py`), scene samplers |
| `navigation/` | the base: A\* planner (`planner.py`, `nav.py`), Driver A (`driver.py`), LiDAR (`sensors.py`), localisation, camera, search (`self_nav.py`) |
| `training/` | recording (`collect.py`), splitting, training (`train.py`), Bayesian optimisation, DAgger |
| `noise_and_delay/` | the noise diagnosis: disturbed recording, noise check, tier-4 split |
| `evaluation/` | `evaluate.py` (test any policy), `super_table.py`, `compare.py`, `analyse_failures.py`, `run_super_laptop.py` (super test S1 and S5) |
| `analysis/` | `report_numbers.py` and the figure scripts |
| `abdullah_code/driver_A/` | Abdullah's recording and training scripts for Driver A and the camera tests, with their own module copies |
| `trained/` | the trained models |
| `results/` | the results, by report section: `0_old_tests/` (steps to model K), `1_main_models/`, `2_obstacles/`, `3_final_tests/` (fresh scenes, super test); `all_results.xlsx` summarises them |
| `benchmark/`, `scenes_obstacles/` | the frozen test scenes |
| `figures/` | figures made by the scripts in section 7 |
| `docs/` | data format, pre-processing and the super-test decisions |
| `scripts/` | the shell scripts used on the team's server to run the experiments, kept as a record (their paths are the server's) |
| `reproduce.sh` | section 1 |

Each code folder has a small `_paths.py` that gives every script the project folder and lets the folders
import each other, so any script runs from `comp0245_bc` as shown.

## 10. When something goes wrong

- `No module named pinocchio` (or pybullet): you are not inside `pixi shell` (section 2).
- `Cannot load URDF file`: the `models` link is missing (section 2).
- `No module named torch`: training needs `pixi shell ... -e ml`.
- `run ... not found in data/...`: that recording set is not downloaded (section 2).
- **Small differences between computers are normal.** Rerunning model R on all 50 T1 scenes gives the same
  92%, but 3 borderline scenes changed outcome (held, crash or timeout) and final errors moved by about
  0.1 mm. Success rates match; individual scenes may not. On one computer every run repeats exactly.
- Simulator warnings are hidden; `export ROBOENV_VERBOSE=1` shows them. A crash prints one line and writes
  the details to `error.log`. Ctrl+C stops cleanly; running the same command again carries on.
