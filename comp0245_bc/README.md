# COMP0245 behaviour cloning: a mobile manipulator learns to reach

A Robotnik four-wheeled base carrying a seven-joint Kinova arm, simulated in PyBullet. The course template
provides an expert controller. We record the expert and train neural networks to copy it: every 1 ms the
learned policy maps the robot's measured state and the target position to 7 arm torques and 4 wheel speeds.
A trial succeeds when the hand comes within **1 cm** of the target and **stays there for 5 s**.

**Headline results**

| System | What is learned | Fresh scenes (held 5 s) | Super test (610 trials) |
|---|---|---|---|
| Expert + A\* planner (ceiling) | nothing | 100% | 96.7% |
| **Final system (S3)**: R-BO arm + Kalman filter + A\* planner | the arm | **90%** | **81.3%** |
| Model R alone, fully learned (3 seeds) | arm and wheels | 86%, 90%, 84% | not run |

Team: David, Abdullah Qureshi, Muneeb Musharaf.

---

## 1. Setup (once)

The code runs inside the course's **RoboEnv** environment (pixi). Linux, WSL or macOS shown; on Windows
PowerShell use the same commands with `\` paths.

```bash
# 1. link the robot models from your RoboEnv checkout into this folder
ln -s /path/to/RoboEnv/models models

# 2. every session: open a shell in RoboEnv's environment, then work from this folder
pixi shell --manifest-path /path/to/RoboEnv/pixi.toml          # simulator, testing, figures
pixi shell --manifest-path /path/to/RoboEnv/pixi.toml -e ml    # the same plus PyTorch, for training
cd /path/to/comp0245_bc
```

Run every command below **from this folder** (`comp0245_bc`). Testing and figures need only the default
environment: trained models are plain NumPy arrays. Training needs PyTorch (`-e ml`).

**What is in git and what is not.** All code, every trained model (`trained/`, about 100 files) and every
result (`results/`) are in git. The recorded expert drives (`data/`) are not, because they are too large:
download them from the team OneDrive into `data/` if you want to **train**, or re-record them (section 5).
Testing a model never needs `data/`.
The Gaussian-process model `trained/gp_R.pkl` (130 MB) is not in git either; `training/train_gp.py` rebuilds it.

## 2. Test any model (one command)

```bash
python3 evaluation/evaluate.py --policy <POLICY> --tiers 1 --n 50 --workers 4 --hold 5
```

`<POLICY>` comes from the table in section 3. Results go to `results/new_runs/` (one CSV per tier, a
summary JSON, a few plots of drives), and a one-line score is printed per tier.

| Option | Meaning |
|---|---|
| `--tiers 1 2 3` | benchmark tiers (50 scenes each, `benchmark/scenes.json`): **1** normal, **2** far or high targets, **3** awkward starts, **4** sensor noise + 1–4 ms delay, **5** changed physics (friction, mass), **6** objects anywhere, **7** everything mixed |
| `--n 10` | only the first 10 scenes of each tier (quick check) |
| `--hold 5` | the team's success rule: reach 1 cm **and hold it 5 s**. Without it, success = first touch within 1 cm (the template's rule) |
| `--scenes benchmark/scenes_fresh.json --tiers 1` | 50 fresh scenes, used once for the final scores |
| `--scenes scenes_obstacles/obstacles_test.json --tiers 10 11 12 13 14 --max-time 90` | obstacle scenes (30 each): 10 clear, 11 sparse, 12 cluttered, 13 gap, 14 blocked |
| `--set obs_noise=0.01` | add sensor noise to every scene. Also `delay_steps=2` (ms), `payload_kg=0.5`, `torque_noise=2`, `objects=3` (a box in the way), `objects=4` (a wall with a doorway) |
| `--only noise` / `--only delay` | tier 4 with only one of its two disturbances (diagnosis) |
| `--plots 0` | skip the drive plots |

To reproduce obstacle, box, door or super-test numbers exactly, first `export NAV_STUCK=0 OBJ_TIME=90`
(the settings the super test used; see `results/2_obstacles/README.md`).

Compare saved results as a table: `python3 evaluation/compare.py --tier 1` (every 50-scene benchmark
result), or `--results <folder>` for one folder. The super-test table: `python3 evaluation/super_table.py`.

To **watch** a drive in a window: `python3 abdullah_code/driver_A/watch.py --policy <POLICY> --tier 1 --index 0`
(`--index` is the scene number 0–49; add `--top` for a view from above; in WSL first `export LIBGL_ALWAYS_SOFTWARE=1`).

## 3. Every model: report name, how to run it, where its results are

A policy is a model file, optionally wrapped: `model.npz:kfv` adds the Kalman filter,
`model.npz:filt20noqd` a 20 ms position filter, `...:nav` David's VFH/Bug 2 obstacle layer,
`planner:<arm>` Abdullah's A\* planner driving the base with `<arm>` on the arm, `driverbrake:<driver>:<arm>`
Abdullah's learned Driver A (+ brake). Result folders are under `results/`.

### The main line (no obstacles, no disturbances)

| Report name | `--policy` | Results |
|---|---|---|
| Expert (ceiling) | `expert` | `1_main_models/expert/` |
| v0 … G (the early ladder) | `trained/mlp_v0.npz` … `trained/vG_both.npz` | `0_old_tests/<name>/` |
| **K** (Stage A final) | `trained/vK_kick.npz` | `1_main_models/K/` |
| **P** (K + hold-still data) | `trained/vP_hold.npz` | `1_main_models/P/` |
| **R** (P + IK inputs, main learned model) | `trained/vR_ik.npz` (seeds: `vR_ik_s1`, `vR_ik_s2`) | `1_main_models/R/` |
| **R-BO** (final arm, Bayesian optimisation) | `trained/bo_t03.npz:kfv` | `1_main_models/R-BO/` |
| Ablations (R minus one input group) | `trained/abl_no_goal.npz`, `abl_no_iknow`, `abl_no_history`, `abl_no_erravg`, `abl_no_ikavg` | `1_main_models/ablations/` |
| Arm swap (P's arm + expert base) | `trained/vP_hold.npz:mlp_arm` (or `:mlp_wheels`) | `1_main_models/arm_swap/` |
| GRU, TCN, attention | `trained/gru_P.npz`, `trained/tcn2.npz`, `trained/attn.npz` | `1_main_models/memory_models/` |
| Linear, random forest, Gaussian process, PCA | `trained/cmp_linear.npz`, `cmp_forest.pkl`, `gp_R.pkl`, `vR_pca15.npz` | `1_main_models/other_learners/` |
| DAgger, HG-DAgger, memory-free expert, Q, S-relative | `trained/vL_dagger1.npz`, `vL_hg1`, `markov_expert`, `vQ_hw3`, `vS_rel` | `1_main_models/side_experiments/` |
| Abdullah's model C (K's inputs, ~50% more data) | `trained/vK_more_hold_arm.npz` | `1_main_models/abdullah_model_C/` |

### Noise and delay (details: [`results/1_main_models/noise_and_delay/README.md`](results/1_main_models/noise_and_delay/README.md))

| Report name | `--policy` | Results |
|---|---|---|
| K-noisy (L), K-filtered (M), K-history-filter (M2), K-clean-labels (M2-noise0), K-delay (N) | `trained/vL_disturbed.npz`, `vM_smooth`, `vM2_smooth_hist`, `vM2_noise0`, `vN_delay` | `noise_and_delay/K_diagnosis/` |
| R + position filter, R + Kalman filter | `trained/vR_ik.npz:filt20noqd`, `trained/vR_ik.npz:kfv` | `noise_and_delay/filters_and_robust_models/` |
| Ensemble, noise augmentation, dropout, weight decay, domain randomisation | `ens:trained/vR_ik.npz,trained/vR_ik_s1.npz,trained/vR_ik_s2.npz`, `trained/vT_aug.npz`, `vU_drop`, `vU_wd`, `vM_dr` | same |

### Obstacles and bases (details: [`results/2_obstacles/README.md`](results/2_obstacles/README.md))

| Report name | `--policy` | Results |
|---|---|---|
| **S1** expert + A\* planner | `planner` | `3_final_tests/super_test/` |
| **S2** R-BO + Kalman + VFH/Bug 2 | `trained/bo_t03.npz:kfv:nav` | `2_obstacles/R-BO+Kalman+VFH_Bug2_nav/` |
| **S3 final** R-BO + Kalman + A\* planner | `planner:trained/bo_t03.npz:kfv` | `3_final_tests/super_test/` |
| **S4** R-BO + Kalman + Driver A + brake | `driverbrake:trained/driver_A.npz:trained/bo_t03.npz:kfv` | `3_final_tests/super_test/` |
| **S5** S3 with the camera finding the target | `cam:planner:trained/bo_t03.npz:kfv` | `3_final_tests/super_test/` |
| Model S (LiDAR inputs) | `trained/vS_lidar.npz:filt20noqd` | `2_obstacles/S_lidar_inputs+position_filter_20ms/` |
| Driver B, off-spot arm | `driver:trained/driver_B.npz`, `planner:trained/vS_arm.npz` | `2_obstacles/abdullah_bases/` |

## 4. Train a model

Training needs the recordings in `data/` and the `-e ml` environment. Each command writes `trained/<name>.npz`;
use a new `--name` so the committed models are never overwritten. Every model also stores the exact
options it was trained with (`meta` inside the `.npz`).

```bash
K="--history deltas --goal --separate --reach-weight 3 --integral"
D="data/runs_nudged data/runs_reach data/runs_kick"
python3 training/train.py --name my_K $K --extra-train $D                                         # K
python3 training/train.py --name my_P $K --extra-train $D data/runs_hold data/runs_extreme        # P
python3 training/train.py --name my_R --ik $K --extra-train $D data/runs_hold data/runs_extreme   # R
python3 training/train.py --name my_RBO --ik --history deltas --goal --separate --integral --reach-weight 2.473 \
        --lr 0.000741 --batch 2048 --hidden 192 192 192 --extra-train $D data/runs_hold data/runs_extreme   # R-BO
python3 training/bayes_opt.py                                       # the 12-trial search that found R-BO (hours)
```

The noise-and-delay models (L, M, M2, M2-noise0, N) are in the noise README; GRU / TCN / attention use
`training/train_gru.py` and `training/train_conv.py`, the Gaussian process `training/train_gp.py`, DAgger
`training/dagger.py`. Every script prints its options with `--help`. Driver A and model C were trained
with Abdullah's own scripts: [`abdullah_code/README.md`](abdullah_code/README.md).

## 5. Record expert data

```bash
python3 training/collect.py --start 0 --end 300 --workers 4                                   # data/runs (clean)
python3 training/collect.py --start 10000 --end 10150 --noise 0.05 0.1 0.15 --out data/runs_nudged
python3 training/collect.py --start 20000 --end 20120 --noise 0.05 0.1 --nudge-when reach --out data/runs_reach
python3 training/collect.py --start 30000 --end 30120 --kick 0.02 --out data/runs_kick
python3 training/collect.py --start 60000 --end 60200 --hold 5 --out data/runs_hold
python3 training/collect.py --start 61000 --end 61150 --hold 5 --z-bands 0.15 0.3 0.95 1.1 --out data/runs_extreme
python3 training/make_split.py                     # train / validation / test split by run (already in splits/)
```

Recording is seeded, so the same command gives the same drives. The benchmark scenes in `benchmark/` were
generated once (`training/make_benchmark.py`) and must never be regenerated or edited.

## 6. Rebuild the figures and tables

```bash
python3 analysis/report_figures.py       # every report figure -> figures/report/
python3 analysis/make_figures.py         # the Stage A figures   -> figures/
python3 analysis/pca_report.py           # PCA of R's inputs     -> figures/pca_inputs.png
python3 evaluation/super_table.py        # the super-test table
python3 evaluation/analyse_failures.py --results results --policies vK_kick vR_ik   # why drives failed
```

## 7. Where everything is

| Folder | What |
|---|---|
| `core/` | the shared engine: the expert (`expert_episode.py`, `robotnik_arm_control.py`), model inputs (`features.py`, `ik_features.py`), running any policy (`rollout.py`), scene samplers |
| `navigation/` | bases and sensing: A\* planner, VFH/Bug 2 layer, Driver A/B, LiDAR, odometry, camera |
| `training/` | recording (`collect.py`), splitting, training every kind of model, Bayesian optimisation, DAgger |
| `noise_and_delay/` | the noise diagnosis: disturbed recording, noise check, tier-4 split |
| `evaluation/` | `evaluate.py` (test any policy), `compare.py`, `super_table.py`, `analyse_failures.py` |
| `analysis/` | figures and data reports |
| `abdullah_code/` | Abdullah's own recording and training scripts for Driver A and model C (kept as he wrote them) |
| `trained/` | every trained model |
| `results/` | every result, by section: `0_old_tests/`, `1_main_models/` (K, P, R, R-BO, noise and delay, side experiments, model C), `2_obstacles/`, `3_final_tests/` (fresh scenes, super test). Model folders with several tiers have `tier1/`, `tier2/` … subfolders |
| `benchmark/`, `scenes_obstacles/` | the frozen test scenes |
| `figures/` | all figures; `figures/report/` holds the report's |
| `docs/` | team notes from the project (history) |
| `scripts/` | the shell scripts the server runs used, kept as a record (their paths are the server's) |

Each code folder has a small `_paths.py`: it gives every script the project folder and lets the folders
import each other, so you can run any script from `comp0245_bc` as shown.

## 8. When something goes wrong

- `No module named pinocchio` (or pybullet): you are not inside `pixi shell` (section 1).
- `Cannot load URDF file`: the `models` link is missing (section 1).
- `No module named torch`: training needs `pixi shell ... -e ml`.
- `run ... not found in data/...`: that recording set is not downloaded (section 1).
- Small differences (a millimetre, a scene or two) between computers are normal: floating point and
  library versions. On one computer every run is exactly repeatable.
- Simulator warnings are hidden; `export ROBOENV_VERBOSE=1` shows them. A crash prints one line and writes
  the details to `error.log`. Ctrl+C stops cleanly; running the same command again carries on.
