# Autonomy: obstacles, lidar, learned driving, camera

Open-ended work beyond the main task: the robot plans around obstacles and can reach the target without being given its
own position. Imitation learning only, no reinforcement learning. Full results: `results/all_results.json`, `figures/`. `python build_report.py` builds a report page at `results/report.html` (not in git).

Shared with the rest of the project, not copied: `../data ../splits ../benchmark ../configs ../models ../RoboEnv`.

Environments: `../RoboEnv/.pixi/envs/default/bin/python` for the simulator, `../RoboEnv/.pixi/envs/ml/bin/python` for training
(uses the GPU when there is one). Viewer on WSL: `export LIBGL_ALWAYS_SOFTWARE=1`.

## Results: frozen test scenes, 150 drives, reached / held 5 s within 1 cm

| Setup | Reached | Held |
|---|---|---|
| Original expert, before | 38 | not measured |
| Planner + expert arm | 149 | 149 |
| Driver A + expert arm, brake (seeds 0, 1) | 145, 143 | 144, 141 |
| Planner + R | 132 | 129 |
| Driver A + R, brake (seeds 0, 1) | 117, 118 | 114, 115 |
| Driver B + expert arm (seeds 0, 1) | 5, 9 | 2, 8 |
| Camera + planner + expert arm, no true position | 138 | 132 |
| Camera + planner + R | 108 | 100 |
| Camera + driver A + R, brake (seeds 0, 1) | 79, 104 | 74, 91 |
| Search, target not given | 57 | 55 |
| Odometry only | 0 | 0 |

Driver A still uses the planner's route and parked check. Camera setups except search are told the target once at the start.

## Results: team benchmark tiers 1-3, no obstacles, 150 drives

| Setup | Reached | Held |
|---|---|---|
| Original expert | 149 | 149 |
| R alone | 145 | 131 |
| Planner + expert arm | 148 | 148 |
| Planner + R | 142 | 128 |
| Driver A + expert arm, brake (seeds 0, 1) | 144, 148 | 142, 148 |
| Camera + planner + expert arm | 118 | 115 |

The camera setup fails all 25 tier 2 scenes with targets at 1.10 m or higher, because they are above its view once parked (top of view 5 degrees above
horizontal). On the other 125 scenes it reached 118 and held 115. The obstacle scenes have no target above 0.99 m, so they do not test this.

## Main files

| File | What it does |
|---|---|
| `obstacle_scenes.py`, `scenes/` | Frozen validation and test scenes: clear, sparse, cluttered, gap, blocked |
| `sensors.py`, `localisation.py` | Lidar, encoders, gyro, accelerometer; Kalman filter odometry |
| `planner.py`, `nav.py` | A* on a lidar map, path follower, `PlannerPolicy` |
| `record_teacher.py`, `driver_features.py`, `train_driver.py`, `driver.py` | Learned wheel driver: recording, inputs, training, driving |
| `record_arm.py`, `train.py` | Arm recordings with off-spot parking; arm training (R's recipe gives vS) |
| `camera.py`, `check_camera.py` | Depth camera target fix (colour, sphere fit); accuracy test |
| `self_nav.py`, `eval_selfnav.py` | No true position: modes `odo`, `cam`, `search`; test |
| `evaluate.py`, `rollout.py` | Tests with true position |
| `watch.py` | Viewer; `--camera` shows the camera image |
| `make_figures.py`, `build_report.py` | Figures, results file and report page |

## Reproduce

```
python evaluate.py --policy planner --scenes scenes/obstacles_test.json --tiers 10 11 12 13 14 --workers 7 --max-time 90 --hold 5 --plots 0 --out results/stage3
python evaluate.py --policy driverbrake:trained/driver_A.npz:trained/vR_ik.npz --scenes scenes/obstacles_test.json --tiers 10 11 12 13 14 --workers 7 --max-time 90 --hold 5 --plots 0 --out results/stage5
python eval_selfnav.py --mode cam --drive trained/driver_A.npz --arm trained/vR_ik.npz --brake --scenes scenes/obstacles_test.json --workers 7 --out results/stage6
```

The scripts `run_stage34.sh`, `run_step5a.sh`, `run_step5b.sh`, `run_final.sh`, `run_final_b.sh` and `run_tiers.sh` hold the exact commands used, in order.
Test scenes are for final scores only; develop on `scenes/obstacles_val.json`.

## Limits

Static obstacles 1.6 m tall, one-plane lidar, our own sensor noise values, the target is the only red object, single training seeds differ a lot, and the
brake idea came after seeing crashes on the test scenes (confirmed on validation before use). Camera mode cannot see targets at 1.10 m or higher once parked.
