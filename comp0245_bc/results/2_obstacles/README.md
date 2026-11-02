# Obstacles and bases

The base has to drive round obstacles to reach a parking spot from which the arm can reach the target.
Run every command from `comp0245_bc` (see the main README).

## The systems

A system is an **arm** (who moves the joints) plus a **base** (who drives the wheels):

| Base | What it is | Code | Learned? |
|---|---|---|---|
| Network's own wheels | the trained model drives the base too | `core/rollout.py` | yes |
| VFH / Bug 2 (`...:nav`) | David's layer: a 180-ray LiDAR finds free directions (vector field histogram) and follows obstacle edges (Bug 2); the network drives while the way is clear | `core/rollout.py` (`NavPolicy`) | no |
| Map + A\* (`...:mapnav`) | the same layer plus an occupancy map and an A\* route | `core/rollout.py` (`MapNavPolicy`) | no |
| A\* planner (`planner:<arm>`) | Abdullah's: LiDAR map, A\* route, straight-segment follower, then the template's own parking controller | `navigation/nav.py`, `navigation/planner.py` | no |
| Driver A (`driverbrake:<driver>:<arm>`) | Abdullah's learned base: a network that imitates the A\* planner from LiDAR inputs, plus a brake | `navigation/driver.py`, `trained/driver_A.npz` | yes |
| Driver B (`driver:trained/driver_B.npz`) | Driver A without the planner's route as an input | same | yes |
| Camera (`cam:<system>`) | no true position: wheel and gyro odometry, and a depth camera that finds the target | `navigation/self_nav.py`, `navigation/camera.py` | no |

The final system **S3** = R-BO arm + Kalman filter + A\* planner: `planner:trained/bo_t03.npz:kfv`. Only the
arm is learned there; the network's wheel outputs are ignored.

## Test scenes

| Scenes | Option | What |
|---|---|---|
| Box, door | `--tiers 1 --set objects=3` / `--set objects=4` | benchmark tier 1 with a box in the path, or a wall with a doorway |
| Obstacle tiers | `--scenes scenes_obstacles/obstacles_test.json --tiers 10 11 12 13 14 --max-time 90` | 30 scenes each: 10 clear, 11 sparse, 12 cluttered, 13 gap, 14 blocked (150 in all). Develop on `obstacles_val.json`; the test file is for final scores only |

Always add `--hold 5` (the team's success rule).

## Folders

| Folder | What |
|---|---|
| `R-BO+Kalman+VFH_Bug2_nav/`, `R+Kalman+VFH_Bug2_nav/`, `R+position_filter_20ms+VFH_Bug2_nav/` | David's navigation layer with different arms (box, door, noise, tiers 5 and 7) |
| `R+position_filter_20ms+map_Astar_nav/` | the map + A\* version |
| `R+position_filter_20ms/`, `expert/` | no navigation layer, on the object scenes (the baseline) |
| `expert+VFH_Bug2_nav/` | the expert arm with David's layer |
| `S_lidar_inputs+position_filter_20ms/` | model S: R trained with LiDAR inputs to drive round obstacles itself (it could not) |
| `door_120s/` | the door scenes with a 120 s limit instead of 40 s |
| `rejected_navigation/` | earlier versions of David's layer that were replaced (flip-flopping route, hysteresis, stuck detector) |
| `abdullah_bases/` | Abdullah's planner and drivers on the obstacle tiers, by his stages (below) |

`abdullah_bases/`: `before_navigation/` (expert and R with no navigation), `stage2_planner/`,
`stage3_planner_and_driver_A/` (planner vs Driver A, expert arm and R), `stage4_driver_B/`,
`stage5_learned_arms/` (Driver A + brake + R, planner + off-spot arm), `stage6_camera_and_odometry/`,
`validation_driver_A_less_data/` (Driver A on 25 / 50 / 75% of its data), `benchmark_tiers_1_to_3/`,
`logs/` (his run logs), and the summaries his figure script reads (`all_results.json`, `camera.json`, `routes.json`).

## Run them

```bash
export NAV_STUCK=0 OBJ_TIME=90          # the super test's settings (see the end of this file)
O="--scenes scenes_obstacles/obstacles_test.json --tiers 10 11 12 13 14 --max-time 90 --hold 5 --workers 4"
python3 evaluation/evaluate.py --policy planner $O                                                  # S1 ceiling
python3 evaluation/evaluate.py --policy planner:trained/bo_t03.npz:kfv $O                           # S3 final
python3 evaluation/evaluate.py --policy driverbrake:trained/driver_A.npz:trained/bo_t03.npz:kfv $O  # S4 Driver A
python3 evaluation/evaluate.py --policy cam:planner:trained/bo_t03.npz:kfv $O                       # S5 camera
python3 evaluation/evaluate.py --policy trained/bo_t03.npz:kfv:nav $O                               # S2 VFH/Bug 2
python3 evaluation/evaluate.py --policy trained/vR_ik.npz:kfv:nav --tiers 1 --set objects=4 --hold 5 --workers 4   # door
python3 abdullah_code/driver_A/make_figures.py      # Abdullah's obstacle figures -> figures/abdullah_bases/
```

**To reproduce the super-test numbers exactly**, set the same two switches the super test used:
`export NAV_STUCK=0 OBJ_TIME=90` (stuck detector of the VFH/Bug 2 layer off, as in its final version, and
90 s for the box and door scenes). The super test was split in two: `python3 evaluation/run_super_laptop.py`
runs S1 and S5 (resumable, sets the switches itself); S2–S4 ran on the server with the same tests
(`scripts/run_super_server.sh`, kept as a record). All results are in `results/3_final_tests/super_test/`;
`python3 evaluation/super_table.py` builds the table.
