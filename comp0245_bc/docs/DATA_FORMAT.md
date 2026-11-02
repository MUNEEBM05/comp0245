# Recorded expert runs: data format

One file per run: `data/runs/run_<seed>.npz` (NumPy compressed). Each file is one episode of the
template's expert controller, recorded at **100 Hz** (every 10th step of the 1 kHz simulation).

## Loading a run

```python
from check_runs import load_run
meta, a = load_run("data/runs/run_00000.npz")
a["torque_arm"].shape      # (n_logged, 7)
meta["outcome"]            # "reached", "unreachable", "step_cap", "crashed", "timeout"
```

## Per-timestep arrays (one row per logged step)

| Key | Shape | Units | Meaning |
|---|---|---|---|
| `t` | (N,) | s | simulation time |
| `base_pos` | (N, 3) | m | base position (x, y, z), world frame, true pose (not odometry) |
| `base_quat` | (N, 4) | – | base orientation quaternion (x, y, z, w) |
| `base_bearing` | (N,) | rad | base heading (yaw) |
| `q_mes` | (N, 17) | rad | all joint angles: 0–3 wheels, **4–10 arm**, 11–16 fingers |
| `qd_mes` | (N, 17) | rad/s | all joint velocities, same layout |
| `ee_pos` | (N, 3) | m | end-effector (hand) position, world frame |
| `target` | (N, 3) | m | the target, repeated on every row |
| **Labels (what the model must predict)** | | | |
| `torque_arm` | (N, 7) | Nm | expert's arm torques (clean, even on noisy runs) |
| `wheel_cmds` | (N, 4) | rad/s | expert's wheel speeds: front-right, front-left, back-left, back-right |
| **Expert's hidden state (not measurable on a real robot; use with care)** | | | |
| `phase` | (N,) int8 | – | base navigation phase: 0 align, 1 drive, 2 orient, 3 done |
| `nav_done` | (N,) bool | – | base has finished; the arm is now reaching |
| `q_des_arm`, `qd_des_arm` | (N, 7) | rad, rad/s | arm reference the PD controller tracks |
| **Extras** | | | |
| `tau_gravity` | (N, 7) | Nm | gravity torque on the arm (Pinocchio) |
| `torque_arm_sent`, `wheel_cmds_sent` | (N, 7), (N, 4) | | command actually executed (differs from the label only on noise runs) |
| `contact_ground`, `contact_self` | (N,) bool | – | arm touching the floor / the base |

## Per-run summary (`meta`)

`seed`, `target`, `start_pose` (x, y, yaw), `base_goal_pos`, `base_goal_bearing`, `standoff`,
`outcome`, `stop_message`, `final_ee_error` (m), `min_ee_error`, `reach_time`, `nav_done_time`,
`nav_timed_out`, `duration`, `n_sim_steps`, `n_logged`, `dt`, `log_every`, `cmd_noise_std`,
`max_steps`, `wall_time_s`, `code_version`, `python`.

## Rules

- **Split by run, never by row.** Rows within a run are near-identical neighbours.
- **Train on `outcome == "reached"` runs** unless you have a reason not to.
- Seeds 0–99,999 are training/validation; seeds from 1,000,000 are benchmark scenes only.
