# Data pre-processing pipeline (for the report)

Every step below is in the code, and most have a measured effect (tier 1 benchmark unless stated).

| # | Step | What we do | Where | Evidence it matters |
|---|---|---|---|---|
| 1 | **Cleaning** | Drop expert drives that failed (crash, never reached). Fix the template labelling a within-1 cm drive as "stalled" (it checks "stopped improving" before "reached"). Flag successful-but-anomalous drives (circling: path > 2x straight line; slow: > 2.5x typical time; navigation timeout). | `check_runs._outcome`, `make_split.py`, `data_report.py` -> `splits/bad_runs.json`, `train.py --exclude` | Only 18/608 drives flagged (3%, all circling near the spot); removing them changed nothing (34 vs 36 /50), i.e. the expert data is clean and remaining errors come from the model |
| 2 | **Distribution check** | Targets and starts plotted in the robot's starting frame (ahead / left / behind), training vs test, failure rate per bin | `data_report.py` -> `figures/data_*.png` | Even coverage in all directions; failures cluster at height extremes -> extra extreme-height recordings |
| 3 | **Splitting** | By DRIVE, not by moment (70/15/15 = 208/45/44 clean drives), so near-identical neighbouring moments never sit on both sides. Extra recordings (nudged, kicked, hold, ...) go to TRAINING only, so validation stays comparable. Benchmark scenes come from a separate seed block (>= 1,000,000); a FRESH test set (`benchmark/scenes_fresh.json`) was used only once, after all choices | `make_split.py`, `samplers.py`, `train.py` | Fresh test confirmed the result (R: 86% held vs 93% on the selection scenes; the gap is the selection bias) |
| 4 | **Resampling** | Simulator runs at 1 kHz; recordings kept at 100 Hz. History / integral / IK averages are updated every 10 ms when driving too, so training and driving inputs match | `expert_episode.py`, `rollout.py` | Live-vs-training inputs verified identical (difference 0.0 over 1,117 moments) |
| 5 | **Coordinate change** | Target and the expert's parking spot expressed as the robot sees them (ahead / left), heading as sin/cos (no 180/-180 jump) | `features.py` | Parking-spot inputs: 2% -> 22% (version A) |
| 6 | **Physics-derived quantities** | Torque targets = expert torque MINUS gravity (Pinocchio); gravity added back when driving. Hand position from joint angles (forward kinematics). Arm IK direction from the Jacobian (`ik_features.py`) | `train.py --torque`, `features.py`, `ik_features.py` | Gravity: v0 -> v1. IK direction: held 26% -> 93% (selection scenes), 40% -> 86% (fresh) |
| 7 | **Time inputs** | How the base moved over 0.1 s / 0.5 s and the arm over 0.1 s; hand error averaged over 0.5 s / 2 s; IK direction averaged over 0.5 s / 2 s | `features.py`, `ik_features.py` | Stage A ladder (history, integral rows) |
| 8 | **Scaling** | Inputs and outputs standardised with TRAINING statistics only; inputs clipped to the range seen in training when driving (anti-windup) | `train.py`, `rollout.py` | Standard practice; prevents runaway inputs when stuck |
| 9 | **Weighting** | Reaching moments count 3x in the loss (they are a small share of each drive). Hold moments: extra weight tested and rejected (12 and 6 vs 17 held) | `train.py --reach-weight --hold-weight` | Ladder version C; hold-weight experiment |
| 10 | **Deliberately NOT inputs** | Raw time, past commands, measured wheel speeds (copycat shortcut), the expert's phase and reference | `features.py` docstring | Wheel-speed ablation: 0/50 and 9/50 |

Pipeline (for the diagram):
recordings (1 kHz sim -> 100 Hz) -> cleaning + outcome fix -> split by drive -> input groups
[basic 26 | history 13 | parking spot 5 | integral 6 | IK 21] + targets (torque - gravity, wheel speeds)
-> standardise (train stats) -> MLP (separate arm / wheel nets, reach weight 3) -> NumPy weights
-> live loop every 1 ms: measure -> same inputs -> model -> + gravity -> clip -> robot.
