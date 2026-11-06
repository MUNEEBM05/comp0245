# COMP0245 behaviour cloning: a mobile manipulator learns to reach

COMP0245 (Machine Learning for Robotics, UCL) group coursework by Muneeb Musharaf, David Wild and Abdullah Qureshi.

A Robotnik four-wheeled base carrying a seven-joint Kinova arm, simulated in PyBullet. We record the course's
expert controller and train neural networks to copy it. A trial succeeds when the hand comes within 1 cm of the
target and stays there for 5 s. The final system (learned arm, Kalman filter, A\* planner) holds 90% of fresh scenes.

**All code, trained models and results are in [`comp0245_bc/`](comp0245_bc/). Its
[README](comp0245_bc/README.md) explains everything: setup from nothing, how to run every model in the report,
what each command outputs, and which external code was used.**

## Quick start

With the course's RoboEnv environment installed (see [section 2 of the README](comp0245_bc/README.md#2-setup-from-nothing) if not):

```bash
cd comp0245_bc
ln -s /path/to/RoboEnv/models models                  # once: the robot models
pixi shell --manifest-path /path/to/RoboEnv/pixi.toml  # every new terminal
cd /path/to/comp0245_bc
bash reproduce.sh check        # 10 s: every model loads, every number in the report is recounted from its result file
bash reproduce.sh list         # every model by its report name (K, P, R, BO, S1-S5, ModelA ...)
bash reproduce.sh run S3 10    # rerun the final system on 10 scenes and compare with the saved result (no window)
bash reproduce.sh watch S3     # watch one drive in the PyBullet window
```
