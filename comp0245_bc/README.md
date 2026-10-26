# COMP0245 Stage 1: Expert data collection

Records the template's expert robot reaching random targets from random starting positions.
Each run is saved as one file; these files are what our models learn from.

**Status:** 300 runs recorded. 298 reached the target (99.3%); 2 failed (seeds 234, 282, see below).
295,795 timesteps at 100 Hz. Data is on OneDrive: *(add link)*. Field list: `DATA_FORMAT.md`.

## How to run (Windows)

**Once:** link the robot models from your RoboEnv folder (PowerShell, inside this folder):
```powershell
New-Item -ItemType Junction -Path .\models -Target "C:\Users\YOU\comp0245\RoboEnv\models"
```

**Every time:** open the RoboEnv environment, then go to this folder:
```powershell
cd C:\Users\YOU\comp0245\RoboEnv
pixi shell
cd C:\path\to\comp0245_bc
```

| Step | Command | Expect |
|---|---|---|
| 1. Test one run | `python expert_episode.py` | `reached`, about 0.21 cm, 8.00 s |
| 2. Record runs | `python collect.py --start 0 --end 300 --workers 3` | about 15 min; `--workers` = CPU cores minus 1 |
| 3. Check them | `python check_runs.py` | summary + `data\runs\check_*.png` plots |

Ctrl+C stops safely; running the same command again carries on. Errors print one line
(details in `error.log` / `failures.log`). To see simulator warnings: `$env:ROBOENV_VERBOSE = "1"`.

## Files

`expert_episode.py` one run · `collect.py` many runs · `check_runs.py` summary and plots ·
`samplers.py` target and start ranges · `robotnik_arm_control.py` template controllers (unchanged) ·
`quiet.py` clean console output · `DATA_FORMAT.md` what is in each file

## Design decisions (be ready to explain these in the demo)

1. **Template controllers unchanged.** The brief is to clone the given expert; changing it would clone something else.
2. **Random start via the simulator's own reset** (`ResetPose`), so its internal state stays consistent.
3. **Record at 100 Hz, not 1,000 Hz.** Neighbouring steps are near-identical; 10x smaller files, same information.
4. **Training ranges:** target height 0.15–1.1 m, distance 1–2 m from the origin; robot starts within 1 m, any heading.
   Higher (1.1–1.25 m) and farther (2–2.5 m) targets are **held back for testing** generalisation.
5. **Uniform over area,** not radius, so targets are not bunched near the centre.
6. **Log before the stop check,** so the final step of every run is kept.
7. **Save the expert's hidden state** (base phase, arm reference). The model may not use it, but it explains the expert's behaviour.
8. **Crash check skips the arm's mounting plate,** which sits inside the base by design and would always count as a collision.
9. **Noise mode (for week 3):** noise goes on the command actually sent; the label stays the expert's clean command, so the model learns to recover.
10. **One file per run, written to a temp file then renamed:** a crash loses only one run, and a half-written file never looks complete.
11. **Seed blocks:** 0–99,999 for training, 1,000,000+ for the benchmark, so test scenes can never leak into training.

## Known expert behaviour

- **Seeds 234, 282 fail:** the robot starts close to its parking spot, misses it, and circles
  (tightest turn about 0.8 m radius) until the 25 s limit. **Excluded from training.**
- **Seed 189** is counted as reached: the template checks "stalled" before "reached", so a
  0.76 cm run was mislabelled. We judge success by final error < 1 cm.
- Arm torques jitter about ±2 Nm while driving and spike to the ±30 Nm limit when reaching starts:
  noisy labels to keep in mind when training.

**Next (Stage 2):** clean the data, split by run (70/15/15), freeze the benchmark scenes.
