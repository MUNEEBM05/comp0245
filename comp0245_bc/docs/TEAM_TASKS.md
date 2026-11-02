# COMP0245 behaviour cloning – team tasks (Thu 29 Oct → submission Fri 6 Nov)

This file is written so you can paste it into your own LLM assistant together with the project files.
Everything you need is in the project folder from `comp0245_code_models_29oct.zip`
(code, trained models, all result CSVs). Read `WHATS_NEW.md` first (one page: the result and how we got it),
then `PREPROCESSING.md` (the data pipeline) and `README.md` (how the code is organised).

---------------------------------------------------------------------------------------------------
## Where we are (one paragraph)

We clone the template's expert (Robotnik base + Kinova 7-joint arm) with a supervised regressor that maps
measurements + target to 7 arm torques + 4 wheel speeds. Our main model is **model R**
(`trained/vR_ik.npz`, an MLP with separate arm and wheel networks). On 50 FRESH test scenes it reaches the
target (within 1 cm) in 94% of scenes (expert 98%) and reaches AND holds still for 5 s in 86% (expert 98%).
Old model K held only 30%. What made the difference: (1) a test that showed the ARM, not the base, was the
problem; (2) recordings where the expert keeps holding still after arriving, plus extra low/high targets;
(3) new inputs from the arm's geometry (Jacobian): which way each joint should move to close the gap.
GRU, convolution (TCN) and attention models all did worse than the MLP with the right inputs.

## Plan to the deadline

| When | Who | What |
|---|---|---|
| Thu 29 – Fri 30 | David (compute) | "Objects anywhere" test (tier 6), regression baselines, Stage B on all models, Bayesian optimisation |
| Thu 29 – Fri 30 | Muneeb | Plots and diagrams (below) |
| Thu 29 – Fri 30 | Abdullah | Reproduce model R on Windows + Stage B tests on model R, then code quality and reproducibility for the markers (below) |
| Sat 31 – Mon 2 | Everyone | Clean-up, understanding, process/code diagrams, results and benchmark figures |
| Mon 2 – Wed 4 | Everyone | **First full draft by Wednesday 4 Nov** (6-page IEEE) for review |
| Thu 5 | Everyone | Review, fix, demo rehearsal |
| Fri 6 | Everyone | Submit (report + code on Moodle; data on OneDrive) |

**Decided:** Stage C (obstacles + safety layer) is SKIPPED – no time. Mention it as future work.

---------------------------------------------------------------------------------------------------
## Muneeb – plots and figures (no simulator needed)

Use Python + matplotlib + pandas, reading the CSVs. Save PNGs into `figures/` at dpi 200, readable in a
two-column IEEE paper (column width ~3.5 in, so keep figures ~3.5 x 2.5 in, fonts >= 8 pt).
`make_figures.py` already draws older versions of some of these and can be extended.

### The result files
- `results/<model>_tier<k>.csv` – one row per test scene, on the 50 frozen benchmark scenes of tier k.
- `results/<model>_hold5_tier<k>.csv` – same, but success = stay within 1 cm for 5 s in a row.
- `results/3_final_tests/fresh_scenes/<model>[_hold5]_tier1.csv` – 50 FRESH tier-1 scenes (the honest final test).
- `<model>_mlp_arm_hold5` = expert drives the base, our model the arm; `_mlp_wheels_` = the reverse.
- `_s1`, `_s2` = the same recipe trained with random seed 1, 2 (no suffix = seed 0).

Columns: `index` (scene), `success` (True/False), `outcome` (reached / reached_not_held / timeout /
crashed_ground / crashed_base / tipped_over), `final_err_cm`, `min_err_cm` (closest the hand got),
`time_s` (drive length), `first_reach_s`, `think_us_per_step` (decision time, microseconds),
`goal_dist_end_m`, `face_err_end_deg`, `torque_jerk`, `wheel_jerk`, `tier`, `policy`.
Success rate = mean of `success`. Use a Wilson 95% interval for error bars on 50 scenes (see `compare.py`).

Tiers: 1 normal, 2 far/high targets (outside training range), 3 unusual starts, 4 sensor noise + command
delay, 5 changed physics (floor friction, base mass), 6 objects anywhere (some unreachable).

### Figures to make (priority order)
1. **Ladder (the story)**: success on tier 1 for each version, "reached" and "held 5 s" side by side.
   Old versions (reached only): `mlp_v0, mlp_v1, mlp_v1_hist, mlp_v2_hist, vA_goal, vD_all, vD2_reachdata,
   vE_integral, vK_kick`. New: `vP_hold` (+ hold/extreme data), `vR_ik` (+ arm geometry inputs), `expert`.
   Hold versions: `vK_kick_hold5, vP_hold_hold5, vR_ik_hold5, expert_hold5`.
2. **Fair model comparison** (same data, same inputs): MLP `vP_hold, vP_hold_s1, vP_hold_s2`;
   convolution `tcn2, tcn2_s1, tcn2_s2`; attention `attn`; GRU `gru_P`; each with `_hold5`. Show each
   training run as a dot and the mean as a bar. Message: the MLP wins; training randomness is large.
3. **Arm vs base diagnosis** (held 5 s, tier 1): expert `expert_hold5` (49/50); our base + expert arm
   `vP_hold_mlp_wheels_hold5` (50/50); expert base + our arm `vP_hold_mlp_arm_hold5` (18/50); ours
   `vP_hold_hold5` (17/50); after the fix `vR_ik_mlp_arm_hold5` and `vR_ik_hold5`. Message: the arm was the
   bottleneck, and the IK inputs fixed it.
4. **Generalisation**: model R vs expert across tier 1, fresh, tier 2, tier 3 (and tier 6 when ready),
   reached and held.
5. **Seeds**: `vR_ik, vR_ik_s1, vR_ik_s2` (held 46, 49, 44) vs `vP_hold*` (17, 12, 9) – consistency.
6. **Offline vs online**: validation error (in each model file's meta, key `val_mse_standardised`; load with
   `json.loads(str(np.load(path)['meta']))`) vs closed-loop success. Message: offline error does not predict
   driving (GRU fits better offline than K but drives far worse).
7. **Pipeline diagram** (draw.io / PowerPoint / matplotlib): use the "Pipeline" section of `PREPROCESSING.md`.
   Show: recordings (1 kHz sim, logged 100 Hz) → cleaning + outcome fix → split by drive → input groups
   [basic 26 | history 13 | parking spot 5 | integral 6 | IK 21] + targets (torque − gravity, wheel speeds)
   → standardise → MLP (arm net + wheel net) → live 1 ms loop (measure → inputs → model → + gravity → clip).
8. Already made, check they still look right: `figures/data_targets_robot_frame.png`,
   `figures/data_coverage.png`, `figures/data_expert_quality.png` (from `data_report.py`).

---------------------------------------------------------------------------------------------------
## Abdullah – reproduce model R on Windows, Stage B tests, then code quality for the markers

Work in the RoboEnv pixi shell, in the project folder (the code zip unzipped on top of the Stage A files).
No training is needed: `trained/vR_ik.npz` is included. Use `--workers 3` (≈ CPU cores − 1).

### 1. Reproducibility check (≈10 min)
```
python evaluation/evaluate.py --policy trained\vR_ik.npz --tiers 1 --n 10 --hold 5 --workers 3 --out results_win
```
Expect about 8–10 of 10 "reached" (held). Report the number. If it is far lower, tell David at once (the
model's inputs are checked to be identical in training and driving on Linux; a mismatch on Windows would
point to a library difference).

### 2. Stage B tests on model R (each 20–40 min)
```
python evaluation/evaluate.py --policy trained\vR_ik.npz --tiers 4 --n 50 --workers 3 --out results
python evaluation/evaluate.py --policy trained\vR_ik.npz --tiers 4 --n 50 --workers 3 --out results --only noise
python evaluation/evaluate.py --policy trained\vR_ik.npz --tiers 4 --n 50 --workers 3 --out results --only delay
python evaluation/evaluate.py --policy trained\vR_ik.npz --tiers 5 --n 50 --workers 3 --out results
python evaluation/compare.py --results results --policies expert vK_kick vR_ik --tier 4
python evaluation/compare.py --results results --policies expert vK_kick vR_ik --tier 5
```
For comparison, already in `results`: tier 4 expert 41/50, K 0/50; noise only K 6/50; delay only
expert 42/50, K 1/50; tier 5 expert 50/50, K 35/50.
Tier 4 = sensor noise (std 0.005/0.01/0.02, fixed per scene, fresh random values every 1 ms, only on what
the controller SEES) + command delay 1–4 ms. Note: the EXPERT itself crashes sometimes with 2–4 ms delay.
Send back the new CSV files (`results/vR_ik_tier4.csv`, `vR_ik_noiseonly_tier4.csv`,
`vR_ik_delayonly_tier4.csv`, `vR_ik_tier5.csv` and the matching `_summary.json`).

### 3. Code quality and reproducibility for the markers (main job after the tests)
Goal: a marker unzips our code, follows ONE page, sees the robot drive and gets our numbers.
What the brief requires in the README: (1) how to run the code and which outputs it generates;
(2) which external code was used and which parts were modified or written from scratch.
Data and trained weights are NOT submitted on Moodle (they go on OneDrive / Hugging Face with a link).

Checklist:
- [ ] **Fresh-folder test**: unzip the code into an EMPTY folder on Windows, follow only the README, and note
      every place you get stuck. Fix the README (or the code) until it works without help.
- [ ] **Quick start in the README (top)**: 3 commands, each with what you should see and how long it takes:
      (a) watch model R drive one scene in the GUI, e.g.
      `python -c "from rollout import *; m,_=rollout(make_policy('trained/vR_ik.npz'), {'seed':0,'target':[0,1.1,0.5],'start_pose':[0,0,0]}, use_gui=True, hold_s=5); print(m['outcome'], m['final_err_cm'], 'cm')"`
      (b) the 10-scene test (`evaluate.py ... --n 10 --hold 5`), (c) the results table (`compare.py`).
      Consider wrapping (a) into a small `demo.py` with `--model` and `--scene` options and a clear printout.
- [ ] **Reproduce section**: the exact commands, in order, from recording data to the final table
      (see "Reproduce R" in `WHATS_NEW.md`), with expected run times on a normal laptop and expected results
      (with our seed-to-seed spread, so a marker knows what "the same" means).
- [ ] **Script map**: one table, every `.py` file, one line on what it does, "core" vs "experiment"
      (experiments: `dagger.py`, `markov_expert.py`, `train_gru.py`, `train_conv.py`, ...). Consider moving
      the batch `run_*.sh` files into an `experiments/` folder and saying they are Linux command lists.
- [ ] **Sources and changes**: template files used unchanged (`robotnik_arm_control.py`, config JSON);
      template main loop turned into `expert_episode.py` (what changed: function, random starts, recording,
      hold/noise options; controllers untouched); everything else written by us; libraries used
      (PyTorch, NumPy, Pinocchio, PyBullet / simulation_and_control, scikit-learn, matplotlib).
- [ ] **Every script runs politely**: `python <script> --help` works and explains the options; a missing
      file gives a one-line message, not a traceback (we use `quiet.run_main` for this).
- [ ] **Dependencies**: say which pixi environment to use; PyTorch is only needed for TRAINING (driving
      uses NumPy only); list versions (`pip list` in the environment).
- [ ] **Determinism**: note that scenes and recordings are seeded (same seed → same scene and noise),
      training seeds are set, and results may differ slightly across machines.
- [ ] **Data link**: README line with the OneDrive / Hugging Face link and the expected folder layout
      (`data/runs`, `data/runs_hold`, ...), and which folders are needed for which model.

### 4. If time is left: report reading list (related work section)
Behaviour cloning compounding errors and DAgger (Ross, Gordon & Bagnell 2011); noise injection DART
(Laskey et al. 2017); HG-DAgger (Kelly et al. 2019); domain randomisation (Tobin et al. 2017); action
chunking / ACT (Zhao et al. 2023); robomimic "What matters in learning from offline human demonstrations"
(Mandlekar et al. 2021); copycat problem (Wen et al. 2020); GRU (Cho et al. 2014); TCN (Bai et al. 2018).
Write 3–4 sentences each on how it relates to what we did (we tried DAgger, HG-DAgger, DART-style nudges,
domain randomisation, chunking, GRU, TCN, attention – see the table in `WHATS_NEW.md`).

---------------------------------------------------------------------------------------------------
## Rules for everyone
- Never edit `benchmark/scenes.json` or `benchmark/scenes_fresh.json` (frozen test scenes).
- Never train or tune on the fresh scenes; they are for final numbers only.
- Always quote "reached" (template rule) and "reached and held 5 s" (our stricter rule), both within 1 cm.
- One training run is not enough: compare means over seeds where we have them.
- Open question for the TA: the arm IK inputs reuse the template's IK formula (physics-informed feature
  engineering, like gravity / forward kinematics). Ask before the report is final.
