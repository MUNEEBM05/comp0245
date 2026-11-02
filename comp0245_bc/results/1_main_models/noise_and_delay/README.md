# Noise and delay

How the models cope with **sensor noise** (every measurement wobbles) and **command delay** (the robot
receives each command 1–4 ms late). Benchmark tier 4 has both: tier-1 scenes plus sensor noise of standard
deviation 0.005–0.02 and a 1–4 ms delay. Run every command from `comp0245_bc` (see the main README).

## The story in one table

| Step | Model | What changed | Result |
|---|---|---|---|
| Problem | K (`vK_kick`) | none | tier 4: **0/50** reached. The expert gets 82% |
| Diagnosis | K split | `noise_and_delay/tier4_split.py` runs tier 4 with only the noise or only the delay | noise-only 8/50, delay-only 1/50: both hurt |
| Diagnosis | noise check | `noise_and_delay/noise_check.py` measures how far noise moves each input | arm speeds and 0.1 s history changes are mostly noise |
| 1 | **L** K-noisy (`vL_disturbed`) | K + recordings of the expert under noise and delay | tier 4: 5, noise-only 13, delay-only 10 |
| 2 | **M** K-filtered (`vM_smooth`) | L + a low-pass filter (τ = 0.05 s) on arm speeds and history | crashes even on clean scenes: the lagged speed brakes the arm too late |
| 3 | **M2** K-history-filter (`vM2_smooth_hist`) | filter on the history inputs only, arm speeds raw | tier 4: 8, noise-only 19, delay-only 14 |
| 4 | **M2-noise0** K-clean-labels (`vM2_noise0`) | M2 trained only on the disturbed recordings made **without** sensor noise | **best K-based model**: tiers 1–4: 40, 30, 36, 14; noise-only 25 |
| 5 | **N** K-delay (`vN_delay`) | M2-noise0 + delay-only recordings (`runs_delay`) | delay-only 25, but tiers 1–4: 37, 28, 32, 14 |
| Fix | R + position filter (`vR_ik:filt20noqd`) | moving average of positions over 20 ms (not speeds: filtering speeds breaks R too, as in M) | noise 0.01, held 5 s: 41/50 (R alone: 4) |
| **Fix used** | R + Kalman filter (`vR_ik:kfv`) | Kalman filter that also estimates speeds | noise 0.01, held 5 s: **46/50** |
| Final | R-BO + Kalman (`bo_t03:kfv`) in the final system S3 | the final arm | noise 0.01 on fresh scenes, held 5 s: 92% (super test) |

Numbers are reached within 1 cm out of 50. Steps 1–5 (Muneeb) were run on Muneeb's laptop, where K scores
31/50 on tier 1 instead of 36 on David's; every result there is from that laptop, so they compare with each
other exactly.

## Folders

| Folder | What |
|---|---|
| `K_diagnosis/tiers/<model>/tier1 … tier4/` | Muneeb's models K, L, M2, M2_noise0, N (and the expert, tier 4) on benchmark tiers 1–4 |
| `K_diagnosis/noise_only/<model>/` | tier 4 with only the sensor noise (`benchmark/tier4_noise_only.json`); also `noise_check.txt` |
| `K_diagnosis/delay_only/<model>/` | tier 4 with only the delay (`benchmark/tier4_delay_only.json`) |
| `filters_and_robust_models/<system>/` | David's tests of R under noise, delay, payload and motor noise: the filters, the Kalman filter, the ensemble, and models trained to be robust (noise augmentation, dropout, weight decay, domain randomisation) |

## Reproduce

**Record** the disturbed expert data (seeded, so identical every time):

```bash
python3 noise_and_delay/collect_noisy.py --start 40000 --end 40160 --obs-noise 0 0.005 0.01 0.02 --delay 0 1 2 3 4 --out data/runs_disturbed
mkdir -p data/runs_disturbed_n0 && cp data/runs_disturbed/run_400{00..156..4}.npz data/runs_disturbed_n0/   # the 40 noise-free ones
python3 noise_and_delay/collect_noisy.py --start 50000 --end 50160 --obs-noise 0 --delay 1 2 3 4 --out data/runs_delay
```

`collect_noisy.py` is the exact recorder used for these sets: each run gets one noise level and one delay
in turn by seed, so every pairing appears equally often. (`training/collect.py` draws them at random instead.)

**Train** (needs `pixi shell ... -e ml`):

```bash
K="--history deltas --goal --separate --reach-weight 3 --integral"
D="data/runs_nudged data/runs_reach data/runs_kick"
python3 training/train.py --name my_L  $K --extra-train $D data/runs_disturbed
python3 training/train.py --name my_M  $K --smooth 0.05 --extra-train $D data/runs_disturbed
python3 training/train.py --name my_M2 $K --smooth 0.05 --smooth-history-only --extra-train $D data/runs_disturbed
python3 training/train.py --name my_M2_noise0 $K --smooth 0.05 --smooth-history-only --extra-train $D data/runs_disturbed_n0
python3 training/train.py --name my_N  $K --smooth 0.05 --smooth-history-only --extra-train $D data/runs_disturbed_n0 data/runs_delay
```

**Test**:

```bash
python3 evaluation/evaluate.py --policy trained/vM2_noise0.npz --tiers 1 2 3 4 --workers 4
python3 noise_and_delay/tier4_split.py           # writes the two split scene files (already in benchmark/)
python3 evaluation/evaluate.py --policy trained/vM2_noise0.npz --tiers 4 --scenes benchmark/tier4_noise_only.json --workers 4
python3 evaluation/evaluate.py --policy trained/vM2_noise0.npz --tiers 4 --scenes benchmark/tier4_delay_only.json --workers 4
python3 evaluation/evaluate.py --policy trained/vR_ik.npz:kfv --tiers 1 --hold 5 --set obs_noise=0.01 --workers 4
python3 noise_and_delay/noise_check.py           # the input-noise table
python3 evaluation/compare.py --results results/1_main_models/noise_and_delay/K_diagnosis/tiers --tier 4
```
