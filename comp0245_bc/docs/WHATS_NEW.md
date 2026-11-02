# What's new since the Stage A zip (27–29 Oct)

## The result
**Model R** (`trained/vR_ik.npz`) on 50 FRESH test scenes (`benchmark/scenes_fresh.json`, never used for any choice):

| Model | Reached (template rule) | Reached AND held within 1 cm for 5 s | Crashes |
|---|---|---|---|
| Expert | 49/50 (98%) | 49/50 (98%) | 0 |
| **R = P + arm IK inputs** | **47/50 (94%)** | **43/50 (86%)** | 1 |
| P = K + hold/extreme data | 41/50 (82%) | 20/50 (40%) | 1 |
| K (Stage A model) | 30/50 (60%) | 15/50 (30%) | 1 |

Three training runs of R on the benchmark scenes: held 46, 49, 44 /50 (P: 17, 12, 9). Decision time 0.38 ms/step.

## How R was obtained (in order)
1. **Hold-still data** – the template stops recording the moment the hand arrives, so the model never saw "arrived, stay put".
   `collect.py --hold 5` keeps recording 5 s after arrival (expert unchanged). → `data/runs_hold` (200 drives)
2. **Extreme heights** – failures clustered at very low / very high targets (`data_report.py`).
   `collect.py --z-bands 0.15 0.3 0.95 1.1` → `data/runs_extreme` (150 drives)
3. **Arm IK inputs** (`ik_features.py`, `train.py --ik`) – the hold test showed the ARM was the problem
   (our base + expert arm: 50/50 held; expert base + our arm: 18/50). New inputs: the joint speeds the arm's
   geometry asks for to close the hand error (Jacobian, damped least squares, the template's IK formula and
   constants) + their 0.5 s / 2 s running averages. Computed from measurements only, like the hand position.
   **Raise with the TA**: we argue this is physics-informed feature engineering (same category as FK / gravity).

Reproduce R:
```
python training/collect.py --start 60000 --end 60200 --hold 5 --out data/runs_hold --workers 3
python training/collect.py --start 61000 --end 61150 --hold 5 --z-bands 0.15 0.3 0.95 1.1 --out data/runs_extreme --workers 3
pixi run -e ml python training/train.py --name vR_ik --ik --history deltas --goal --separate --reach-weight 3 --integral \
    --extra-train data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme
python evaluation/evaluate.py --policy trained/vR_ik.npz --tiers 1 --hold 5 --scenes benchmark/scenes_fresh.json --out results/3_final_tests/fresh_scenes --workers 3
python evaluation/evaluate.py --policy trained/vR_ik.npz:mlp_arm --tiers 1 --n 50 --hold 5 --out results   # arm-only test
```
(First training with `--ik` computes the IK inputs for every recording, ~4 min, cached in `data/.ik_cache`.)

## New tools
| File | What |
|---|---|
| `evaluate.py --hold S` | stricter success: stay within 1 cm for S seconds (outcome `reached_not_held` otherwise) |
| `evaluate.py --only noise/delay` | tier-4 diagnosis: one disturbance at a time |
| `evaluate.py --scenes FILE` | e.g. the fresh test scenes |
| `rollout.py` | GRU, TCN (convolution) and attention drivers; hold rule; tier-5 physics; `markov_expert` policy |
| `ik_features.py` | arm IK inputs (above) |
| `data_report.py` | where targets are seen from the robot, coverage vs failures, expert-drive quality → `splits/bad_runs.json` |
| `train.py --exclude/--hold-weight/--ik` | drop bad drives / weight hold rows / IK inputs |
| `train_gru.py`, `train_conv.py --arch tcn|attn` | memory models (resumable checkpoints) |
| `dagger.py`, `markov_expert.py` | DAgger, HG-DAgger, memory-free expert labeller |
| `collect.py --hold/--z-bands/--obs-noise/--delay/--friction/--mass` | hold data, height bands, domain randomisation |
| `make_benchmark.py --add-tier 5` | tier 5 = changed physics (friction 0.5–1.3, base mass x0.7–1.3) |
| `run_*.sh` | the exact batch scripts used (Linux; read them as command lists on Windows) |

## Experiments that did NOT beat the MLP (for the report)
| Experiment | Result (tier 1, /50) | Why |
|---|---|---|
| Plain DAgger | 1 reached | expert has hidden memory (phase, timeout) → contradictory labels |
| HG-DAgger | 25 | fixed the contradictions, too little new data |
| DAgger with memory-free expert (2 rounds) | 28, 29 | base fixed; arm's creeping reference still hidden |
| Domain randomisation (model M) | 32; tier-4 noise-only 11 vs 6 | slightly more noise-robust, jumpier elsewhere |
| Everything mixed (model O) | 19 | extra data pulls the fit away from the clean expert |
| Bad-drive removal | 34 vs 36 | only 3% flagged (circling near the spot); no effect |
| Hold rows weighted x3 / x6 | held 12 / 6 vs 17 | "almost there" and "arrived" look alike → freezes early |
| GRU (same data as P) | 12 reached, 5 held | learned memory < hand-made memory at equal effort |
| Convolution v2 (3 runs) | 31/22/23 reached, 18/8/3 held | smoother holding on its best run only |
| Attention (transformer) | 16 reached, 1 held | worst of the window models while driving |

## Findings worth a figure
- Offline error does not predict driving (GRU: better offline fit, worse driving).
- Training randomness ≈ design changes (P: 23–40 reached over 3 runs) → report means over runs.
- The expert itself crashes with 2–4 ms command delay; physics changes (tier 5) barely hurt K (35 vs 36).
- Direction of the target does not matter; height extremes do (`figures/data_coverage.png`).

## Next: Stage B on model R
Noise, delay, torque noise, payload, physics (tiers 4–5) for R, P, K and the expert; then Bayesian optimisation.
