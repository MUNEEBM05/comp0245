"""Stage B noise check: how much tier-4 sensor noise shakes each of K's inputs.

For clean recorded runs, build K's 50 inputs twice: once from the true measurements and once
after adding the same noise tier 4 adds while driving (rollout.py: 0.02 on base x/y, yaw and arm
angles; 0.2 on arm speeds). The difference, divided by the input's spread in K's training data
(x_std, what the network actually sees), says how many "training widths" each input jitters by.
Above about 0.3 the network is being fed mostly noise on that input.

The hand position is kept noise-free here (recomputing it needs the simulator), so the err_*
rows are a lower bound.

    python noise_check.py
"""
import glob
import json

import numpy as np

from features import run_xy_full

SIGMA = 0.02          # tier-4 obs_noise (benchmark/scenes.json)
N_RUNS = 30

model = np.load("trained/vK_kick.npz", allow_pickle=True)
meta = json.loads(str(model["meta"]))
names, x_std = meta["feature_names"], model["x_std"].ravel()
rng = np.random.default_rng(0)

diffs = []
for path in sorted(glob.glob("data/runs/run_*.npz"))[:N_RUNS]:
    clean = dict(np.load(path, allow_pickle=True))
    noisy = dict(clean)
    n = len(clean["t"])
    noisy["base_pos"] = clean["base_pos"].copy()
    noisy["base_pos"][:, :2] += rng.normal(0, SIGMA, (n, 2))
    noisy["base_bearing"] = clean["base_bearing"] + rng.normal(0, SIGMA, n)
    noisy["q_mes"] = clean["q_mes"] + rng.normal(0, SIGMA, clean["q_mes"].shape)
    noisy["qd_mes"] = clean["qd_mes"] + rng.normal(0, 10 * SIGMA, clean["qd_mes"].shape)
    kw = dict(history=True, goal=True, integral=True)   # K's input groups
    Xc, _ = run_xy_full(clean, **kw)
    Xn, _ = run_xy_full(noisy, **kw)
    diffs.append(Xn - Xc)

ratio = np.concatenate(diffs).std(axis=0) / x_std
for i in np.argsort(-ratio):
    print(f"{names[i]:>18s}  {ratio[i]:6.2f}" + ("   <-- mostly noise" if ratio[i] > 0.3 else ""))