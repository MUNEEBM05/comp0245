"""Gaussian process regressor on model R's exact inputs and targets (David's feature file).

    python train_gp.py                         # 4,000 rows -> trained/gp_R.pkl (about 8 min)
    python train_gp.py --rows 2000 --name gp_R_2k

A GP stores its training rows and compares every new moment with all of them, so training cost grows
with rows^2 and every decision with rows: only a few thousand of R's ~1 million rows fit. The feature
file already drew reaching moments 3x more likely (like R's reach weight), so rows are drawn evenly
from it by default; --reach-weight tilts them further towards reaching.
The saved .pkl drives through evaluate.py exactly like the random forest (rollout.MLPPolicy).
"""
import argparse
import os
import pickle
import time

import numpy as np

from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))

# Which inputs R uses, so rollout.py rebuilds the same 71 when driving (see train.py meta)
R_META = {"model": "gp", "torque": "minus_gravity", "history_features": "deltas", "goal_features": True,
          "integral_features": True, "ik_features": True, "action": "torque"}


def main():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(HERE, "data", "R_features_for_regressors.npz"))
    ap.add_argument("--rows", type=int, default=4000, help="training rows the GP keeps")
    ap.add_argument("--reach-weight", type=float, default=1.0,
                    help="how much likelier a reaching row is drawn (on top of the file's own 3x)")
    ap.add_argument("--name", default="gp_R")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not os.path.exists(a.data):
        print(f"Feature file not found: {a.data} (David's R_features_for_regressors.npz)")
        return

    d = np.load(a.data, allow_pickle=True)
    x_mean, x_std, y_mean, y_std = d["x_mean"], d["x_std"], d["y_mean"], d["y_std"]
    xtr, ytr = (d["X_train"] - x_mean) / x_std, (d["Y_train"] - y_mean) / y_std
    xva, yva = (d["X_val"] - x_mean) / x_std, (d["Y_val"] - y_mean) / y_std
    reach_va = d["reaching_val"].astype(bool)

    rng = np.random.default_rng(a.seed)
    p = np.where(d["reaching_train"].astype(bool), a.reach_weight, 1.0)
    idx = rng.choice(len(xtr), size=a.rows, replace=False, p=p / p.sum())

    # RBF kernel: moments with nearby (standardised) inputs get similar outputs. One shared length
    # scale for all 71 inputs; a separate scale per input needs more memory than a laptop has.
    # WhiteKernel = measurement noise. Both are tuned by maximising the marginal likelihood.
    kernel = ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(1e-2, noise_level_bounds=(1e-8, 1.0))
    gp = GaussianProcessRegressor(kernel=kernel, random_state=a.seed)
    print(f"{a.name}: GP on {a.rows} rows ({d['reaching_train'][idx].mean():.0%} reaching), fitting ...", flush=True)
    t0 = time.time()
    gp.fit(xtr[idx], ytr[idx])
    fit_s = time.time() - t0

    pred = gp.predict(xva)
    val_mse = float(((pred - yva) ** 2).mean())
    reach_mse = float(((pred[reach_va] - yva[reach_va]) ** 2).mean())
    mae = (np.abs(pred - yva) * y_std).mean(0)
    times = []
    for i in range(200):                       # one decision = one row, as when driving
        t1 = time.perf_counter()
        gp.predict(xva[i:i + 1])
        times.append(time.perf_counter() - t1)
    think_ms = 1e3 * float(np.median(times))

    print(f"  kernel after tuning: {gp.kernel_}")
    print(f"  validation standardised MSE {val_mse:.4f} (model R 0.0057) | reaching rows only {reach_mse:.4f}")
    print(f"  MAE: torque minus gravity {mae[:7].mean():.3f} N m (R 0.121) | wheels {mae[7:].mean():.3f} rad/s (R 0.026)")
    print(f"  one decision {think_ms:.2f} ms (budget 1 ms, plus building the inputs) | fit {fit_s:.0f} s")

    meta = dict(R_META, name=a.name, rows=a.rows, reach_weight=a.reach_weight, seed=a.seed,
                kernel=str(gp.kernel_), val_mse_standardised=val_mse, val_mse_reaching=reach_mse,
                val_mae=dict(zip([str(n) for n in d["output_names"]], map(float, mae))),
                predict_ms=think_ms, fit_s=fit_s)
    out = os.path.join(HERE, "trained", f"{a.name}.pkl")
    with open(out, "wb") as fh:
        pickle.dump({"forest": gp, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                     "x_lo": d["X_train"].min(0), "x_hi": d["X_train"].max(0), "meta": meta}, fh)
    print(f"  saved {os.path.relpath(out, HERE)} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    run_main(main, HERE)
