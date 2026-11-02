"""Train the wheel network on the teacher's recordings (record_teacher.py): sensor inputs -> the planner's wheel commands.

    pixi run -e ml python train_driver.py --name driver_A            # version A: lidar + next waypoint + parking spot + motion
    pixi run -e ml python train_driver.py --name driver_B --no-waypoint        # version B: no waypoint, no planner when driving
    pixi run -e ml python train_driver.py --name driver_A_p50 --fraction 0.5   # learning curve: use half of the drives

Same learner as train.py: MLP (3 x 256, ReLU), Adam, weighted MSE, early stopping on validation. Only successful drives are used.
Inputs and outputs are standardised with training statistics; validation is data/auto_teacher_val.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from driver_features import NAMES, WAYPOINT_COLUMNS, from_run
from quiet import run_main
from train import forward, train_mlp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(folder, fraction=1.0, seed=0):
    files = sorted(glob.glob(os.path.join(folder, "run_*.npz")))
    good = []
    for f in files:
        with np.load(f) as z:
            if json.loads(str(z["meta"]))["outcome"] == "reached":
                good.append(f)
    if fraction < 1.0:
        keep = np.random.default_rng(seed).permutation(len(good))[: int(round(len(good) * fraction))]
        good = [good[i] for i in sorted(keep)]
    X, Y = [], []
    for f in good:
        with np.load(f) as z:
            a = {k: z[k] for k in z.files if k != "meta"}
        X.append(from_run(a))
        Y.append(a["wheel_cmds"])
    return np.concatenate(X), np.concatenate(Y), len(good), len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", default=[os.path.join(ROOT, "data", d) for d in ("auto_teacher", "auto_teacher_nudged")])
    ap.add_argument("--val", default=os.path.join(ROOT, "data", "auto_teacher_val"))
    ap.add_argument("--no-waypoint", action="store_true", help="version B: drive without the planner's waypoint")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    out = os.path.join(HERE, "trained", f"{a.name}.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; pick a new --name")
    np.random.seed(a.seed)
    t0 = time.time()
    parts = [load(d, a.fraction, a.seed) for d in a.data]
    Xtr, Ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    Xva, Yva, n_val, _ = load(a.val)
    cols = [i for i in range(len(NAMES)) if not (a.no_waypoint and i in WAYPOINT_COLUMNS)]
    Xtr, Xva = Xtr[:, cols], Xva[:, cols]
    used = [(os.path.basename(os.path.normpath(d)), p[2], p[3]) for d, p in zip(a.data, parts)]
    print(f"{a.name}: {len(Xtr)} training rows from {used}, {len(Xva)} validation rows, {len(cols)} inputs")
    x_mean, x_std, y_mean, y_std = Xtr.mean(0), Xtr.std(0) + 1e-6, Ytr.mean(0), Ytr.std(0) + 1e-6
    x_lo, x_hi = Xtr.min(0), Xtr.max(0)
    xtr, ytr = ((Xtr - x_mean) / x_std).astype(np.float32), ((Ytr - y_mean) / y_std).astype(np.float32)
    xva, yva = ((Xva - x_mean) / x_std).astype(np.float32), ((Yva - y_mean) / y_std).astype(np.float32)
    Ws, bs, history = train_mlp(xtr, ytr, xva, yva, a)
    pred = forward(Ws, bs, xva) * y_std + y_mean
    mae = float(np.abs(pred - Yva).mean())
    meta = {"name": a.name, "model": "mlp_driver", "version": "B" if a.no_waypoint else "A", "columns": cols,
            "feature_names": [NAMES[i] for i in cols], "hidden": a.hidden, "seed": a.seed, "fraction": a.fraction,
            "train_rows": int(len(Xtr)), "drives": used, "val_mae_rad_s": mae, "history": history}
    weights = {f"W{i}": W for i, W in enumerate(Ws)}
    weights.update({f"b{i}": b for i, b in enumerate(bs)})
    np.savez(out, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, x_lo=x_lo, x_hi=x_hi, n_layers=np.array(len(Ws)),
             meta=np.array(json.dumps(meta)), **weights)
    print(f"  validation error {mae:.3f} rad/s per wheel | best epoch {int(np.argmin([h['val_mse'] for h in history])) + 1} "
          f"of {len(history)} | saved {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    run_main(main, HERE)
