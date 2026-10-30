"""Train an MLP with DAgger recordings and near-target weighting. Saves to trained_dagger/ only.

    pixi run -e ml python train_dagger.py --name vK_dagger1 --history deltas --goal --separate --reach-weight 3 \
        --integral --extra-train data/runs_nudged data/runs_reach data/runs_kick \
        --dagger data/runs_dagger1 --near-weight 3

Same learner, inputs and outputs as train.py (and it reuses train.py's functions), with three changes:
    --dagger DIR...        DAgger recordings (dagger_collect.py) added to TRAINING. Unlike --extra-train,
                           runs that did not reach the target are KEPT: the states where the model got
                           stuck are exactly what DAgger is for.
    --dagger-radius R      keep only the DAgger rows where the arm is reaching and the hand is within
                           R m of the target (the final approach); default: all rows
    --noise-aug S...       add one noisy copy of every TRAINING run (validation stays clean). Each copy draws
                           one sigma from S and perturbs the measurements exactly like rollout.py's
                           tier-4 sensor noise: base x/y, heading and joint angles +-sigma, joint
                           speeds +-10 sigma, hand position recomputed from the noisy joints (FK).
                           Labels stay the expert's clean commands: "noisy input -> what the expert did"
    --near-weight W        rows where the arm is reaching AND the hand is within --near-radius of the
    --near-radius R        target get an extra weight W in the loss (default R = 0.03 m)
    --arm-only DIR...      expert recordings used for the ARM network only (collect_more.py --scene arm)
    --integral-taus T...   averaging windows (s) for the --integral inputs, e.g. 0.5 2 5 10
                           (default 0.5 2, as train.py); saved in the model so rollout.py rebuilds them
    --wheel-taus T...      the WHEEL network only sees these windows (e.g. 0.5 2); the arm sees all of
                           --integral-taus. Saved as per-network input columns in the model
    --cut-hold             drop the rows recorded after the first reach in --hold recordings, so the
                           same files also give the "more data, no hold" comparison
    --out-dir              default trained_dagger/. Writing into trained/ is refused, and an existing
                           model is never overwritten (pick a new --name)

Validation is still the same clean runs (splits/split.json), so val MSE is comparable with train.py.
MLP only (torque outputs); the linear / forest / reference ablations stay in train.py.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from check_runs import _outcome, load_run
from features import INTEGRAL_TAUS_S, feature_names_full, integral_names, output_names, run_integral
from quiet import run_main
from train import _find_run, forward, run_data, train_mlp

HERE = os.path.dirname(os.path.abspath(__file__))


_FK_WORLD = None


def noisy_copy(arrays, sigma, rng):
    """The run as a tier-4 sensor would have reported it (same noise model as rollout.py)."""
    global _FK_WORLD
    if _FK_WORLD is None:
        from rollout import World
        _FK_WORLD = World([0.0, 0.0, 0.0], [1.0, 0.0, 0.5])
    n = len(arrays["t"])
    bp = arrays["base_pos"].astype(float).copy()
    bp[:, :2] += rng.normal(0, sigma, (n, 2))
    yaw = arrays["base_bearing"].astype(float) + rng.normal(0, sigma, n)
    quat = np.column_stack([np.zeros(n), np.zeros(n), np.sin(yaw / 2), np.cos(yaw / 2)])   # x y z w
    q = arrays["q_mes"].astype(float) + rng.normal(0, sigma, arrays["q_mes"].shape)
    qd = arrays["qd_mes"].astype(float) + rng.normal(0, 10 * sigma, arrays["qd_mes"].shape)
    ee = np.array([_FK_WORLD.forward_kinematics(bp[i], quat[i], q[i])[0] for i in range(n)])
    out = dict(arrays)
    out.update(base_pos=bp.astype(np.float32), base_bearing=yaw.astype(np.float32), base_quat=quat.astype(np.float32),
               q_mes=q.astype(np.float32), qd_mes=qd.astype(np.float32), ee_pos=ee.astype(np.float32))
    return out


def rows(arrays, a, meta, arm_only=False, noise_rng=None):
    """X, Y, reaching flag, hand-to-target distance and arm-only flag for one run.

    With noise_rng (training runs, --noise-aug) the clean rows are followed by one noisy copy
    (the distance used for --near-weight stays the true one)."""
    if noise_rng is not None and a.noise_aug:
        clean = rows(arrays, a, meta, arm_only)
        noisy = rows(noisy_copy(arrays, float(noise_rng.choice(a.noise_aug)), noise_rng), a, meta, arm_only)
        noisy = noisy[:3] + (clean[3],) + noisy[4:]
        return tuple(np.concatenate([c, n]) for c, n in zip(clean, noisy))
    X, Y, R = run_data(arrays, a)
    if a.integral_taus:
        # run_data appends the default 0.5 s / 2 s averages last (no wheel-speed inputs here): swap them
        X = np.column_stack([X[:, :-3 * len(INTEGRAL_TAUS_S)], run_integral(arrays, taus=a.integral_taus)])
    D = np.linalg.norm(arrays["target"] - arrays["ee_pos"], axis=1).astype(np.float32)
    if a.cut_hold and meta.get("hold_after_reach_s") and meta.get("reach_time") is not None:
        # Drop the rows recorded after the first reach (as if recorded without --hold).
        # Features are built on the whole run first, so the kept rows are exactly as without the cut.
        keep = arrays["t"] <= meta["reach_time"] + 1e-6
        X, Y, R, D = X[keep], Y[keep], R[keep], D[keep]
    return X, Y, R, D, np.full(len(X), float(arm_only), np.float32)


def near_only(part, radius):
    """Keep the rows of one run where the arm is reaching and the hand is within radius."""
    X, Y, R, D, A = part
    k = (R > 0) & (D < radius)
    return X[k], Y[k], R[k], D[k], A[k]


def load_dir(d, a, keep_failed, arm_only=False):
    parts, n = [], 0
    for f in sorted(glob.glob(os.path.join(d, "run_*.npz"))):
        try:
            meta, arrays = load_run(f)
        except Exception:
            continue
        if not keep_failed and _outcome(meta) != "reached":
            continue
        parts.append(rows(arrays, a, meta, arm_only, a.noise_rng))
        n += 1
    return parts, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="model name, saved as <out-dir>/<name>.npz")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "trained_dagger"))
    ap.add_argument("--torque", choices=["minus_gravity", "raw"], default="minus_gravity")
    ap.add_argument("--history", choices=["none", "deltas"], default="none")
    ap.add_argument("--goal", action="store_true")
    ap.add_argument("--integral", action="store_true")
    ap.add_argument("--integral-taus", type=float, nargs="+", default=None,
                    help="averaging windows in s for --integral (default 0.5 2)")
    ap.add_argument("--wheel-taus", type=float, nargs="+", default=None,
                    help="averaging windows the wheel network sees (default: all of --integral-taus)")
    ap.add_argument("--separate", action="store_true")
    ap.add_argument("--reach-weight", type=float, default=1.0)
    ap.add_argument("--dagger-radius", type=float, default=None,
                    help="keep only DAgger rows while reaching with the hand within this many m")
    ap.add_argument("--noise-aug", type=float, nargs="+", default=None,
                    help="sensor-noise sigmas for one noisy copy of each training run, e.g. 0.005 0.01 0.02")
    ap.add_argument("--near-weight", type=float, default=1.0)
    ap.add_argument("--near-radius", type=float, default=0.03)
    ap.add_argument("--runs", nargs="+", default=[os.path.join(HERE, "data", "runs")])
    ap.add_argument("--split", default=os.path.join(HERE, "splits", "split.json"))
    ap.add_argument("--extra-train", nargs="*", default=[], help="extra expert recordings (successful runs only)")
    ap.add_argument("--dagger", nargs="*", default=[], help="DAgger recordings (all runs kept)")
    ap.add_argument("--arm-only", nargs="*", default=[],
                    help="expert recordings used for the ARM network only (e.g. data/runs_arm); needs --separate")
    ap.add_argument("--cut-hold", action="store_true",
                    help="drop rows recorded after the first reach in --hold recordings (collect_more.py)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    a.action, a.wheel_vel = "torque", False          # what run_data / train_mlp expect from train.py's args

    out_dir = os.path.abspath(a.out_dir)
    if out_dir == os.path.join(HERE, "trained"):
        raise SystemExit("refusing to write into trained/ (the original models); use the default trained_dagger/")
    out = os.path.join(out_dir, f"{a.name}.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; pick a new --name (models are never overwritten)")
    if a.integral_taus and not a.integral:
        raise SystemExit("--integral-taus needs --integral")
    if a.wheel_taus and not (a.integral_taus and a.separate and set(a.wheel_taus) <= set(a.integral_taus)):
        raise SystemExit("--wheel-taus needs --separate and must be a subset of --integral-taus")
    if a.arm_only and not a.separate:
        raise SystemExit("--arm-only needs --separate (one network for the arm, one for the wheels)")
    np.random.seed(a.seed)
    t0 = time.time()

    with open(a.split) as fh:
        split = json.load(fh)
    def split_rows(seed):
        meta, arrays = load_run(_find_run(seed, a.runs))
        return rows(arrays, a, meta)
    a.noise_rng = np.random.default_rng(1000 + a.seed) if a.noise_aug else None
    def train_rows(seed):
        meta, arrays = load_run(_find_run(seed, a.runs))
        return rows(arrays, a, meta, noise_rng=a.noise_rng)
    train_parts = [train_rows(s) for s in split["train"]]
    val_parts = [split_rows(s) for s in split["val"]]          # validation stays clean
    n_extra = n_dagger = n_arm = 0
    for d in a.extra_train:
        p, n = load_dir(d, a, keep_failed=False)
        train_parts += p
        n_extra += n
    for d in a.arm_only:
        p, n = load_dir(d, a, keep_failed=False, arm_only=True)
        train_parts += p
        n_arm += n
    dagger_rows = 0
    for d in a.dagger:
        p, n = load_dir(d, a, keep_failed=True)
        if a.dagger_radius:
            p = [near_only(x, a.dagger_radius) for x in p]
        train_parts += p
        n_dagger += n
        dagger_rows += sum(len(x[0]) for x in p)
    Xtr, Ytr, Rtr, Dtr, Atr = (np.concatenate(c) for c in zip(*train_parts))
    Xva, Yva, _, _, _ = (np.concatenate(c) for c in zip(*val_parts))
    print(f"{a.name}: {len(Xtr)} training rows ({n_extra} extra runs, {n_dagger} DAgger runs = {dagger_rows} rows, {n_arm} arm-only runs = {int(Atr.sum())} rows), "
          f"{len(Xva)} validation rows")

    # Standardise with TRAINING statistics; input ranges are saved for clipping at drive time (as train.py)
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    x_lo, x_hi = Xtr.min(0), Xtr.max(0)
    y_mean, y_std = Ytr.mean(0), Ytr.std(0) + 1e-6
    xtr, ytr = (Xtr - x_mean) / x_std, (Ytr - y_mean) / y_std
    xva, yva = (Xva - x_mean) / x_std, (Yva - y_mean) / y_std

    # Per-row loss weights: 1 while driving, reach_weight while reaching, + near_weight close to the target
    near = (Rtr > 0) & (Dtr < a.near_radius)
    wtr = 1.0 + (a.reach_weight - 1.0) * Rtr + (a.near_weight - 1.0) * near
    print(f"  near-target rows (< {a.near_radius * 100:.0f} cm while reaching): {int(near.sum())} "
          f"({near.mean() * 100:.1f}%), weight {a.near_weight}")

    groups = {"all": slice(0, 11)} if not a.separate else {"arm": slice(0, 7), "wheels": slice(7, 11)}
    group_inputs = {}
    if a.wheel_taus:   # drop the integral columns of the windows the wheel net should not see (block is last)
        n_int = 3 * len(a.integral_taus)
        start = Xtr.shape[1] - n_int
        drop = {start + 3 * i + j for i, t in enumerate(a.integral_taus) if t not in a.wheel_taus for j in range(3)}
        group_inputs["wheels"] = [c for c in range(Xtr.shape[1]) if c not in drop]
        print(f"  wheel network sees {len(group_inputs['wheels'])} of {Xtr.shape[1]} inputs (windows {a.wheel_taus} s)")
    nets, history = {}, {}
    for g, sl in groups.items():
        use = Atr == 0 if g == "wheels" else np.ones(len(xtr), bool)   # arm-only runs: not for the wheels
        cols = group_inputs.get(g, slice(None))
        Ws, bs, h = train_mlp(xtr[use][:, cols].astype(np.float32), ytr[use][:, sl].astype(np.float32),
                              xva[:, cols].astype(np.float32), yva[:, sl].astype(np.float32), a, wtr[use])
        nets[g], history[g] = (Ws, bs), h

    pred = np.column_stack([forward(*nets[g], xva[:, group_inputs.get(g, slice(None))]) for g in groups]) * y_std + y_mean
    mae = np.abs(pred - Yva).mean(0)
    val_mse = float((((pred - Yva) / y_std) ** 2).mean())

    names = output_names("torque")
    meta = {"name": a.name, "model": "mlp", "torque": a.torque, "history_features": a.history,
            "goal_features": a.goal, "integral_features": a.integral, "action": "torque",
            "wheel_vel_features": False, "separate": a.separate, "reach_weight": a.reach_weight,
            "near_weight": a.near_weight, "near_radius": a.near_radius,
            "groups": list(groups),
            "feature_names": (feature_names_full(a.history == "deltas", a.goal, False)
                              + (integral_names(a.integral_taus or INTEGRAL_TAUS_S) if a.integral else [])),
            "integral_taus": list(a.integral_taus) if a.integral_taus else None,
            "group_inputs": group_inputs,
            "output_names": names, "hidden": a.hidden,
            "val_mse_standardised": val_mse, "val_mae": dict(zip(names, map(float, mae))),
            "history": history, "train_rows": int(len(Xtr)), "seed": a.seed,
            "runs": [os.path.basename(os.path.normpath(r)) for r in a.runs],
            "extra_train": [os.path.basename(os.path.normpath(r)) for r in a.extra_train],
            "extra_train_runs": n_extra,
            "dagger": [os.path.basename(os.path.normpath(r)) for r in a.dagger], "dagger_runs": n_dagger,
            "dagger_radius": a.dagger_radius, "noise_aug": a.noise_aug,
            "arm_only": [os.path.basename(os.path.normpath(r)) for r in a.arm_only], "arm_only_runs": n_arm,
            "cut_hold": a.cut_hold}
    weights = {}
    for g, (Ws, bs) in nets.items():
        prefix = "" if g == "all" else f"{g}_"
        weights.update({f"{prefix}W{i}": W for i, W in enumerate(Ws)})
        weights.update({f"{prefix}b{i}": b for i, b in enumerate(bs)})
        weights[f"{prefix}n_layers"] = np.array(len(Ws))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(out, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, x_lo=x_lo, x_hi=x_hi,
             meta=np.array(json.dumps(meta)), **weights)

    print(f"  validation error (MAE): torque minus gravity {mae[:7].mean():.3f} Nm | wheels {mae[7:].mean():.3f} rad/s "
          f"| standardised MSE {val_mse:.4f}")
    print(f"  saved {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    run_main(main, HERE)
