"""Train a model that copies the expert: measurements + target -> arm torques + wheel speeds.

    python train.py --name mlp_v0 --torque raw                       # v0: the plain copy
    python train.py --name mlp_v1                                    # v1: torque minus gravity (default)
    python train.py --name linear_v1 --model linear                  # same setup, linear regression
    python train.py --name final --history deltas --goal --separate --reach-weight 3 --integral \
                    --extra-train data/runs_nudged data/runs_reach data/runs_kick     # the final model (K)

Every design choice from the report is a switch here, so any version of the ladder can be rebuilt:
    --torque minus_gravity|raw   learn only the torque left over after gravity (physics supplies gravity,
                                 like the lab-1 regression), or the whole torque
    --history none|deltas        add how the robot moved over the last 0.1 s and 0.5 s
    --goal                       add where the expert's parking spot is, seen from the robot
    --integral                   add the hand-to-target error averaged over 0.5 s and 2 s (integral action)
    --separate                   one network for the arm, one for the wheels
    --reach-weight W             weight rows where the arm is reaching W times more in the loss
    --extra-train DIR...         extra recordings (nudged / reach-nudged / kicked) added to TRAINING only;
                                 validation stays the same clean runs so all versions are comparable
    --model mlp|linear|forest    the learner (forest = scikit-learn random forest, saved as .pkl)
    --wheel-vel                  ablation: add measured wheel speeds as inputs (causes the copycat shortcut)
    --action torque|reference    ablation: predict torques (the brief) or an arm reference for a fixed PD

MLP training needs PyTorch (in RoboEnv: pixi run -e ml python train.py ...). Linear / forest do not.
Inputs and outputs are standardised with TRAINING statistics; the scaling, the input ranges and the
switches used are saved inside the model file, so rollout.py rebuilds the exact same inputs when driving.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from check_runs import load_run, _outcome
from features import feature_names_full, output_names, run_xy_full
from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # data/, splits/, benchmark/, configs/, models/ and RoboEnv/ are shared in the parent folder


def _find_run(seed, run_dirs):
    for d in run_dirs:
        f = os.path.join(d, f"run_{seed:05d}.npz")
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"run {seed} not found in {run_dirs}")


def run_data(arrays, a):
    """Inputs X, outputs Y and a per-row 'reaching' flag for one recorded run.

    The flag (base navigation done = the arm is reaching) is what --reach-weight uses to make
    the loss care more about the final approach, which is a small fraction of every run.
    """
    X, Y = run_xy_full(arrays, history=a.history == "deltas", goal=a.goal, integral=a.integral, action=a.action,
                       wheel_vel=a.wheel_vel)
    if a.action == "torque" and a.torque == "minus_gravity":
        Y = Y.copy()
        Y[:, :7] -= arrays["tau_gravity"]
    return X, Y, arrays["nav_done"].astype(np.float32)


def load_split(split_path, run_dirs, part, a):
    with open(split_path) as fh:
        seeds = json.load(fh)[part]
    parts = [run_data(load_run(_find_run(s, run_dirs))[1], a) for s in seeds]
    return tuple(np.concatenate(p) for p in zip(*parts))


def train_forest(xtr, ytr, a, rng):
    """Random forest (scikit-learn). Saved with pickle; driven through sklearn at test time."""
    from sklearn.ensemble import RandomForestRegressor
    idx = rng.choice(len(xtr), size=min(a.forest_rows, len(xtr)), replace=False)
    forest = RandomForestRegressor(n_estimators=a.trees, max_depth=a.max_depth, min_samples_leaf=5,
                                   n_jobs=-1, random_state=a.seed)
    forest.fit(xtr[idx], ytr[idx])
    return forest


def train_linear(xtr, ytr, ridge):
    """Ridge regression in closed form: W = (X'X + ridge*I)^-1 X'Y.

    On standardised data no bias term is needed (inputs and outputs are centred). Saved as a
    one-layer "network" so rollout.py drives it exactly like an MLP.
    """
    d = xtr.shape[1]
    W = np.linalg.solve(xtr.T @ xtr + ridge * np.eye(d), xtr.T @ ytr)
    return [W.astype(np.float32)], [np.zeros(ytr.shape[1], np.float32)], []


def train_mlp(xtr, ytr, xva, yva, a, wtr=None):
    """MLP with (optionally weighted) MSE loss; early stopping on validation MSE."""
    import torch
    from torch import nn
    torch.manual_seed(a.seed)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)
    wtr_t = torch.from_numpy(np.ones(len(xtr), np.float32) if wtr is None else wtr.astype(np.float32))
    xva_t, yva_t = torch.from_numpy(xva), torch.from_numpy(yva)

    # Fully connected network: inputs -> hidden (ReLU) x N -> outputs, no activation on the output
    # because torques and wheel speeds are unbounded real numbers.
    layers, d = [], xtr.shape[1]
    for h in a.hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, ytr.shape[1]))
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    loss_fn = nn.MSELoss()  # MSE: punishes large torque errors most (those make the arm overshoot)

    # Train in mini-batches; keep the weights of the epoch with the lowest VALIDATION loss and stop
    # early once it has not improved for `patience` epochs (guards against overfitting).
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(xtr_t))
        total = 0.0
        for i in range(0, len(xtr_t), a.batch):
            idx = perm[i:i + a.batch]
            opt.zero_grad()
            err = ((net(xtr_t[idx]) - ytr_t[idx]) ** 2).mean(1)      # MSE per row
            loss = (wtr_t[idx] * err).sum() / wtr_t[idx].sum()      # weighted average
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(xva_t), yva_t).item()
        history.append({"epoch": epoch, "train_mse": total / len(xtr_t), "val_mse": val})
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if a.verbose:
            print(f"  epoch {epoch:3d} | train MSE {total / len(xtr_t):.4f} | val MSE {val:.4f}", flush=True)
        if bad >= a.patience:
            break
    net.load_state_dict(best_state)
    # Export the weights as plain NumPy arrays: driving the robot then needs no PyTorch at all
    lin = [m for m in net if isinstance(m, nn.Linear)]
    Ws = [m.weight.detach().numpy().T.astype(np.float32) for m in lin]
    bs = [m.bias.detach().numpy().astype(np.float32) for m in lin]
    return Ws, bs, history


def forward(Ws, bs, x):
    h = x
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="model name, e.g. mlp_v1 (saved as trained/<name>.npz)")
    ap.add_argument("--model", choices=["mlp", "linear", "forest"], default="mlp")
    ap.add_argument("--trees", type=int, default=50, help="forest: number of trees")
    ap.add_argument("--max-depth", type=int, default=16, help="forest: depth limit (keeps prediction fast)")
    ap.add_argument("--forest-rows", type=int, default=150000,
                    help="forest: train on a random subset this big (full data is slow and huge on disk)")
    ap.add_argument("--torque", choices=["minus_gravity", "raw"], default="minus_gravity")
    ap.add_argument("--history", choices=["none", "deltas"], default="none",
                    help="deltas: add how the robot moved over the last 0.1 s and 0.5 s (v3)")
    ap.add_argument("--goal", action="store_true",
                    help="add where the parking spot is, seen from the robot (variant A)")
    ap.add_argument("--integral", action="store_true",
                    help="add the hand-to-target error averaged over the last 0.5 s and 2 s (the expert's "
                         "'keep pushing while an error remains' trick, built from measurements)")
    ap.add_argument("--wheel-vel", action="store_true",
                    help="add the measured wheel speeds as inputs (ablation: may cause the copycat shortcut)")
    ap.add_argument("--action", choices=["torque", "reference"], default="torque",
                    help="torque: the model outputs arm torques (the brief). reference: the model outputs where "
                         "the arm should go, and the template's PD law makes the torque (ask the lecturer first)")
    ap.add_argument("--separate", action="store_true",
                    help="train separate networks for the arm and the wheels (variant B)")
    ap.add_argument("--reach-weight", type=float, default=1.0,
                    help="training weight of rows where the arm is reaching (variant C), e.g. 3")
    ap.add_argument("--runs", nargs="+", default=[os.path.join(ROOT, "data", "runs")],
                    help="one or more run folders (e.g. clean + nudged recordings); the split lists their seeds")
    ap.add_argument("--split", default=os.path.join(ROOT, "splits", "split.json"))
    ap.add_argument("--extra-train", nargs="*", default=[],
                    help="extra run folders (e.g. data/runs_nudged) whose successful runs are added to "
                         "TRAINING only; validation stays the same clean runs, so versions stay comparable")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6, help="stop after this many epochs without improvement")
    ap.add_argument("--ridge", type=float, default=1.0, help="regularisation strength for linear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print every epoch")
    a = ap.parse_args()
    np.random.seed(a.seed)

    t0 = time.time()
    Xtr, Ytr, Rtr = load_split(a.split, a.runs, "train", a)
    Xva, Yva, _ = load_split(a.split, a.runs, "val", a)
    n_extra = 0
    for d in a.extra_train:
        for f in sorted(glob.glob(os.path.join(d, "run_*.npz"))):
            try:
                meta, arrays = load_run(f)
            except Exception:
                continue
            if _outcome(meta) != "reached":
                continue
            X, Y, R = run_data(arrays, a)
            Xtr, Ytr, Rtr = np.concatenate([Xtr, X]), np.concatenate([Ytr, Y]), np.concatenate([Rtr, R])
            n_extra += 1
    if a.extra_train:
        print(f"  + {n_extra} extra training runs from {', '.join(a.extra_train)}")
    print(f"{a.name}: {a.model}, torque {a.torque} | {len(Xtr)} training rows, {len(Xva)} validation rows")

    # Standardise with TRAINING statistics only (no peeking at validation data)
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    # Range of every input seen in training: at test time inputs are clipped to it, so a drive that
    # wanders somewhere unseen (e.g. integral inputs winding up while stuck) cannot feed the network
    # values it never learned from. Anti-windup, in control terms.
    x_lo, x_hi = Xtr.min(0), Xtr.max(0)
    y_mean, y_std = Ytr.mean(0), Ytr.std(0) + 1e-6
    xtr, ytr = (Xtr - x_mean) / x_std, (Ytr - y_mean) / y_std
    xva, yva = (Xva - x_mean) / x_std, (Yva - y_mean) / y_std

    if a.model == "forest":
        import pickle
        rng = np.random.default_rng(a.seed)
        forest = train_forest(xtr.astype(np.float32), ytr.astype(np.float32), a, rng)
        pred = forest.predict(xva) * y_std + y_mean
        mae = np.abs(pred - Yva).mean(0)
        val_mse = float((((pred - Yva) / y_std) ** 2).mean())
        names = feature_names_full(a.history == "deltas", a.goal, a.integral, a.wheel_vel)
        meta = {"name": a.name, "model": "forest", "torque": a.torque, "history_features": a.history,
                "goal_features": a.goal, "integral_features": a.integral, "action": a.action,
                "wheel_vel_features": a.wheel_vel, "separate": False, "groups": ["all"],
                "feature_names": names, "output_names": output_names(a.action), "trees": a.trees,
                "max_depth": a.max_depth, "val_mse_standardised": val_mse,
                "val_mae": dict(zip(output_names(a.action), map(float, mae))), "train_rows": int(len(Xtr)),
                "seed": a.seed}
        out = os.path.join(ROOT, "trained", f"{a.name}.pkl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump({"forest": forest, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                         "x_lo": x_lo, "x_hi": x_hi, "meta": meta}, fh)
        print(f"  validation error (MAE): torque minus gravity {mae[:7].mean():.3f} Nm | wheels {mae[7:].mean():.3f} rad/s "
              f"| standardised MSE {val_mse:.4f}")
        print(f"  saved {out} ({os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t0:.0f} s)")
        return

    # Per-row loss weights: 1 while driving, reach_weight while the arm is reaching
    wtr = 1.0 + (a.reach_weight - 1.0) * Rtr
    # --separate trains one network per output group (arm, wheels), each with its own early
    # stopping; otherwise one network predicts all outputs.
    n_arm = 14 if a.action == "reference" else 7
    n_out = n_arm + 4
    groups = {"all": slice(0, n_out)} if not a.separate else {"arm": slice(0, n_arm), "wheels": slice(n_arm, n_out)}
    nets, history = {}, {}
    for g, sl in groups.items():
        if a.model == "linear":
            nets[g] = train_linear(xtr, ytr[:, sl], a.ridge)[:2]
            history[g] = []
        else:
            Ws, bs, h = train_mlp(xtr.astype(np.float32), ytr[:, sl].astype(np.float32),
                                  xva.astype(np.float32), yva[:, sl].astype(np.float32), a, wtr)
            nets[g], history[g] = (Ws, bs), h

    pred = np.column_stack([forward(*nets[g], xva) for g in groups]) * y_std + y_mean
    mae = np.abs(pred - Yva).mean(0)               # in real units
    val_mse = float((((pred - Yva) / y_std) ** 2).mean())

    names = feature_names_full(a.history == "deltas", a.goal, a.integral, a.wheel_vel)
    OUTPUT_NAMES = output_names(a.action)
    meta = {"name": a.name, "model": a.model, "torque": a.torque, "history_features": a.history,
            "goal_features": a.goal, "integral_features": a.integral, "action": a.action,
            "wheel_vel_features": a.wheel_vel,
            "separate": a.separate, "reach_weight": a.reach_weight,
            "groups": list(groups), "feature_names": names,
            "output_names": OUTPUT_NAMES, "hidden": a.hidden if a.model == "mlp" else [],
            "val_mse_standardised": val_mse, "val_mae": dict(zip(OUTPUT_NAMES, map(float, mae))),
            "history": history, "train_rows": int(len(Xtr)), "seed": a.seed,
            "runs": [os.path.basename(os.path.normpath(r)) for r in a.runs],
            "extra_train": [os.path.basename(os.path.normpath(r)) for r in a.extra_train],
            "extra_train_runs": n_extra}
    out = os.path.join(ROOT, "trained", f"{a.name}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    weights = {}
    for g, (Ws, bs) in nets.items():
        prefix = "" if g == "all" else f"{g}_"
        weights.update({f"{prefix}W{i}": W for i, W in enumerate(Ws)})
        weights.update({f"{prefix}b{i}": b for i, b in enumerate(bs)})
        weights[f"{prefix}n_layers"] = np.array(len(Ws))
    np.savez(out, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, x_lo=x_lo, x_hi=x_hi,
             meta=np.array(json.dumps(meta)), **weights)

    if a.action == "reference":
        print(f"  validation error (MAE): arm reference offset {mae[:7].mean():.4f} rad | reference speed "
              f"{mae[7:14].mean():.3f} rad/s | wheels {mae[14:].mean():.3f} rad/s | standardised MSE {val_mse:.4f}")
    else:
        what = "torque minus gravity" if a.torque == "minus_gravity" else "torque"
        print(f"  validation error (MAE): {what} {mae[:7].mean():.3f} Nm | wheels {mae[7:].mean():.3f} rad/s "
              f"| standardised MSE {val_mse:.4f}")
    print(f"  saved {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    run_main(main, HERE)
