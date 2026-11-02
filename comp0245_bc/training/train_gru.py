"""Train a GRU (recurrent network with memory) that copies the expert.

    python training/train_gru.py --name gru_basic --extra-train data/runs_nudged data/runs_reach data/runs_kick
    python training/train_gru.py --name gru_full --history deltas --goal --integral --extra-train ...

Why: the expert has memory that the measurements do not show (its phase, its creeping arm reference,
"keep pushing while the hand is still off"). The MLP only gets hand-made memory (history and integral
inputs). A GRU (Cho et al., 2014) carries a learned memory, a vector h of `--hidden` numbers, from one
10 ms step to the next:

    r = sigmoid(W_r x + U_r h + b_r)          reset gate: how much old memory to use
    z = sigmoid(W_z x + U_z h + b_z)          update gate: how much old memory to keep
    n = tanh(W_n x + r * (U_n h + b_hn) + b_in)
    h_new = (1 - z) * n + z * h
    output = head(h_new)                       a small MLP: hidden -> 128 (ReLU) -> 11 outputs

Same inputs (by default just the 26 basic ones: no hand-made memory), same outputs (7 torques minus
gravity, 4 wheel speeds), same split, same extra data and same reach weighting as model K, so the
comparison is fair. Training uses windows of `--window` steps (10 ms each) cut at random from the
recorded drives; the first `--burn-in` steps of a window only warm up the memory and are not scored,
unless the window starts at the very beginning of a drive (where the memory really is empty).

Saved like the MLPs (trained/<name>.npz, NumPy weights, meta["model"] = "gru"); rollout.py drives it.
Needs PyTorch.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import glob
import json
import os
import time

import numpy as np

from check_runs import load_run, _outcome
from features import feature_names_full
from quiet import run_main
from train import _find_run, run_data

HERE = ROOT


def load_runs(a):
    """Lists of per-run (X, Y, reaching flag) arrays for training and validation."""
    with open(a.split) as fh:
        split = json.load(fh)
    tr = [run_data(load_run(_find_run(s, a.runs))[1], a) for s in split["train"]]
    va = [run_data(load_run(_find_run(s, a.runs))[1], a) for s in split["val"]]
    n_extra = 0
    for d in a.extra_train:
        for f in sorted(glob.glob(os.path.join(d, "run_*.npz"))):
            try:
                meta, arrays = load_run(f)
            except Exception:
                continue
            if not meta.get("dagger") and _outcome(meta) != "reached":
                continue
            X, Y, R = run_data(arrays, a)
            if "train_mask" in arrays:      # HG-DAgger rows the expert did not drive: do not score them
                R = np.where(arrays["train_mask"].astype(bool), R, -1.0)
            tr.append((X, Y, R))
            n_extra += 1
    return tr, va, n_extra


def windows(runs, length, burn_in, rng, n):
    """n random windows (X, Y, loss weight) of `length` steps; padded with weight 0 if a run is short."""
    d_in, d_out = runs[0][0].shape[1], runs[0][1].shape[1]
    X = np.zeros((n, length, d_in), np.float32)
    Y = np.zeros((n, length, d_out), np.float32)
    W = np.zeros((n, length), np.float32)
    lens = np.array([len(r[0]) for r in runs])
    pick = rng.choice(len(runs), size=n, p=lens / lens.sum())   # longer drives picked more often
    for k, i in enumerate(pick):
        x, y, w = runs[i]
        start = int(rng.integers(-burn_in, max(1, len(x) - length + 1)))
        start = max(0, start)                                    # some windows start at the beginning
        seg = slice(start, min(len(x), start + length))
        m = seg.stop - seg.start
        X[k, :m], Y[k, :m], W[k, :m] = x[seg], y[seg], w[seg]
        if start > 0:
            W[k, :burn_in] = 0.0                                 # warm-up steps are not scored
    return X, Y, W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--history", choices=["none", "deltas"], default="none")
    ap.add_argument("--goal", action="store_true")
    ap.add_argument("--integral", action="store_true")
    ap.add_argument("--runs", nargs="+", default=[os.path.join(HERE, "data", "runs")])
    ap.add_argument("--split", default=os.path.join(HERE, "splits", "split.json"))
    ap.add_argument("--extra-train", nargs="*", default=[])
    ap.add_argument("--reach-weight", type=float, default=3.0)
    ap.add_argument("--hidden", type=int, default=128, help="size of the GRU memory")
    ap.add_argument("--head", type=int, default=128, help="hidden layer of the output MLP")
    ap.add_argument("--window", type=int, default=300, help="steps per training window (10 ms each)")
    ap.add_argument("--burn-in", type=int, default=100, help="warm-up steps per window (not scored)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--windows-per-epoch", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    a.torque, a.action, a.wheel_vel = "minus_gravity", "torque", False     # what run_data expects

    import torch
    from torch import nn
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    tr, va, n_extra = load_runs(a)
    Xall = np.concatenate([r[0] for r in tr])
    Yall = np.concatenate([r[1] for r in tr])
    x_mean, x_std = Xall.mean(0), Xall.std(0) + 1e-6
    x_lo, x_hi = Xall.min(0), Xall.max(0)
    y_mean, y_std = Yall.mean(0), Yall.std(0) + 1e-6

    def prep(runs):
        out = []
        for X, Y, R in runs:
            w = np.where(R < 0, 0.0, 1.0 + (a.reach_weight - 1.0) * np.clip(R, 0, 1)).astype(np.float32)
            out.append((((X - x_mean) / x_std).astype(np.float32), ((Y - y_mean) / y_std).astype(np.float32), w))
        return out
    trn, val = prep(tr), prep(va)
    print(f"{a.name}: GRU {a.hidden} + head {a.head} | {Xall.shape[1]} inputs | {len(trn)} training drives "
          f"({len(Xall)} rows, {n_extra} extra) | {len(val)} validation drives", flush=True)

    class Net(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.gru = nn.GRU(d_in, a.hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(a.hidden, a.head), nn.ReLU(), nn.Linear(a.head, d_out))

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(h)

    net = Net(Xall.shape[1], Yall.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    # Validation: whole clean drives from the start, memory starting empty, exactly as when driving
    def validate():
        net.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val:
                p = net(torch.from_numpy(x)[None])[0].numpy()
                se += float(((p - y) ** 2).sum())
                n += y.size
        return se / n

    # Checkpoint after every epoch, so a restarted machine carries on where it stopped
    ckpt = os.path.join(HERE, "trained", f"{a.name}.ckpt.pt")
    best, best_state, bad, history, first = np.inf, None, 0, [], 1
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"])
        best, best_state, bad, history = c["best"], c["best_state"], c["bad"], c["history"]
        rng.bit_generator.state = c["rng"]
        first = len(history) + 1
        print(f"  resuming from checkpoint after epoch {first - 1}", flush=True)
    for epoch in range(first, a.epochs + 1):
        if bad >= a.patience:
            break
        net.train()
        X, Y, W = windows(trn, a.window, a.burn_in, rng, a.windows_per_epoch)
        X, Y, W = map(torch.from_numpy, (X, Y, W))
        total = 0.0
        for i in range(0, len(X), a.batch):
            opt.zero_grad()
            err = ((net(X[i:i + a.batch]) - Y[i:i + a.batch]) ** 2).mean(-1)
            w = W[i:i + a.batch]
            loss = (w * err).sum() / w.sum().clamp(min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # recurrent nets can have exploding gradients
            opt.step()
            total += loss.item()
        v = validate()
        history.append({"epoch": epoch, "train_mse": total / max(1, len(X) // a.batch), "val_mse": v})
        print(f"  epoch {epoch:3d} | train {history[-1]['train_mse']:.4f} | val {v:.4f} | {time.time() - t0:.0f} s",
              flush=True)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "best": best, "best_state": best_state,
                    "bad": bad, "history": history, "rng": rng.bit_generator.state}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
    net.load_state_dict(best_state)

    # Validation error in real units (same report as train.py)
    net.eval()
    with torch.no_grad():
        P = np.concatenate([net(torch.from_numpy(x)[None])[0].numpy() for x, _, _ in val]) * y_std + y_mean
    Yva = np.concatenate([r[1] for r in va])
    mae = np.abs(P - Yva).mean(0)

    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    names = feature_names_full(a.history == "deltas", a.goal, a.integral, False)
    meta = {"name": a.name, "model": "gru", "torque": "minus_gravity", "history_features": a.history,
            "goal_features": a.goal, "integral_features": a.integral, "action": "torque",
            "wheel_vel_features": False, "separate": False, "groups": ["all"], "reach_weight": a.reach_weight,
            "hidden": a.hidden, "head": a.head, "window": a.window, "burn_in": a.burn_in,
            "feature_names": names, "val_mse_standardised": float(best),
            "val_mae": {"torque_mean": float(mae[:7].mean()), "wheels_mean": float(mae[7:].mean())},
            "history": history, "train_drives": len(trn), "seed": a.seed,
            "extra_train": [os.path.basename(os.path.normpath(r)) for r in a.extra_train]}
    out = os.path.join(HERE, "trained", f"{a.name}.npz")
    np.savez(out, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, x_lo=x_lo, x_hi=x_hi,
             W_ih=sd["gru.weight_ih_l0"], W_hh=sd["gru.weight_hh_l0"], b_ih=sd["gru.bias_ih_l0"],
             b_hh=sd["gru.bias_hh_l0"], H_W0=sd["head.0.weight"].T, H_b0=sd["head.0.bias"],
             H_W1=sd["head.2.weight"].T, H_b1=sd["head.2.bias"], meta=np.array(json.dumps(meta)))
    print(f"  validation error (MAE): torque minus gravity {mae[:7].mean():.3f} Nm | wheels {mae[7:].mean():.3f} rad/s "
          f"| standardised MSE {best:.4f}")
    print(f"  saved {out} ({time.time() - t0:.0f} s)")
    os.remove(ckpt)


if __name__ == "__main__":
    run_main(main, HERE)
