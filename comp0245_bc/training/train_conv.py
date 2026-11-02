"""Train a temporal convolution model (TCN) that copies the expert: a WINDOW of past inputs in,
a CHUNK of future commands out (the TA's "vectors of inputs and outputs").

    python training/train_conv.py --name tcn --history deltas --goal --integral \
        --extra-train data/runs_nudged data/runs_reach data/runs_kick data/runs_hold data/runs_extreme

Inputs:  the last ~0.6 s of input rows (one row per 10 ms), through 4 causal convolution layers with
         dilations 1, 2, 4, 8 and kernel 5 (receptive field 61 rows). "Causal" = each layer only looks
         at the present and the past, so it can run live (Bai et al. 2018, temporal convolutional networks).
Outputs: the expert's next `--chunk` commands (default 10 rows = 0.1 s), each 11 numbers.
         When driving, the chunks predicted at the last few rows all say something about "now"; they
         are blended (temporal ensembling, as in ACT, Zhao et al. 2023), which smooths jitter, the
         thing that makes the hand drift out of the 1 cm zone.

Same features, outputs (torque minus gravity, wheel speeds), split and weighting as the MLPs, so the
comparison is fair. Saved as trained/<name>.npz with meta["model"] = "tcn"; rollout.py drives it with
NumPy, computing only the newest step of each layer (past steps are cached), so it is cheap.
Checkpoints after every epoch; re-running the same command resumes.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import json
import os
import time

import numpy as np

from features import feature_names_full
from quiet import run_main
from train_gru import load_runs

HERE = ROOT
KERNEL, DILATIONS = 5, (1, 2, 4, 8)
RECEPTIVE = 1 + (KERNEL - 1) * sum(DILATIONS)          # 61 rows = 0.61 s


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
    ap.add_argument("--hold-weight", type=float, default=1.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--head", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=10, help="future commands predicted per step (10 ms each)")
    ap.add_argument("--ensemble-k", type=float, default=0.5,
                    help="blend weight exp(-k * age) for older chunks when driving (stored in the model file)")
    ap.add_argument("--arch", choices=["tcn", "attn"], default="tcn",
                    help="tcn = causal dilated convolutions; attn = small transformer (self-attention) over the window")
    ap.add_argument("--d-model", type=int, default=64, help="attn: token size")
    ap.add_argument("--heads", type=int, default=4, help="attn: attention heads")
    ap.add_argument("--blocks", type=int, default=2, help="attn: transformer blocks")
    ap.add_argument("--skip", type=int, default=1,
                    help="1 = also feed the CURRENT input row straight to the output layer (a skip connection), so the "
                         "window only has to add context on top of what a plain MLP would do")
    ap.add_argument("--now-weight", type=float, default=5.0,
                    help="loss weight of the command for NOW relative to each later command in the chunk")
    ap.add_argument("--rows-per-epoch", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    a.torque, a.action, a.wheel_vel = "minus_gravity", "torque", False

    import torch
    from torch import nn
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    tr, va, n_extra = load_runs(a)

    def stack(runs):
        X = np.concatenate([r[0] for r in runs]).astype(np.float32)
        Y = np.concatenate([r[1] for r in runs]).astype(np.float32)
        R = np.concatenate([r[2] for r in runs]).astype(np.float32)
        lens = np.array([len(r[0]) for r in runs])
        start = np.repeat(np.cumsum(lens) - lens, lens)          # first row of each row's own run
        end = np.repeat(np.cumsum(lens) - 1, lens)               # last row of each row's own run
        return X, Y, R, start, end
    Xtr, Ytr, Rtr, Str, Etr = stack(tr)
    Xva, Yva, Rva, Sva, Eva = stack(va)
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    x_lo, x_hi = Xtr.min(0), Xtr.max(0)
    y_mean, y_std = Ytr.mean(0), Ytr.std(0) + 1e-6
    xtr = torch.from_numpy((Xtr - x_mean) / x_std); ytr = torch.from_numpy((Ytr - y_mean) / y_std)
    xva = torch.from_numpy((Xva - x_mean) / x_std); yva = torch.from_numpy((Yva - y_mean) / y_std)
    # row weights: reaching x reach_weight; HG-DAgger rows not driven by the expert get 0
    wtr = torch.from_numpy(np.where(Rtr < 0, 0.0, 1.0 + (a.reach_weight - 1.0) * Rtr).astype(np.float32))
    past = torch.arange(RECEPTIVE - 1, -1, -1)                   # 60, 59, ..., 0 rows back
    ahead = torch.arange(a.chunk)
    d_in, d_out = Xtr.shape[1], Ytr.shape[1]
    print(f"{a.name}: {a.arch.upper()} | {a.channels} ch / d_model {a.d_model}, receptive field {RECEPTIVE} rows ({RECEPTIVE * 10} ms), chunk {a.chunk} "
          f"| {d_in} inputs | {len(tr)} training drives ({len(Xtr)} rows, {n_extra} extra) | {len(va)} val drives",
          flush=True)

    def batch(x, y, w, S, E, rows):
        rows = torch.as_tensor(rows)
        S, E = torch.as_tensor(S)[rows], torch.as_tensor(E)[rows]
        idx_in = torch.maximum(rows[:, None] - past[None, :], S[:, None])      # repeat a drive's first row
        idx_out = rows[:, None] + ahead[None, :]
        valid = (idx_out <= E[:, None]).float()                                # no targets past the drive's end
        idx_out = torch.minimum(idx_out, E[:, None])
        wy = valid * (w[idx_out] if w is not None else 1.0)
        return x[idx_in].transpose(1, 2), y[idx_out], wy                       # (B, F, T), (B, C, 11), (B, C)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            c_in = d_in
            for d in DILATIONS:
                self.convs.append(nn.Conv1d(c_in, a.channels, KERNEL, dilation=d))
                c_in = a.channels
            self.head = nn.Sequential(nn.Linear(a.channels + (d_in if a.skip else 0), a.head), nn.ReLU(),
                                      nn.Linear(a.head, a.head), nn.ReLU(), nn.Linear(a.head, a.chunk * d_out))

        def forward(self, x):                      # x: (B, F, 61); valid convolutions shrink it to length 1
            h = x
            for conv in self.convs:
                h = torch.relu(conv(h))
            z = torch.cat([h[:, :, -1], x[:, :, -1]], 1) if a.skip else h[:, :, -1]
            return self.head(z).view(-1, a.chunk, d_out)

    class Block(nn.Module):
        """Pre-norm transformer block: x + attention(LN(x)), then x + feed-forward(LN(x)). Causal: each
        moment only attends to itself and older moments, so cached past results never change."""

        def __init__(self, d, h):
            super().__init__()
            self.h = h
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
            self.ff1, self.ff2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

        def forward(self, x, mask):
            B, T, D = x.shape
            hd = D // self.h
            z = self.ln1(x)
            q = self.q(z).view(B, T, self.h, hd).transpose(1, 2)
            k = self.k(z).view(B, T, self.h, hd).transpose(1, 2)
            v = self.v(z).view(B, T, self.h, hd).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / hd ** 0.5 + mask        # which past moments matter for this one
            out = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, T, D)
            x = x + self.o(out)
            return x + self.ff2(torch.relu(self.ff1(self.ln2(x))))

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            d = a.d_model
            self.emb = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.zeros(RECEPTIVE, d))          # learned "how long ago" embedding
            self.blocks = nn.ModuleList(Block(d, a.heads) for _ in range(a.blocks))
            self.lnf = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d + (d_in if a.skip else 0), a.head), nn.ReLU(),
                                      nn.Linear(a.head, a.head), nn.ReLU(), nn.Linear(a.head, a.chunk * d_out))
            self.register_buffer("mask", torch.triu(torch.full((RECEPTIVE, RECEPTIVE), -1e9), 1))

        def forward(self, x):                      # x: (B, F, 61), oldest first, newest (= now) last
            xt = x.transpose(1, 2)
            h = self.emb(xt) + self.pos
            for b in self.blocks:
                h = b(h, self.mask)
            z = self.lnf(h[:, -1])
            z = torch.cat([z, xt[:, -1]], 1) if a.skip else z
            return self.head(z).view(-1, a.chunk, d_out)

    net = TCN() if a.arch == "tcn" else Attn()
    chunk_w = torch.ones(a.chunk)
    chunk_w[0] = a.now_weight
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def validate():
        net.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(xva), 4096):
                rows = np.arange(i, min(i + 4096, len(xva)))
                xb, yb, wb = batch(xva, yva, None, Sva, Eva, rows)
                p = net(xb)[:, 0]                  # the command for NOW, as when driving
                se += float(((p - yb[:, 0]) ** 2).sum()); n += yb[:, 0].numel()
        return se / n

    ckpt = os.path.join(HERE, "trained", f"{a.name}.ckpt.pt")
    best, best_state, bad, history, first = np.inf, None, 0, [], 1
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"])
        best, best_state, bad, history = c["best"], c["best_state"], c["bad"], c["history"]
        rng.bit_generator.state = c["rng"]
        first = len(history) + 1
        print(f"  resuming after epoch {first - 1}", flush=True)
    for epoch in range(first, a.epochs + 1):
        if bad >= a.patience:
            break
        net.train()
        rows_all = rng.choice(len(xtr), size=min(a.rows_per_epoch, len(xtr)), replace=False)
        total, nb = 0.0, 0
        for i in range(0, len(rows_all), a.batch):
            xb, yb, wb = batch(xtr, ytr, wtr, Str, Etr, rows_all[i:i + a.batch])
            opt.zero_grad()
            err = ((net(xb) - yb) ** 2).mean(-1)              # (B, C)
            wbn = wb * chunk_w                                 # the command for NOW counts most
            loss = (wbn * err).sum() / wbn.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        v = validate()
        history.append({"epoch": epoch, "train_mse": total / nb, "val_mse": v})
        print(f"  epoch {epoch:3d} | train {total / nb:.4f} | val (command now) {v:.4f} | {time.time() - t0:.0f} s",
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

    # validation error in real units for the command "now"
    net.eval()
    with torch.no_grad():
        P = np.concatenate([net(batch(xva, yva, None, Sva, Eva, np.arange(i, min(i + 4096, len(xva))))[0])[:, 0].numpy()
                            for i in range(0, len(xva), 4096)]) * y_std + y_mean
    mae = np.abs(P - Yva).mean(0)

    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    weights = {}
    if a.arch == "attn":
        weights = {"A__" + k.replace(".", "__"): v for k, v in sd.items() if k != "mask"}
    for i in range(len(DILATIONS) if a.arch == "tcn" else 0):
        weights[f"C{i}_W"] = sd[f"convs.{i}.weight"]            # (out, in, kernel)
        weights[f"C{i}_b"] = sd[f"convs.{i}.bias"]
    weights.update(H_W0=sd["head.0.weight"].T, H_b0=sd["head.0.bias"], H_W1=sd["head.2.weight"].T, H_b1=sd["head.2.bias"],
                   H_W2=sd["head.4.weight"].T, H_b2=sd["head.4.bias"])
    meta = {"name": a.name, "model": a.arch, "d_model": a.d_model, "heads": a.heads, "blocks": a.blocks, "torque": "minus_gravity", "history_features": a.history,
            "goal_features": a.goal, "integral_features": a.integral, "action": "torque", "wheel_vel_features": False,
            "separate": False, "groups": ["all"], "reach_weight": a.reach_weight, "hold_weight": a.hold_weight,
            "kernel": KERNEL, "dilations": list(DILATIONS), "chunk": a.chunk, "ensemble_k": a.ensemble_k, "skip": bool(a.skip),
            "now_weight": a.now_weight,
            "channels": a.channels, "head": a.head,
            "feature_names": feature_names_full(a.history == "deltas", a.goal, a.integral, False),
            "val_mse_standardised": float(best),
            "val_mae": {"torque_mean": float(mae[:7].mean()), "wheels_mean": float(mae[7:].mean())},
            "history": history, "seed": a.seed,
            "extra_train": [os.path.basename(os.path.normpath(r)) for r in a.extra_train]}
    out = os.path.join(HERE, "trained", f"{a.name}.npz")
    np.savez(out, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, x_lo=x_lo, x_hi=x_hi,
             meta=np.array(json.dumps(meta)), **weights)
    print(f"  validation error (MAE): torque minus gravity {mae[:7].mean():.3f} Nm | wheels {mae[7:].mean():.3f} rad/s "
          f"| standardised MSE {best:.4f}")
    print(f"  saved {out} ({time.time() - t0:.0f} s)")
    os.remove(ckpt)


if __name__ == "__main__":
    run_main(main, HERE)
