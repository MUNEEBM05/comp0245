"""Training curves and learning curve for the report.

    python plot_curves.py                                   # both figures into figures/
    python plot_curves.py --model trained/vK_kick.npz       # training curves of another model

training_curves.png   loss per epoch for ONE training run (arm and wheel nets): is the model still
                      learning, or overfitting? Uses the per-epoch history saved in the model file.
learning_curve.png    benchmark success against the amount of training data (learning_curve.py):
                      would recording more runs help? Points not yet run are left out.
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/, splits/, benchmark/, configs/, models/ and RoboEnv/ are shared in the parent folder
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def wilson(k, n, z=1.96):
    """95% confidence range for k successes out of n (same idea as compare.py's range)."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fig_training(model, out):
    meta = json.loads(str(np.load(model)["meta"]))
    hist = meta["history"]
    fig, axes = plt.subplots(1, len(hist), figsize=(5.2 * len(hist), 3.8), facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    for ax, (group, h) in zip(axes, hist.items()):
        ep = [r["epoch"] for r in h]
        tr, va = [r["train_mse"] for r in h], [r["val_mse"] for r in h]
        best = int(np.argmin(va))
        style(ax)
        ax.plot(ep, tr, color=BLUE, lw=2, label="training")
        ax.plot(ep, va, color=ORANGE, lw=2, label="validation")
        ax.axvline(ep[best], color=GREY, lw=1, ls="--")
        ax.plot(ep[best], va[best], "o", ms=8, color=ORANGE, mec="#fcfcfb", mew=2, zorder=3)
        ax.annotate(f"kept: epoch {ep[best]}", (ep[best], va[best]), xytext=(-8, 30), textcoords="offset points",
                    ha="right", fontsize=9, color=INK2)
        ax.text(ep[-1], tr[-1], "  training", color=INK2, fontsize=9, va="center")
        ax.text(ep[-1], va[-1], "  validation", color=INK2, fontsize=9, va="center")
        ax.set_yscale("log")
        ax.set_xlim(0.5, ep[-1] + 0.5 + 0.18 * ep[-1])
        ax.set_xlabel("epoch", color=INK2, fontsize=9)
        ax.set_ylabel("mean squared error (standardised, log scale)", color=INK2, fontsize=9)
        ax.set_title(f"{group} network", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    fig.suptitle(f"Training curves: {meta.get('name', os.path.basename(model))}", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


def rates(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    k1 = sum(r["success"] == "True" for r in rows)
    k2 = sum(float(r["min_err_cm"]) <= 2.0 for r in rows)
    return n, k1, k2


def fig_learning(out):
    pts = []
    for f in glob.glob(os.path.join(HERE, "results_lc", "vK_lc_p*_s0_tier1.csv")):
        pts.append((int(re.search(r"_p(\d+)_", f).group(1)), *rates(f)))
    # 100% = K's recipe retrained on this machine (learning_curve.py --fractions 1.0); the original
    # K file is only a stand-in until then (same data and recipe, different PyTorch build)
    k_csv = os.path.join(HERE, "results50", "vK_kick_tier1.csv")
    if not any(p[0] == 100 for p in pts) and os.path.exists(k_csv):
        pts.append((100, *rates(k_csv)))
    if len(pts) < 2:
        print("learning curve: fewer than 2 points so far (run learning_curve.py); skipped")
        return
    pts.sort()
    x = np.array([p[0] for p in pts])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="#fcfcfb")
    style(ax)
    for idx, color, label in ((2, BLUE, "within 1 cm (task tolerance)"), (3, ORANGE, "within 2 cm")):
        n = np.array([p[1] for p in pts])
        k = np.array([p[idx] for p in pts])
        lo, hi = zip(*(wilson(ki, ni) for ki, ni in zip(k, n)))
        y = 100 * k / n
        ax.fill_between(x, 100 * np.array(lo), 100 * np.array(hi), color=color, alpha=0.12, lw=0)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, label=label)
        ax.text(x[-1] + 2, y[-1], f"{label.split(' (')[0]}", color=INK2, fontsize=9, va="center")
    exp = os.path.join(HERE, "results50", "expert_tier1.csv")
    if os.path.exists(exp):
        n, k1, _ = rates(exp)
        ax.axhline(100 * k1 / n, color=GREY, lw=1, ls="--")
        ax.text(x[0], 100 * k1 / n + 1.5, f"expert: {k1}/{n} within 1 cm", color=INK2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}%" + ("\n(original data)" if v == 100 else "\n(+ new runs)" if v > 100 else "")
                        for v in x])
    ax.set_xlim(x[0] - 8, x[-1] + 0.3 * (x[-1] - x[0]))
    ax.set_ylim(0, 105)
    ax.set_xlabel("share of recorded runs used for training", color=INK2, fontsize=9)
    ax.set_ylabel("benchmark scenes reached (%, tier 1, n = 50)", color=INK2, fontsize=9)
    ax.set_title("Learning curve: success vs amount of training data", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    fig.text(0.01, 0.01, "Shaded: 95% confidence range. Same recipe, same 45 validation runs, same 50 scenes.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "trained", "vK_kick.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_training(a.model, os.path.join(a.out, "training_curves.png"))
    fig_learning(os.path.join(a.out, "learning_curve.png"))


if __name__ == "__main__":
    main()
