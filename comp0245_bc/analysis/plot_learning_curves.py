"""Learning curves of the main arm networks (K, P, R and the BO network), one IEEE column wide.

    python3 analysis/plot_learning_curves.py

Left panel: validation MSE of the arm network after every training epoch.
Right panel: each network's best validation MSE against its held success
(filled dot: T1 benchmark, hollow dot: fresh scenes).

Every value is read from files already in the repo, nothing is typed in:
  - the curves come from the training history that training/train.py saves in each model's
    'meta' field (trained/<name>.npz -> meta -> history -> arm). The validation loss is the plain,
    unweighted MSE of the standardised arm torques on the held-out clean expert drives
    (training/train.py, nn.MSELoss), the same drives for all four networks.
  - held success is counted from the per-scene result files (results/..., outcome 'reached'
    in a hold-5 s run). These are the Table III numbers of the report.

Outputs (in figures/):
  learning_curves.pdf / .png                   with K, P, R, BO written next to each line and dot
  learning_curves_nolabels.pdf / .png          no text inside the plots, for labelling by hand
  learning_curves_data.csv                     every plotted number, for checking in Excel
"""
from _paths import ROOT  # noqa: F401  (project folder; lets the code folders import each other)
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

OUT = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

# network -> (model file, T1 held result, fresh-scene held result)
MODELS = {
    "K": ("vK_kick", "1_main_models/K/tier1/vK_kick_hold5_tier1.csv",
          "3_final_tests/fresh_scenes/K/vK_kick_hold5_summary.json"),
    "P": ("vP_hold", "1_main_models/P/vP_hold_hold5_tier1.csv",
          "3_final_tests/fresh_scenes/P/vP_hold_hold5_summary.json"),
    "R": ("vR_ik", "1_main_models/R/tier1/vR_ik_hold5_tier1.csv",
          "3_final_tests/fresh_scenes/R/vR_ik_hold5_summary.json"),
    # BO network: T1 with the 29 ms position filter it was tuned with, fresh scenes in the final
    # system (Kalman filter and A* base), as in Table III
    "BO": ("bo_t03", "1_main_models/R-BO/bo_t03_filt29noqd_hold5_tier1.csv",
           "3_final_tests/super_test/fresh_scenes/planner_bo_t03_kfv_hold5_summary.json"),
}

# Colours checked for colour-blind separation; every line and dot is also labelled by name.
COLORS = {"K": "#eda100", "P": "#1baf7a", "R": "#e34948", "BO": "#2a78d6"}
INK, GRID = "#0b0b0b", "#d9d8d4"


def held_from_csv(path):
    """Held success in % from a per-scene result file (one row per scene)."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return 100.0 * sum(r["outcome"] == "reached" for r in rows) / len(rows), len(rows)


def held_from_summary(path):
    """Held success in % from a summary file (tier '1' of that test set)."""
    with open(path) as fh:
        s = json.load(fh)["1"]
    return 100.0 * s["success_rate"], s["scenes"]


def load():
    hist, best, t1, fresh = {}, {}, {}, {}
    for name, (model, t1_file, fresh_file) in MODELS.items():
        meta = np.load(os.path.join(ROOT, "trained", model + ".npz"), allow_pickle=True)["meta"]
        meta = meta.item() if meta.shape == () else meta
        meta = json.loads(meta.decode() if isinstance(meta, bytes) else str(meta))
        h = meta["history"]["arm"]
        hist[name] = ([e["epoch"] for e in h], [e["val_mse"] for e in h])
        best[name] = min(e["val_mse"] for e in h)  # early stopping keeps the best epoch
        t1[name], n1 = held_from_csv(os.path.join(RES, t1_file))
        fresh[name], n2 = held_from_summary(os.path.join(RES, fresh_file))
        print(f"{name:3s} {model:8s} epochs {len(h):2d}  best arm val MSE {best[name]:.5f}  "
              f"T1 held {t1[name]:.0f}% ({n1} scenes)  fresh held {fresh[name]:.0f}% ({n2} scenes)")
    return hist, best, t1, fresh


def write_csv(hist, best, t1, fresh):
    path = os.path.join(OUT, "learning_curves_data.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch"] + [f"{k} val MSE" for k in MODELS])
        for ep in range(1, max(len(hist[k][0]) for k in MODELS) + 1):
            w.writerow([ep] + [f"{hist[k][1][ep - 1]:.6f}" if ep <= len(hist[k][1]) else "" for k in MODELS])
        w.writerow([])
        w.writerow(["network", "best val MSE", "T1 held %", "fresh held %"])
        for k in MODELS:
            w.writerow([k, f"{best[k]:.6f}", f"{t1[k]:.0f}", f"{fresh[k]:.0f}"])
    print("wrote", os.path.relpath(path, ROOT))


def plot(hist, best, t1, fresh, labels, name):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.55), gridspec_kw={"width_ratios": [1.35, 1]})

    for k in MODELS:
        ep, val = hist[k]
        a1.plot(ep, val, color=COLORS[k], lw=1.3, solid_capstyle="round", solid_joinstyle="round")
        if labels:
            a1.annotate(k, (ep[-1], val[-1]), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=INK)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.04]))
    a1.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.02", "0.04"]))
    a1.yaxis.set_minor_locator(NullLocator())
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Arm validation MSE")
    a1.set_xlim(0, 46 if labels else 41)

    for k in MODELS:
        a2.scatter(best[k], fresh[k], s=20, facecolor="white", edgecolor=COLORS[k], lw=1.2, zorder=3)
        a2.scatter(best[k], t1[k], s=22, color=COLORS[k], edgecolor="white", lw=0.8, zorder=4)
        if labels:
            dx = -5 if k == "K" else 5
            a2.annotate(k, (best[k], (t1[k] + fresh[k]) / 2), xytext=(dx, 0), textcoords="offset points",
                        ha="right" if dx < 0 else "left", va="center", fontsize=6.5, color=INK)
    a2.set_xlabel("Best validation MSE")
    a2.set_ylabel("Held (%)")
    a2.set_ylim(0, 105)
    a2.set_xlim(0.0055, 0.0175)
    a2.set_xticks([0.006, 0.010, 0.014])

    for a in (a1, a2):
        a.grid(True, color=GRID, lw=0.5)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print("wrote", f"figures/{name}.pdf and .png")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.labelcolor": INK, "pdf.fonttype": 42,
    })
    hist, best, t1, fresh = load()
    write_csv(hist, best, t1, fresh)
    plot(hist, best, t1, fresh, True, "learning_curves")
    plot(hist, best, t1, fresh, False, "learning_curves_nolabels")


if __name__ == "__main__":
    main()
