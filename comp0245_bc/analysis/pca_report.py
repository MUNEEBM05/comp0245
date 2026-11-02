"""PCA on model R's 71 inputs (Unsupervised Methods lecture): how many directions carry the variance, and do
the signals that decide success survive?

    python analysis/pca_report.py          # -> figures/pca_inputs.png, prints the numbers, writes splits/pca_summary.json

Uses the R_features export if present (R_features_for_regressors.npz), otherwise rebuilds the inputs.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = ROOT


def main():
    path = os.path.join(HERE, "data", "R_features_for_regressors.npz")
    d = np.load(path)
    X, names = d["X_train"], [str(n) for n in d["feature_names"]]
    x = (X - d["x_mean"]) / d["x_std"]
    _, sv, Vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum()
    cum = np.cumsum(ev)
    k = {p: int(np.searchsorted(cum, p) + 1) for p in (0.8, 0.9, 0.95, 0.99)}
    # How much of each input survives when keeping k95 components (share of its variance reconstructed)
    k95 = k[0.95]
    V = Vt[:k95].T
    rec = (x @ V) @ V.T
    kept = 1 - ((x - rec) ** 2).mean(0) / (x ** 2).mean(0).clip(1e-9)
    groups = {"IK direction now": [n for n in names if n.startswith("qd_ik_") and "avg" not in n],
              "IK averages": [n for n in names if n.startswith("qd_ik_avg")],
              "hand error": ["err_x", "err_y", "err_z"],
              "joint speeds": [f"qd{i}" for i in range(1, 8)],
              "joint angles": [f"q{i}" for i in range(1, 8)],
              "parking spot": [n for n in names if n.startswith(("goal_", "face_err"))]}
    gk = {g: float(np.mean([kept[names.index(n)] for n in ns])) for g, ns in groups.items()}
    summary = {"components_for": {f"{int(p * 100)}%": v for p, v in k.items()}, "k95": k95,
               "share_of_each_group_kept_at_k95": gk}
    with open(os.path.join(HERE, "splits", "pca_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(np.arange(1, len(cum) + 1), 100 * cum, "o-", ms=3)
    for p, c in ((0.9, "#e67e22"), (0.95, "#c0392b")):
        ax[0].axvline(k[p], color=c, ls=":", lw=1)
        ax[0].text(k[p] + 0.5, 50, f"{int(p * 100)}%: {k[p]} comps", color=c, fontsize=8)
    ax[0].set(xlabel="number of principal components", ylabel="input variance explained (%)",
              title=f"PCA of model R's {len(names)} inputs")
    ax[0].grid(alpha=0.3)
    ax[1].barh(list(gk), [100 * v for v in gk.values()], color="#2f6fd6")
    ax[1].set(xlim=(0, 105), xlabel=f"% of the group's variation kept with {k95} components",
              title="What survives the compression?")
    for i, v in enumerate(gk.values()):
        ax[1].text(100 * v + 1, i, f"{100 * v:.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "pca_inputs.png"), dpi=160)


if __name__ == "__main__":
    main()
