"""Figures for the IEEE report, drawn from the saved results tables (no simulator needed).

    python report_figures.py                  # every figure into figures/report/<group>/
    python report_figures.py --only ladder    # just one figure

To add a model to a figure, add one line to that figure's list below. Results are found by the
naming rule <results>/<model>[_hold5]_tier<k>.csv, so nothing else needs to change.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare import wilson
from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "report")

# ---- What goes in each figure: (model name in the results files, label in the figure) ----------
LADDER = [
    ("mlp_v0", "v0 copy"),
    ("mlp_v1", "v1 +gravity"),
    ("mlp_v1_hist", "+history"),
    ("mlp_v2_hist", "v2 +nudged"),
    ("vA_goal", "A +spot"),
    ("vD_all", "D +sep. nets"),
    ("vD2_reachdata", "D2 +reach data"),
    ("vE_integral", "E +integral"),
    ("vK_kick", "K +kick data"),
    ("vP_hold", "P +hold data"),
    ("vR_ik", "R +arm IK"),
    ("expert", "expert"),
]

# ---- One look for every figure: IEEE column width (3.5 in), text >= 8 pt ----------------------
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 200})
COL_W, COL_H = 3.5, 2.5
REACHED, HELD = "#2f6fd6", "#e67e22"


def success(model, tier=1, hold=False, results="results50"):
    """(successes, scenes) for one model on one tier, or None if that test was never run."""
    path = os.path.join(HERE, results, f"{model}{'_hold5' if hold else ''}_tier{tier}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return sum(r["success"] == "True" for r in rows), len(rows)


def bars(ax, x, counts, color, width, label):
    """Bars in % with Wilson 95% error bars; a model without that result gets no bar."""
    have = [(xi, c) for xi, c in zip(x, counts) if c is not None]
    if not have:
        return
    xs = [xi for xi, _ in have]
    pct = np.array([100 * k / n for _, (k, n) in have])
    lo, hi = np.array([wilson(k, n) for _, (k, n) in have]).T * 100
    ax.bar(xs, pct, width, color=color, label=label)
    ax.errorbar(xs, pct, yerr=[pct - lo, hi - pct], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)


def save(fig, group, name):
    folder = os.path.join(OUT, group)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


def fig_ladder():
    """Tier-1 success of each version in story order: reached, and reached + held 5 s where tested."""
    x = np.arange(len(LADDER))
    w = 0.4
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    bars(ax, x - w / 2, [success(m) for m, _ in LADDER], REACHED, w, "reached (1 cm)")
    bars(ax, x + w / 2, [success(m, hold=True) for m, _ in LADDER], HELD, w, "reached + held 5 s")
    ax.set_xticks(x, [lab for _, lab in LADDER], rotation=45, ha="right")
    ax.set_ylabel("success on 50 tier-1 scenes (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "story", "ladder")


FIGURES = {"ladder": fig_ladder}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=FIGURES, help="draw just this figure")
    a = ap.parse_args()
    for name, draw in FIGURES.items():
        if a.only in (None, name):
            draw()


if __name__ == "__main__":
    run_main(main, HERE)