"""Figures for the IEEE report, drawn from the saved results tables (no simulator needed).

    python analysis/report_figures.py                  # every figure into figures/report/<group>/
    python analysis/report_figures.py --only ladder    # just one figure (names: see FIGURES at the bottom)

To add a model to a figure, add one line to that figure's list below. Results are found by the
naming rule <results>/<model>[_hold5]_tier<k>.csv, so nothing else needs to change.
"""
from _paths import ROOT, find_result  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from compare import wilson
from quiet import run_main

HERE = ROOT
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

# Same data and inputs (model P's), different learners: (label, [one results name per training run])
MODEL_FAMILIES = [
    ("MLP", ["vP_hold", "vP_hold_s1", "vP_hold_s2"]),
    ("TCN", ["tcn2", "tcn2_s1", "tcn2_s2"]),
    ("attention", ["attn"]),
    ("GRU", ["gru_P"]),
]

# The same recipe trained with three random seeds: is the result repeatable?
SEED_FAMILIES = [
    ("model P\n(MLP + hold data)", ["vP_hold", "vP_hold_s1", "vP_hold_s2"]),
    ("model R\n(P + arm IK inputs)", ["vR_ik", "vR_ik_s1", "vR_ik_s2"]),
]

# Who drives what, judged on "reached + held 5 s". One group per arm driver:
# (label, results name with the expert driving the base, results name with the same driver on the base)
# _mlp_arm = expert drives the base, our model the arm; _mlp_wheels = the reverse.
ARM_VS_BASE = [
    ("expert", "expert", "vP_hold_mlp_wheels"),
    ("model P", "vP_hold_mlp_arm", "vP_hold"),
    ("model R", "vR_ik_mlp_arm", "vR_ik"),
]

# Test sets for the generalisation figure: (label, tier, results folder)
TEST_SETS = [
    ("tier 1\nnormal", 1, "results"),
    ("fresh\nscenes", 1, "results/3_final_tests/fresh_scenes"),
    ("tier 2\nfar/high", 2, "results"),
    ("tier 3\nodd starts", 3, "results"),
    ("tier 6\nobjects", 6, "results"),
]
GENERAL_MODELS = [("vR_ik", "model R"), ("expert", "expert")]

# ---- One look for every figure: IEEE column width (3.5 in), text >= 8 pt ----------------------
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
COL_W, COL_H = 3.5, 2.5   # one column
PAGE_W = 7.16             # both columns (figure* in LaTeX)
REACHED, HELD = "#2f6fd6", "#e67e22"
EXPERT, EXPERT_HELD = "#a6a6a6", "#595959"
LEARNERS = {"mlp": ("MLP", "#2f6fd6"), "tcn": ("TCN", "#e67e22"), "attn": ("attention", "#8e44ad"),
            "gru": ("GRU", "#27ae60"), "linear": ("linear", "#7f8c8d")}  # meta["model"] -> (label, colour)


def success(model, tier=1, hold=False, results="results"):
    """(successes, scenes) for one model on one tier, or None if that test was never run."""
    path = find_result(os.path.join(HERE, results), f"{model}{'_hold5' if hold else ''}_tier{tier}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return sum(r["success"] == "True" for r in rows), len(rows)


CI_LABEL = "95% confidence range"


def bars(ax, x, counts, color, width, label, ci_label=None):
    """Bars in % with Wilson 95% error bars; a model without that result gets no bar."""
    have = [(xi, c) for xi, c in zip(x, counts) if c is not None]
    if not have:
        return
    xs = [xi for xi, _ in have]
    pct = np.array([100 * k / n for _, (k, n) in have])
    lo, hi = np.array([wilson(k, n) for _, (k, n) in have]).T * 100
    ax.bar(xs, pct, width, color=color, label=label)
    ax.errorbar(xs, pct, yerr=[pct - lo, hi - pct], fmt="none", ecolor="black", elinewidth=0.6,
                capsize=1.5, label=ci_label)


def missing(ax, x, counts, y=3, text="not\ntested"):
    """Write 'not tested' where a result does not exist, so it is not read as 0%."""
    for xi, c in zip(x, counts):
        if c is None:
            ax.text(xi, y, text, ha="center", va="bottom", fontsize=6, color="0.4")


def legend_above(ax, ncol=2):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False)


def runs_plot(families, group, name):
    """One dot per training run, bar = mean over runs; reached and held side by side."""
    x = np.arange(len(families))
    w = 0.4
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    dot_label = "one training run (seed)"
    for dx, hold, color, label in ((-w / 2, False, REACHED, "reached (1 cm), mean"),
                                   (w / 2, True, HELD, "reached + held 5 s, mean")):
        for xi, (_, runs) in zip(x + dx, families):
            pct = [100 * k / n for k, n in filter(None, (success(r, hold=hold) for r in runs))]
            if not pct:
                continue
            ax.bar(xi, np.mean(pct), w, color=color, label=label)
            ax.scatter([xi] * len(pct), pct, s=8, color="black", zorder=3, clip_on=False, label=dot_label)
            label = dot_label = None  # legend entry only once each
    ax.set_xticks(x, [lab for lab, _ in families])
    ax.set_ylabel("success on 50 tier-1 scenes (%)")
    ax.set_ylim(0, 105)
    legend_above(ax)
    save(fig, group, name)


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
    held = [success(m, hold=True) for m, _ in LADDER]
    bars(ax, x - w / 2, [success(m) for m, _ in LADDER], REACHED, w, "reached (1 cm)", CI_LABEL)
    bars(ax, x + w / 2, held, HELD, w, "reached + held 5 s")
    first = next(i for i, c in enumerate(held) if c is not None)   # the hold test starts here
    ax.text(1.5, 55, f"held 5 s not tested\nbefore {LADDER[first][1].split()[0]}", ha="center",
            fontsize=6, color="0.4")
    ax.set_xticks(x, [lab for _, lab in LADDER], rotation=45, ha="right")
    ax.set_ylabel("success on 50 tier-1 scenes (%)")
    ax.set_ylim(0, 105)
    legend_above(ax)
    save(fig, "story", "ladder")


def fig_models():
    runs_plot(MODEL_FAMILIES, "models", "model_comparison")


def fig_seeds():
    runs_plot(SEED_FAMILIES, "story", "seeds")


def fig_arm_vs_base():
    """Swap test: does success follow whoever drives the ARM, whoever drives the base doesn't matter?"""
    x = np.arange(len(ARM_VS_BASE))
    w = 0.38
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    bars(ax, x - w / 2, [success(m, hold=True) for _, m, _ in ARM_VS_BASE], EXPERT, w,
         "base driven by the expert")
    bars(ax, x + w / 2, [success(m, hold=True) for _, _, m in ARM_VS_BASE], HELD, w,
         "base driven by our model P / R", CI_LABEL)
    ax.set_xticks(x, [lab for lab, _, _ in ARM_VS_BASE])
    ax.set_xlabel("who drives the arm")
    ax.set_ylabel("reached + held 5 s, tier 1 (%)")
    ax.set_ylim(0, 105)
    legend_above(ax, ncol=1)
    save(fig, "diagnosis", "arm_vs_base")


def fig_generalisation():
    """Model R vs the expert on each test set; top = reached, bottom = reached + held 5 s."""
    x = np.arange(len(TEST_SETS))
    w = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.6))
    for ax, hold, ylabel in ((axes[0], False, "reached 1 cm (%)"), (axes[1], True, "reached + held 5 s (%)")):
        counts = {}
        for (model, label), dx, color in zip(GENERAL_MODELS, (-w / 2, w / 2), (REACHED, EXPERT)):
            counts[model] = [success(model, tier, hold, folder) for _, tier, folder in TEST_SETS]
            bars(ax, x + dx, counts[model], color, w, label, CI_LABEL if model == "expert" else None)
        missing(ax, x, [None if all(c[i] is None for c in counts.values()) else 1 for i in x])
        ax.set_xticks(x, [lab for lab, _, _ in TEST_SETS])
        ax.set_ylim(0, 105)
        ax.set_ylabel(ylabel)
    legend_above(axes[0], ncol=3)
    fig.tight_layout(h_pad=0.8)
    save(fig, "generalisation", "generalisation")


def fig_offline_online():
    """Validation error (offline) vs tier-1 reached (closed loop), for every trained model with both."""
    pts = []
    for path in sorted(glob.glob(os.path.join(HERE, "trained", "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        meta = json.loads(str(np.load(path, allow_pickle=True)["meta"]))
        res = success(name)
        if res is not None and meta.get("val_mse_standardised") is not None:
            pts.append((meta["val_mse_standardised"], 100 * res[0] / res[1], meta.get("model", "mlp")))
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    for kind, (label, color) in LEARNERS.items():
        sel = [(e, s) for e, s, k in pts if k == kind]
        if sel:
            ax.scatter(*zip(*sel), s=12, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("validation error (standardised MSE, log)")
    ax.set_ylabel("reached, tier 1 (%)")
    ax.set_ylim(-3, 100)
    ax.legend(frameon=False)
    save(fig, "models", "offline_vs_online")


# Pipeline diagram: one list per row, boxes left to right. Top = training, bottom = driving.
PIPELINE_TRAIN = [
    "recordings\n1 kHz sim,\nlogged 100 Hz",
    "cleaning\n+ outcome fix",
    "split by drive\n70 / 15 / 15",
    "71 inputs\nbasic 26, hist 13\nspot 5, integ 6\nIK 21",
    "standardise\ntargets: torque\n- gravity, wheels",
    "train MLP\narm net +\nwheel net",
]
PIPELINE_DRIVE = [
    "measure\nrobot state",
    "same 71 inputs\n(updated\nevery 10 ms)",
    "standardise\n+ clip to\ntrain range",
    "MLP\n(NumPy only)",
    "+ gravity\n(Pinocchio)",
    "clip torques\n-> robot",
]


def fig_pipeline():
    """Data pipeline (top) and the live control loop (bottom), page width."""
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.6))
    ax.set_xlim(-0.05, 6)
    ax.set_ylim(-0.05, 2.25)
    ax.axis("off")
    bw, bh = 0.84, 0.75
    rows = ((PIPELINE_TRAIN, 1.65, "#dbe7fb", "training (offline)"),
            (PIPELINE_DRIVE, 0.4, "#fbe3cf", "driving (every 1 ms)"))
    for boxes, cy, color, title in rows:
        ax.text(0.03, cy + bh / 2 + 0.08, title, fontsize=8, weight="bold")
        for col, text in enumerate(boxes):
            cx = col + 0.45
            ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh, boxstyle="round,pad=0.02",
                                        fc=color, ec="black", lw=0.6))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=7)
            if col:
                ax.annotate("", (cx - bw / 2, cy), (cx - 1 + bw / 2, cy), arrowprops=dict(arrowstyle="->", lw=0.7))
    # trained weights go from the training MLP (top, last box) to the driving MLP (bottom, 4th box)
    top, bottom, mid = 1.65 - bh / 2 - 0.03, 0.4 + bh / 2 + 0.03, 1.03
    ax.plot([5.45, 5.45, 3.45], [top, mid, mid], color="black", lw=0.7, ls="--")
    ax.annotate("", (3.45, bottom), (3.45, mid), arrowprops=dict(arrowstyle="->", lw=0.7, ls="--"))
    ax.text(4.45, mid + 0.04, "trained weights", ha="center", va="bottom", fontsize=7)
    save(fig, "diagrams", "pipeline")


FIGURES = {
    "ladder": fig_ladder,
    "models": fig_models,
    "seeds": fig_seeds,
    "arm_vs_base": fig_arm_vs_base,
    "generalisation": fig_generalisation,
    "offline_online": fig_offline_online,
    "pipeline": fig_pipeline,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=FIGURES, help="draw just this figure")
    a = ap.parse_args()
    for name, draw in FIGURES.items():
        if a.only in (None, name):
            draw()


if __name__ == "__main__":
    run_main(main, HERE)
