"""Make the report figures from the saved results tables.

    python make_figures.py                      # everything into figures/

Figures:
    ladder.png            success rate per version (1 cm / 2 cm / 5 cm), the Stage A story in one chart
    failure_reasons.png   why each version failed, stacked bars (crash / not parked / not settled ...)
    offline_vs_online.png validation error (offline) vs closed-loop success: the behaviour-cloning gap
    models.png            linear vs random forest vs MLP on the same inputs
    tiers.png             the final model and the expert across the benchmark tiers
    drive_success.png / drive_failure.png   one example drive of each kind (final model)
Every figure reads only files in results50/ and trained/, so it can be regenerated on any machine.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse_failures import REASONS, reason
from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))

# The Stage A ladder: (name of the results file, label in the figure). Order = order of the story.
LADDER = [
    ("mlp_v0", "v0 plain copy"),
    ("mlp_v1", "v1 + gravity by physics"),
    ("mlp_v1_hist", "+ history"),
    ("mlp_v2_hist", "v2 + nudged data"),
    ("vA_goal", "A + parking spot"),
    ("vD_all", "D + separate nets, reach wt"),
    ("vD2_reachdata", "D2 + reach-nudged data"),
    ("vE_integral", "E + integral input"),
    ("vK_kick", "K + velocity-kick data"),
    ("expert", "expert (ceiling)"),
]
MODELS = [("cmp_linear", "linear regression"), ("cmp_forest", "random forest"), ("vK_kick", "MLP")]
COLORS = {"1 cm": "#2f6fd6", "2 cm": "#7fa8e6", "5 cm": "#c9daf5"}
REASON_COLORS = ["#c0392b", "#e67e22", "#f1c40f", "#8e44ad", "#7f8c8d"]


def load(results, name, tier=1):
    path = os.path.join(results, f"{name}_tier{tier}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def rates(rows):
    n = len(rows)
    closest = np.array([float(r["min_err_cm"]) for r in rows])
    ok = sum(r["success"] == "True" for r in rows)
    return n, ok / n, (closest <= 2).mean(), (closest <= 5).mean()


def fig_ladder(results, out):
    labels, r1, r2, r5, ns = [], [], [], [], []
    for name, label in LADDER:
        rows = load(results, name)
        if rows is None:
            continue
        n, a, b, c = rates(rows)
        labels.append(label); r1.append(a); r2.append(b); r5.append(c); ns.append(n)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 1.5))
    ax.barh(y, r5, color=COLORS["5 cm"], label="within 5 cm")
    ax.barh(y, r2, color=COLORS["2 cm"], label="within 2 cm")
    ax.barh(y, r1, color=COLORS["1 cm"], label="within 1 cm (the task's tolerance)")
    for i, (a, n) in enumerate(zip(r1, ns)):
        ax.text(a + 0.01, i, f"{100 * a:.0f}%  (n={n})", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("success rate on the frozen tier-1 benchmark")
    ax.set_title("Stage A: each fix, tested on the same scenes")
    ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_failures(results, out):
    names = [(n, l) for n, l in LADDER if n != "expert"]
    labels, mat = [], []
    for name, label in names:
        rows = load(results, name)
        if rows is None:
            continue
        counts = {k: 0 for k, _ in REASONS}
        for r in rows:
            if r["success"] != "True":
                counts[reason(r) if "goal_dist_end_m" in r else "crash"] += 1
        labels.append(label); mat.append([counts[k] / len(rows) for k, _ in REASONS])
    mat = np.array(mat)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 1.8))
    left = np.zeros(len(labels))
    for j, (k, text) in enumerate(REASONS):
        ax.barh(y, mat[:, j], left=left, color=REASON_COLORS[j], label=text)
        left += mat[:, j]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of benchmark scenes")
    ax.set_title("What went wrong in the failed drives, per version")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_offline_online(results, out):
    pts, seen = [], set()
    for name, label in LADDER + MODELS:
        if name == "expert" or name in seen:      # the MLP row of MODELS is the same model as K
            continue
        seen.add(name)
        rows = load(results, name)
        model_file = os.path.join(HERE, "trained", name + (".pkl" if name == "cmp_forest" else ".npz"))
        if rows is None or not os.path.exists(model_file):
            continue
        if name == "cmp_forest":
            import pickle
            with open(model_file, "rb") as fh:
                meta = pickle.load(fh)["meta"]
        else:
            meta = json.loads(str(np.load(model_file)["meta"]))
        pts.append((meta["val_mse_standardised"], rates(rows)[1], label))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for x, y, label in pts:
        ax.scatter(x, y, s=50, color="#2f6fd6")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("offline validation error (standardised MSE, lower is better)")
    ax.set_ylabel("closed-loop success within 1 cm")
    ax.set_title("Offline accuracy does not predict driving success")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_models(results, out):
    labels, r1, r5, crash, think = [], [], [], [], []
    for name, label in MODELS:
        rows = load(results, name)
        if rows is None:
            continue
        n, a, _, c = rates(rows)
        labels.append(f"{label}\n(n={n})"); r1.append(a); r5.append(c)
        crash.append(np.mean([r["outcome"].startswith("crashed") for r in rows]))
        think.append(np.mean([float(r["think_us_per_step"]) for r in rows if r.get("think_us_per_step")]))
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(x - 0.2, r1, 0.2, label="within 1 cm", color=COLORS["1 cm"])
    a1.bar(x, r5, 0.2, label="within 5 cm", color=COLORS["5 cm"])
    a1.bar(x + 0.2, crash, 0.2, label="crashed", color="#c0392b")
    a1.set_xticks(x, labels); a1.set_ylim(0, 1); a1.set_title("Same inputs, same data: who can drive?")
    a1.legend(fontsize=8); a1.grid(axis="y", alpha=0.3)
    a2.bar(x, think, color="#7f8c8d")
    a2.axhline(1000, color="r", ls="--", lw=1, label="1 ms budget per decision")
    a2.set_xticks(x, labels); a2.set_yscale("log"); a2.set_ylabel("decision time (microseconds)")
    a2.set_title("Efficiency"); a2.legend(fontsize=8); a2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_tiers(results, out):
    tiers = [(1, "1 in-distribution"), (2, "2 OOD targets"), (3, "3 unusual starts"), (4, "4 disturbances")]
    data = {}
    for name, label in [("expert", "expert"), ("vK_kick", "final model (K)")]:
        data[label] = []
        for t, _ in tiers:
            rows = load(results, name, t)
            data[label].append(rates(rows)[1] if rows else np.nan)
    x = np.arange(len(tiers))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, data["expert"], 0.36, label="expert", color="#7f8c8d")
    ax.bar(x + 0.18, data["final model (K)"], 0.36, label="final model (K)", color=COLORS["1 cm"])
    ax.set_xticks(x, [l for _, l in tiers]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("success within 1 cm"); ax.set_title("Benchmark tiers: how far does it generalise?")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_drives(results, outdir):
    """One success and one failure of the final model, re-driven so the plot is fresh."""
    from evaluate import plot_drive
    from rollout import MLPPolicy, rollout
    rows = load(results, "vK_kick")
    if rows is None:
        return
    with open(os.path.join(HERE, "benchmark", "scenes.json")) as fh:
        scenes = {(s["tier"], s["index"]): s for s in json.load(fh)["scenes"]}
    ok = next((r for r in rows if r["success"] == "True"), None)
    bad = next((r for r in rows if r["success"] != "True"), None)
    pol = MLPPolicy(os.path.join(HERE, "trained", "vK_kick.npz"))
    for r, fname in ((ok, "drive_success.png"), (bad, "drive_failure.png")):
        if r is None:
            continue
        scene = scenes[(1, int(r["index"]))]
        m, traj = rollout(pol, scene)
        plot_drive(m, traj, scene, os.path.join(outdir, fname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results50"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--no-drives", action="store_true", help="skip the two example drives (needs the simulator)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_ladder(a.results, os.path.join(a.out, "ladder.png"))
    fig_failures(a.results, os.path.join(a.out, "failure_reasons.png"))
    fig_offline_online(a.results, os.path.join(a.out, "offline_vs_online.png"))
    fig_models(a.results, os.path.join(a.out, "models.png"))
    fig_tiers(a.results, os.path.join(a.out, "tiers.png"))
    if not a.no_drives:
        fig_drives(a.results, a.out)
    print(f"wrote figures to {a.out}: " + ", ".join(sorted(os.listdir(a.out))))


if __name__ == "__main__":
    run_main(main, HERE)
