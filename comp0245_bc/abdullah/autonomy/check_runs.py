"""Sanity-check recorded runs: a summary table plus plots of a few runs.

    python check_runs.py                      # summarise data/runs, plot the first 4 runs
    python check_runs.py --out data/runs --plot 8

Writes data/runs/summary.csv and data/runs/check_*.png.
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quiet import run_main

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # data/ and benchmark/ are shared with the parent folder
REACH_TOLERANCE = 0.01  # m, the template's success threshold
PHASE_NAMES = ["align", "drive", "orient", "done"]


def load_run(path):
    """Load one run: returns (meta dict, dict of arrays)."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    return meta, arrays


def plot_run(meta, a, path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    t = a["t"]

    # Top-down view of the base path
    ax[0, 0].plot(a["base_pos"][:, 0], a["base_pos"][:, 1], "b-", label="base path")
    ax[0, 0].plot(*meta["start_pose"][:2], "go", label="start")
    ax[0, 0].plot(*meta["base_goal_pos"], "ks", label="parking goal")
    ax[0, 0].plot(*meta["target"][:2], "r*", ms=14, label="target (x, y)")
    ax[0, 0].plot(a["ee_pos"][:, 0], a["ee_pos"][:, 1], "m:", label="hand path")
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_title(f"seed {meta['seed']}: {meta['outcome']}")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    # Distance from hand to target, with the base phase shaded
    err = np.linalg.norm(a["target"] - a["ee_pos"], axis=1)
    ax[0, 1].plot(t, err * 100, "g-")
    ax[0, 1].axhline(1.0, color="r", ls="--", lw=0.8, label="1 cm tolerance")
    ax2 = ax[0, 1].twinx()
    ax2.step(t, a["phase"], "k-", alpha=0.3, where="post")
    ax2.set_yticks(range(4), PHASE_NAMES)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel("hand-to-target distance (cm)")
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)

    # Labels the model will learn
    ax[1, 0].plot(t, a["wheel_cmds"])
    ax[1, 0].set_title("wheel commands (rad/s)")
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 1].plot(t, a["torque_arm"])
    ax[1, 1].set_title("arm torques (Nm)")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _outcome(meta):
    """Success is judged by final error: the template can label a within-tolerance run a stall."""
    if meta["outcome"] == "unreachable" and meta["final_ee_error"] < REACH_TOLERANCE:
        return "reached"
    return meta["outcome"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "runs"))
    ap.add_argument("--plot", type=int, default=4, help="how many runs to plot")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.out, "run_*.npz")))
    if not files:
        print(f"No runs found in {a.out}. Record some first with collect.py.")
        return

    rows, skipped, plots_done, warnings = [], [], 0, []
    for f in files:
        try:
            meta, arrays = load_run(f)
        except Exception as exc:
            skipped.append(os.path.basename(f))
            warnings.append(f"could not read {os.path.basename(f)} ({type(exc).__name__}); skipped it")
            continue
        outcome = _outcome(meta)
        nan = any(np.isnan(v).any() for v in arrays.values() if v.dtype.kind == "f")
        rows.append({
            "seed": meta["seed"], "outcome": outcome, "template_outcome": meta["outcome"],
            "final_err_cm": round(meta["final_ee_error"] * 100, 2),
            "reach_time_s": meta["reach_time"] if meta["reach_time"] is not None
            else (round(meta["duration"], 2) if outcome == "reached" else None),
            "duration_s": round(meta["duration"], 2),
            "nav_done_s": meta["nav_done_time"], "nav_timed_out": meta["nav_timed_out"],
            "n_logged": meta["n_logged"], "has_nan": nan, "wall_s": round(meta["wall_time_s"], 1),
            "target_z": round(meta["target"][2], 3),
        })
        if plots_done < a.plot:
            png = os.path.join(a.out, f"check_{meta['seed']:05d}.png")
            try:
                plot_run(meta, arrays, png)
            except OSError:
                plt.close("all")
                warnings.append(f"could not save {os.path.basename(png)}: is it open in another program? "
                                f"Close it and run again")
            plots_done += 1

    if not rows:
        print("No readable runs found.")
        return

    csv_path = os.path.join(a.out, "summary.csv")
    try:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        wrote_csv = True
    except PermissionError:
        warnings.append("could not write summary.csv: is it open in Excel? Close it and run again")
        wrote_csv = False

    outcomes = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    relabelled = [r["seed"] for r in rows if r["outcome"] != r["template_outcome"]]
    reached = [r for r in rows if r["outcome"] == "reached"]
    print(f"{len(rows)} runs: {outcomes}")
    if relabelled:
        print(f"  relabelled as reached (within 1 cm, template said stall): seeds {relabelled}")
    failed = [r["seed"] for r in rows if r["outcome"] != "reached"]
    if failed:
        print(f"  not reached (exclude from training): seeds {failed}")
    if reached:
        rt = np.array([r["reach_time_s"] for r in reached], dtype=float)
        print(f"reached: time to reach {rt.mean():.1f} s on average (min {rt.min():.1f}, max {rt.max():.1f})")
    print(f"runs with NaNs: {sum(r['has_nan'] for r in rows)}; nav timeouts: {sum(r['nav_timed_out'] for r in rows)}")
    print(f"total rows: {sum(r['n_logged'] for r in rows)}; mean wall time per run {np.mean([r['wall_s'] for r in rows]):.1f} s")
    if wrote_csv:
        print(f"wrote {csv_path} and {plots_done} plot(s)")
    for w_ in warnings:
        print(f"note: {w_}")


if __name__ == "__main__":
    run_main(main, HERE)
