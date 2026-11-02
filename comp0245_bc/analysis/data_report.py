"""Where are the targets and starts, where does the model fail, and which expert drives are 'bad'?

    python analysis/data_report.py                         # figures into figures/, bad-drive list into splits/
    python analysis/data_report.py --policy vK_kick        # whose benchmark failures to overlay (default vK_kick)

Figures (all in the robot's STARTING frame: x = ahead of the robot, y = to its left, because what matters
for the model is where the target is relative to the robot, not where it is on the map):
    data_targets_robot_frame.png   training targets (grey) vs test targets (green = model reached, red = failed)
    data_coverage.png              histograms: target direction (ahead / side / behind), distance, height,
                                   training vs test, with the model's failure rate per bin
    data_expert_quality.png        the expert's own drives: time taken and path length; the 'bad' ones marked
Also writes splits/bad_runs.json: expert drives that succeeded but badly (see BAD_RULES), for --exclude in train.py.
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

from check_runs import _outcome, load_run
from quiet import run_main

HERE = ROOT
DRIVE_DIRS = ["runs", "runs_nudged", "runs_reach", "runs_kick"]   # model K's training data

# A successful expert drive counts as "bad" (an anomaly the model should not copy) if:
BAD_RULES = {
    "slow": "took longer than 2.5x the median time for its distance",
    "detour": "base path more than 2x the straight-line distance to the parking spot (circling)",
    "nav_timeout": "the base never parked properly; the 25 s navigation timeout ended the drive phase",
}


def to_start_frame(target, start):
    """Target (x, y) as seen from the robot's start pose: (ahead, left)."""
    dx, dy = target[0] - start[0], target[1] - start[1]
    c, s = np.cos(start[2]), np.sin(start[2])
    return c * dx + s * dy, -s * dx + c * dy


def drive_stats(path):
    meta, a = load_run(path)
    bp = a["base_pos"][:, :2]
    path_len = float(np.linalg.norm(np.diff(bp, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(np.array(meta["base_goal_pos"]) - np.array(meta["start_pose"][:2])))
    ahead, left = to_start_frame(meta["target"], meta["start_pose"])
    return {"file": os.path.relpath(path, HERE), "seed": meta["seed"], "outcome": _outcome(meta),
            "time": float(meta["duration"]), "path": path_len, "straight": straight,
            "nav_timeout": bool(meta.get("nav_timed_out")), "ahead": ahead, "left": left,
            "height": meta["target"][2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="vK_kick")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- the expert's drives -------------------------------------------------------------
    with open(os.path.join(HERE, "splits", "split.json")) as fh:
        split = json.load(fh)
    stats = []
    for d in DRIVE_DIRS:
        for f in sorted(glob.glob(os.path.join(HERE, "data", d, "run_*.npz"))):
            s = drive_stats(f)
            s["set"] = d
            if d == "runs":
                s["set"] = "val" if s["seed"] in split["val"] else "test" if s["seed"] in split["test"] else "train"
            stats.append(s)
    ok = [s for s in stats if s["outcome"] == "reached"]

    # time vs distance: a drive is "slow" if it took > 2.5x the median time per metre (+ fixed overheads)
    dist = np.array([s["straight"] for s in ok]); tt = np.array([s["time"] for s in ok])
    A = np.column_stack([np.ones_like(dist), dist])
    coef = np.linalg.lstsq(A, tt, rcond=None)[0]                      # expected time = a + b * distance
    for s in ok:
        expected = coef[0] + coef[1] * s["straight"]
        reasons = []
        if s["time"] > 2.5 * expected:
            reasons.append("slow")
        if s["path"] > 2.0 * max(s["straight"], 0.3):
            reasons.append("detour")
        if s["nav_timeout"]:
            reasons.append("nav_timeout")
        s["bad"] = reasons
    bad = [s for s in ok if s["bad"]]
    bad_train = [s for s in bad if s["set"] not in ("val", "test")]
    with open(os.path.join(HERE, "splits", "bad_runs.json"), "w") as fh:
        json.dump({"rules": BAD_RULES, "note": "successful expert drives judged anomalous; exclude from TRAINING "
                   "only (validation and test stay unchanged)",
                   "runs": [{"file": s["file"], "seed": s["seed"], "set": s["set"], "why": s["bad"],
                             "time_s": round(s["time"], 1), "path_m": round(s["path"], 2),
                             "straight_m": round(s["straight"], 2)} for s in bad]}, fh, indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    good = [s for s in ok if not s["bad"]]
    for group, col, lab in ((good, "#7f8c8d", "normal drive"), (bad, "#c0392b", "flagged as bad")):
        ax[0].scatter([s["straight"] for s in group], [s["time"] for s in group], s=8, c=col, label=lab, alpha=0.7)
        ax[1].scatter([s["straight"] for s in group], [s["path"] for s in group], s=8, c=col, label=lab, alpha=0.7)
    xs = np.linspace(0, dist.max(), 50)
    ax[0].plot(xs, coef[0] + coef[1] * xs, "k--", lw=1, label="typical time")
    ax[0].plot(xs, 2.5 * (coef[0] + coef[1] * xs), "r:", lw=1, label="2.5x typical")
    ax[1].plot(xs, 2 * np.maximum(xs, 0.3), "r:", lw=1, label="2x straight line")
    ax[0].set(xlabel="straight-line distance to parking spot (m)", ylabel="drive time (s)",
              title="Expert drives: time taken")
    ax[1].set(xlabel="straight-line distance to parking spot (m)", ylabel="base path length (m)",
              title="Expert drives: path length (circling shows up high)")
    for x in ax:
        x.legend(fontsize=8); x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "data_expert_quality.png"), dpi=130); plt.close(fig)

    # ---- benchmark scenes + the model's results ------------------------------------------
    with open(os.path.join(HERE, "benchmark", "scenes.json")) as fh:
        scenes = {(s["tier"], s["index"]): s for s in json.load(fh)["scenes"]}
    res_path = find_result(a.results, f"{a.policy}_tier1.csv")
    with open(res_path) as fh:
        rows = list(csv.DictReader(fh))
    test = []
    for r in rows:
        sc = scenes[(1, int(r["index"]))]
        ah, le = to_start_frame(sc["target"], sc["start_pose"])
        test.append({"ahead": ah, "left": le, "height": sc["target"][2], "dist": float(np.hypot(ah, le)),
                     "ok": r["success"] == "True"})
    train = [s for s in ok if s["set"] not in ("val", "test")]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter([s["left"] for s in train], [s["ahead"] for s in train], s=6, c="#bdc3c7", label="training targets")
    for flag, col, lab in ((True, "#27ae60", f"test: {a.policy} reached"), (False, "#c0392b", f"test: {a.policy} failed")):
        pts = [t for t in test if t["ok"] == flag]
        ax.scatter([t["left"] for t in pts], [t["ahead"] for t in pts], s=40, c=col, edgecolor="k", lw=0.4, label=lab)
    ax.scatter([0], [0], marker="^", s=160, c="k", label="robot at start (facing up)")
    ax.invert_xaxis()                                      # left of the robot is on the left of the plot
    ax.set(xlabel="target to the robot's left (m)", ylabel="target ahead of the robot (m)",
           title="Where the target is, seen from the robot's start")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "data_targets_robot_frame.png"), dpi=130); plt.close(fig)

    # Coverage histograms with failure rate per bin
    def direction(s):
        return np.degrees(np.arctan2(s["left"], s["ahead"]))    # 0 = straight ahead, +-180 = behind
    feats = [("direction of target from robot (deg; 0 = ahead, +-180 = behind)", direction, np.linspace(-180, 180, 9)),
             ("distance from robot start (m)", lambda s: float(np.hypot(s["ahead"], s["left"])), np.linspace(0, 3.2, 9)),
             ("target height (m)", lambda s: s["height"], np.linspace(0.1, 1.15, 8))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    summary = {}
    for x, (label, fn, bins) in zip(ax, feats):
        tr = np.array([fn(s) for s in train]); te = np.array([fn(t) for t in test])
        okm = np.array([t["ok"] for t in test])
        x.hist(tr, bins=bins, color="#bdc3c7", label=f"training drives (n={len(tr)})", density=True)
        x.hist(te, bins=bins, histtype="step", color="k", lw=1.5, label=f"test scenes (n={len(te)})", density=True)
        x2 = x.twinx()
        centres, fail = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (te >= lo) & (te < hi)
            if m.sum() >= 3:
                centres.append((lo + hi) / 2); fail.append(1 - okm[m].mean())
        x2.plot(centres, fail, "o-", color="#c0392b", label=f"{a.policy} failure rate")
        x2.set_ylim(0, 1); x2.set_ylabel("failure rate (bins with >= 3 test scenes)", color="#c0392b")
        x.set_xlabel(label); x.set_ylabel("share of drives"); x.grid(alpha=0.3)
        x.legend(fontsize=7, loc="upper left"); x2.legend(fontsize=7, loc="upper right")
        summary[label] = {"bins": [round(float(c), 2) for c in centres], "failure_rate": [round(float(f), 2) for f in fail]}
    fig.suptitle("Is the training data covering the test scenes, and where does the model fail?")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "data_coverage.png"), dpi=130); plt.close(fig)

    behind = [t for t in test if abs(direction(t)) > 90]
    print(f"expert drives checked: {len(stats)} | succeeded: {len(ok)} | flagged bad: {len(bad)} "
          f"({len(bad_train)} in training data) -> splits/bad_runs.json")
    for k in BAD_RULES:
        print(f"   {k:12s}: {sum(k in s['bad'] for s in bad)}  ({BAD_RULES[k]})")
    print(f"test targets behind the robot at the start: {len(behind)}/{len(test)}, "
          f"{a.policy} failed {sum(not t['ok'] for t in behind)} of them; "
          f"ahead: failed {sum(not t['ok'] for t in test if abs(direction(t)) <= 90)}/{len(test) - len(behind)}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    run_main(main, HERE)
