"""Collects every result into results/all_results.json and draws figures/*.png for the report."""
import csv
import glob
import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
F = os.path.join(HERE, "figures")
SCENES = ["clear", "sparse", "cluttered", "gap", "blocked"]
TIER = dict(zip(range(10, 15), SCENES))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "figure.dpi": 100})

# (label, group, pattern, kind)
SETUPS = [
    ("Original expert", "Before", "before/expert", "ev"),
    ("R alone", "Before", "before/vR_ik", "ev"),
    ("Planner + expert arm", "True position", "stage3/planner_expert_hold5", "ev"),
    ("A seed 0 + expert, brake", "True position", "stage3/brake_driver_A_expert_hold5", "ev"),
    ("A seed 1 + expert, brake", "True position", "stage3/brake_driver_A_s1_expert_hold5", "ev"),
    ("A seed 0 + expert", "True position", "stage3/driver_A_expert_hold5", "ev"),
    ("A seed 1 + expert", "True position", "stage3/driver_A_s1_expert_hold5", "ev"),
    ("Planner + R", "True position", "stage3/planner_vR_ik_hold5", "ev"),
    ("Planner + vS seed 0", "True position", "stage5/planner_vS_arm_hold5", "ev"),
    ("A seed 0 + R", "True position", "stage3/driver_A_vR_ik_hold5", "ev"),
    ("A seed 1 + R", "True position", "stage3/driver_A_s1_vR_ik_hold5", "ev"),
    ("A seed 0 + R, brake", "True position", "stage5/brake_driver_A_vR_ik_hold5", "ev"),
    ("A seed 1 + R, brake", "True position", "stage5/brake_driver_A_s1_vR_ik_hold5", "ev"),
    ("B seed 0 + expert", "True position", "stage4/driver_B_expert_hold5", "ev"),
    ("B seed 1 + expert", "True position", "stage4/driver_B_s1_expert_hold5", "ev"),
    ("B seed 0 + R", "True position", "stage4/driver_B_vR_ik_hold5", "ev"),
    ("Camera + planner + expert", "No true position", "stage6/cam_planner_expert_test", "sn"),
    ("Camera + A seed 0 + expert", "No true position", "stage6/cam_driver_A_expert_test", "sn"),
    ("Camera + A seed 1 + expert", "No true position", "stage6/cam_driver_A_s1_expert_test", "sn"),
    ("Camera + planner + R", "No true position", "stage6/cam_planner_vR_ik_test", "sn"),
    ("Camera + planner + vS seed 0", "No true position", "stage6/cam_planner_vS_arm_test", "sn"),
    ("Camera + A seed 0 + R, brake", "No true position", "stage6/cam_driver_A_vR_ik_test", "sn"),
    ("Camera + A seed 1 + R, brake", "No true position", "stage6/cam_driver_A_s1_vR_ik_test", "sn"),
    ("Search + planner + expert", "No true position", "stage6/search_planner_expert_test", "sn"),
    ("Odometry only + planner + expert", "No true position", "stage6/odo_planner_expert_test", "sn"),
]


def score(rows):
    return {"n": len(rows), "reached": sum(1 for r in rows if r.get("first_reach_s") not in ("", None)), "held": sum(1 for r in rows if r["outcome"] == "reached")}


def load(pattern, kind):
    rows = []
    if kind == "ev":
        for t, sc in TIER.items():
            with open(os.path.join(R, f"{pattern}_tier{t}.csv")) as fh:
                rows += [{**r, "scenario": sc} for r in csv.DictReader(fh)]
    else:
        with open(os.path.join(R, pattern + ".csv")) as fh:
            rows = list(csv.DictReader(fh))
    return rows


def collect():
    out = []
    for label, group, pat, kind in SETUPS:
        rows = load(pat, kind)
        o = Counter(r["outcome"] for r in rows)
        out.append({"label": label, "group": group, **score(rows), "scenes": {s: score([r for r in rows if r["scenario"] == s]) for s in SCENES},
                    "timeouts": o["timeout"], "arm_crashes": o["crashed_base"] + o["crashed_ground"], "obstacle_hits": o["crashed_obstacle"], "not_held": o["reached_not_held"]})
    with open(os.path.join(R, "all_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def fig_overview(res):
    keep = ["Original expert", "Planner + expert arm", "A seed 0 + expert, brake", "A seed 1 + expert, brake", "Planner + R", "A seed 0 + R, brake", "A seed 1 + R, brake",
            "Camera + planner + expert", "Camera + A seed 0 + expert", "Camera + A seed 1 + expert", "Camera + planner + R", "Camera + A seed 0 + R, brake", "Camera + A seed 1 + R, brake", "Search + planner + expert", "Odometry only + planner + expert"]
    by = {r["label"]: r for r in res}
    rows = [by[k] for k in keep][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [r["reached"] / r["n"] * 100 for r in rows], 0.34, color=BLUE, label="Reached")
    ax.barh(y - 0.19, [0 if r["group"] == "Before" else r["held"] / r["n"] * 100 for r in rows], 0.34, color=ORANGE, label="Held 5 s")
    for i, r in enumerate(rows):
        ax.text(max(r["reached"], r["held"]) / r["n"] * 100 + 1, i, str(r['reached']) if r["group"] == "Before" else f"{r['reached']} / {r['held']}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, [r["label"] for r in rows], color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xlabel("% of 150 frozen test drives (labels: reached / held)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "overview.png"))


def fig_scenes(res):
    keep = ["Planner + expert arm", "A seed 1 + expert, brake", "Planner + R", "A seed 1 + R, brake", "Camera + planner + expert", "Camera + planner + R", "Camera + planner + vS seed 0", "Camera + A seed 0 + R, brake", "Camera + A seed 1 + R, brake", "Search + planner + expert"]
    by = {r["label"]: r for r in res}
    m = np.array([[by[k]["scenes"][s]["held"] for s in SCENES] for k in keep])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fc", "#2a78d6", "#123f78"]), vmin=0, vmax=30, aspect="auto")
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v >= 14 else INK)
    ax.set_xticks(range(5), SCENES, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(keep)), keep, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "scenes_held.png"))


def fig_camera():
    with open(os.path.join(R, "camera.json")) as fh:
        rows = json.load(fh)["rows"]
    d = [r["distance_m"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(d, [r["median_cm"] for r in rows], "-o", color=BLUE, lw=2, ms=4, label="Median")
    ax.plot(d, [r["p95_cm"] for r in rows], "-o", color=ORANGE, lw=2, ms=4, label="95th percentile")
    ax.set_xlabel("Distance to target (m)")
    ax.set_ylabel("Target position error (cm)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "camera_error.png"))


def dev_score(name):
    rows = []
    for t in TIER:
        with open(os.path.join(R, "dev", f"{name}_tier{t}.csv")) as fh:
            rows += list(csv.DictReader(fh))
    return score(rows)


def fig_learning():
    fr = {25: "brake_driver_A_p25_expert_hold5", 50: "brake_driver_A_p50_expert_hold5", 75: "brake_driver_A_p75_expert_hold5", 100: "brake_driver_A_expert_hold5"}
    held = {f: dev_score(n)["held"] for f, n in fr.items()}
    s1 = dev_score("brake_driver_A_s1_expert_hold5")["held"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(list(held), list(held.values()), "-o", color=BLUE, lw=2, ms=5, label="Seed 0")
    ax.plot([100], [s1], "o", color=ORANGE, ms=6, label="Seed 1")
    ax.set_xticks(list(held), ["25%\n(99 drives)", "50%\n(199)", "75%\n(299)", "100%\n(398)"])
    ax.set_ylim(50, 75)
    ax.set_ylabel("Held 5 s, of 75 validation drives")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "learning_curve.png"))
    return held, s1


def fig_training():
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name, c, lab in (("train_driver_A", BLUE, "A seed 0"), ("train_driver_A_s1", ORANGE, "A seed 1")):
        with open(os.path.join(R, name + ".txt")) as fh:
            v = [float(x) for x in re.findall(r"val MSE ([0-9.]+)", fh.read())]
        ax.plot(range(1, len(v) + 1), v, color=c, lw=2, label=lab)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (standardised)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "training_curve.png"))


def draw_obstacle(ax, o):
    if o["shape"] == "cylinder":
        ax.add_patch(plt.Circle(o["xy"], o["radius"], color="#b9b8b2"))
    else:
        hx, hy = o["half_extents"][:2]
        c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
        pts = [(sx * hx, sy * hy) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        ax.add_patch(plt.Polygon([(o["xy"][0] + c * x - s * y, o["xy"][1] + s * x + c * y) for x, y in pts], color="#b9b8b2"))


def fig_routes():
    path = os.path.join(R, "routes.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        d = json.load(fh)
    style = {"planner": (BLUE, "Planner (true position)"), "driverbrake:trained/driver_A_s1.npz": (ORANGE, "Learned driver A"), "cam": (AQUA, "Camera, no true position")}
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5), sharex=True, sharey=True)
    for ax, sc in zip(axes, SCENES):
        s = next(x for x in d["scenes"] if x["scenario"] == sc)
        for o in s.get("obstacles") or []:
            draw_obstacle(ax, o)
        for r in d["drives"]:
            if r["scenario"] == sc:
                xy = np.array(r["xy"])
                ax.plot(xy[:, 0], xy[:, 1], color=style[r["setup"]][0], lw=1.6, label=style[r["setup"]][1])
        ax.plot(*s["start_pose"][:2], "s", color=INK, ms=4)
        ax.plot(*s["target"][:2], "*", color="#e34948", ms=9)
        ax.set_title(sc, fontsize=9, color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.grid(color=GRID, lw=0.4)
    axes[0].set_ylabel("y (m)")
    axes[2].set_xlabel("x (m). Square: start, star: target, grey: obstacles")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(F, "routes.png"))


def main():
    os.makedirs(F, exist_ok=True)
    res = collect()
    fig_overview(res)
    fig_scenes(res)
    fig_camera()
    print(fig_learning())
    fig_training()
    fig_routes()
    for r in res:
        print(f"{r['label']:36s} {r['reached']:3d} {r['held']:3d} t{r['timeouts']:3d} a{r['arm_crashes']:3d} o{r['obstacle_hits']:3d} nh{r['not_held']:2d}")


if __name__ == "__main__":
    main()
