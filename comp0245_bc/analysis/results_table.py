"""Every saved result in one spreadsheet: results/all_results.xlsx.

    python analysis/results_table.py                       # writes results/all_results.xlsx
    python analysis/results_table.py --out /tmp/test.xlsx  # somewhere else

The first sheets are comparison tables in the style of the viva tables (test tiers, headline, ladder,
arm swap, memory models, input ablation, PCA, other learners, Bayesian optimisation, super test,
obstacles, side experiments, noise and delay). Each has a title, a "say it like this" line and a
source line, and every number is read from a result file. The only typed-in numbers are the offline
validation errors of the other regressors, which no result file holds. Cells whose result file is
missing are left blank and listed when the script finishes.

The raw sheets after them hold every per-drive CSV under results/ (one row per CSV, so one per system,
condition and tier), sorted by results folder. Each row is worked out from the drives exactly as
evaluate.py's summarise() does. The CSVs are used rather than *_summary.json because a summary keeps
only the last tier that was run. The last sheet, All scenes, has one row per drive from the same CSVs.
    Main models (raw)       results/1_main_models/ except noise_and_delay/
    Noise and delay (raw)   results/1_main_models/noise_and_delay/
    Obstacles (raw)         results/2_obstacles/ except abdullah_bases/
    Abdullah bases (raw)    results/2_obstacles/abdullah_bases/
    Final tests (raw)       results/3_final_tests/
    Old tests (raw)         results/0_old_tests/
The test condition (sensor noise, delay, torque noise, payload, noise only, delay only) and the
success rule (held 5 s or reached) are read from the file name.
Uses only the Python standard library (an .xlsx file is a zip of XML files), so no install is needed.
"""
from _paths import ROOT  # noqa: F401  (project folder; lets the code folders import each other)
import argparse
import csv
import glob
import json
import math
import os
import re
import zipfile
from xml.sax.saxutils import escape

RESULTS = os.path.join(ROOT, "results")

# (sheet name, folder under results/, subfolders left out because another sheet takes them)
RAW_SHEETS = [
    ("Main models (raw)", "1_main_models", ("noise_and_delay",)),
    ("Noise and delay (raw)", "1_main_models/noise_and_delay", ()),
    ("Obstacles (raw)", "2_obstacles", ("abdullah_bases",)),
    ("Abdullah bases (raw)", "2_obstacles/abdullah_bases", ()),
    ("Final tests (raw)", "3_final_tests", ()),
    ("Old tests (raw)", "0_old_tests", ()),
]

# File-name parts that name the test condition, as (pattern, label). Removed from the name to give the policy.
CONDITIONS = [
    (r"_obs_noise([0-9.]+)", "sensor noise {}"),
    (r"_delay_steps(\d+)", "delay {} steps"),
    (r"_torque_noise([0-9.]+)", "torque noise {}"),
    (r"_payload_kg([0-9.]+)", "payload {} kg"),
    (r"_noiseonly", "noise only"),
    (r"_delayonly", "delay only"),
]
OUTCOMES = ["reached", "reached_not_held", "timeout", "crashed_base", "crashed_ground",
            "crashed_obstacle", "crashed_object", "tipped_over"]
COLUMNS = ["System (folder)", "Policy", "Condition", "Success rule", "Tier", "Scenes", "Success",
           *OUTCOMES, "Median final error (cm)", "Mean time to reach (s)", "Torque jerk", "Wheel jerk",
           "Decision time (us)", "File"]


def parse_name(name):
    """Policy, condition and success rule from a result file name (without _tierN.csv)."""
    rule = "held 5 s" if "_hold5" in name else "reached"
    name = name.replace("_hold5", "")
    found = []
    for pattern, label in CONDITIONS:
        m = re.search(pattern, name)
        if m:
            found.append(label.format(*m.groups()))
            name = name[:m.start()] + name[m.end():]
    return name, ", ".join(found) or "clean", rule


def summarise(drives):
    """The same numbers as evaluate.py's summarise(), from the rows of one per-drive CSV."""
    def num(r, key):
        try:
            return float(r[key])
        except (KeyError, TypeError, ValueError):
            return None

    def mean(key, subset):
        vals = [v for v in (num(r, key) for r in subset) if v is not None]
        return sum(vals) / len(vals) if vals else None

    # Abdullah's stage CSVs have no success column: there success means the outcome is "reached".
    ok = [r for r in drives if (r["success"] == "True" if "success" in r else r["outcome"] == "reached")]
    errs = sorted(v for v in (num(r, "final_err_cm") for r in drives) if v is not None)
    mid = len(errs) // 2
    median = (errs[mid] if len(errs) % 2 else (errs[mid - 1] + errs[mid]) / 2) if errs else None
    n = len(drives)
    return {"scenes": n, "success_rate": len(ok) / n if n else 0.0, "median_final_err_cm": median,
            "mean_time_to_reach_s": mean("time_s", ok), "torque_jerk": mean("torque_jerk", drives),
            "wheel_jerk": mean("wheel_jerk", drives), "think_us_per_step": mean("think_us_per_step", drives),
            "outcomes": {k: sum(r["outcome"] == k for r in drives) for k in OUTCOMES}}


def result_files(folder, skip):
    """Every per-drive CSV under results/<folder>, minus the skipped subfolders, with its parsed name."""
    base = os.path.join(RESULTS, folder)
    for path in sorted(glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)):
        rel = os.path.relpath(path, base)
        if rel.split(os.sep)[0] in skip:
            continue
        stem = os.path.basename(path)[:-len(".csv")]
        m = re.search(r"_tier(\d+)$", stem)
        tier = int(m.group(1)) if m else ""
        policy, condition, rule = parse_name(stem[:m.start()] if m else stem)
        system = os.path.dirname(rel).replace(os.sep, "/") or "."
        parts = system.split("/")
        if parts[-1].startswith("tier") and parts[-1][4:].isdigit():
            system = "/".join(parts[:-1]) or "."      # K/tier4/vK_kick_tier4.csv belongs to system K
        for part, label in (("noise_only", "noise only"), ("delay_only", "delay only")):
            if condition == "clean" and part in parts:
                condition = label                     # K diagnosis keeps the condition in the folder name
        if tier == 4 and condition == "clean":
            condition = "noise + delay (tier 4)"
        yield path, system, policy, condition, rule, tier, stem[:m.start()] if m else stem


def load_summary(path):
    """summarise() of one per-drive CSV, or None if it has no drives."""
    with open(path, newline="") as f:
        drives = list(csv.DictReader(f))
    return summarise(drives) if drives and "outcome" in drives[0] else None


SCENE_FIELDS = ["index", "seed", "outcome", "success", "final_err_cm", "min_err_cm", "time_s", "base_path_m",
                "torque_jerk", "wheel_jerk", "think_us_per_step"]
SCENE_COLUMNS = ["System (folder)", "Policy", "Condition", "Success rule", "Tier", "Scene index", "Seed", "Outcome",
                 "Success", "Final error (cm)", "Closest (cm)", "Time (s)", "Base path (m)", "Torque jerk",
                 "Wheel jerk", "Decision time (us)", "File"]


def scene_rows():
    """One row per drive in every per-drive CSV under results/."""
    rows = []
    for path, system, policy, condition, rule, tier, _ in result_files("", ()):
        with open(path, newline="") as f:
            drives = list(csv.DictReader(f))
        if not drives or "outcome" not in drives[0]:
            continue
        file = os.path.relpath(path, ROOT).replace(os.sep, "/")
        for d in drives:
            if "success" not in d:
                d["success"] = str(d["outcome"] == "reached")   # Abdullah's stage CSVs
            vals = []
            for k in SCENE_FIELDS:
                v = d.get(k, "")
                try:
                    v = float(v) if k not in ("outcome", "success") and v != "" else (v or None)
                    if k in ("index", "seed") and isinstance(v, float):
                        v = int(v)
                except ValueError:
                    pass
                vals.append(v)
            rows.append([system, policy, condition, rule, tier, *vals, file])
    return rows


def raw_rows(folder, skip):
    """One row per per-drive CSV under results/<folder>."""
    rows = []
    for path, system, policy, condition, rule, tier, _ in result_files(folder, skip):
        s = load_summary(path)
        if s is None:
            continue
        rows.append([system, policy, condition, rule, tier, s["scenes"], s["success_rate"],
                     *[s["outcomes"][k] for k in OUTCOMES], s["median_final_err_cm"],
                     s["mean_time_to_reach_s"], s["torque_jerk"], s["wheel_jerk"], s["think_us_per_step"],
                     os.path.relpath(path, ROOT).replace(os.sep, "/")])
    return rows


# --- comparison tables (David's viva-table style) --------------------------------------------------
# Each table is looked up by (system folder under results/, file name without _tierN, tier), so it
# does not matter which tierN/ subfolder a CSV sits in. Lookups that find no file are printed at the end.
INDEX = {}
MISSING = []
F = "1_main_models/noise_and_delay/filters_and_robust_models"
KD = "1_main_models/noise_and_delay/K_diagnosis"
ST = "3_final_tests/super_test"
FR = "3_final_tests/fresh_scenes"


class Pct(float):
    """A success rate: shown as a percentage."""


def build_index():
    for path, system, _, _, _, tier, name in result_files("", ()):
        s = load_summary(path)
        if s is not None:
            INDEX[(system, name, tier)] = s


def get(system, name, tier=1):
    s = INDEX.get((system, name, tier))
    if s is None:
        MISSING.append(f"{system}/{name} tier {tier}")
    return s


def rate(system, name, tier=1):
    s = get(system, name, tier)
    return Pct(s["success_rate"]) if s else None


def pooled(items):
    """One success rate over several (system, name, tier) results, as super_table.py does."""
    found = [get(*it) for it in items]
    if not all(found):
        return None
    return Pct(sum(s["success_rate"] * s["scenes"] for s in found) / sum(s["scenes"] for s in found))


def mean(values):
    values = [v for v in values if v is not None]
    return Pct(sum(values) / len(values)) if values else None


def seeds(system, names, tier=1):
    """Rates of each seed (None where missing), padded to 3, then their mean."""
    rates = [rate(system, n, tier) for n in names] + [None] * (3 - len(names))
    return [*rates, mean(rates)]


# Each table: (title, "say it like this" line, header, rows, source line).
def tier_tables():
    E, R = "1_main_models/expert", "1_main_models/R"
    tiers = [
        ("1 Normal", "Same recipe as training, unseen seeds", "1-2 m away, 0.15-1.10 m high",
         "Within 1 m of the origin, any facing", "None", 1),
        ("2 New targets", "Objects outside the training range", "Half too high (1.10-1.25 m), half too far (2-2.5 m)",
         "Normal", "None", 2),
        ("3 Odd starts", "Unusual starting positions", "Normal",
         "Half far (1-2 m) facing away, half very close (0.3-0.7 m)", "None", 3),
        ("4 Disturbances", "Sensor noise and command delay (see the Noise and delay sheet)", "Normal", "Normal",
         "Noise 0.005-0.02, delay 1-4 steps", 4),
        ("5 Physics", "Different floor and robot weight", "Normal", "Normal",
         "Floor friction 0.5-1.3, base mass x0.7-1.3", 5),
        ("6 Objects anywhere", "Objects far, near, high or on the floor", "0.5-3 m away, 0.05-1.6 m high (some unreachable)",
         "Up to 2 m from the origin", "None", 6),
        ("7 Everything mixed", "All disturbances at random", "Normal", "Normal",
         "Noise, delay, motor noise, payload 0-0.5 kg, friction, mass", 7),
    ]
    rows = []
    for label, tests, target, start, dist, t in tiers:
        r_reached = rate(R, "vR_ik", t) if t <= 3 else (
            rate("1_main_models/abdullah_model_C/benchmark_50_scenes/R", "vR_ik", 4) if t == 4 else None)
        rows.append([label, tests, target, start, dist,
                     rate(E, "expert", t) if t <= 5 else None,
                     rate(E, "expert_hold5", t) if t != 4 else None,
                     r_reached, rate(R, "vR_ik_hold5", t) if t != 4 else None])
    rows.append(["Fresh scenes", "Final test: new tier-1 scenes, each used once", "Like tier 1", "Like tier 1", "None",
                 rate(FR + "/expert", "expert"), rate(FR + "/expert", "expert_hold5"),
                 seeds(FR + "/R", ["vR_ik", "vR_ik_s1", "vR_ik_s2"])[3],
                 seeds(FR + "/R", ["vR_ik_hold5", "vR_ik_s1_hold5", "vR_ik_s2_hold5"])[3]])
    return [("Test tiers: what each tier checks (50 frozen scenes per tier)",
             "Tier 1 is like training. Every other tier changes one thing: targets, starts, disturbances, "
             "physics, objects anywhere, or everything at once.",
             ["Tier", "What it tests", "Object (target)", "Robot start", "Disturbance", "Expert reached",
              "Expert held 5 s", "Model R reached", "Model R held 5 s"], rows,
             "Source: results/1_main_models/expert and R (seed 0). Tier 4 R reached is from Abdullah's rerun of R. "
             "Fresh scenes: R is the mean of 3 seeds. Blank = not run.")]


def headline_tables():
    rows = [["Expert (teacher)", *seeds(FR + "/expert", ["expert"]), *seeds(FR + "/expert", ["expert_hold5"])],
            ["Model K", *seeds(FR + "/K", ["vK_kick"]), *seeds(FR + "/K", ["vK_kick_hold5"])],
            ["Model P", *seeds(FR + "/P", ["vP_hold"]), *seeds(FR + "/P", ["vP_hold_hold5"])],
            ["Model R (arm and base learned)", *seeds(FR + "/R", ["vR_ik", "vR_ik_s1", "vR_ik_s2"]),
             *seeds(FR + "/R", ["vR_ik_hold5", "vR_ik_s1_hold5", "vR_ik_s2_hold5"])],
            ["R + Kalman + VFH/Bug 2", *[None] * 4, *seeds(FR + "/R+Kalman+VFH_Bug2_nav", ["vR_ik_kfv_nav_hold5"])],
            ["R-BO + Kalman + VFH/Bug 2", *[None] * 4,
             *seeds(FR + "/R-BO+Kalman+VFH_Bug2_nav", ["bo_t03_kfv_nav_hold5"])],
            ["Final system: R-BO + Kalman + A* planner", *[None] * 4,
             *seeds(ST + "/fresh_scenes", ["planner_bo_t03_kfv_hold5"])]]
    return [("Headline: fresh, never-seen tier-1 scenes (50 scenes)",
             "Model R holds most fresh scenes, close to the expert and far ahead of K and P. "
             "The final system adds the A* planner for the base.",
             ["Model", "Reached seed 0", "Reached seed 1", "Reached seed 2", "Reached mean",
              "Held 5 s seed 0", "Held 5 s seed 1", "Held 5 s seed 2", "Held 5 s mean"], rows,
             "Source: results/3_final_tests/fresh_scenes and super_test/fresh_scenes (scenes_fresh.json, used once "
             "per model). Success = hand within 1 cm of the target, held 5 s unless 'reached' is stated.")]


def ladder_tables():
    O, M = "0_old_tests", "1_main_models"
    steps = [("v0", "Copies the expert's raw commands, including the gravity torque", O + "/v0", "mlp_v0", None),
             ("v1", "Learns torque minus gravity (gravity from Pinocchio)", O + "/v1", "mlp_v1", None),
             ("v1 + history", "+ history inputs (motion over 0.1 s and 0.5 s)", O + "/v1_history", "mlp_v1_hist", None),
             ("v2", "+ nudged recordings (DART)", O + "/v2_history", "mlp_v2_hist", None),
             ("A", "+ parking-spot inputs", O + "/A", "vA_goal", None),
             ("D", "+ separate arm and wheel networks, reach weight 3", O + "/D", "vD_all", None),
             ("D2", "+ reach-nudged recordings", O + "/D2", "vD2_reachdata", None),
             ("E", "+ error-average inputs over 0.5 s and 2 s", O + "/E", "vE_integral", None),
             ("Model K", "+ velocity-kick recordings", M + "/K", "vK_kick", "vK_kick_hold5"),
             ("Model P", "+ hold-still and extreme-height recordings", M + "/P", "vP_hold", "vP_hold_hold5"),
             ("Model R", "+ arm IK inputs (71 inputs, two 3x256 MLPs)", M + "/R", "vR_ik", "vR_ik_hold5"),
             ("R-BO", "Bayesian-optimised settings (3x192) + 29 ms position filter", M + "/R-BO", None,
              "bo_t03_filt29noqd_hold5")]
    rows = [[label, what, rate(sys_, reached) if reached else None, rate(sys_, held) if held else None]
            for label, what, sys_, reached, held in steps]
    return [("The ladder: one change at a time (benchmark tier 1, 50 scenes, seed 0)",
             "Each step adds one change to the step before. The arm IK inputs in model R gave the largest jump.",
             ["Model", "What changed", "Reached", "Held 5 s"], rows,
             "Source: results/0_old_tests and results/1_main_models. Blank = not run with that rule.")]


def arm_swap_tables():
    A = "1_main_models/arm_swap"
    rows = [["P arm + P base (model P)", *seeds("1_main_models/P", ["vP_hold_hold5"])],
            ["P arm + expert base", *seeds(A + "/P_arm+expert_base", ["vP_hold_mlp_arm_hold5"])],
            ["Expert arm + P base", *seeds(A + "/expert_arm+P_base", ["vP_hold_mlp_wheels_hold5"])],
            ["R arm + expert base", *seeds(A + "/R_arm+expert_base",
                                           ["vR_ik_mlp_arm_hold5", "vR_ik_s1_mlp_arm_hold5", "vR_ik_s2_mlp_arm_hold5"])],
            ["R arm + R base (model R)", *seeds("1_main_models/R", ["vR_ik_hold5", "vR_ik_s1_hold5", "vR_ik_s2_hold5"])]]
    return [("Arm swap: which half made the difference? (tier 1, held 5 s)",
             "P's base is as good as the expert's. P's arm fails with either base, so the arm was the bottleneck. "
             "R's arm fixed it.",
             ["Combination", "Seed 0", "Seed 1", "Seed 2", "Mean"], rows,
             "Source: results/1_main_models/arm_swap. The other half is always the expert's own controller.")]


def memory_tables():
    MM = "1_main_models/memory_models"
    rows = [["MLP (model P)", "Feed-forward + hand-built history inputs", 3,
             seeds("1_main_models/P", ["vP_hold", "vP_hold_s1", "vP_hold_s2"])[3],
             seeds("1_main_models/P", ["vP_hold_hold5", "vP_hold_s1_hold5", "vP_hold_s2_hold5"])[3]],
            ["TCN", "Convolution over time", 3,
             seeds(MM + "/temporal_convolution", ["tcn2", "tcn2_s1", "tcn2_s2"])[3],
             seeds(MM + "/temporal_convolution", ["tcn2_hold5", "tcn2_s1_hold5", "tcn2_s2_hold5"])[3]],
            ["Attention", "Self-attention over time", 1, rate(MM + "/attention", "attn"),
             rate(MM + "/attention", "attn_hold5")],
            ["GRU", "Recurrent network", 1, rate(MM + "/GRU_P", "gru_P"), rate(MM + "/GRU_P", "gru_P_hold5")],
            ["Earlier GRU, no hand-built inputs", "Recurrent network (K's data)", 1,
             rate(MM + "/GRU_basic", "gru_basic"), None],
            ["Earlier GRU, all inputs", "Recurrent network (K's data)", 1, rate(MM + "/GRU_full", "gru_full"), None],
            ["Earlier GRU, longer window", "Recurrent network (K's data)", 1,
             rate(MM + "/GRU_full_long", "gru_full_long"), None],
            ["First TCN (no skip connection)", "Convolution over time", 1,
             rate(MM + "/temporal_convolution_first", "tcn"), rate(MM + "/temporal_convolution_first", "tcn_hold5")]]
    return [("Learned memory: does a memory network help? (tier 1, same data as P)",
             "The MLP with hand-built memory inputs beat the GRU, TCN and attention networks trained on the same data.",
             ["Model", "Type", "Seeds", "Reached (mean)", "Held 5 s (mean)"], rows,
             "Source: results/1_main_models/memory_models and P. The first four rows use P's data and inputs, "
             "without IK inputs, so the comparison is about memory alone.")]


def ablation_tables():
    AB, full = "1_main_models/ablations", rate("1_main_models/R", "vR_ik_hold5")
    groups = [("Without IK averages", "R_no_IK_averages", "abl_no_ikavg_hold5"),
              ("Without error averages", "R_no_error_averages", "abl_no_erravg_hold5"),
              ("Without history", "R_no_history", "abl_no_history_hold5"),
              ("Without IK direction now", "R_no_IK_direction", "abl_no_iknow_hold5"),
              ("Without parking spot (goal)", "R_no_parking_spot", "abl_no_goal_hold5")]
    rows = [["Full model R (71 inputs)", full, Pct(0) if full is not None else None]]
    for label, folder, name in groups:
        r = rate(f"{AB}/{folder}", name)
        rows.append([label, r, Pct(full - r) if full is not None and r is not None else None])
    return [("Input ablation: remove one group, retrain, test (tier 1, held 5 s)",
             "The parking spot, IK direction and history inputs carry the result. The two averages are spare.",
             ["Inputs", "Held 5 s", "Drop vs full R (points)"], rows, "Source: results/1_main_models/ablations and R (seed 0).")]


def pca_tables():
    with open(os.path.join(ROOT, "splits", "pca_summary.json")) as f:
        comps = {v: float(k.rstrip("%")) / 100 for k, v in json.load(f)["components_for"].items()}
    O = "1_main_models/other_learners"
    rows = [["All 71 inputs (model R)", Pct(1), rate("1_main_models/R", "vR_ik_hold5")]]
    for n in sorted(comps, reverse=True):
        r = rate(f"{O}/R_PCA{n}", f"vR_pca{n}_hold5") if n in (15, 25) else None
        rows.append([f"{n} PCA components", Pct(comps[n]), r])
    return [("Dimension reduction with PCA (tier 1, held 5 s)",
             "25 components keep 95% of the input variance but the robot fails. The small signals PCA drops "
             "are the ones that steer the hand.",
             ["Inputs", "Variance kept", "Held 5 s"], rows,
             "Source: splits/pca_summary.json and results/1_main_models/other_learners. Blank = not driven.")]


def learner_tables():
    O, mlp = "1_main_models/other_learners", 0.0057
    learners = [("MLP (model R)", "1_main_models/R", "vR_ik", mlp, "71 inputs, two 3x256 MLPs"),
                ("Gaussian process", O + "/Gaussian_process", "gp_R", 0.079, "4,000 rows, 0.74 ms per decision"),
                ("Kernel ridge (Nystroem)", None, None, 0.066, "Offline only"),
                ("Linear ridge", None, None, 0.125, "Offline only"),
                ("Linear model (D2's inputs)", O + "/linear_regression", "cmp_linear", None, ""),
                ("Random forest (D2's inputs)", O + "/random_forest", "cmp_forest", None, "20 scenes")]
    rows = []
    for label, sys_, name, mse, note in learners:
        s = get(sys_, name) if sys_ else None
        rows.append([label, Pct(s["success_rate"]) if s else None, s["scenes"] if s else None, mse,
                     round(mse / mlp, 1) if mse else None, note])
    return [("Other regression models (tier 1, reached)",
             "On the same inputs, linear and kernel methods had 12 to 22 times the MLP's error, so the MLP was kept.",
             ["Model", "Reached", "Scenes", "Validation MSE (standardised)", "Error vs MLP (x)", "Note"], rows,
             "Source: results/1_main_models/other_learners. Validation MSE typed in from Muneeb's offline tests on "
             "R's inputs (same validation set). Ridge models were not driven.")]


def bo_tables():
    with open(os.path.join(RESULTS, "1_main_models", "R-BO", "bayes_opt_trials", "trials.json")) as f:
        trials = json.load(f)
    rows = [[f"Trial {t['trial']}" + (" (R's settings)" if t["trial"] == 0 else "") +
             (" (winner, R-BO)" if t["trial"] == 3 else ""), t["lr"], t["batch"], f"{t['depth']}x{t['width']}",
             round(t["reach_weight"], 2), t["filter_ms"], Pct(t["held_clean"]), Pct(t["held_noisy"]),
             Pct(t["score"]), t["minutes"]] for t in trials]
    return [("Bayesian optimisation of R's settings (skopt, 12 trials)",
             "A Gaussian process picks each next setting to try. Trial 3 scored best and became R-BO.",
             ["Trial", "Learning rate", "Batch", "Net (layers x width)", "Reach weight", "Filter (ms)",
              "Held clean", "Held noisy", "Score", "Minutes"], rows,
             "Source: results/1_main_models/R-BO/bayes_opt_trials/trials.json. Scored on 20 separate tuning scenes "
             "(benchmark/scenes_bo.json). Score = mean of held clean and held with noise 0.01.")]


def super_tables():
    systems = [("S1 Expert + A* planner (ceiling)", "planner_expert"), ("S2 R-BO + Kalman + VFH/Bug 2", "bo_t03_kfv_nav"),
               ("S3 R-BO + Kalman + A* planner (final)", "planner_bo_t03_kfv"),
               ("S4 R-BO + Kalman + Driver A (brake)", "driverbrake_driver_A_bo_t03_kfv"),
               ("S5 R-BO + Kalman + planner + camera", "selfnav_cam_planner_bo_t03_kfv")]
    cols = [("Fresh", lambda n: [(ST + "/fresh_scenes", f"{n}_hold5", 1)]),
            ("Hard targets (T2)", lambda n: [(ST + "/benchmark_tiers", f"{n}_hold5", 2)]),
            ("Odd starts (T3)", lambda n: [(ST + "/benchmark_tiers", f"{n}_hold5", 3)]),
            ("Noise 0.01", lambda n: [(ST + "/fresh_scenes", f"{n}_obs_noise0.01_hold5", 1)]),
            ("Noise + delay (T4)", lambda n: [(ST + "/benchmark_tiers", f"{n}_hold5", 4)]),
            ("Obstacles (T10-14, 150)", lambda n: [(ST + "/obstacles", f"{n}_hold5", t) for t in range(10, 15)]),
            ("Box", lambda n: [(ST + "/benchmark_tiers", f"{n}_objects3_hold5", 1)]),
            ("Door", lambda n: [(ST + "/benchmark_tiers", f"{n}_objects4_hold5", 1)])]
    rows = []
    for label, n in systems:
        v = [pooled(items(n)) for _, items in cols]
        stages = [mean(v[0:3]) if all(x is not None for x in v[0:3]) else None,
                  mean(v[3:5]) if all(x is not None for x in v[3:5]) else None,
                  mean(v[5:8]) if all(x is not None for x in v[5:8]) else None]
        rows.append([label, *v, *stages, mean(v) if all(x is not None for x in v) else None])
    return [("Super test: every final system on every test (held 5 s)",
             "The final system (S3) is the best learned system overall. Driver A and the camera lose the most. "
             "Every learned system still fails under noise plus delay.",
             ["System", *[c for c, _ in cols], "Stage 1 accuracy", "Stage 2 robustness", "Stage 3 obstacles", "Overall"],
             rows, "Source: results/3_final_tests/super_test, pooled as in evaluation/super_table.py. "
                   "Overall = mean of the 8 test columns.")]


def obstacle_tables():
    O = "2_obstacles"
    systems = [("Expert", O + "/expert", "expert", None),
               ("Expert + VFH/Bug 2", O + "/expert+VFH_Bug2_nav", "expert_nav", None),
               ("R + 20 ms filter (no navigation)", O + "/R+position_filter_20ms", "vR_ik_filt20noqd", None),
               ("R + 20 ms filter + VFH/Bug 2", O + "/R+position_filter_20ms+VFH_Bug2_nav", "vR_ik_filt20noqd_nav", 1),
               ("R + 20 ms filter + map A*", O + "/R+position_filter_20ms+map_Astar_nav", "vR_ik_filt20noqd_mapnav", 1),
               ("R + Kalman + VFH/Bug 2", O + "/R+Kalman+VFH_Bug2_nav", "vR_ik_kfv_nav", 1),
               ("R-BO + Kalman + VFH/Bug 2", O + "/R-BO+Kalman+VFH_Bug2_nav", "bo_t03_kfv_nav", 1),
               ("Model S (LiDAR inputs) + 20 ms filter", O + "/S_lidar_inputs+position_filter_20ms", "vS_lidar_filt20noqd", 1)]
    rows = []
    for label, sys_, name, clean in systems:
        rows.append([label, rate(sys_, f"{name}_hold5") if clean else None,
                     *[rate(sys_, f"{name}_objects{k}_hold5") if (k > 2 or name in ("expert", "vR_ik_filt20noqd"))
                       else None
                       for k in (1, 2, 3, 4)]])
    rows.append(["R + Kalman + VFH/Bug 2, 120 s limit", None, None, None, None,
                 rate(O + "/door_120s/R+Kalman+VFH_Bug2_nav", "vR_ik_kfv_nav_objects4_hold5")])
    return [("Obstacles: tier 1 scenes with objects added (held 5 s)",
             "Without a navigation layer even the expert fails the box and the door. "
             "With one, the learned arm holds most of them.",
             ["System", "No objects", "Objects 1", "Objects 2", "Box (objects 3)", "Door (objects 4)"], rows,
             "Source: results/2_obstacles. Blank = not run. The super-test sheet has the final systems.")]


def side_tables():
    S = "1_main_models/side_experiments"
    exps = [("DAgger", "One DAgger round: the expert labels the model's own states", "DAgger", "vL_dagger1", None, None),
            ("HG-DAgger", "Human-gated DAgger: the expert takes over only when the model goes wrong",
             "human_gated_DAgger", "vL_hg1", None, None),
            ("Memory-free DAgger, round 1", "DAgger with a memory-free expert as labeller", "memory_free_DAgger",
             "vN_mdag1", None, None),
            ("Memory-free DAgger, round 2", "Second round", "memory_free_DAgger", "vN_mdag2", None, None),
            ("Memory-free expert", "The expert rewritten without hidden memory (not learned)", "memory_free_expert",
             "markov_expert", None, None),
            ("Hold weight 3", "Hold rows weighted x3 in the loss", "Q_hold_weight3", "vQ_hw3", "vQ_hw3_hold5", None),
            ("Hold weight 6", "Hold rows weighted x6 in the loss", "Q_hold_weight6", "vQ_hw6", "vQ_hw6_hold5", None),
            ("Relative target", "R with the target given relative to the robot", "S_relative_target", None,
             "vS_rel_hold5", "vS_rel_hold5"),
            ("Wide targets", "Relative target + wide data, including unreachable targets", "S_wide_targets", None,
             "vS_wide_hold5", "vS_wide_hold5")]
    rows = [[label, what, rate(f"{S}/{d}", r) if r else None, rate(f"{S}/{d}", h) if h else None,
             rate(f"{S}/{d}", t6, 6) if t6 else None] for label, what, d, r, h, t6 in exps]
    rows.append(["Model R (for comparison)", "", rate("1_main_models/R", "vR_ik"),
                 rate("1_main_models/R", "vR_ik_hold5"), rate("1_main_models/R", "vR_ik_hold5", 6)])
    return [("Side experiments that did not beat R",
             "DAgger failed because the expert's hidden memory gives contradictory labels. "
             "The other ideas did not beat model R.",
             ["Experiment", "What changed", "Tier 1 reached", "Tier 1 held 5 s", "Tier 6 held 5 s"], rows,
             "Source: results/1_main_models/side_experiments. Blank = not run.")]


def noise_tables():
    E, R = "1_main_models/expert", "1_main_models/R"
    dist = [("None (clean)", lambda s, n: (s, n + "_hold5", 1)),
            ("Physics (tier 5)", lambda s, n: (s, n + "_hold5", 5)),
            ("Sensor noise 0.005", lambda s, n: (F + "/" + s.split("/")[-1], n + "_obs_noise0.005_hold5", 1)),
            ("Sensor noise 0.01", lambda s, n: (F + "/" + s.split("/")[-1], n + "_obs_noise0.01_hold5", 1)),
            ("Motor (torque) noise 2", lambda s, n: (F + "/" + s.split("/")[-1], n + "_torque_noise2_hold5", 1)),
            ("Delay 2 steps", lambda s, n: (F + "/" + s.split("/")[-1], n + "_delay_steps2_hold5", 1)),
            ("Payload 0.5 kg", lambda s, n: (F + "/" + s.split("/")[-1], n + "_payload_kg0.5_hold5", 1)),
            ("Everything mixed (tier 7)", lambda s, n: (s, n + "_hold5", 7))]
    gap = []
    for label, where in dist:
        e, r = rate(*where(E, "expert")), rate(*where(R, "vR_ik"))
        kal = where(F + "/R+Kalman", "vR_ik_kfv") if label not in ("Motor (torque) noise 2", "Delay 2 steps",
                                                                   "Payload 0.5 kg") else None
        gap.append([label, e, r, rate(*kal) if kal else None,
                    Pct(e - r) if e is not None and r is not None else None])
    gap_table = ("Disturbances: expert vs model R (tier 1, held 5 s)",
                 "R matches the expert when the physics change. Sensor noise, delay and payload break it because "
                 "the expert uses feedback and R copies.",
                 ["Disturbance", "Expert held", "Model R held", "R + Kalman held", "Gap expert - R (points)"], gap,
                 "Source: results/1_main_models/expert, R and noise_and_delay/filters_and_robust_models. "
                 "Blank = not run.")

    diag = [("Expert", "expert", "The template's controller", "expert"),
            ("Model K", "vK_kick", "Stage A final model", "K"),
            ("K-noisy", "vL_disturbed", "K + recordings with sensor noise and delay", "L"),
            ("K-filtered", "vM_smooth", "K-noisy + low-pass filter on arm speeds and history", "M"),
            ("K-history-filter", "vM2_smooth_hist", "Filter on history inputs only", "M2"),
            ("K-clean-labels", "vM2_noise0", "Disturbed recordings made without sensor noise", "M2_noise0"),
            ("K-delay", "vN_delay", "K-clean-labels + 130 delay-only recordings", "N")]
    rows = []
    for label, name, what, d in diag:
        tiers = [rate(f"{KD}/tiers/{d}", name, t) if (f"{KD}/tiers/{d}", name, t) in INDEX else None
                 for t in (1, 2, 3, 4)]
        if d == "K":
            tiers[3] = rate("1_main_models/K", "vK_kick", 4)
        rows.append([label, name, what, *tiers,
                     rate(f"{KD}/noise_only/{d}", name, 4) if d != "expert" else None,
                     rate(f"{KD}/delay_only/{d}", name, 4) if d not in ("expert", "M") else None])
    rows.append(["Domain randomisation (David)", "vM_dr", "Recordings with random noise, delay, friction and mass",
                 rate(F + "/domain_randomisation", "vM_dr"), None, None, rate(F + "/domain_randomisation", "vM_dr", 4),
                 rate(F + "/domain_randomisation", "vM_dr_noiseonly", 4), None])
    rows.append(["Domain randomisation + DAgger (David)", "vO_both", "Domain randomisation + both DAgger rounds",
                 rate(F + "/domain_randomisation+DAgger", "vO_both"), None, None,
                 rate(F + "/domain_randomisation+DAgger", "vO_both", 4), None, None])
    diag_table = ("K diagnosis: which disturbance breaks the model? (reached, 50 scenes)",
                  "K fails tier 4 mainly through delay, not noise. Clean labels and delay recordings help, "
                  "but filtering the arm speeds adds a lag that destabilises the arm.",
                  ["Model", "File", "What changed", "Tier 1", "Tier 2", "Tier 3", "Tier 4 (noise + delay)",
                   "Noise only", "Delay only"], rows,
                  "Source: results/1_main_models/noise_and_delay/K_diagnosis (Muneeb's laptop, where K scores tier 1 "
                  "R 31 instead of 36). K tier 4 is from results/1_main_models/K. Blank = not run.")

    conds = [("Clean", None), ("Noise 0.005", "_obs_noise0.005"), ("Noise 0.01", "_obs_noise0.01"),
             ("Delay 2 steps", "_delay_steps2"), ("Torque noise 2", "_torque_noise2"), ("Payload 0.5 kg", "_payload_kg0.5")]
    fixes = [("Model R, no fix", F + "/R", "vR_ik", (R, "vR_ik")),
             ("Position filter 10 ms", F + "/R+position_filter_10ms", "vR_ik_filt10noqd", None),
             ("Position filter 20 ms", F + "/R+position_filter_20ms", "vR_ik_filt20noqd", None),
             ("Base-only filter 20 ms", F + "/R+base_filter_20ms", "vR_ik_filt20base", None),
             ("Kalman filter (positions and speeds)", F + "/R+Kalman", "vR_ik_kfv", None),
             ("Kalman filter (positions only)", F + "/R+Kalman_no_speeds", "vR_ik_kf", None),
             ("R-BO + 29 ms position filter", "1_main_models/R-BO", "bo_t03_filt29noqd", None),
             ("Ensemble of 3 seeds", F + "/R_ensemble_of_3_seeds", "ens_vR_ik+vR_ik_s1+vR_ik_s2", None),
             ("Ensemble + 10 ms filter", F + "/R_ensemble_of_3_seeds+position_filter_10ms",
              "ens_vR_ik+vR_ik_s1+vR_ik_s2_filt10noqd", None),
             ("Weight decay", F + "/weight_decay", "vU_wd", None),
             ("Dropout", F + "/dropout", "vU_drop", None),
             ("Noise augmentation", F + "/noise_augmentation", "vT_aug", None),
             ("Noise augmentation + messy recordings", F + "/noise_augmentation+messy_recordings", "vT_robust", None),
             ("Expert (for comparison)", F + "/expert", "expert", (E, "expert"))]
    rows = []
    for label, sys_, name, clean_at in fixes:
        row = [label]
        for _, suffix in conds:
            if suffix is None:
                s, n = clean_at or (sys_, name)
                key = (s, n + "_hold5", 1)
            else:
                key = (sys_, name + suffix + "_hold5", 1)
            row.append(rate(*key) if key in INDEX else None)
        rows.append(row)
    fix_table = ("Fixing sensor noise and delay on model R (tier 1, held 5 s)",
                 "Filtering positions, not speeds, and the Kalman filter recover R under noise. "
                 "No fix solves the 2-step delay or the payload.",
                 ["Fix", *[c for c, _ in conds]], rows,
                 "Source: results/1_main_models/noise_and_delay/filters_and_robust_models, R and R-BO (seed 0). "
                 "Blank = not run.")
    return [gap_table, diag_table, fix_table]


COMPARISONS = [("Test tiers", tier_tables, [16, 34, 30, 30, 30, 10, 10, 10, 10]),
               ("Headline", headline_tables, [40, 10, 10, 10, 10, 10, 10, 10, 10]),
               ("Ladder", ladder_tables, [12, 52, 10, 10]),
               ("Arm swap", arm_swap_tables, [32, 10, 10, 10, 10]),
               ("Memory models", memory_tables, [34, 38, 8, 12, 12]),
               ("Input ablation", ablation_tables, [32, 10, 14]),
               ("PCA", pca_tables, [28, 12, 10]),
               ("Other learners", learner_tables, [30, 10, 8, 14, 12, 32]),
               ("Bayesian optimisation", bo_tables, [26, 12, 8, 12, 10, 9, 10, 10, 9, 9]),
               ("Super test", super_tables, [38, 9, 10, 10, 10, 10, 12, 8, 8, 10, 11, 11, 9]),
               ("Obstacles", obstacle_tables, [38, 11, 11, 11, 12, 12]),
               ("Side experiments", side_tables, [28, 52, 12, 12, 12]),
               ("Noise and delay", noise_tables, [38, 18, 44, 10, 10, 10, 14, 10, 10])]


def block_rows(tables):
    """The styled rows of one comparison sheet: each table as title, 'say it' line, header, rows, source."""
    out = []
    for title, say, header, rows, source in tables:
        out += [[(title, 4)], [("Say it like this: " + say, 5)], [(h, 1) for h in header]]
        for row in rows:
            out.append([(v, 2 if isinstance(v, Pct) else 7 if isinstance(v, str) else 0) for v in row])
        out += [[(source, 6)], []]
    return out


# --- minimal .xlsx writer --------------------------------------------------------------------------
# Cell styles (index into cellXfs in STYLES): 0 plain, 1 header, 2 percent, 3 two decimals,
# 4 table title, 5 'say it like this' line, 6 source note, 7 wrapped text.
STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.00"/></numFmts>
<fonts count="5"><font><sz val="10"/><name val="Calibri"/></font>
<font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>
<font><b/><sz val="12"/><color rgb="FF1F3A5F"/><name val="Calibri"/></font>
<font><i/><sz val="10"/><color rgb="FF1F3A5F"/><name val="Calibri"/></font>
<font><sz val="9"/><color rgb="FF5B6676"/><name val="Calibri"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FF1F3A5F"/></patternFill></fill></fills>
<borders count="1"><border/></borders>
<cellStyleXfs count="1"><xf/></cellStyleXfs>
<cellXfs count="8"><xf/>
<xf fontId="1" fillId="2" applyFont="1" applyFill="1"><alignment wrapText="1" vertical="center"/></xf>
<xf numFmtId="9" applyNumberFormat="1"/>
<xf numFmtId="164" applyNumberFormat="1"/>
<xf fontId="2" applyFont="1"/>
<xf fontId="3" applyFont="1"/>
<xf fontId="4" applyFont="1"/>
<xf applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def col_letter(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def cell(ref, value, style=0):
    st = f' s="{style}"' if style else ""
    if isinstance(value, float) and not math.isfinite(value):
        value = None                                  # e.g. mean time to reach when nothing reached
    if value is None:
        return f'<c r="{ref}"{st}/>'
    if isinstance(value, bool):
        value = str(value)
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"{st}><v>{value}</v></c>'
    return f'<c r="{ref}"{st} t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def sheet_xml(header, rows, styles, widths):
    """One worksheet: a navy header row, frozen and filterable, then the rows."""
    lines = [f'<row r="1">' + "".join(cell(f"{col_letter(c)}1", h, 1) for c, h in enumerate(header)) + "</row>"]
    for r, row in enumerate(rows, start=2):
        lines.append(f'<row r="{r}">' + "".join(
            cell(f"{col_letter(c)}{r}", v, styles.get(c, 0)) for c, v in enumerate(row)) + "</row>")
    last = f"{col_letter(len(header) - 1)}{len(rows) + 1}"
    cols = "".join(f'<col min="{c + 1}" max="{c + 1}" width="{w}" customWidth="1"/>' for c, w in enumerate(widths))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/>'
            '</sheetView></sheetViews>'
            f'<cols>{cols}</cols><sheetData>{"".join(lines)}</sheetData>'
            f'<autoFilter ref="A1:{last}"/></worksheet>')


def block_sheet_xml(styled_rows, widths):
    """One comparison worksheet from rows of (value, style) pairs."""
    lines = []
    for r, row in enumerate(styled_rows, start=1):
        lines.append(f'<row r="{r}">' + "".join(
            cell(f"{col_letter(c)}{r}", v, st) for c, (v, st) in enumerate(row)) + "</row>")
    cols = "".join(f'<col min="{c + 1}" max="{c + 1}" width="{w}" customWidth="1"/>' for c, w in enumerate(widths))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
            f'<cols>{cols}</cols><sheetData>{"".join(lines)}</sheetData></worksheet>')


def write_xlsx(path, sheets):
    """sheets: list of (name, worksheet xml, has a filter row)."""
    n = len(sheets)
    overrides = "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" ContentType="application/'
                        'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(n))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/'
                   'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/'
                   f'vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>{overrides}</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                   'relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        names = "".join(f'<sheet name="{escape(name)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
                        for i, (name, _, _) in enumerate(sheets))
        filters = "".join(f'<definedName name="_xlnm._FilterDatabase" localSheetId="{i}" hidden="1">'
                          f"'{name}'!A1</definedName>" for i, (name, _, filt) in enumerate(sheets) if filt)
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   f'<sheets>{names}</sheets><definedNames>{filters}</definedNames></workbook>')
        rels = "".join(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/'
                       f'2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>' for i in range(n))
        rels += (f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/officeDocument/'
                 '2006/relationships/styles" Target="styles.xml"/>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}'
                   '</Relationships>')
        z.writestr("xl/styles.xml", STYLES)
        for i, (_, xml, _) in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", xml)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RESULTS, "all_results.xlsx"))
    args = ap.parse_args()

    styles = {COLUMNS.index("Success"): 2, **{COLUMNS.index(c): 3 for c in
              ("Median final error (cm)", "Mean time to reach (s)", "Torque jerk", "Wheel jerk")}}
    widths = [34, 30, 18, 10, 6, 7, 8, *[9] * len(OUTCOMES), 11, 11, 9, 9, 10, 70]
    build_index()
    sheets = []
    for name, tables, col_widths in COMPARISONS:
        built = tables()
        sheets.append((name, block_sheet_xml(block_rows(built), col_widths), False))
        print(f"{name:24s} {sum(len(t[3]) for t in built):4d} rows")
    for name, folder, skip in RAW_SHEETS:
        rows = raw_rows(folder, skip)
        sheets.append((name, sheet_xml(COLUMNS, rows, styles, widths), True))
        print(f"{name:24s} {len(rows):4d} rows")
    rows = scene_rows()
    sheets.append(("All scenes", sheet_xml(SCENE_COLUMNS, rows, {}, [34, 30, 18, 10, 6, 8, 10, 16, 8, 10, 10, 9, 10,
                                                                   10, 10, 11, 70]), True))
    print(f"{'All scenes':24s} {len(rows):4d} rows (one per drive)")
    write_xlsx(args.out, sheets)
    print("wrote", args.out)
    if MISSING:
        print(f"{len(MISSING)} comparison cells have no result file (left blank):")
        for m in MISSING:
            print("  ", m)


if __name__ == "__main__":
    main()
