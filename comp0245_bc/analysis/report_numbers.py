"""Every number quoted in the report, read straight from the saved per-scene result files.

    python3 analysis/report_numbers.py

For each table or claim it prints the value the report gives, the value counted now from the result
file, and the file it came from. A scene counts as a success when its 'outcome' column is 'reached'
(in a hold-5 s run that means the hand stayed within 1 cm for 5 s). Needs no simulator and no data/.
Exit code 1 if any number no longer matches.
"""
from _paths import ROOT  # noqa: F401  (project folder; lets the code folders import each other)
import csv
import os
import subprocess
import sys

RES = os.path.join(ROOT, "results")


def count(*files):
    """(successes, scenes) summed over one or more per-scene CSV files."""
    ok = tot = 0
    for f in files:
        with open(os.path.join(RES, f)) as fh:
            rows = list(csv.DictReader(fh))
        ok += sum(r["outcome"] == "reached" for r in rows)
        tot += len(rows)
    return ok, tot


def tiers(prefix, suffix=".csv"):
    """The five obstacle tiers (10 clear, 11 sparse, 12 cluttered, 13 gap, 14 blocked) of one system."""
    return [f"{prefix}_tier{t}{suffix}" for t in range(10, 15)]


A = "2_obstacles/abdullah_bases/"
# (report part, what, value in the report, how the report states it, files)
CHECKS = [
    ("Choosing a model", "linear regression, T1 reached", 0, "n", ["1_main_models/other_learners/linear_regression/cmp_linear_tier1.csv"]),
    ("Choosing a model", "random forest, T1 reached (20 scenes)", 0, "n", ["1_main_models/other_learners/random_forest/cmp_forest_tier1.csv"]),
    ("Data collection", "DAgger on K, T1 reached", 1, "n", ["1_main_models/side_experiments/DAgger/vL_dagger1_tier1.csv"]),
    ("Model K", "v0 plain copy, T1 reached %", 0, "%", ["0_old_tests/v0/mlp_v0_tier1.csv"]),
    ("Model K", "+ gravity split (v1), T1 reached %", 0, "%", ["0_old_tests/v1/mlp_v1_tier1.csv"]),
    ("Model K", "+ history, nudged data, T1 reached %", 2, "%", ["0_old_tests/v2_history/mlp_v2_hist_tier1.csv"]),
    ("Model K", "+ parking spot (A), T1 reached %", 22, "%", ["0_old_tests/A/vA_goal_tier1.csv"]),
    ("Model K", "+ separate networks (D), T1 reached %", 40, "%", ["0_old_tests/D/vD_all_tier1.csv"]),
    ("Model K", "+ error average (E), T1 reached %", 64, "%", ["0_old_tests/E/vE_integral_tier1.csv"]),
    ("Model K", "K, T1 reached %", 72, "%", ["1_main_models/K/tier1/vK_kick_tier1.csv"]),
    ("Noise and delay", "K on T4 (noise + delay), reached %", 0, "%", ["1_main_models/K/tier4/vK_kick_tier4.csv"]),
    ("Noise and delay", "expert on T4, reached %", 82, "%", ["1_main_models/expert/tier4/expert_tier4.csv"]),
    ("Noise and delay", "K, T4 noise only, reached", 8, "n", ["1_main_models/noise_and_delay/K_diagnosis/noise_only/K/vK_kick_tier4.csv"]),
    ("Noise and delay", "K, T4 delay only, reached", 1, "n", ["1_main_models/noise_and_delay/K_diagnosis/delay_only/K/vK_kick_tier4.csv"]),
    ("Noise and delay", "K-clean-labels, T4 noise only, reached", 25, "n", ["1_main_models/noise_and_delay/K_diagnosis/noise_only/M2_noise0/vM2_noise0_tier4.csv"]),
    ("Model R", "K, T1 held %", 20, "%", ["1_main_models/K/tier1/vK_kick_hold5_tier1.csv"]),
    ("Model R", "K, fresh held %", 30, "%", ["3_final_tests/fresh_scenes/K/vK_kick_hold5_tier1.csv"]),
    ("Model R", "P, T1 reached %", 80, "%", ["1_main_models/P/vP_hold_tier1.csv"]),
    ("Model R", "P, T1 held %", 34, "%", ["1_main_models/P/vP_hold_hold5_tier1.csv"]),
    ("Model R", "P, fresh held %", 40, "%", ["3_final_tests/fresh_scenes/P/vP_hold_hold5_tier1.csv"]),
    ("Model R", "expert arm + P base, T1 held %", 100, "%", ["1_main_models/arm_swap/expert_arm+P_base/vP_hold_mlp_wheels_hold5_tier1.csv"]),
    ("Model R", "P arm + expert base, T1 held %", 36, "%", ["1_main_models/arm_swap/P_arm+expert_base/vP_hold_mlp_arm_hold5_tier1.csv"]),
    ("Model R", "R, T1 reached %", 96, "%", ["1_main_models/R/tier1/vR_ik_tier1.csv"]),
    ("Model R", "R, T1 held %", 92, "%", ["1_main_models/R/tier1/vR_ik_hold5_tier1.csv"]),
    ("Model R", "R seed 0, fresh held %", 86, "%", ["3_final_tests/fresh_scenes/R/vR_ik_hold5_tier1.csv"]),
    ("Model R", "R seed 1, fresh held %", 90, "%", ["3_final_tests/fresh_scenes/R/vR_ik_s1_hold5_tier1.csv"]),
    ("Model R", "R seed 2, fresh held %", 84, "%", ["3_final_tests/fresh_scenes/R/vR_ik_s2_hold5_tier1.csv"]),
    ("Model R", "R without parking-spot inputs, T1 held %", 46, "%", ["1_main_models/ablations/R_no_parking_spot/abl_no_goal_hold5_tier1.csv"]),
    ("Model R", "R without current IK direction, T1 held %", 58, "%", ["1_main_models/ablations/R_no_IK_direction/abl_no_iknow_hold5_tier1.csv"]),
    ("Model R", "R without history, T1 held %", 60, "%", ["1_main_models/ablations/R_no_history/abl_no_history_hold5_tier1.csv"]),
    ("BO + Kalman", "BO + 29 ms filter, T1 held %", 98, "%", ["1_main_models/R-BO/bo_t03_filt29noqd_hold5_tier1.csv"]),
    ("BO + Kalman", "BO + 29 ms filter, noise 0.01, T1 held %", 94, "%", ["1_main_models/R-BO/bo_t03_filt29noqd_obs_noise0.01_hold5_tier1.csv"]),
    ("BO + Kalman", "BO + Kalman filter, T1 held %", 100, "%", ["1_main_models/R-BO/bo_t03_kfv_hold5_tier1.csv"]),
    ("BO + Kalman", "BO + Kalman filter, noise 0.01, T1 held %", 98, "%", ["1_main_models/R-BO/bo_t03_kfv_obs_noise0.01_hold5_tier1.csv"]),
    ("Scenes and obstacles", "expert alone, 150 obstacle drives reached", 38, "n", tiers(A + "before_navigation/expert")),
    ("Scenes and obstacles", "expert alone on box scenes, held %", 0, "%", ["2_obstacles/expert/expert_objects3_hold5_tier1.csv"]),
    ("Scenes and obstacles", "expert alone on door scenes, held %", 0, "%", ["2_obstacles/expert/expert_objects4_hold5_tier1.csv"]),
    ("Expert planner", "A* planner + expert arm, 150 drives held", 149, "n", tiers(A + "stage3_planner_and_driver_A/planner_expert_hold5")),
    ("Model A", "Driver A + brake + expert arm, seed 0, held", 144, "n", tiers(A + "stage3_planner_and_driver_A/brake_driver_A_expert_hold5")),
    ("Model A", "Driver A + brake + expert arm, seed 1, held", 141, "n", tiers(A + "stage3_planner_and_driver_A/brake_driver_A_s1_expert_hold5")),
    ("Mixing parts", "A* planner + R, 150 drives held %", 86, "%", tiers(A + "stage3_planner_and_driver_A/planner_vR_ik_hold5")),
    ("Mixing parts", "Driver A + brake + R, seed 0, held %", 76, "%", tiers(A + "stage5_learned_arms/brake_driver_A_vR_ik_hold5")),
    ("Camera", "Kalman odometry + camera, planner + expert arm, held", 132, "n", [A + "stage6_camera_and_odometry/cam_planner_expert_test.csv"]),
    ("Localisation", "Kalman odometry only, planner + expert arm, held", 0, "n", [A + "stage6_camera_and_odometry/odo_planner_expert_test.csv"]),
    ("Search mode", "search mode, planner + expert arm, held", 55, "n", [A + "stage6_camera_and_odometry/search_planner_expert_test.csv"]),
    ("Final models", "camera, planner + R, held", 100, "n", [A + "stage6_camera_and_odometry/cam_planner_vR_ik_test.csv"]),
    ("Final models", "camera, Driver A + R, seed 0, held", 74, "n", [A + "stage6_camera_and_odometry/cam_driver_A_vR_ik_test.csv"]),
    ("Final models", "camera, Driver A + R, seed 1, held", 91, "n", [A + "stage6_camera_and_odometry/cam_driver_A_s1_vR_ik_test.csv"]),
]


def main():
    bad = 0
    print(f"{'report part':24s} {'what':54s} {'report':>7s} {'files':>7s}  source")
    for where, what, want, kind, files in CHECKS:
        ok, tot = count(*files)
        got = round(100 * ok / tot) if kind == "%" else ok
        unit = "%" if kind == "%" else f"/{tot}"
        flag = "" if got == want else "   <-- DIFFERENT"
        bad += got != want
        src = files[0] if len(files) == 1 else files[0].replace("_tier10", "_tier10..14")
        print(f"{where:24s} {what:54s} {str(want) + unit:>7s} {str(got) + unit:>7s}  {src}{flag}")
    print("\nSuper test (S1-S5), rebuilt by evaluation/super_table.py:\n", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "super_table.py")], check=True)
    print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} report numbers match their result files.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
