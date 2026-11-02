"""Super test, laptop half (1 Nov): S1 expert arm + A* planner, and S5 camera version. Works on Windows, Linux, macOS.

From this folder, inside the RoboEnv pixi shell:
    python run_super_laptop.py              (7 workers; use --workers N to change)
Idempotent: if it stops, run it again and it skips finished parts. Results go to results_super/.
Same rule as the server half: held 5 s within 1 cm, 40 s limit, or 90 s with obstacles.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["planner", "cam:planner:trained/bo_t03.npz:kfv"]
NAMES = {"planner": "planner_expert", "cam:planner:trained/bo_t03.npz:kfv": "selfnav_cam_planner_bo_t03_kfv"}
FRESH, OBST = os.path.join("benchmark", "scenes_fresh.json"), os.path.join("scenes_obstacles", "obstacles_test.json")
TESTS = [  # (out folder, csv suffix, label, extra args)
    ("fresh", "_hold5_tier1", "1 fresh", ["--scenes", FRESH, "--tiers", "1"]),
    ("bench", "_hold5_tier2", "1 hard targets", ["--tiers", "2"]),
    ("bench", "_hold5_tier3", "1 awkward starts", ["--tiers", "3"]),
    ("fresh", "_obs_noise0.01_hold5_tier1", "2 noise 0.01", ["--scenes", FRESH, "--tiers", "1", "--set", "obs_noise=0.01"]),
    ("bench", "_hold5_tier4", "2 noise+delay", ["--tiers", "4"]),
    *[("obst", f"_hold5_tier{t}", f"3 obstacles tier {t}", ["--scenes", OBST, "--tiers", str(t), "--max-time", "90"]) for t in range(10, 15)],
    ("bench", "_objects3_hold5_tier1", "3 box", ["--tiers", "1", "--set", "objects=3"]),
    ("bench", "_objects4_hold5_tier1", "3 door", ["--tiers", "1", "--set", "objects=4"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    env = dict(os.environ, ROBOENV_HEADLESS="1", NAV_STUCK="0", OBJ_TIME="90")
    os.chdir(HERE)
    for spec in SYSTEMS:
        print(f"== {spec}", flush=True)
        for out, suf, label, extra in TESTS:
            folder = os.path.join("results_super", out)
            if os.path.exists(os.path.join(folder, NAMES[spec] + suf + ".csv")):
                continue
            t0 = time.time()
            print(f"  {time.strftime('%H:%M')} {label} ...", flush=True)
            r = subprocess.run([sys.executable, "evaluate.py", "--policy", spec, "--workers", str(a.workers), "--plots", "0",
                                "--hold", "5", "--out", folder, *extra], env=env, capture_output=True, text=True)
            line = next((l for l in r.stdout.splitlines() if "->" in l), (r.stdout + r.stderr)[-400:])
            print(f"    {line.strip()}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("== done. Zip the results_super folder and send it back.", flush=True)


if __name__ == "__main__":
    main()
