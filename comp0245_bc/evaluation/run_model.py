"""Run or watch any model from the report by its report name (K, P, R, BO, S1-S5, ModelA ...).

    python3 evaluation/run_model.py list            # every name and what it is
    python3 evaluation/run_model.py run             # asks which model and how many scenes
    python3 evaluation/run_model.py run S3 10       # the final system on the first 10 of its scenes
    python3 evaluation/run_model.py watch S3        # open the simulator window on its first scene
    python3 evaluation/run_model.py watch S3 55     # ... or on a chosen scene number

'run' uses the same policy, scenes and settings as the saved result, writes the rerun to
results/reproduced/run/<NAME>/, and compares it scene by scene with the saved result.
Normally called through reproduce.sh (bash reproduce.sh run / list / watch).
"""
from _paths import ROOT  # noqa: F401  (project folder; lets the code folders import each other)
import csv
import glob
import os
import subprocess
import sys

R = "results/"
T1 = ["--tiers", "1"]
FRESH = ["--scenes", "benchmark/scenes_fresh.json", "--tiers", "1"]
OBST = ["--scenes", "scenes_obstacles/obstacles_test.json", "--tiers", "10", "11", "12", "13", "14", "--max-time", "90"]
ST = R + "3_final_tests/super_test/fresh_scenes/"
AB = R + "2_obstacles/abdullah_bases/stage3_planner_and_driver_A/"

# name: (what it is, report figure for all scenes, policy, scene options, hold 5 s?, saved result file(s))
MODELS = {
    "expert": ("the template's expert controller (the ceiling)", "98% held on T1 and on fresh scenes",
               "expert", T1, True, [R + "1_main_models/expert/tier1/expert_hold5_tier1.csv"]),
    "v0": ("plain copy of the expert", "0% reached", "trained/mlp_v0.npz", T1, False, [R + "0_old_tests/v0/mlp_v0_tier1.csv"]),
    "v1": ("v0 + gravity split", "0% reached", "trained/mlp_v1.npz", T1, False, [R + "0_old_tests/v1/mlp_v1_tier1.csv"]),
    "v2": ("v1 + history inputs, nudged data", "2% reached", "trained/mlp_v2_hist.npz", T1, False, [R + "0_old_tests/v2_history/mlp_v2_hist_tier1.csv"]),
    "A": ("v2 + parking-spot inputs", "22% reached", "trained/vA_goal.npz", T1, False, [R + "0_old_tests/A/vA_goal_tier1.csv"]),
    "D": ("A + separate arm and wheel networks", "40% reached", "trained/vD_all.npz", T1, False, [R + "0_old_tests/D/vD_all_tier1.csv"]),
    "E": ("D + error-average inputs", "64% reached", "trained/vE_integral.npz", T1, False, [R + "0_old_tests/E/vE_integral_tier1.csv"]),
    "K": ("model K: E + kicked data", "20% held (72% reached)", "trained/vK_kick.npz", T1, True, [R + "1_main_models/K/tier1/vK_kick_hold5_tier1.csv"]),
    "P": ("model P: K + hold and extreme-height data", "34% held", "trained/vP_hold.npz", T1, True, [R + "1_main_models/P/vP_hold_hold5_tier1.csv"]),
    "R": ("model R: P + IK inputs", "92% held", "trained/vR_ik.npz", T1, True, [R + "1_main_models/R/tier1/vR_ik_hold5_tier1.csv"]),
    "BO": ("the BO network with the Kalman filter (the final arm)", "100% held (98% with noise 0.01)",
           "trained/bo_t03.npz:kfv", T1, True, [R + "1_main_models/R-BO/bo_t03_kfv_hold5_tier1.csv"]),
    "S1": ("super test S1: expert arm + A* planner (ceiling)", "100% held on fresh scenes",
           "planner", FRESH, True, [ST + "planner_expert_hold5_tier1.csv"]),
    "S2": ("super test S2: BO network + Kalman + VFH/Bug 2", "90% held on fresh scenes",
           "trained/bo_t03.npz:kfv:nav", FRESH, True, [ST + "bo_t03_kfv_nav_hold5_tier1.csv"]),
    "S3": ("super test S3, THE FINAL SYSTEM: BO network + Kalman + A* planner", "90% held on fresh scenes",
           "planner:trained/bo_t03.npz:kfv", FRESH, True, [ST + "planner_bo_t03_kfv_hold5_tier1.csv"]),
    "S4": ("super test S4: BO network + Kalman + Driver A (model A) + brake", "72% held on fresh scenes",
           "driverbrake:trained/driver_A.npz:trained/bo_t03.npz:kfv", FRESH, True, [ST + "driverbrake_driver_A_bo_t03_kfv_hold5_tier1.csv"]),
    "S5": ("super test S5: S3 with no true position (odometry + camera)", "76% held on fresh scenes",
           "cam:planner:trained/bo_t03.npz:kfv", FRESH, True, [ST + "selfnav_cam_planner_bo_t03_kfv_hold5_tier1.csv"]),
    "ModelA": ("model A, the learned driver + brake, with the expert arm", "144 of 150 obstacle drives held",
               "driverbrake:trained/driver_A.npz:expert", OBST, True,
               [AB + f"brake_driver_A_expert_hold5_tier{t}.csv" for t in range(10, 15)]),
}
# The eight super-test conditions (as in scripts/run_super_server.sh), for S1-S5:
# name: (what, scene options, --set values, super_test subfolder, result-file suffixes)
TESTS = {
    "fresh": ("50 fresh scenes", FRESH, [], "fresh_scenes", ["_hold5_tier1"]),
    "hard": ("hard targets (T2)", ["--tiers", "2"], [], "benchmark_tiers", ["_hold5_tier2"]),
    "awkward": ("awkward starts (T3)", ["--tiers", "3"], [], "benchmark_tiers", ["_hold5_tier3"]),
    "noise": ("sensor noise 0.01 on the fresh scenes", FRESH, ["obs_noise=0.01"], "fresh_scenes", ["_obs_noise0.01_hold5_tier1"]),
    "noisedelay": ("noise 0.005-0.02 and a 1-4 ms delay (T4)", ["--tiers", "4"], [], "benchmark_tiers", ["_hold5_tier4"]),
    "obstacles": ("the 150 obstacle scenes", OBST, [], "obstacles", [f"_hold5_tier{t}" for t in range(10, 15)]),
    "box": ("a 0.5 m box in the path (T1)", T1, ["objects=3"], "benchmark_tiers", ["_objects3_hold5_tier1"]),
    "door": ("a 1 m wall with a doorway (T1)", T1, ["objects=4"], "benchmark_tiers", ["_objects4_hold5_tier1"]),
}
SUPER = ["S1", "S2", "S3", "S4", "S5"]

GROUPS = [("Arm models, 50 benchmark scenes (T1)", ["expert", "v0", "v1", "v2", "A", "D", "E", "K", "P", "R", "BO"]),
          ("Super-test systems, 50 fresh scenes", ["S1", "S2", "S3", "S4", "S5"]),
          ("Obstacles, 150 obstacle scenes", ["ModelA"])]
ALIASES = {k.lower(): k for k in MODELS} | {"modela": "ModelA", "model_a": "ModelA", "drivera": "ModelA", "final": "S3"}


def find(name):
    key = ALIASES.get(name.strip().lower())
    if key is None:
        sys.exit(f"unknown model '{name}'. Names: " + ", ".join(MODELS))
    return key


def show_list():
    for title, names in GROUPS:
        print(f"\n{title}")
        for n in names:
            what, figure = MODELS[n][:2]
            print(f"  {n:7s} {what}   [report: {figure}]")
    print("\nSuper-test conditions for S1-S5 (default: fresh):")
    for t, v in TESTS.items():
        print(f"  {t:11s} {v[0]}")
    print("\nRun one:   bash reproduce.sh run S3 10          (name, how many scenes; default 10)")
    print("           bash reproduce.sh run S3 10 door     (a super-test condition, for S1-S5)")
    print("Watch one: bash reproduce.sh watch S3           (or: watch S3 door, watch S3 55 for scene 55)")


def ask(prompt, default=None):
    try:
        got = input(prompt).strip()
    except EOFError:
        got = ""
    return got or default


def outcomes(files, n_first):
    """{(file number, scene index): outcome} for the first n scenes of each saved or rerun file."""
    out = {}
    for i, f in enumerate(files):
        with open(f) as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: int(r["index"]))[:n_first]
        for r in rows:
            out[(i, r["index"])] = r["outcome"]
    return out


def super_test(key, test):
    """Scene options and saved result files of super-test condition `test` for system `key` (S1-S5)."""
    if test not in TESTS:
        sys.exit(f"unknown test '{test}'. Tests: " + ", ".join(TESTS))
    if key not in SUPER:
        sys.exit(f"the super-test conditions are for S1-S5; {key} runs on its own scenes (bash reproduce.sh run {key})")
    what, scenes, sets, folder, suffixes = TESTS[test]
    prefix = os.path.basename(MODELS[key][5][0]).split("_hold5_tier1")[0]
    saved = [f"{R}3_final_tests/super_test/{folder}/{prefix}{s}.csv" for s in suffixes]
    return what, scenes + (["--set"] + sets if sets else []), saved


def run(name, n, test=None):
    key = find(name)
    what, figure, policy, scenes, hold, saved = MODELS[key]
    if test:
        cond, scenes, saved = super_test(key, test)
        figure = f"see the super-test table: bash reproduce.sh check   ({cond})"
    out = os.path.join("results", "reproduced", "run", key + (f"_{test}" if test else ""))
    for f in glob.glob(os.path.join(out, "*")):
        os.remove(f)
    per = f"first {n} scenes" + (" of each of the 5 obstacle tiers" if "10" in scenes else "")
    print(f"\n{key}: {what}" + (f"\ntest: {TESTS[test][0]}" if test else "") +
          f"\nrunning on the {per}, success = {'held 5 s within 1 cm' if hold else 'reached 1 cm'}\n")
    env = dict(os.environ, ROBOENV_HEADLESS="1", NAV_STUCK="0", OBJ_TIME="90")
    cmd = [sys.executable, "evaluation/evaluate.py", "--policy", policy, "--n", str(n), "--plots", "0",
           "--workers", os.environ.get("WORKERS", "4"), "--out", out] + scenes + (["--hold", "5"] if hold else [])
    print("command: python3 " + " ".join(cmd[1:]) + "\n")
    subprocess.run(cmd, env=env, check=True)
    new = sorted(f for f in glob.glob(os.path.join(out, "*_tier*.csv")))
    new.sort(key=lambda f: int(f.rsplit("_tier", 1)[1].split(".")[0]))
    s, r = outcomes(saved, n), outcomes(new, n)
    same = sum(s.get(k) == v for k, v in r.items())
    ok_s, ok_r = sum(v == "reached" for v in s.values()), sum(v == "reached" for v in r.values())
    word = "held" if hold else "reached"
    print(f"\n{key} {word}: rerun {ok_r}/{len(r)}, saved result {ok_s}/{len(s)} on the same scenes, "
          f"same outcome in {same}/{len(r)} scenes")
    print(f"report (all scenes): {figure}")
    print(f"rerun saved in {out}/  (saved result: {saved[0] if len(saved) == 1 else saved[0].replace('_tier10', '_tier10..14')})")


def watch(name, scene_index=None, test=None):
    """One drive in the PyBullet window, run by the same code as 'run' (core/rollout.py) with the window on."""
    import json
    key = find(name)
    policy, scenes = MODELS[key][2], MODELS[key][3]
    sets = []
    if test:
        _, scenes, _ = super_test(key, test)
        if "--set" in scenes:
            sets = scenes[scenes.index("--set") + 1:]
    scene_file = scenes[scenes.index("--scenes") + 1] if "--scenes" in scenes else "benchmark/scenes.json"
    tier = int(scenes[scenes.index("--tiers") + 1])
    if "10" in scenes:
        tier = 12                                     # obstacles: a cluttered scene, so the obstacles show
    with open(scene_file) as fh:
        pool = [s for s in json.load(fh)["scenes"] if s["tier"] == tier]
    first = min(s["index"] for s in pool)            # fresh scenes are numbered from 50
    idx = first if scene_index is None else int(scene_index)
    match = [s for s in pool if s["index"] == idx]
    if not match:
        sys.exit(f"no scene {idx}; scenes in this set: {first}-{max(s['index'] for s in pool)}")
    scene = dict(match[0])
    for kv in sets:                                   # the same disturbance evaluate.py --set adds
        k, v = kv.split("=")
        scene["disturbance"] = dict(scene.get("disturbance") or {}, **{k: (int(v) if k == "delay_steps" else float(v))})
    max_time = 90.0 if tier >= 10 else 40.0
    os.environ.update(NAV_STUCK="0", OBJ_TIME="90", ROBOENV_HEADLESS="0")
    from rollout import make_policy, rollout
    print(f"\n{key}: {MODELS[key][0]}" + (f"\ntest: {TESTS[test][0]}" if test else "") +
          f"\nscene {idx} of {scene_file} (tier {tier}). The PyBullet window opens now;")
    print("the run ends by itself after the 5 s hold or the time limit. In WSL first run: export LIBGL_ALWAYS_SOFTWARE=1\n")
    metrics, _ = rollout(make_policy(policy), scene, max_time=max_time, use_gui=True, hold_s=5.0)
    out = metrics.get("outcome")
    print(f"{key}, scene {idx}: {'held 5 s within 1 cm (success)' if out == 'reached' else out}, "
          f"final error {metrics.get('final_err_cm', float('nan')):.2f} cm")


def main():
    os.chdir(ROOT)
    a = sys.argv[1:]
    mode = a[0] if a else "list"
    if mode == "list":
        show_list()
    elif mode == "run":
        if len(a) < 2:
            show_list()
            name = ask("\nModel name: ")
            if not name:
                sys.exit("no model given")
            n = ask("How many scenes? [10]: ", "10")
            test = None
            if find(name) in SUPER:
                test = ask(f"Which test? ({', '.join(TESTS)}) [fresh]: ", "fresh")
            run(name, int(n), None if test == "fresh" else test)
        else:   # run NAME [N] [TEST], in any order after the name
            rest = a[2:]
            n = next((x for x in rest if x.isdigit()), "10")
            test = next((x.lower() for x in rest if not x.isdigit()), None)
            run(a[1], int(n), None if test == "fresh" else test)
    elif mode == "watch":   # watch NAME [TEST] [SCENE]
        name = a[1] if len(a) > 1 else ask("Model name: ")
        rest = a[2:]
        scene = next((x for x in rest if x.isdigit()), None)
        test = next((x.lower() for x in rest if not x.isdigit()), None)
        watch(name, scene, None if test == "fresh" else test)
    else:
        sys.exit("usage: run_model.py [list | run [NAME [N] [TEST]] | watch NAME [TEST] [SCENE]]")


if __name__ == "__main__":
    main()
